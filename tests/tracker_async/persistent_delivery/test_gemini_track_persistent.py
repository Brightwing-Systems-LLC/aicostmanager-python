import asyncio
import json
import os
import threading
import time
import urllib.request
import uuid

import pytest

genai = pytest.importorskip("google.genai")

from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from aicostmanager.usage_utils import get_usage_from_response

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _wait_for_cost_event(aicm_api_key: str, response_id: str, timeout: int = 30):
    headers = {
        "Authorization": f"Bearer {aicm_api_key}",
        "Content-Type": "application/json",
    }
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/api/v1/cost-events/{response_id}",
                headers=headers,
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.load(resp)
                    print(f"Response data: {data}")

                    # Handle both array and single object responses
                    if isinstance(data, list) and len(data) > 0:
                        # Server returns array of cost events
                        first_event = data[0]
                        event_id = first_event.get("event_id") or first_event.get(
                            "uuid"
                        )
                        if event_id:
                            uuid.UUID(str(event_id))
                            return data
                    else:
                        # Handle single object response (fallback)
                        event_id = data.get("event_id") or data.get(
                            "cost_event", {}
                        ).get("event_id")
                        if event_id:
                            uuid.UUID(str(event_id))
                            return data
        except Exception as e:
            print(f"Error checking cost event: {e}")
        time.sleep(1)
    raise AssertionError(f"cost event for {response_id} not found")


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "gemini-2.0-flash"],
)
def test_gemini_track_non_streaming(model, google_api_key, aicm_api_key, tmp_path):
    if not google_api_key:
        pytest.skip("GOOGLE_API_KEY not set in .env file")
    ini = IniManager(str(tmp_path / "ini"))
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE,
        dconfig,
        db_path=str(tmp_path / "gemini_queue.db"),
        poll_interval=0.1,
        batch_interval=0.1,
    )
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = genai.Client(api_key=google_api_key)

        resp = client.models.generate_content(model=model, contents="Say hi")
        response_id = getattr(resp, "id", None) or getattr(resp, "response_id", None)
        if not response_id:
            # Gemini doesn't provide response IDs, generate our own for tracking
            import uuid as _uuid

            response_id = _uuid.uuid4().hex
            print(f"No response_id from Gemini; using generated id: {response_id}")

        print(f"Response ID: {response_id}")
        print(f"Response type: {type(resp)}")
        print(f"Response dir: {dir(resp)[:20]}...")  # First 20 attributes
        usage = get_usage_from_response(resp, "gemini")
        print(f"Usage result: {usage}")
        print(f"Usage type: {type(usage)}")
        asyncio.run(
            tracker.track_async(f"google::{model}", usage, response_id=response_id)
        )
        _wait_for_cost_event(aicm_api_key, response_id)


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "gemini-2.0-flash"],
)
def test_gemini_track_streaming(model, google_api_key, aicm_api_key, tmp_path):
    if not google_api_key:
        pytest.skip("GOOGLE_API_KEY not set in .env file")
    ini = IniManager(str(tmp_path / "ini2"))
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE,
        dconfig,
        db_path=str(tmp_path / "gemini_streaming_queue.db"),
        poll_interval=0.1,
        batch_interval=0.1,
    )
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = genai.Client(api_key=google_api_key)

        response_id = None
        usage_payload = {}

        stream = client.models.generate_content_stream(model=model, contents=["Say hi"])

        # Process streaming events to accumulate usage data
        final_event = None
        for evt in stream:
            final_event = evt
            # Get response_id from the first event that has it
            if response_id is None:
                response_id = getattr(evt, "id", None) or getattr(
                    evt, "response_id", None
                )

            # Extract usage from each event and accumulate
            from aicostmanager.usage_utils import get_streaming_usage_from_response

            up = get_streaming_usage_from_response(evt, "gemini")
            if isinstance(up, dict) and up:
                # Merge usage data (later events may have more complete info)
                usage_payload.update(up)

        if not usage_payload and final_event is not None:
            # Some clients only include usage in the last event; try one more time
            up = get_streaming_usage_from_response(final_event, "gemini")
            if isinstance(up, dict) and up:
                usage_payload.update(up)

        if not usage_payload:
            pytest.skip("No usage returned in streaming events; skipping")

        if not response_id:
            # Gemini doesn't provide response IDs, generate our own for tracking
            import uuid as _uuid

            response_id = _uuid.uuid4().hex
            print(
                f"No response_id from Gemini streaming; using generated id: {response_id}"
            )

        # Track the usage and get the actual response_id that was used
        asyncio.run(
            tracker.track_async(
                f"google::{model}",
                usage_payload,
                response_id=response_id,
            )
        )

        # If no response_id was provided, we need to get it from the persistent delivery
        # The persistent delivery generates its own ID when none is provided
        if response_id is None:
            # Temporarily stop the worker thread to prevent it from processing our message
            tracker.delivery._stop_event.set()
            tracker.delivery._worker.join(timeout=1.0)

            # Query the database to get the response_id
            import sqlite3

            conn = sqlite3.connect(tracker.delivery.db_path)

            # First, let's see what's in the queue
            cursor = conn.execute("SELECT id, payload FROM queue")
            rows = cursor.fetchall()
            print(f"Queue contents: {len(rows)} rows")
            for row in rows:
                print(f"  Row {row[0]}: {row[1][:100]}...")

            # Now try to get the most recent payload
            cursor = conn.execute("SELECT payload FROM queue ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                payload_data = json.loads(row[0])
                print(f"Payload data: {payload_data}")
                # The local storage format has response_id directly in the payload
                if "response_id" in payload_data:
                    response_id = payload_data["response_id"]
                    print(f"Found response_id: {response_id}")
            else:
                print("No rows found in queue")
            conn.close()

            # If we still don't have a response_id, we can't proceed
            if response_id is None:
                pytest.fail("Failed to get response_id from persistent delivery")

            # Restart the worker thread
            tracker.delivery._stop_event.clear()
            tracker.delivery._worker = threading.Thread(
                target=tracker.delivery._run_worker, daemon=True
            )
            tracker.delivery._worker.start()

        print(f"Using response_id: {response_id}")
        _wait_for_cost_event(aicm_api_key, response_id)
