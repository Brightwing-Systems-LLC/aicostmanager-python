import json
import os
import threading
import time
import urllib.request
import uuid

import pytest

openai = pytest.importorskip("openai")

from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from aicostmanager.usage_utils import get_usage_from_response

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _wait_for_cost_event(aicm_api_key: str, response_id: str, timeout: int = 30):
    """Poll the API until a cost event for ``response_id`` is available."""
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


def _make_client(api_key: str):
    return openai.OpenAI(api_key=api_key)


def test_openai_chat_track_non_streaming(aicm_api_key, tmp_path):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in .env file")

    ini = IniManager(str(tmp_path / "ini"))
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE,
        dconfig,
        db_path=str(tmp_path / "openai_chat_queue.db"),
        poll_interval=0.1,
        batch_interval=0.1,
    )
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = _make_client(api_key)

        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say hi"}],
            max_completion_tokens=20,
        )
        response_id = getattr(resp, "id", None)
        usage = get_usage_from_response(resp, "openai_chat")
        tracker.track("openai::gpt-5-mini", usage, response_id=response_id)
        _wait_for_cost_event(aicm_api_key, response_id)


def test_openai_chat_track_streaming(aicm_api_key, tmp_path):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in .env file")

    ini = IniManager(str(tmp_path / "ini2"))
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE,
        dconfig,
        db_path=str(tmp_path / "openai_chat_streaming_queue.db"),
        poll_interval=0.1,
        batch_interval=0.1,
    )
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = _make_client(api_key)

        response_id = None
        usage_payload = {}

        stream = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": "Say hi"}],
            max_completion_tokens=20,
            stream=True,
            stream_options={"include_usage": True},
        )

        # Process streaming events to accumulate usage data
        for evt in stream:
            # Get response_id from the first event that has it
            if response_id is None:
                response_id = getattr(evt, "id", None)

            # Extract usage from each event and accumulate
            from aicostmanager.usage_utils import get_streaming_usage_from_response

            up = get_streaming_usage_from_response(evt, "openai_chat")
            if isinstance(up, dict) and up:
                # Merge usage data (later events may have more complete info)
                usage_payload.update(up)

        if not usage_payload:
            pytest.skip("No usage returned in streaming events; skipping")

        # Track the usage and get the actual response_id that was used
        tracker.track("openai::gpt-5-mini", usage_payload, response_id=response_id)

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
