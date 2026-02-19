import asyncio
import os
import time

import pytest

openai = pytest.importorskip("openai")

from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from aicostmanager.usage_utils import get_usage_from_response

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _extract_response_id(used_id, fallback):
    if isinstance(used_id, dict):
        return used_id.get("response_id") or fallback
    return used_id or fallback


def _make_client(api_key: str):
    return openai.OpenAI(api_key=api_key)


def test_openai_responses_track_non_streaming(aicm_api_key):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in .env file")
    ini = IniManager("ini")
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE, dconfig, poll_interval=0.1, batch_interval=0.1
    )
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = _make_client(api_key)

        resp = client.responses.create(model="gpt-5-mini", input="Say hi")
        response_id = getattr(resp, "id", None)
        usage = get_usage_from_response(resp, "openai_responses")
        asyncio.run(
            tracker.track_async(
                "openai::gpt-5-mini",
                usage,
                response_id=response_id,
            )
        )
        # Queue-based tracking: ensure queue drained
        deadline = time.time() + 10
        while time.time() < deadline:
            stats = getattr(tracker.delivery, "stats", lambda: {})()
            if stats.get("queued", 0) == 0:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("delivery queue did not drain")


def test_openai_responses_track_streaming(aicm_api_key):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in .env file")
    ini = IniManager("ini2")
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE, dconfig, poll_interval=0.1, batch_interval=0.1
    )
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = _make_client(api_key)

        response_id = None
        usage_payload = {}

        stream = client.responses.create(
            model="gpt-5-mini", input="Say hi", stream=True
        )

        # Process streaming events to accumulate usage data
        for evt in stream:
            # Get response_id from the first event that has it
            if response_id is None:
                response_id = getattr(evt, "id", None)

            # Extract usage from each event and accumulate
            from aicostmanager.usage_utils import get_streaming_usage_from_response

            up = get_streaming_usage_from_response(evt, "openai_responses")
            if isinstance(up, dict) and up:
                # Merge usage data (later events may have more complete info)
                usage_payload.update(up)

        if not usage_payload:
            pytest.skip("No usage returned in streaming events; skipping")

        # Track the usage and get the actual response_id that was used
        used_id = asyncio.run(
            tracker.track_async(
                "openai::gpt-5-mini",
                usage_payload,
                response_id=response_id,
            )
        )

        final_id = _extract_response_id(used_id, response_id)
        print(f"Using response_id: {final_id}")
        # Background queue: ensure queue drained
        deadline = time.time() + 10
        while time.time() < deadline:
            stats = getattr(tracker.delivery, "stats", lambda: {})()
            if stats.get("queued", 0) == 0:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("delivery queue did not drain")
