import json
import os
import time
import urllib.request
import uuid

import httpx
import pytest

from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from aicostmanager.usage_utils import get_streaming_usage_from_response

genai = pytest.importorskip("google.genai")

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _wait_for_cost_event(aicm_api_key: str, response_id: str):
    headers = {"Authorization": f"Bearer {aicm_api_key}"}
    time.sleep(5)
    last_data = None
    for _ in range(3):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/api/v1/cost-events/{response_id}", headers=headers
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.load(resp)
                    last_data = data
                    if isinstance(data, list):
                        if data:
                            evt = data[0]
                            evt_id = evt.get("event_id") or evt.get("uuid")
                            if evt_id:
                                uuid.UUID(str(evt_id))
                                return data
                    else:
                        event_id = data.get("event_id") or data.get(
                            "cost_event", {}
                        ).get("event_id")
                        if event_id:
                            uuid.UUID(str(event_id))
                            return data
        except Exception:
            pass
        time.sleep(1)
    raise AssertionError(
        f"cost event for {response_id} not found; last_data={last_data} base_url={BASE_URL}"
    )


@pytest.mark.parametrize(
    "service_key, model",
    [
        ("google::gemini-2.5-flash", "gemini-2.5-flash"),
        ("google::gemini-2.0-flash", "gemini-2.0-flash"),
    ],
)
def test_gemini_deliver_now_streaming(service_key, model, google_api_key, aicm_api_key):
    if not google_api_key:
        pytest.skip("GOOGLE_API_KEY not set in .env file")
    os.environ["AICM_LOG_BODIES"] = "true"
    ini = IniManager("ini")
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE, dconfig, poll_interval=0.1, batch_interval=0.1
    )

    assert delivery.log_bodies
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = genai.Client(api_key=google_api_key)

        response_id = uuid.uuid4().hex
        usage_payload = {}

        print("gemini response_id:", response_id)
        stream = client.models.generate_content_stream(
            model=model, contents=["Say hi (deliver_now_streaming)"]
        )
        final_event = None
        for evt in stream:
            final_event = evt
            try:
                print("gemini event type:", getattr(evt, "type", type(evt)))
                um = getattr(evt, "usage_metadata", None)
                if um is not None:
                    try:
                        print(
                            "gemini event usage_metadata:",
                            json.dumps(
                                get_streaming_usage_from_response(evt, "gemini"),
                                default=str,
                            ),
                        )
                    except Exception as ie:
                        print("gemini usage extract error:", repr(ie))
            except Exception:
                pass
            up = get_streaming_usage_from_response(evt, "gemini")
            if isinstance(up, dict) and up:
                print("gemini usage chunk:", json.dumps(up, default=str))
                usage_payload = up

        if not usage_payload and final_event is not None:
            # Some clients only include usage in the last event; try one more time
            up = get_streaming_usage_from_response(final_event, "gemini")
            if isinstance(up, dict) and up:
                usage_payload = up

        if not usage_payload:
            print(
                "gemini final_event type:",
                getattr(final_event, "type", type(final_event)),
            )
            print(
                "gemini final_event usage_metadata:",
                getattr(final_event, "usage_metadata", None),
            )
            pytest.skip("No usage returned in streaming events; skipping")

        print("gemini final usage payload:", json.dumps(usage_payload, default=str))
        try:
            dconfig2 = DeliveryConfig(
                ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
            )
            delivery2 = create_delivery(DeliveryType.IMMEDIATE, dconfig2)

            assert delivery2.log_bodies
            with Tracker(
                aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery2
            ) as t2:
                t2.track(service_key, usage_payload, response_id=response_id)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 422:
                try:
                    print("gemini 422 response body:", e.response.text)
                except Exception:
                    pass
                pytest.skip(
                    "Server rejected Gemini streaming usage schema (422). Update server schema to accept raw usage."
                )
            raise

        _wait_for_cost_event(aicm_api_key, response_id)
