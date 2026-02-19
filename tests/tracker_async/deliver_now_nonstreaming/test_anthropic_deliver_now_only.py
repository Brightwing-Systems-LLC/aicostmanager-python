import asyncio
import json
import os
import time
import urllib.request
import uuid

import pytest

from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from aicostmanager.usage_utils import get_usage_from_response

anthropic = pytest.importorskip("anthropic")

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
        ("anthropic::claude-sonnet-4-20250514", "claude-sonnet-4-20250514"),
        ("anthropic::claude-sonnet-4-0", "claude-sonnet-4-20250514"),
    ],
)
def test_anthropic_deliver_now_only(
    service_key, model, anthropic_api_key, aicm_api_key, tmp_path
):
    if not anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY not set in .env file")
    os.environ["AICM_LOG_BODIES"] = "true"
    ini = IniManager(str(tmp_path / "ini"))
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

    assert delivery.log_bodies
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = anthropic.Anthropic(api_key=anthropic_api_key)

        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi (deliver_now_only)"}],
            max_tokens=20,
        )
        response_id = getattr(resp, "id", None)
        usage_payload = get_usage_from_response(resp, "anthropic")

        asyncio.run(
            tracker.track_async(service_key, usage_payload, response_id=response_id)
        )
        _wait_for_cost_event(aicm_api_key, response_id)
