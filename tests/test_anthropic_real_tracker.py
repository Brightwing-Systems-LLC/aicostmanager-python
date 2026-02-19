import json
import os
import time
import urllib.request
import uuid

import pytest

anthropic = pytest.importorskip("anthropic")
from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _usage_to_payload(usage):
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(usage, attr, None)
        if callable(fn):
            try:
                data = fn()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return usage


def _wait_for_empty(delivery, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        stats = getattr(delivery, "stats", lambda: {})()
        queued = stats.get("queued", 0)
        if queued == 0:
            return True
        time.sleep(0.05)
    return False


def _wait_for_cost_event(aicm_api_key: str, response_id: str, timeout: int = 30):
    """Wait 2s then try up to 3 fetches for a cost event."""
    headers = {"Authorization": f"Bearer {aicm_api_key}"}
    time.sleep(2)
    attempts = 3
    last_data = None
    for _ in range(attempts):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/api/v1/cost-events/{response_id}",
                headers=headers,
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
def test_anthropic_tracker(
    service_key, model, anthropic_api_key, aicm_api_key, tmp_path
):
    if not anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY not set in .env file")
    os.environ["AICM_LOG_BODIES"] = "true"
    ini = IniManager(str(tmp_path / "ini"))
    dconfig = DeliveryConfig(
        ini_manager=ini,
        aicm_api_key=aicm_api_key,
        aicm_api_base=BASE_URL,
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE,
        dconfig,
        db_path=str(tmp_path / "anthropic_queue.db"),
        poll_interval=0.1,
        batch_interval=0.1,
    )

    assert delivery.log_bodies
    tracker = Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    )
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    # Background tracking via queue
    resp = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi"}],
        max_tokens=20,
    )
    response_id = getattr(resp, "id", None)
    usage_payload = _usage_to_payload(getattr(resp, "usage", None))
    tracker.track(service_key, usage_payload, response_id=response_id)
    # Wait for the queue to flush
    assert _wait_for_empty(tracker.delivery, timeout=10.0)
    _wait_for_cost_event(aicm_api_key, response_id)

    # Immediate delivery
    resp2 = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi again"}],
        max_tokens=20,
    )
    response_id2 = getattr(resp2, "id", None)
    usage_payload2 = _usage_to_payload(getattr(resp2, "usage", None))
    # Immediate delivery via explicit delivery configuration
    dconfig2 = DeliveryConfig(
        ini_manager=ini,
        aicm_api_key=aicm_api_key,
        aicm_api_base=BASE_URL,
    )
    delivery2 = create_delivery(DeliveryType.IMMEDIATE, dconfig2)

    assert delivery2.log_bodies
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery2
    ) as t2:
        t2.track(service_key, usage_payload2, response_id=response_id2)
    _wait_for_cost_event(aicm_api_key, response_id2)

    tracker.close()
