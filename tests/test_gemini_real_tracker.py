import os
import time

import pytest

genai = pytest.importorskip("google.genai")

from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from tests.track_asserts import assert_track_result_payload

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _to_dict(obj, *, by_alias: bool = False):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                if attr == "model_dump":
                    try:
                        data = fn(by_alias=by_alias)
                    except TypeError:
                        data = fn()
                else:
                    data = fn()
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return obj


def _extract_usage_payload(resp) -> dict:
    for attr in ("usageMetadata", "usage_metadata", "usage"):
        val = getattr(resp, attr, None)
        if val is not None:
            data = _to_dict(val, by_alias=True)
            if isinstance(data, dict) and data:
                return data
    try:
        data = _to_dict(resp, by_alias=True)
        for key in ("usageMetadata", "usage_metadata", "usage"):
            if isinstance(data, dict) and key in data:
                return _to_dict(data.get(key), by_alias=True)

        # Heuristic nested search
        def find_usage(d: dict):
            for k, v in d.items():
                if isinstance(v, dict):
                    keys = set(v.keys())
                    if {
                        "promptTokenCount",
                        "candidatesTokenCount",
                        "totalTokenCount",
                    } & keys or {
                        "prompt_token_count",
                        "candidates_token_count",
                        "total_token_count",
                    } & keys:
                        return v
                    found = find_usage(v)
                    if found is not None:
                        return found
            return None

        if isinstance(data, dict):
            found = find_usage(data)
            if found is not None:
                return _to_dict(found, by_alias=True)
    except Exception:
        pass
    return {}


def _wait_for_empty(delivery, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        stats = getattr(delivery, "stats", lambda: {})()
        queued = stats.get("queued", 0)
        if queued == 0:
            return True
        time.sleep(0.05)
    return False


def _extract_response_id(used_id, fallback):
    if isinstance(used_id, dict):
        return used_id.get("response_id") or fallback
    return used_id or fallback


@pytest.mark.parametrize(
    "service_key, model",
    [
        ("google::gemini-2.5-flash", "gemini-2.5-flash"),
        ("google::gemini-2.0-flash", "gemini-2.0-flash"),
    ],
)
def test_gemini_tracker(service_key, model, google_api_key, aicm_api_key, tmp_path):
    if not google_api_key:
        pytest.skip("GOOGLE_API_KEY not set in .env file")
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
        db_path=str(tmp_path / "gemini_queue.db"),
        poll_interval=0.1,
        batch_interval=0.1,
    )

    assert delivery.log_bodies
    tracker = Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    )
    client = genai.Client(api_key=google_api_key)

    # Background tracking via queue
    resp = client.models.generate_content(model=model, contents="Say hi")
    # Try helper path by feeding usage via track(); if no id, generate and track
    response_id = getattr(resp, "id", None) or getattr(resp, "response_id", None)
    usage_payload = _extract_usage_payload(resp)
    used_id = tracker.track(service_key, usage_payload, response_id=response_id)
    final_id = _extract_response_id(used_id, response_id)
    assert _wait_for_empty(tracker.delivery, timeout=10.0)

    # Immediate delivery
    resp2 = client.models.generate_content(model=model, contents="Say hi again")
    response_id2 = getattr(resp2, "id", None) or getattr(resp2, "response_id", None)
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
        usage2 = _extract_usage_payload(resp2)
        used2 = t2.track(service_key, usage2, response_id=response_id2)
    final2 = _extract_response_id(used2, response_id2)
    assert isinstance(used2, dict)
    assert_track_result_payload(used2.get("result", {}))

    tracker.close()
