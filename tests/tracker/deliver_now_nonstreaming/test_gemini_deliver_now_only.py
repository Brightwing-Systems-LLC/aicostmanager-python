import json
import os
import time
import urllib.request
import uuid

import pytest

genai = pytest.importorskip("google.genai")
from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker

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
                # Prefer alias/camelCase keys when supported (pydantic models)
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
    # Try common locations
    for attr in ("usageMetadata", "usage_metadata", "usage"):
        val = getattr(resp, attr, None)
        if val is not None:
            data = _to_dict(val, by_alias=True)
            if isinstance(data, dict) and data:
                return data
    # Try from full dump
    try:
        data = _to_dict(resp, by_alias=True)
        # Direct keys
        for key in ("usageMetadata", "usage_metadata", "usage"):
            if isinstance(data, dict) and key in data:
                return _to_dict(data.get(key), by_alias=True)

        # Heuristic: find a nested dict with token counts
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
                return _to_dict(found)
    except Exception:
        pass
    return {}


def _normalize_gemini_usage(usage: dict) -> dict:
    if not isinstance(usage, dict):
        return {}
    # Map snake_case keys to the expected camelCase keys
    key_map = {
        "prompt_token_count": "promptTokenCount",
        "candidates_token_count": "candidatesTokenCount",
        "total_token_count": "totalTokenCount",
    }
    normalized: dict = {}
    for k, v in usage.items():
        if k in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount"):
            normalized[k] = v
        elif k in key_map:
            normalized[key_map[k]] = v
    # Only include keys allowed by server schema to avoid "additional properties" errors
    allowed = {"promptTokenCount", "candidatesTokenCount", "totalTokenCount"}
    return {k: v for k, v in normalized.items() if k in allowed}


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
def test_gemini_deliver_now_only(service_key, model, google_api_key, aicm_api_key):
    if not google_api_key:
        pytest.skip("GOOGLE_API_KEY not set in .env file")
    os.environ["AICM_LOG_BODIES"] = "true"
    ini = IniManager("ini")
    dconfig = DeliveryConfig(
        ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
    )
    delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

    assert delivery.log_bodies
    response_id = None
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        client = genai.Client(api_key=google_api_key)

        resp = client.models.generate_content(
            model=model, contents="Say hi (deliver_now_only)"
        )
        response_id = getattr(resp, "id", None) or getattr(resp, "response_id", None)
        if not response_id:
            # Debug-print available attributes to help diagnose schema differences
            try:
                print("gemini response type:", type(resp))
                print(
                    "gemini response dir sample:",
                    [a for a in dir(resp) if not a.startswith("__")][:30],
                )
            except Exception:
                pass
            # Fallback: generate our own correlation id for tracking
            import uuid as _uuid

            response_id = _uuid.uuid4().hex
            print("No response_id from Gemini; using generated id:", response_id)

        # Build usage payload from Gemini response
        usage_payload = _extract_usage_payload(resp)
        # usage_payload = _normalize_gemini_usage(raw_usage_payload)
        # print(
        #     "raw usage payload:", json.dumps(raw_usage_payload, indent=2, default=str)
        # )
        print(
            "normalized usage payload:",
            json.dumps(usage_payload, indent=2, default=str),
        )

        tracker.track(service_key, usage_payload, response_id=response_id)
    _wait_for_cost_event(aicm_api_key, response_id)
