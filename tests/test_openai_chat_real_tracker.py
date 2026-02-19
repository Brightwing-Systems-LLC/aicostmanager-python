import os

import pytest

openai = pytest.importorskip("openai")
from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from aicostmanager.usage_utils import get_usage_from_response
from tests.track_asserts import assert_track_result_payload

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _make_openai_client(api_key: str):
    return openai.OpenAI(api_key=api_key)


def _make_xai_client(api_key: str):
    return openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")


@pytest.mark.parametrize(
    "service_key, model, key_env, maker",
    [
        ("openai::gpt-5-mini", "gpt-5-mini", "OPENAI_API_KEY", _make_openai_client),
        ("xai::grok-3-mini", "grok-3-mini", "GROK_API_KEY", _make_xai_client),
    ],
)
def test_openai_chat_tracker(
    service_key, model, key_env, maker, aicm_api_key, tmp_path
):
    api_key = os.environ.get(key_env)
    if not api_key:
        pytest.skip(f"{key_env} not set in .env file")
    os.environ["AICM_LOG_BODIES"] = "true"
    ini = IniManager(str(tmp_path / "ini"))
    dconfig = DeliveryConfig(
        ini_manager=ini,
        aicm_api_key=aicm_api_key,
        aicm_api_base=BASE_URL,
    )
    delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

    assert delivery.log_bodies
    client = maker(api_key)

    # Immediate delivery using context manager
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    ) as tracker:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_completion_tokens=20,
        )
        response_id = getattr(resp, "id", None)
        usage_payload = get_usage_from_response(resp, "openai_chat")
        result = tracker.track(service_key, usage_payload, response_id=response_id)
        if not result or result.get("result") is None:
            pytest.fail(
                "Server rejected tracking request - check server logs for validation errors"
            )
        assert_track_result_payload(result.get("result", {}))

    # Immediate delivery
    resp2 = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hi again"}],
        max_completion_tokens=20,
    )
    response_id2 = getattr(resp2, "id", None)
    # Immediate delivery using an explicit delivery
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
        usage2 = get_usage_from_response(resp2, "openai_chat")
        result2 = t2.track(service_key, usage2, response_id=response_id2)
        if not result2 or result2.get("result") is None:
            pytest.fail(
                "Server rejected tracking request - check server logs for validation errors"
            )
        assert_track_result_payload(result2.get("result", {}))

    # No explicit close needed; context managers handled shutdown
