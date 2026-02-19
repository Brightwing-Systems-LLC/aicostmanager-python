import os
import time

import pytest

openai = pytest.importorskip("openai")
from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from aicostmanager.usage_utils import get_usage_from_response
from tests.track_asserts import assert_track_result_payload

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _wait_for_empty(delivery, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        stats = getattr(delivery, "stats", lambda: {})()
        queued = stats.get("queued", 0)
        if queued == 0:
            return True
        time.sleep(0.05)
    return False


@pytest.mark.parametrize(
    "service_key, model",
    [("openai::gpt-5-mini", "gpt-5-mini")],
)
def test_openai_responses_tracker(
    service_key, model, openai_api_key, aicm_api_key, tmp_path
):
    if not openai_api_key:
        pytest.skip("OPENAI_API_KEY not set in .env file")
    # Ensure PersistentDelivery logs request/response bodies from the server
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
        db_path=str(tmp_path / "openai_responses_queue.db"),
        poll_interval=0.1,
        batch_interval=0.1,
    )

    assert delivery.log_bodies
    tracker = Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    )
    client = openai.OpenAI(api_key=openai_api_key)

    # Background tracking via queue
    resp = client.responses.create(model=model, input="Say hi")
    response_id = getattr(resp, "id", None)
    usage_payload = get_usage_from_response(resp, "openai_responses")
    track_res = tracker.track(service_key, usage_payload, response_id=response_id)
    if not track_res:
        pytest.fail(
            "Server rejected tracking request - check server logs for validation errors"
        )
    assert track_res.get("queued", 0) >= 0
    assert _wait_for_empty(tracker.delivery, timeout=10.0)

    # Immediate delivery
    resp2 = client.responses.create(model=model, input="Say hi again")
    response_id2 = getattr(resp2, "id", None)
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
        usage2 = get_usage_from_response(resp2, "openai_responses")
        result2 = t2.track(service_key, usage2, response_id=response_id2)
        if not result2 or result2.get("result") is None:
            pytest.fail(
                "Server rejected tracking request - check server logs for validation errors"
            )
        assert_track_result_payload(result2.get("result", {}))

    tracker.close()
