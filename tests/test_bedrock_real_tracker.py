import json
import os
import time
import urllib.request
import uuid

import pytest

boto3 = pytest.importorskip("boto3")

from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager
from aicostmanager.tracker import Tracker
from aicostmanager.usage_utils import get_usage_from_response

BASE_URL = os.environ.get("AICM_API_BASE", "http://127.0.0.1:8890")


def _wait_for_cost_event(aicm_api_key: str, response_id: str, timeout: int = 30):
    headers = {"Authorization": f"Bearer {aicm_api_key}"}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{BASE_URL}/api/v1/cost-events/{response_id}",
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    data = json.load(resp)
                    # Support both array and object responses
                    if isinstance(data, list) and data:
                        first = data[0]
                        event_id = first.get("event_id") or first.get("uuid")
                        if event_id:
                            uuid.UUID(str(event_id))
                            return data
                    elif isinstance(data, dict):
                        event_id = data.get("event_id") or data.get(
                            "cost_event", {}
                        ).get("event_id")
                        if event_id:
                            uuid.UUID(str(event_id))
                            return data
        except Exception:
            pass
        time.sleep(1)
    raise AssertionError(f"cost event for {response_id} not found")


@pytest.mark.parametrize(
    "service_key, model",
    [
        (
            "amazon-bedrock::us.meta.llama3-3-70b-instruct-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
        ),
        ("amazon-bedrock::us.amazon.nova-pro-v1:0", "us.amazon.nova-pro-v1:0"),
    ],
)
def test_bedrock_tracker(service_key, model, aws_region, aicm_api_key):
    if not aws_region:
        pytest.skip("AWS_DEFAULT_REGION not set in .env file")
    ini = IniManager("ini")
    dconfig = DeliveryConfig(
        ini_manager=ini,
        aicm_api_key=aicm_api_key,
        aicm_api_base=BASE_URL,
    )
    delivery = create_delivery(
        DeliveryType.PERSISTENT_QUEUE, dconfig, poll_interval=0.1, batch_interval=0.1
    )
    tracker = Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery
    )
    client = boto3.client("bedrock-runtime", region_name=aws_region)

    body = {
        "messages": [{"role": "user", "content": [{"text": "Say hi"}]}],
        "inferenceConfig": {"maxTokens": 50},
    }
    # Background tracking via queue
    try:
        resp = client.converse(modelId=model, **body)
    except Exception as e:
        msg = str(e)
        if (
            "on-demand throughput isn't supported" in msg
            or "on-demand throughput isn’nt supported" in msg
        ):
            pytest.skip(
                "Bedrock model requires provisioned throughput; skipping this case"
            )
        raise
    response_id = resp.get("ResponseMetadata", {}).get("RequestId") or resp.get(
        "output", {}
    ).get("message", {}).get("id")
    usage_payload = get_usage_from_response(resp, "amazon-bedrock")
    tracker.track(service_key, usage_payload, response_id=response_id)
    _wait_for_cost_event(aicm_api_key, response_id)

    # Immediate delivery
    body2 = {
        "messages": [{"role": "user", "content": [{"text": "Say hi again"}]}],
        "inferenceConfig": {"maxTokens": 50},
    }
    try:
        resp2 = client.converse(modelId=model, **body2)
    except Exception as e:
        msg = str(e)
        if (
            "on-demand throughput isn't supported" in msg
            or "on-demand throughput isn’nt supported" in msg
        ):
            pytest.skip(
                "Bedrock model requires provisioned throughput; skipping this case"
            )
        raise
    response_id2 = resp2.get("ResponseMetadata", {}).get("RequestId") or resp2.get(
        "output", {}
    ).get("message", {}).get("id")
    dconfig2 = DeliveryConfig(
        ini_manager=ini,
        aicm_api_key=aicm_api_key,
        aicm_api_base=BASE_URL,
    )
    delivery2 = create_delivery(DeliveryType.IMMEDIATE, dconfig2)
    with Tracker(
        aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery2
    ) as t2:
        usage2 = get_usage_from_response(resp2, "amazon-bedrock")
        t2.track(service_key, usage2, response_id=response_id2)
    _wait_for_cost_event(aicm_api_key, response_id2)

    tracker.close()
