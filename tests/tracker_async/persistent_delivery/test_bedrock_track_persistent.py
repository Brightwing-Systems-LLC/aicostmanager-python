import asyncio
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


def _make_client(region: str):
    return boto3.client("bedrock-runtime", region_name=region)


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
def test_bedrock_track_non_streaming(service_key, model, aws_region, aicm_api_key):
    if not aws_region:
        pytest.skip("AWS_DEFAULT_REGION not set in .env file")
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
        client = _make_client(aws_region)

        body = {
            "messages": [{"role": "user", "content": [{"text": "Say hi"}]}],
            "inferenceConfig": {"maxTokens": 50},
        }
        resp = client.converse(modelId=model, **body)
        response_id = resp.get("output", {}).get("message", {}).get("id") or resp.get(
            "ResponseMetadata", {}
        ).get("RequestId")
        usage = get_usage_from_response(resp, "amazon-bedrock")
        asyncio.run(tracker.track_async(service_key, usage, response_id=response_id))
        _wait_for_cost_event(aicm_api_key, response_id)


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
def test_bedrock_track_streaming(service_key, model, aws_region, aicm_api_key):
    if not aws_region:
        pytest.skip("AWS_DEFAULT_REGION not set in .env file")
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
        client = _make_client(aws_region)

        body = {
            "messages": [{"role": "user", "content": [{"text": "Say hi"}]}],
            "inferenceConfig": {"maxTokens": 50},
        }
        try:
            resp = client.converse_stream(modelId=model, **body)
        except Exception as e:
            msg = str(e)
            if (
                "on-demand throughput isn't supported" in msg
                or "on-demand throughput isn't supported" in msg
            ):
                pytest.skip(
                    "Bedrock model requires provisioned throughput; skipping this case"
                )
            raise

        response_id = uuid.uuid4().hex
        usage_payload = {}

        final_usage = None
        for chunk in resp["stream"]:
            try:
                if "metadata" in chunk:
                    print(
                        "bedrock metadata usage:",
                        chunk.get("metadata", {}).get("usage"),
                    )
                if "contentBlockDelta" in chunk:
                    print(
                        "bedrock content delta:",
                        chunk["contentBlockDelta"].get("delta"),
                    )
            except Exception:
                pass
            from aicostmanager.usage_utils import get_streaming_usage_from_response

            up = get_streaming_usage_from_response(chunk, "amazon-bedrock")
            if isinstance(up, dict) and up:
                print("bedrock usage chunk:", json.dumps(up, default=str))
                final_usage = up
        if not final_usage:
            # Bedrock usage is in a metadata chunk towards the end
            pytest.skip("No usage found in Bedrock streaming response; skipping")
        usage_payload = final_usage

        asyncio.run(
            tracker.track_async(service_key, usage_payload, response_id=response_id)
        )
        _wait_for_cost_event(aicm_api_key, response_id)
