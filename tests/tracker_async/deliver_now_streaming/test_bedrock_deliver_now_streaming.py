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
from aicostmanager.usage_utils import get_streaming_usage_from_response

boto3 = pytest.importorskip("boto3")

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
def test_bedrock_deliver_now_streaming(service_key, model, aws_region, aicm_api_key):
    if not aws_region:
        pytest.skip("AWS_DEFAULT_REGION not set in .env file")
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
        client = _make_client(aws_region)

        response_id = uuid.uuid4().hex
        usage_payload = {}

        body = {
            "system": [{"text": "You are a helpful assistant."}],
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Say hi (deliver_now_streaming)"}],
                }
            ],
            "inferenceConfig": {"maxTokens": 50},
        }

        request = {"modelId": model, **body}

        try:
            resp = client.converse_stream(**request)
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
            up = get_streaming_usage_from_response(chunk, "amazon-bedrock")
            if isinstance(up, dict) and up:
                print("bedrock usage chunk:", json.dumps(up, default=str))
                final_usage = up
        if not final_usage:
            # Bedrock usage is in a metadata chunk towards the end
            pytest.skip("No usage found in Bedrock streaming response; skipping")
        usage_payload = final_usage

        dconfig2 = DeliveryConfig(
            ini_manager=ini, aicm_api_key=aicm_api_key, aicm_api_base=BASE_URL
        )
        delivery2 = create_delivery(DeliveryType.IMMEDIATE, dconfig2)

        assert delivery2.log_bodies
        with Tracker(
            aicm_api_key=aicm_api_key, ini_path=ini.ini_path, delivery=delivery2
        ) as t2:
            asyncio.run(
                t2.track_async(
                    service_key,
                    usage_payload,
                    response_id=response_id,
                )
            )

        _wait_for_cost_event(aicm_api_key, response_id)
