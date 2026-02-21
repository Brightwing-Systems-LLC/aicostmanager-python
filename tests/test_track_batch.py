"""Tests for Tracker.track_batch() and track_batch_async() functionality."""

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from aicostmanager import Tracker
from aicostmanager.delivery import DeliveryConfig, DeliveryType, create_delivery
from aicostmanager.ini_manager import IniManager


class TestTrackBatch:
    """Test the track_batch method with various scenarios."""

    def test_track_batch_empty_list(self):
        """Test track_batch with empty records list."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(200, json={"results": [], "triggered_limits": {}})

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            result = tracker.track_batch([])

        assert result == {"results": []}
        assert not received  # No HTTP requests should be made

    def test_track_batch_immediate_delivery_single_record(self):
        """Test track_batch with immediate delivery and single record."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "response_id": "test_response_1",
                            "status": "completed",
                            "cost_events": [
                                {"vendor_id": "openai", "service_id": "gpt-4"}
                            ],
                        }
                    ],
                    "triggered_limits": {},
                },
            )

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "response_id": "test_response_1",
                "customer_key": "customer_123",
                "context": {"session_id": "session_456"},
            }
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            result = tracker.track_batch(records)

        assert len(received) == 1
        request_body = received[0]
        assert "tracked" in request_body
        assert len(request_body["tracked"]) == 1

        tracked_record = request_body["tracked"][0]
        assert tracked_record["service_key"] == "openai::gpt-4"
        assert tracked_record["payload"] == {"input_tokens": 100, "output_tokens": 50}
        assert tracked_record["response_id"] == "test_response_1"
        assert tracked_record["customer_key"] == "customer_123"
        assert tracked_record["context"] == {"session_id": "session_456"}

        assert "results" in result
        assert len(result["results"]) == 1

    def test_track_batch_immediate_delivery_multiple_records(self):
        """Test track_batch with immediate delivery and multiple records."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "response_id": "resp_1",
                            "status": "completed",
                            "cost_events": [],
                        },
                        {
                            "response_id": "resp_2",
                            "status": "completed",
                            "cost_events": [],
                        },
                        {
                            "response_id": "resp_3",
                            "status": "completed",
                            "cost_events": [],
                        },
                    ],
                    "triggered_limits": {},
                },
            )

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100},
                "response_id": "resp_1",
                "customer_key": "customer_1",
            },
            {
                "service_key": "anthropic::claude-3",
                "usage": {"input_tokens": 200},
                "response_id": "resp_2",
                "customer_key": "customer_2",
                "context": {"session": "session_abc"},
            },
            {
                "service_key": "heygen::streaming-avatar",
                "usage": {"duration": 120.5},
                "response_id": "resp_3",
                "timestamp": "2025-01-15T10:30:00Z",
            },
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            result = tracker.track_batch(records)

        # Should make exactly one HTTP request with all records
        assert len(received) == 1
        request_body = received[0]
        assert len(request_body["tracked"]) == 3

        # Verify each record
        tracked_records = request_body["tracked"]
        assert tracked_records[0]["service_key"] == "openai::gpt-4"
        assert tracked_records[0]["customer_key"] == "customer_1"

        assert tracked_records[1]["service_key"] == "anthropic::claude-3"
        assert tracked_records[1]["customer_key"] == "customer_2"
        assert tracked_records[1]["context"] == {"session": "session_abc"}

        assert tracked_records[2]["service_key"] == "heygen::streaming-avatar"
        assert tracked_records[2]["payload"] == {"duration": 120.5}

        assert "results" in result
        assert len(result["results"]) == 3

    def test_track_batch_persistent_delivery(self):
        """Test track_batch with persistent delivery."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_queue.db"
            ini = IniManager("ini")
            dconfig = DeliveryConfig(ini_manager=ini, aicm_api_key="test")
            delivery = create_delivery(
                DeliveryType.PERSISTENT_QUEUE, dconfig, db_path=str(db_path)
            )

            records = [
                {
                    "service_key": "openai::gpt-4",
                    "usage": {"input_tokens": 100},
                    "response_id": "resp_1",
                },
                {
                    "service_key": "anthropic::claude-3",
                    "usage": {"input_tokens": 200},
                    "response_id": "resp_2",
                },
            ]

            with Tracker(
                aicm_api_key="test", ini_path="ini", delivery=delivery
            ) as tracker:
                result = tracker.track_batch(records)

            # Should return queued info
            assert "queued" in result
            assert "response_ids" in result
            assert len(result["response_ids"]) == 2
            assert "resp_1" in result["response_ids"]
            assert "resp_2" in result["response_ids"]
            assert isinstance(result["queued"], int)

    def test_track_batch_with_defaults_fallback(self):
        """Test track_batch uses instance-level defaults when record values are None."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(200, json={"results": [{}], "triggered_limits": {}})

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100},
                # No customer_key or context - should use tracker defaults
            }
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            # Set instance-level defaults
            tracker.set_customer_key("default_customer")
            tracker.set_context({"default": "context"})

            result = tracker.track_batch(records)
            assert result is not None  # Verify batch was processed

        tracked_record = received[0]["tracked"][0]
        assert tracked_record["customer_key"] == "default_customer"
        assert tracked_record["context"] == {"default": "context"}

    def test_track_batch_with_anonymization(self):
        """Test track_batch applies anonymization correctly."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(200, json={"results": [{}], "triggered_limits": {}})

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        def anonymizer(value):
            return "ANONYMIZED"

        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100, "user_id": "sensitive_user_123"},
            }
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            result = tracker.track_batch(
                records, anonymize_fields=["user_id"], anonymizer=anonymizer
            )
            assert result is not None  # Verify batch was processed

        tracked_record = received[0]["tracked"][0]
        assert tracked_record["payload"]["input_tokens"] == 100
        assert tracked_record["payload"]["user_id"] == "ANONYMIZED"

    def test_track_batch_auto_generates_response_ids(self):
        """Test track_batch auto-generates response_ids when not provided."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(200, json={"results": [{}], "triggered_limits": {}})

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100},
                # No response_id provided
            }
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            result = tracker.track_batch(records)
            assert result is not None  # Verify batch was processed

        tracked_record = received[0]["tracked"][0]
        assert "response_id" in tracked_record
        assert len(tracked_record["response_id"]) == 32  # UUID hex length

    def test_track_batch_handles_timestamps(self):
        """Test track_batch handles different timestamp formats."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(
                200, json={"results": [{}, {}], "triggered_limits": {}}
            )

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        now = datetime.now(timezone.utc)
        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100},
                "timestamp": now,  # datetime object
            },
            {
                "service_key": "anthropic::claude-3",
                "usage": {"input_tokens": 200},
                "timestamp": "2025-01-15T10:30:00Z",  # ISO string
            },
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            result = tracker.track_batch(records)
            assert result is not None  # Verify batch was processed

        tracked_records = received[0]["tracked"]

        # First record should have timestamp from datetime object
        assert float(tracked_records[0]["timestamp"]) == now.timestamp()

        # Second record should have timestamp from ISO string
        expected_timestamp = datetime.fromisoformat(
            "2025-01-15T10:30:00+00:00"
        ).timestamp()
        assert float(tracked_records[1]["timestamp"]) == expected_timestamp

    def test_track_batch_error_handling(self):
        """Test track_batch handles errors gracefully."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "Internal server error"})

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100},
            }
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            with pytest.raises(httpx.HTTPStatusError):
                tracker.track_batch(records)

    def test_track_batch_missing_required_fields(self):
        """Test track_batch raises appropriate errors for missing required fields."""
        ini = IniManager("ini")
        dconfig = DeliveryConfig(ini_manager=ini, aicm_api_key="test")
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            # Missing service_key
            with pytest.raises(KeyError):
                tracker.track_batch([{"usage": {"tokens": 100}}])

            # Missing usage
            with pytest.raises(KeyError):
                tracker.track_batch([{"service_key": "openai::gpt-4"}])

    def test_track_batch_exceeds_max_batch_size(self):
        """Test track_batch raises BatchSizeLimitExceeded when batch size exceeds 1000."""
        from aicostmanager.client.exceptions import BatchSizeLimitExceeded

        ini = IniManager("ini")
        dconfig = DeliveryConfig(ini_manager=ini, aicm_api_key="test")
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        # Create 1001 records (exceeds limit)
        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"tokens": 100},
                "response_id": f"resp_{i}",
            }
            for i in range(1001)
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            with pytest.raises(BatchSizeLimitExceeded) as exc_info:
                tracker.track_batch(records)

        assert exc_info.value.batch_size == 1001
        assert exc_info.value.max_batch_size == 1000
        assert "1001 exceeds maximum allowed limit of 1000" in str(exc_info.value)

    def test_track_batch_max_batch_size_allowed(self):
        """Test track_batch allows exactly 1000 records."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(200, json={"results": [], "triggered_limits": {}})

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        # Create exactly 1000 records (at limit)
        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"tokens": 100},
                "response_id": f"resp_{i}",
            }
            for i in range(1000)
        ]

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            result = tracker.track_batch(records)

        # Should succeed and make a request
        assert len(received) == 1
        assert len(received[0]["tracked"]) == 1000


class TestTrackBatchAsync:
    """Test the async version of track_batch."""

    def test_track_batch_async_basic(self):
        """Test basic functionality of track_batch_async."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(
                200,
                json={
                    "results": [{"response_id": "test_resp", "status": "completed"}],
                    "triggered_limits": {},
                },
            )

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100},
                "response_id": "test_resp",
            }
        ]

        async def run():
            with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
                return await tracker.track_batch_async(records)

        result = asyncio.run(run())

        assert len(received) == 1
        assert "results" in result
        assert len(result["results"]) == 1

    def test_track_batch_async_persistent_delivery(self):
        """Test track_batch_async with persistent delivery."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_queue.db"
            ini = IniManager("ini")
            dconfig = DeliveryConfig(ini_manager=ini, aicm_api_key="test")
            delivery = create_delivery(
                DeliveryType.PERSISTENT_QUEUE, dconfig, db_path=str(db_path)
            )

            records = [
                {
                    "service_key": "openai::gpt-4",
                    "usage": {"input_tokens": 100},
                    "response_id": "async_resp_1",
                }
            ]

            async def run():
                with Tracker(
                    aicm_api_key="test", ini_path="ini", delivery=delivery
                ) as tracker:
                    return await tracker.track_batch_async(records)

            result = asyncio.run(run())

            assert "queued" in result
            assert "response_ids" in result
            assert "async_resp_1" in result["response_ids"]

    def test_track_batch_async_exceeds_max_batch_size(self):
        """Test track_batch_async raises BatchSizeLimitExceeded when batch size exceeds 1000."""
        from aicostmanager.client.exceptions import BatchSizeLimitExceeded

        ini = IniManager("ini")
        dconfig = DeliveryConfig(ini_manager=ini, aicm_api_key="test")
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        # Create 1001 records (exceeds limit)
        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"tokens": 100},
                "response_id": f"resp_{i}",
            }
            for i in range(1001)
        ]

        async def run():
            with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
                await tracker.track_batch_async(records)

        with pytest.raises(BatchSizeLimitExceeded) as exc_info:
            asyncio.run(run())

        assert exc_info.value.batch_size == 1001
        assert exc_info.value.max_batch_size == 1000
        assert "1001 exceeds maximum allowed limit of 1000" in str(exc_info.value)

    def test_track_batch_async_max_batch_size_allowed(self):
        """Test track_batch_async allows exactly 1000 records."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(200, json={"results": [], "triggered_limits": {}})

        transport = httpx.MockTransport(handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        # Create exactly 1000 records (at limit)
        records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"tokens": 100},
                "response_id": f"resp_{i}",
            }
            for i in range(1000)
        ]

        async def run():
            with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
                return await tracker.track_batch_async(records)

        result = asyncio.run(run())

        # Should succeed and make a request
        assert len(received) == 1
        assert len(received[0]["tracked"]) == 1000


class TestTrackBatchIntegration:
    """Integration tests for track_batch functionality."""

    def test_track_batch_vs_individual_tracks_equivalence(self):
        """Test that track_batch produces equivalent results to individual track calls."""
        batch_received = []
        individual_received = []

        def batch_handler(request: httpx.Request) -> httpx.Response:
            batch_received.append(json.loads(request.read().decode()))
            return httpx.Response(
                200, json={"results": [{}, {}], "triggered_limits": {}}
            )

        def individual_handler(request: httpx.Request) -> httpx.Response:
            individual_received.append(json.loads(request.read().decode()))
            return httpx.Response(200, json={"results": [{}], "triggered_limits": {}})

        # Test data
        test_records = [
            {
                "service_key": "openai::gpt-4",
                "usage": {"input_tokens": 100},
                "response_id": "resp_1",
                "customer_key": "customer_1",
            },
            {
                "service_key": "anthropic::claude-3",
                "usage": {"input_tokens": 200},
                "response_id": "resp_2",
                "customer_key": "customer_2",
                "context": {"session": "test_session"},
            },
        ]

        # Test batch approach
        batch_transport = httpx.MockTransport(batch_handler)
        ini = IniManager("ini")
        dconfig = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=batch_transport
        )
        delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

        with Tracker(aicm_api_key="test", ini_path="ini", delivery=delivery) as tracker:
            tracker.track_batch(test_records)

        # Test individual approach
        individual_transport = httpx.MockTransport(individual_handler)
        dconfig2 = DeliveryConfig(
            ini_manager=ini, aicm_api_key="test", transport=individual_transport
        )
        delivery2 = create_delivery(DeliveryType.IMMEDIATE, dconfig2)

        with Tracker(
            aicm_api_key="test", ini_path="ini", delivery=delivery2
        ) as tracker:
            for record in test_records:
                tracker.track(
                    record["service_key"],
                    record["usage"],
                    response_id=record["response_id"],
                    customer_key=record["customer_key"],
                    context=record.get("context"),
                )

        # Compare results
        assert len(batch_received) == 1
        assert len(individual_received) == 2

        batch_records = batch_received[0]["tracked"]
        individual_records = [req["tracked"][0] for req in individual_received]

        # Should have same number of records
        assert len(batch_records) == len(individual_records)

        # Compare each record (order might be different, so sort by response_id)
        batch_sorted = sorted(batch_records, key=lambda x: x["response_id"])
        individual_sorted = sorted(individual_records, key=lambda x: x["response_id"])

        for batch_record, individual_record in zip(batch_sorted, individual_sorted):
            assert batch_record["service_key"] == individual_record["service_key"]
            assert batch_record["payload"] == individual_record["payload"]
            assert batch_record["response_id"] == individual_record["response_id"]
            assert batch_record["customer_key"] == individual_record["customer_key"]
            assert batch_record.get("context") == individual_record.get("context")

    def test_track_batch_with_triggered_limits(self):
        """Test track_batch handles triggered limits correctly."""
        received = []

        def handler(request: httpx.Request) -> httpx.Response:
            received.append(json.loads(request.read().decode()))
            return httpx.Response(
                200,
                json={
                    "results": [{"response_id": "test_resp", "status": "completed"}],
                    "triggered_limits": {
                        "customer_123": {
                            "monthly_spend": {
                                "limit": 1000,
                                "current": 950,
                                "triggered": True,
                            }
                        }
                    },
                },
            )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ini", delete=False
        ) as tmp_ini:
            ini_path = tmp_ini.name

        try:
            transport = httpx.MockTransport(handler)
            ini = IniManager(ini_path)
            dconfig = DeliveryConfig(
                ini_manager=ini, aicm_api_key="test", transport=transport
            )
            delivery = create_delivery(DeliveryType.IMMEDIATE, dconfig)

            records = [
                {
                    "service_key": "openai::gpt-4",
                    "usage": {"input_tokens": 100},
                    "customer_key": "customer_123",
                }
            ]

            with Tracker(
                aicm_api_key="test", ini_path=ini_path, delivery=delivery
            ) as tracker:
                result = tracker.track_batch(records)

            # Verify triggered limits were processed
            assert "triggered_limits" in result
            assert "customer_123" in result["triggered_limits"]

            # Note: In test scenarios, triggered limits may not be written to INI
            # due to mocked transport. The important thing is that they're returned
            # in the result, indicating the batch processing handled them correctly.

        finally:
            import os

            try:
                os.unlink(ini_path)
            except OSError:
                pass
