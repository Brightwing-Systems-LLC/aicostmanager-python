from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import uuid4

from .client.exceptions import BatchSizeLimitExceeded
from .delivery import (
    Delivery,
    DeliveryConfig,
    DeliveryType,
    create_delivery,
)
from .ini_manager import IniManager
from .logger import create_logger
from .usage_utils import (
    API_TO_VENDOR,
    VENDOR_TO_API,
    get_streaming_usage_from_response,
    get_usage_from_response,
)


class Tracker:
    """Lightweight usage tracker for the new ``/track`` endpoint."""

    def __init__(
        self,
        *,
        aicm_api_key: str | None = None,
        ini_path: str | None = None,
        delivery: Delivery | None = None,
        delivery_type: DeliveryType | str | None = None,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ) -> None:
        self.ini_manager = IniManager(IniManager.resolve_path(ini_path))
        self.ini_path = self.ini_manager.ini_path
        self.aicm_api_key = aicm_api_key or os.getenv("AICM_API_KEY")
        ini_dir = Path(self.ini_path).resolve().parent

        def _get(option: str, default: str | None = None) -> str | None:
            val = self.ini_manager.get_option("tracker", option)
            if val is not None:
                return val
            return os.getenv(option, default)

        api_base = _get("AICM_API_BASE", "https://aicostmanager.com")
        api_url = _get("AICM_API_URL", "/api/v1")
        log_file = _get("AICM_LOG_FILE", str(ini_dir / "aicm.log"))
        log_level = _get("AICM_LOG_LEVEL")
        timeout = float(_get("AICM_TIMEOUT", "10.0"))
        poll_interval = float(_get("AICM_POLL_INTERVAL", "0.1"))
        batch_interval = float(_get("AICM_BATCH_INTERVAL", "0.5"))
        immediate_pause_seconds = float(_get("AICM_IMMEDIATE_PAUSE_SECONDS", "5.0"))
        max_attempts = int(_get("AICM_MAX_ATTEMPTS", "3"))
        max_retries = int(_get("AICM_MAX_RETRIES", "5"))
        max_batch_size = int(_get("AICM_MAX_BATCH_SIZE", "1000"))
        log_bodies_val = _get("AICM_LOG_BODIES", "false")
        log_bodies = str(log_bodies_val).lower() in {"1", "true", "yes", "on"}

        raise_on_error_val = _get("AICM_RAISE_ON_ERROR", "false")
        raise_on_error = str(raise_on_error_val).lower() in {"1", "true", "yes", "on"}

        db_path = _get("AICM_DB_PATH", str(ini_dir / "queue.db"))
        delivery_name_cfg = _get("AICM_DELIVERY_TYPE")

        self.logger = create_logger(__name__, log_file, log_level)

        if delivery is not None:
            self.delivery = delivery
            resolved_type = getattr(delivery, "type", None)
        else:
            delivery_name_arg = (
                delivery_type.value
                if isinstance(delivery_type, DeliveryType)
                else delivery_type
            )
            if delivery_name_arg is not None:
                resolved_type = DeliveryType(str(delivery_name_arg).lower())
            elif delivery_name_cfg:
                resolved_type = DeliveryType(delivery_name_cfg.lower())
            else:
                resolved_type = DeliveryType.IMMEDIATE

            final_db_path = (
                db_path if resolved_type == DeliveryType.PERSISTENT_QUEUE else None
            )

            dconfig = DeliveryConfig(
                ini_manager=self.ini_manager,
                aicm_api_key=self.aicm_api_key,
                aicm_api_base=api_base,
                aicm_api_url=api_url,
                timeout=timeout,
                log_file=log_file,
                log_level=log_level,
                immediate_pause_seconds=immediate_pause_seconds,
            )
            self.delivery = create_delivery(
                resolved_type,
                dconfig,
                db_path=final_db_path,
                poll_interval=poll_interval,
                batch_interval=batch_interval,
                max_attempts=max_attempts,
                max_retries=max_retries,
                max_batch_size=max_batch_size,
                log_bodies=log_bodies,
                raise_on_error=raise_on_error,
            )
        if resolved_type is not None:
            self.ini_manager.set_option(
                "tracker", "AICM_DELIVERY_TYPE", resolved_type.value.upper()
            )

        # Instance-level tracking metadata
        self.customer_key: Optional[str] = None
        self.context: Optional[Dict[str, Any]] = None
        self._anonymize_fields = {str(field) for field in anonymize_fields or []}
        self._anonymizer = anonymizer or self._default_anonymizer

    # ------------------------------------------------------------------
    # Configuration Methods
    # ------------------------------------------------------------------
    def set_customer_key(self, key: str | None) -> None:
        """Update the ``customer_key`` used for tracking."""
        self.customer_key = key

    def set_context(self, context: Dict[str, Any] | None) -> None:
        """Update the ``context`` dictionary used for tracking."""
        self.context = context

    def set_anonymize_fields(self, fields: Iterable[str] | None) -> None:
        """Configure usage payload fields that should be anonymized before sending."""

        self._anonymize_fields = {str(field) for field in fields or []}

    def set_anonymizer(self, anonymizer: Callable[[Any], Any] | None) -> None:
        """Set the callable used to anonymize sensitive values."""

        self._anonymizer = anonymizer or self._default_anonymizer

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    def _get_vendor_api_mapping(self, service_key: str) -> Tuple[str, str]:
        """Extract vendor and api_id from service_key.

        Returns:
            Tuple of (vendor, api_id)
        """
        if "::" in service_key:
            vendor = service_key.split("::")[0]
            api_id = VENDOR_TO_API.get(vendor, vendor)
            return vendor, api_id
        else:
            # If no "::" found, assume it's already an api_id
            return service_key, service_key

    def _build_final_service_key(
        self, service_key: str, api_id: str, response_or_chunk: Any
    ) -> str:
        """Build the final service_key from api_id and model if needed."""
        if "::" not in service_key:
            # service_key was actually an api_id, extract model from response
            model = getattr(response_or_chunk, "model", None)
            if model:
                vendor = API_TO_VENDOR.get(api_id, api_id)
                return f"{vendor}::{model}"
        return service_key

    def _resolve_tracking_params(
        self, customer_key: Optional[str], context: Optional[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Resolve tracking parameters using instance defaults as fallbacks."""
        resolved_key = customer_key if customer_key is not None else self.customer_key
        resolved_context = context if context is not None else self.context
        return resolved_key, resolved_context

    def _apply_anonymization(
        self,
        usage: Dict[str, Any],
        *,
        anonymize_fields: Iterable[str] | None,
        anonymizer: Callable[[Any], Any] | None,
    ) -> Dict[str, Any]:
        fields = set(self._anonymize_fields)
        if anonymize_fields:
            fields.update(str(field) for field in anonymize_fields)
        if not fields:
            return usage

        anonymizer_fn = anonymizer or self._anonymizer
        if anonymizer_fn is None:
            return usage

        if not isinstance(usage, Mapping):
            return usage

        return self._anonymize_mapping(usage, fields, anonymizer_fn)

    def _anonymize_mapping(
        self,
        data: Mapping[str, Any],
        fields: set[str],
        anonymizer: Callable[[Any], Any],
    ) -> Dict[str, Any]:
        sanitized: Dict[str, Any] = {}
        for key, value in data.items():
            if key in fields:
                sanitized[key] = self._apply_anonymizer(value, anonymizer)
            else:
                sanitized[key] = self._anonymize_value(value, fields, anonymizer)
        return sanitized

    def _anonymize_value(
        self,
        value: Any,
        fields: set[str],
        anonymizer: Callable[[Any], Any],
    ) -> Any:
        if isinstance(value, Mapping):
            return self._anonymize_mapping(value, fields, anonymizer)
        if isinstance(value, list):
            return [self._anonymize_value(item, fields, anonymizer) for item in value]
        if isinstance(value, tuple):
            return tuple(
                self._anonymize_value(item, fields, anonymizer) for item in value
            )
        if isinstance(value, set):
            return {self._anonymize_value(item, fields, anonymizer) for item in value}
        return value

    def _apply_anonymizer(self, value: Any, anonymizer: Callable[[Any], Any]) -> Any:
        try:
            return anonymizer(value)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.warning("Anonymizer failed: %s", exc)
            return value

    @staticmethod
    def _default_anonymizer(value: Any) -> Any:
        if isinstance(value, str):
            return "[REDACTED]"
        if isinstance(value, bytes):
            return b"[REDACTED]"
        if isinstance(value, bytearray):
            return bytearray(b"[REDACTED]")
        if isinstance(value, Mapping):
            return {
                key: Tracker._default_anonymizer(inner) for key, inner in value.items()
            }
        if isinstance(value, list):
            return [Tracker._default_anonymizer(inner) for inner in value]
        if isinstance(value, tuple):
            return tuple(Tracker._default_anonymizer(inner) for inner in value)
        if isinstance(value, set):
            return {
                f"[REDACTED_{idx}]"
                if isinstance(inner, str)
                else Tracker._default_anonymizer(inner)
                for idx, inner in enumerate(value)
            }
        return value

    def _build_record(
        self,
        service_key: str,
        usage: Dict[str, Any],
        *,
        response_id: Optional[str],
        timestamp: str | datetime | None,
        customer_key: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "service_key": service_key,
            "response_id": response_id or uuid4().hex,
            "timestamp": (
                str(timestamp.timestamp())
                if isinstance(timestamp, datetime)
                else str(datetime.fromisoformat(timestamp).timestamp())
                if isinstance(timestamp, str)
                else str(datetime.now(timezone.utc).timestamp())
            ),
            "payload": usage,
        }
        if customer_key is not None:
            record["customer_key"] = customer_key
        if context is not None:
            record["context"] = context
        return record

    # ------------------------------------------------------------------
    # Core Tracking Methods
    # ------------------------------------------------------------------
    def track_batch(
        self,
        records: list[dict],
        *,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Track multiple usage records in a single batch request.

        This method sends multiple tracking records in a single HTTP request,
        which is more efficient than individual track() calls and avoids
        issues with persistent queue API key mixing.

        Note:
            Maximum batch size is 1000 records. Batches exceeding this limit
            will raise BatchSizeLimitExceeded.

        Args:
            records: List of dictionaries, each containing:
                - service_key (str): The service key for tracking
                - usage (Dict[str, Any]): Usage data to track
                - response_id (Optional[str]): Response ID (auto-generated if None)
                - timestamp (Optional[str | datetime]): Timestamp (current time if None)
                - customer_key (Optional[str]): Customer key (uses instance default if None)
                - context (Optional[Dict[str, Any]]): Context data (uses instance default if None)
            anonymize_fields: Fields to anonymize across all records
            anonymizer: Anonymization function to apply to all records

        Returns:
            For immediate delivery: dict with "results" and "triggered_limits" keys
            For queued delivery: dict with "queued" count

        Example:
            records = [
                {
                    "service_key": "openai::gpt-4",
                    "usage": {"tokens": 100},
                    "customer_key": "customer1",
                    "response_id": "resp1"
                },
                {
                    "service_key": "anthropic::claude-3",
                    "usage": {"tokens": 200},
                    "customer_key": "customer2",
                    "context": {"session": "abc123"}
                }
            ]
            result = tracker.track_batch(records)
        """
        if len(records) > 1000:
            raise BatchSizeLimitExceeded(len(records), 1000)

        if not records:
            return (
                {"results": []} if hasattr(self.delivery, "_enqueue") else {"queued": 0}
            )

        # Build all records first
        built_records = []
        for record_data in records:
            # Extract required fields
            service_key = record_data["service_key"]
            usage = record_data["usage"]

            # Extract optional fields with defaults
            response_id = record_data.get("response_id")
            timestamp = record_data.get("timestamp")
            customer_key = record_data.get("customer_key")
            context = record_data.get("context")

            # Use instance-level values as fallbacks if parameters are None
            resolved_customer_key, resolved_context = self._resolve_tracking_params(
                customer_key, context
            )

            # Apply anonymization to this record's usage
            anonymized_usage = self._apply_anonymization(
                usage, anonymize_fields=anonymize_fields, anonymizer=anonymizer
            )

            # Build the record
            record = self._build_record(
                service_key,
                anonymized_usage,
                response_id=response_id,
                timestamp=timestamp,
                customer_key=resolved_customer_key,
                context=resolved_context,
            )
            built_records.append(record)

        # For immediate delivery, send all records in one batch
        if hasattr(self.delivery, "type") and self.delivery.type.value == "immediate":
            body = {self.delivery._body_key: built_records}
            try:
                data = self.delivery._post_with_retry(body, max_attempts=3)
                if self.delivery._limits_enabled() and isinstance(data, dict):
                    tl_data = data.get("triggered_limits")
                    if tl_data:
                        # Use a ConfigManager that preserves existing INI settings
                        from .config_manager import ConfigManager

                        cfg = ConfigManager(
                            ini_path=self.ini_manager.ini_path, load=True
                        )
                        try:
                            cfg.write_triggered_limits(tl_data)
                        except Exception as exc:  # pragma: no cover
                            self.logger.error("Triggered limits update failed: %s", exc)
                return data
            except Exception as exc:
                self.logger.error("Batch delivery failed: %s", exc)
                raise
        else:
            # For queued delivery, enqueue all records
            total_queued = 0
            response_ids = []
            for record in built_records:
                result = self.delivery.enqueue(record)
                if isinstance(result, int):
                    total_queued = result  # Last queue length
                response_ids.append(record["response_id"])

            return {"queued": total_queued, "response_ids": response_ids}

    def track(
        self,
        service_key: str,
        usage: Dict[str, Any],
        *,
        response_id: Optional[str] = None,
        timestamp: str | datetime | None = None,
        customer_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Track usage data.

        For immediate delivery, returns a dict with ``result`` and
        ``triggered_limits`` keys. For queued delivery, returns a dict
        ``{"queued": <count>}`` indicating the queue length.
        """
        # Use instance-level values as fallbacks if parameters are None
        customer_key, context = self._resolve_tracking_params(customer_key, context)

        anonymized_usage = self._apply_anonymization(
            usage, anonymize_fields=anonymize_fields, anonymizer=anonymizer
        )

        record = self._build_record(
            service_key,
            anonymized_usage,
            response_id=response_id,
            timestamp=timestamp,
            customer_key=customer_key,
            context=context,
        )
        result = self.delivery.enqueue(record)
        if isinstance(result, int):
            return {"queued": result, "response_id": record["response_id"]}
        return result

    async def track_batch_async(
        self,
        records: list[dict],
        *,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Async version of track_batch.

        Note:
            Maximum batch size is 1000 records. Batches exceeding this limit
            will raise BatchSizeLimitExceeded.
        """
        if len(records) > 1000:
            raise BatchSizeLimitExceeded(len(records), 1000)

        return await asyncio.to_thread(
            self.track_batch,
            records,
            anonymize_fields=anonymize_fields,
            anonymizer=anonymizer,
        )

    async def track_async(
        self,
        service_key: str,
        usage: Dict[str, Any],
        *,
        response_id: Optional[str] = None,
        timestamp: str | datetime | None = None,
        customer_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self.track,
            service_key,
            usage,
            response_id=response_id,
            timestamp=timestamp,
            customer_key=customer_key,
            context=context,
            anonymize_fields=anonymize_fields,
            anonymizer=anonymizer,
        )

    # ------------------------------------------------------------------
    # LLM Tracking Methods
    # ------------------------------------------------------------------
    def track_llm_usage(
        self,
        service_key: str,
        response: Any,
        *,
        response_id: Optional[str] = None,
        timestamp: str | datetime | None = None,
        customer_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Extract usage from an LLM response and enqueue it.

        Parameters are identical to :meth:`track` except that ``response`` is
        the raw LLM client response.  Usage information is obtained via
        :func:`get_usage_from_response` using the provided ``service_key``.
        ``response`` is returned to allow call chaining. If a ``response_id`` was
        not provided and one is generated, it is attached to the response as
        ``response.aicm_response_id`` for convenience.
        """
        vendor, api_id = self._get_vendor_api_mapping(service_key)

        usage = get_usage_from_response(response, api_id)
        if isinstance(usage, dict) and usage:
            final_service_key = self._build_final_service_key(
                service_key, api_id, response
            )

            track_result = self.track(
                final_service_key,
                usage,
                response_id=response_id,
                timestamp=timestamp,
                customer_key=customer_key,
                context=context,
                anonymize_fields=anonymize_fields,
                anonymizer=anonymizer,
            )
            self._attach_tracking_metadata(response, track_result)
        return response

    def _attach_tracking_metadata(self, response: Any, track_result: Any) -> None:
        """Safely attach tracking metadata to response object."""
        try:
            setattr(
                response,
                "aicm_response_id",
                track_result.get("result", {}).get("response_id"),
            )
            setattr(response, "aicm_track_result", track_result)
        except Exception:
            pass

    async def track_llm_usage_async(
        self,
        service_key: str,
        response: Any,
        *,
        response_id: Optional[str] = None,
        timestamp: str | datetime | None = None,
        customer_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Async version of :meth:`track_llm_usage`."""
        return await asyncio.to_thread(
            self.track_llm_usage,
            service_key,
            response,
            response_id=response_id,
            timestamp=timestamp,
            customer_key=customer_key,
            context=context,
            anonymize_fields=anonymize_fields,
            anonymizer=anonymizer,
        )

    def track_llm_stream_usage(
        self,
        service_key: str,
        stream: Any,
        *,
        response_id: Optional[str] = None,
        timestamp: str | datetime | None = None,
        customer_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ):
        """Yield streaming events while tracking usage.

        ``stream`` should be an iterable of events from an LLM SDK.  Usage
        information is extracted from events using
        :func:`get_streaming_usage_from_response` and sent via :meth:`track` once
        available.
        """
        vendor, api_id = self._get_vendor_api_mapping(service_key)

        usage_sent = False
        for chunk in stream:
            if not usage_sent:
                usage = get_streaming_usage_from_response(chunk, api_id)
                if isinstance(usage, dict) and usage:
                    # For streaming, try to get model from both stream and chunk
                    chunk_with_model = type(
                        "MockResponse",
                        (),
                        {
                            "model": getattr(stream, "model", None)
                            or getattr(chunk, "model", None)
                        },
                    )()
                    final_service_key = self._build_final_service_key(
                        service_key, api_id, chunk_with_model
                    )

                    self.track(
                        final_service_key,
                        usage,
                        response_id=response_id,
                        timestamp=timestamp,
                        customer_key=customer_key,
                        context=context,
                        anonymize_fields=anonymize_fields,
                        anonymizer=anonymizer,
                    )
                    usage_sent = True
            yield chunk

    async def track_llm_stream_usage_async(
        self,
        service_key: str,
        stream: Any,
        *,
        response_id: Optional[str] = None,
        timestamp: str | datetime | None = None,
        customer_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ):
        """Asynchronous version of :meth:`track_llm_stream_usage`."""
        vendor, api_id = self._get_vendor_api_mapping(service_key)

        usage_sent = False
        async for chunk in stream:
            if not usage_sent:
                usage = get_streaming_usage_from_response(chunk, api_id)
                if isinstance(usage, dict) and usage:
                    # For streaming, try to get model from both stream and chunk
                    chunk_with_model = type(
                        "MockResponse",
                        (),
                        {
                            "model": getattr(stream, "model", None)
                            or getattr(chunk, "model", None)
                        },
                    )()
                    final_service_key = self._build_final_service_key(
                        service_key, api_id, chunk_with_model
                    )

                    await self.track_async(
                        final_service_key,
                        usage,
                        response_id=response_id,
                        timestamp=timestamp,
                        customer_key=customer_key,
                        context=context,
                        anonymize_fields=anonymize_fields,
                        anonymizer=anonymizer,
                    )
                    usage_sent = True
            yield chunk

    # ------------------------------------------------------------------
    # Context Manager and Lifecycle Methods
    # ------------------------------------------------------------------
    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - automatically closes the tracker."""
        self.close()

    def close(self) -> None:
        """Release any resources associated with the tracker.

        A ``Tracker`` may be constructed with an explicitly provided delivery
        instance.  Historically such deliveries were treated as externally
        managed and were not stopped when the tracker was closed.  This led to
        surprising behaviour where queued deliveries (like
        :class:`PersistentDelivery`) could continue running after the tracker
        context exited, leaving callers unsure when the queue had fully drained.

        To provide predictable shutdown semantics, ``close()`` now always stops
        the associated delivery instance.  Callers that wish to reuse a single
        delivery across multiple trackers should manage the delivery lifecycle
        separately instead of relying on the tracker's context manager.
        """
        if getattr(self, "delivery", None) is not None:
            self.delivery.stop()
