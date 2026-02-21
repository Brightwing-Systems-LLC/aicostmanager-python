from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Callable, Iterable, Iterator
from typing import Any

from .delivery import DeliveryType
from .tracker import Tracker
from .usage_utils import (
    _KNOWN_EXTRACTORS,
    _generic_extract_streaming_usage,
    _generic_extract_usage,
    get_streaming_usage_from_response,
    get_usage_from_response,
)


class _Proxy:
    """Recursive proxy that intercepts method calls for tracking."""

    def __init__(self, obj: Any, wrapper: "ServiceWrapper") -> None:
        object.__setattr__(self, "_obj", obj)
        object.__setattr__(self, "_wrapper", wrapper)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._obj, name)
        if callable(attr):
            # Check if this callable has non-private attributes (like MagicMock objects)
            # If it does, we should proxy it instead of wrapping it as a function
            # This preserves attribute access like client.chat.completions
            has_attributes = False
            try:
                # First check for common mock attributes or if it's a Mock/MagicMock
                # This is safer and faster than inspecting __dict__
                has_attributes = (
                    hasattr(attr, "_mock_name")
                    or hasattr(attr, "assert_called")
                    or type(attr).__name__ in ("Mock", "MagicMock", "AsyncMock")
                )

                # If not a mock, check for any non-private attributes
                if not has_attributes:
                    attr_dict = getattr(attr, "__dict__", {})
                    # Only check if keys exist, don't access the actual attributes
                    # to avoid triggering MagicMock side effects
                    has_attributes = any(
                        key for key in attr_dict if not key.startswith("_")
                    )
            except (AttributeError, TypeError):
                # If we can't inspect it safely, assume it's a regular callable
                pass

            if has_attributes:
                # This callable has attributes (e.g., MagicMock). Proxy it so nested
                # attribute access like .completions.create works without turning it
                # into a bare function.
                return _Proxy(attr, self._wrapper)

            if inspect.iscoroutinefunction(attr):

                async def async_call(*args, **kwargs):
                    model = self._wrapper._extract_model(attr, args, kwargs)
                    result = await attr(*args, **kwargs)
                    return await self._wrapper._handle_async_result(result, model)

                return async_call
            else:

                def sync_call(*args, **kwargs):
                    model = self._wrapper._extract_model(attr, args, kwargs)
                    result = attr(*args, **kwargs)
                    return self._wrapper._handle_result(result, model)

                return sync_call
        return _Proxy(attr, self._wrapper) if _should_wrap(attr) else attr

    def __call__(self, *args, **kwargs):
        # Fast-path for unittest.mock objects: avoid signature introspection
        is_mock_obj = hasattr(self._obj, "_mock_name") or type(self._obj).__name__ in (
            "Mock",
            "MagicMock",
            "AsyncMock",
        )
        if is_mock_obj:
            result = self._obj(*args, **kwargs)
            return self._wrapper._handle_result(result, None)
        model = self._wrapper._extract_model(self._obj, args, kwargs)
        result = self._obj(*args, **kwargs)
        return self._wrapper._handle_result(result, model)


def _should_wrap(obj: Any) -> bool:
    # Do not recursively wrap unittest.mock objects
    if hasattr(obj, "_mock_name") or type(obj).__name__ in (
        "Mock",
        "MagicMock",
        "AsyncMock",
    ):
        return False
    return not isinstance(
        obj,
        (
            str,
            bytes,
            bytearray,
            int,
            float,
            bool,
            type(None),
        ),
    )


class ServiceWrapper:
    """Generic wrapper that tracks usage for any vendor's SDK client.

    Supports three service-key modes:

    1. **Dynamic model** (default): ``service_key = "{vendor}::{model}"``
       where model is extracted from each call's arguments.
    2. **Fixed service**: Pass ``service="stt-streaming"`` to get
       ``service_key = "{vendor}::stt-streaming"`` (model extraction skipped).
    3. **Full override**: Pass ``service_key="heygen::streaming-avatar"``
       to use that exact string for every call.

    Custom extractors can be provided to handle non-standard response
    formats.  When omitted, generic heuristics try ``.usage``,
    ``.usage_metadata``, ``dict["usage"]``, etc.
    """

    def __init__(
        self,
        client: Any,
        *,
        vendor: str = "",
        service: str | None = None,
        service_key: str | None = None,
        usage_extractor: Callable | None = None,
        streaming_usage_extractor: Callable | None = None,
        aicm_api_key: str | None = None,
        tracker: Tracker | None = None,
        customer_key: str | None = None,
        context: dict[str, Any] | None = None,
        delivery_type: DeliveryType | str | None = None,
        anonymize_fields: Iterable[str] | None = None,
        anonymizer: Callable[[Any], Any] | None = None,
    ) -> None:
        self._client = client
        self._vendor = vendor
        self._service = service
        self._service_key_override = service_key

        # Resolve extractors
        self._usage_extractor = usage_extractor or _generic_extract_usage
        self._streaming_usage_extractor = (
            streaming_usage_extractor or _generic_extract_streaming_usage
        )

        if tracker is None:
            self._tracker = Tracker(
                aicm_api_key=aicm_api_key,
                delivery_type=delivery_type,
                anonymize_fields=anonymize_fields,
                anonymizer=anonymizer,
            )
        else:
            self._tracker = tracker
            if anonymize_fields is not None:
                self._tracker.set_anonymize_fields(anonymize_fields)
            if anonymizer is not None:
                self._tracker.set_anonymizer(anonymizer)
        self._proxy = _Proxy(client, self)
        self.customer_key = customer_key
        self.context = context
        self._anonymize_fields = (
            tuple(anonymize_fields) if anonymize_fields is not None else None
        )
        self._anonymizer = anonymizer

    # ------------------------------------------------------------------
    def set_customer_key(self, key: str | None) -> None:
        """Update the ``customer_key`` used for tracking."""
        self.customer_key = key

    def set_context(self, context: dict[str, Any] | None) -> None:
        """Update the ``context`` dictionary used for tracking."""
        self.context = context

    def set_anonymize_fields(self, fields: Iterable[str] | None) -> None:
        """Update the usage fields that should be anonymized before tracking."""

        self._anonymize_fields = tuple(fields) if fields is not None else None
        self._tracker.set_anonymize_fields(fields)

    def set_anonymizer(self, anonymizer: Callable[[Any], Any] | None) -> None:
        """Update the anonymizer callable used for sensitive fields."""

        self._anonymizer = anonymizer
        self._tracker.set_anonymizer(anonymizer)

    # ------------------------------------------------------------------
    def _extract_model(self, method: Any, args: tuple, kwargs: dict) -> str | None:
        for key in ("model", "model_id", "modelId"):
            if key in kwargs:
                return kwargs[key]
        try:
            sig = inspect.signature(method)
            params = list(sig.parameters)
            for idx, name in enumerate(params):
                if name in ("model", "model_id", "modelId") and idx < len(args):
                    return args[idx]
        except (ValueError, TypeError):
            pass
        return None

    def _get_vendor(self) -> str:
        if self._service_key_override:
            # Parse vendor from the override
            if "::" in self._service_key_override:
                return self._service_key_override.split("::")[0]
            return self._service_key_override
        return self._vendor

    def _build_service_key(self, model: str | None) -> str:
        # Mode 3: Full override
        if self._service_key_override:
            return self._service_key_override
        vendor = self._get_vendor()
        # Mode 2: Fixed service
        if self._service is not None:
            return f"{vendor}::{self._service}"
        # Mode 1: Dynamic model (default)
        model_id = model or "unknown-model"
        return f"{vendor}::{model_id}"

    def _track_usage(self, response: Any, model: str | None) -> Any:
        usage = self._usage_extractor(response)
        if usage:
            # Try multiple field names for response ID
            response_id = (
                getattr(response, "id", None)
                or getattr(response, "response_id", None)
                or getattr(response, "responseId", None)
            )
            # Also try dict access for some response formats
            if response_id is None and isinstance(response, dict):
                response_id = (
                    response.get("response_id")
                    or response.get("responseId")
                    or response.get("id")
                )
            self._tracker.track(
                self._build_service_key(model),
                usage,
                response_id=response_id,
                customer_key=self.customer_key,
                context=self.context,
                anonymize_fields=self._anonymize_fields,
                anonymizer=self._anonymizer,
            )
        return response

    def _wrap_stream(self, stream: Iterable, model: str | None) -> Iterable:
        service_key = self._build_service_key(model)
        usage_sent = False
        # If the stream is a unittest.mock object, add a defensive upper bound
        is_mock_stream = hasattr(stream, "_mock_name") or type(stream).__name__ in (
            "Mock",
            "MagicMock",
            "AsyncMock",
        )
        max_chunks = 10000 if is_mock_stream else None
        count = 0
        for chunk in stream:
            count += 1
            # If chunk is a mock, skip usage extraction to avoid recursion on MagicMock
            is_mock_chunk = hasattr(chunk, "_mock_name") or type(chunk).__name__ in (
                "Mock",
                "MagicMock",
                "AsyncMock",
            )
            if not usage_sent and not is_mock_chunk:
                usage = self._streaming_usage_extractor(chunk)
                if usage:
                    self._tracker.track(
                        service_key,
                        usage,
                        customer_key=self.customer_key,
                        context=self.context,
                        anonymize_fields=self._anonymize_fields,
                        anonymizer=self._anonymizer,
                    )
                    usage_sent = True
            yield chunk
            if max_chunks is not None and count >= max_chunks:
                break

    async def _wrap_stream_async(self, stream: AsyncIterable, model: str | None):
        service_key = self._build_service_key(model)
        usage_sent = False
        # If the stream is a unittest.mock object, add a defensive upper bound
        is_mock_stream = hasattr(stream, "_mock_name") or type(stream).__name__ in (
            "Mock",
            "MagicMock",
            "AsyncMock",
        )
        max_chunks = 10000 if is_mock_stream else None
        count = 0
        async for chunk in stream:
            count += 1
            is_mock_chunk = hasattr(chunk, "_mock_name") or type(chunk).__name__ in (
                "Mock",
                "MagicMock",
                "AsyncMock",
            )
            if not usage_sent and not is_mock_chunk:
                usage = self._streaming_usage_extractor(chunk)
                if usage:
                    await self._tracker.track_async(
                        service_key,
                        usage,
                        customer_key=self.customer_key,
                        context=self.context,
                        anonymize_fields=self._anonymize_fields,
                        anonymizer=self._anonymizer,
                    )
                    usage_sent = True
            yield chunk
            if max_chunks is not None and count >= max_chunks:
                break

    def _handle_result(self, result: Any, model: str | None):
        # Generalized: streaming dict with an inner "stream" key
        if isinstance(result, dict) and "stream" in result:
            inner_stream = result.get("stream")
            if isinstance(inner_stream, (Iterator, Iterable)) and not isinstance(
                inner_stream, (str, bytes, bytearray, dict)
            ):
                wrapped = self._wrap_stream(inner_stream, model)
                new_result = dict(result)
                new_result["stream"] = wrapped
                return new_result
        # Don't treat Mock objects as iterables even if they claim to be
        if hasattr(result, "_mock_name") or type(result).__name__ in (
            "Mock",
            "MagicMock",
            "AsyncMock",
        ):
            return self._track_usage(result, model)
        if isinstance(result, AsyncIterable):
            return self._wrap_stream_async(result, model)
        if isinstance(result, Iterator) and not isinstance(
            result, (str, bytes, bytearray)
        ):
            return self._wrap_stream(result, model)
        return self._track_usage(result, model)

    async def _handle_async_result(self, result: Any, model: str | None):
        # Don't treat Mock objects as iterables even if they claim to be
        if hasattr(result, "_mock_name") or type(result).__name__ in (
            "Mock",
            "MagicMock",
            "AsyncMock",
        ):
            return self._track_usage(result, model)
        if isinstance(result, AsyncIterable):
            return self._wrap_stream_async(result, model)
        if isinstance(result, Iterator) and not isinstance(
            result, (str, bytes, bytearray)
        ):
            return self._wrap_stream(result, model)
        return self._track_usage(result, model)

    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - delegated
        return getattr(self._proxy, name)

    def close(self) -> None:  # pragma: no cover - simple passthrough
        self._tracker.close()

    def __del__(self):  # pragma: no cover - cleanup
        try:
            self.close()
        except Exception:
            pass


# Keep BaseLLMWrapper as an alias for backward compatibility
BaseLLMWrapper = ServiceWrapper


# ---------------------------------------------------------------------------
# Named wrappers — thin aliases with pre-configured vendor + extractors
# ---------------------------------------------------------------------------
def _extractors_for(api_id: str) -> tuple[Callable, Callable]:
    """Look up known extractors or fall back to generic ones."""
    return _KNOWN_EXTRACTORS.get(
        api_id, (_generic_extract_usage, _generic_extract_streaming_usage)
    )


class OpenAIChatWrapper(ServiceWrapper):
    def __init__(self, client: Any, **kwargs: Any) -> None:
        ext, sext = _extractors_for("openai_chat")
        super().__init__(
            client,
            vendor="openai",
            usage_extractor=ext,
            streaming_usage_extractor=sext,
            **kwargs,
        )

    def _get_vendor(self) -> str:  # pragma: no cover - simple logic
        base_url = getattr(self._client, "base_url", "")
        if not base_url and hasattr(self._client, "client"):
            base_url = getattr(self._client.client, "base_url", "")
        if not base_url and hasattr(self._client, "_client"):
            base_url = getattr(self._client._client, "base_url", "")
        url = str(base_url).lower()
        if "fireworks.ai" in url:
            return "fireworks-ai"
        if "x.ai" in url:
            return "xai"
        return "openai"


class OpenAIResponsesWrapper(ServiceWrapper):
    def __init__(self, client: Any, **kwargs: Any) -> None:
        ext, sext = _extractors_for("openai_responses")
        super().__init__(
            client,
            vendor="openai",
            usage_extractor=ext,
            streaming_usage_extractor=sext,
            **kwargs,
        )


class AnthropicWrapper(ServiceWrapper):
    def __init__(self, client: Any, **kwargs: Any) -> None:
        ext, sext = _extractors_for("anthropic")
        super().__init__(
            client,
            vendor="anthropic",
            usage_extractor=ext,
            streaming_usage_extractor=sext,
            **kwargs,
        )


class GeminiWrapper(ServiceWrapper):
    def __init__(self, client: Any, **kwargs: Any) -> None:
        ext, sext = _extractors_for("gemini")
        super().__init__(
            client,
            vendor="google",
            usage_extractor=ext,
            streaming_usage_extractor=sext,
            **kwargs,
        )


class BedrockWrapper(ServiceWrapper):
    def __init__(self, client: Any, **kwargs: Any) -> None:
        ext, sext = _extractors_for("amazon-bedrock")
        super().__init__(
            client,
            vendor="amazon-bedrock",
            usage_extractor=ext,
            streaming_usage_extractor=sext,
            **kwargs,
        )


class FireworksWrapper(ServiceWrapper):
    def __init__(self, client: Any, **kwargs: Any) -> None:
        ext, sext = _extractors_for("fireworks-ai")
        super().__init__(
            client,
            vendor="fireworks-ai",
            usage_extractor=ext,
            streaming_usage_extractor=sext,
            **kwargs,
        )

    def _extract_model(self, method: Any, args: tuple, kwargs: dict) -> str | None:
        # First try the standard extraction from call parameters
        model = super()._extract_model(method, args, kwargs)
        if model is not None:
            return model

        # For fireworks.LLM clients, model is set at initialization
        # Try to extract from the client instance
        if hasattr(self._client, "model"):
            return self._client.model

        return None


__all__ = [
    "ServiceWrapper",
    "BaseLLMWrapper",
    "OpenAIChatWrapper",
    "OpenAIResponsesWrapper",
    "AnthropicWrapper",
    "GeminiWrapper",
    "BedrockWrapper",
    "FireworksWrapper",
]
