"""Anthropic streams record their whole usage, once.

``messages.stream()`` returns a context manager, which the proxy could not read
usage from, so those calls were never tracked. ``messages.create(stream=True)``
returned a stream whose first usage-bearing event (``message_start``) was
tracked, so every call was recorded with about one output token. These tests
drive the real Anthropic SDK over a fake transport.
"""

import asyncio
import json

import pytest

anthropic = pytest.importorskip("anthropic")
httpx = pytest.importorskip("httpx")

from aicostmanager.wrappers import AnthropicWrapper  # noqa: E402

MESSAGE_ID = "msg_stream_usage"
MODEL = "claude-haiku-5-5"
FINAL_USAGE = {
    "input_tokens": 12,
    "cache_creation_input_tokens": 300,
    "cache_read_input_tokens": 2000,
    "output_tokens": 42,
}


def _events():
    return [
        {
            "type": "message_start",
            "message": {
                "id": MESSAGE_ID,
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 12,
                    "cache_creation_input_tokens": 300,
                    "cache_read_input_tokens": 2000,
                    "output_tokens": 1,
                },
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello there"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 42},
        },
        {"type": "message_stop"},
    ]


def _sse_body():
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in _events()
    ).encode()


def _handler(request):
    if not json.loads(request.content).get("stream"):
        message = dict(_events()[0]["message"])
        message.update(
            content=[{"type": "text", "text": "Hello there"}],
            stop_reason="end_turn",
            usage=FINAL_USAGE,
        )
        return httpx.Response(200, json=message)
    return httpx.Response(
        200, content=_sse_body(), headers={"content-type": "text/event-stream"}
    )


class RecordingTracker:
    def __init__(self):
        self.calls = []
        self.error = None

    def track(self, service_key, usage, **kwargs):
        if self.error is not None:
            raise self.error
        self.calls.append({"service_key": service_key, "usage": usage, **kwargs})

    async def track_async(self, service_key, usage, **kwargs):
        self.track(service_key, usage, **kwargs)

    def close(self):
        pass


@pytest.fixture
def tracker():
    return RecordingTracker()


def _wrapped(tracker):
    client = anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    wrapper = AnthropicWrapper(client, tracker=tracker)
    wrapper.set_customer_key("customer-1")
    wrapper.set_context({"run": "r1"})
    return wrapper


def _async_wrapped(tracker):
    client = anthropic.AsyncAnthropic(
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
    )
    wrapper = AnthropicWrapper(client, tracker=tracker)
    wrapper.set_customer_key("customer-1")
    wrapper.set_context({"run": "r1"})
    return wrapper


def _assert_one_final_event(tracker, context=None):
    assert len(tracker.calls) == 1, tracker.calls
    call = tracker.calls[0]
    assert call["service_key"] == f"anthropic::{MODEL}"
    assert call["response_id"] == MESSAGE_ID
    assert call["customer_key"] == "customer-1"
    assert call["context"] == (context or {"run": "r1"})
    for field, value in FINAL_USAGE.items():
        assert call["usage"][field] == value, field


def _request():
    return {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Hi"}],
    }


def test_stream_helper_records_the_final_message_once(tracker):
    wrapper = _wrapped(tracker)

    with wrapper.messages.stream(**_request()) as stream:
        message = stream.get_final_message()

    assert message.content[0].text == "Hello there"
    _assert_one_final_event(tracker)


def test_stream_helper_text_iteration_records_once(tracker):
    wrapper = _wrapped(tracker)

    with wrapper.messages.stream(**_request()) as stream:
        text = "".join(stream.text_stream)

    assert text == "Hello there"
    _assert_one_final_event(tracker)


def test_stream_helper_left_early_records_what_was_billed(tracker):
    wrapper = _wrapped(tracker)

    with wrapper.messages.stream(**_request()) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                break

    assert len(tracker.calls) == 1, tracker.calls
    usage = tracker.calls[0]["usage"]
    assert usage["input_tokens"] == 12
    assert usage["cache_read_input_tokens"] == 2000


def test_stream_helper_keeps_the_context_of_the_call(tracker):
    wrapper = _wrapped(tracker)

    manager = wrapper.messages.stream(**_request())
    wrapper.set_context({"run": "a-later-run"})
    with manager as stream:
        stream.get_final_message()

    _assert_one_final_event(tracker)


def test_stream_helper_tracking_error_does_not_hide_the_callers_error(tracker):
    wrapper = _wrapped(tracker)
    tracker.error = RuntimeError("tracking down")

    with pytest.raises(ValueError, match="caller"):
        with wrapper.messages.stream(**_request()) as stream:
            next(iter(stream))
            raise ValueError("caller")


def test_async_stream_helper_records_the_final_message_once(tracker):
    wrapper = _async_wrapped(tracker)

    async def run():
        async with wrapper.messages.stream(**_request()) as stream:
            return await stream.get_final_message()

    message = asyncio.run(run())

    assert message.content[0].text == "Hello there"
    _assert_one_final_event(tracker)


def test_raw_stream_records_the_final_usage_once(tracker):
    wrapper = _wrapped(tracker)

    events = list(wrapper.messages.create(stream=True, **_request()))

    assert [e.type for e in events][-1] == "message_stop"
    _assert_one_final_event(tracker)


def test_async_raw_stream_records_the_final_usage_once(tracker):
    wrapper = _async_wrapped(tracker)

    async def run():
        stream = await wrapper.messages.create(stream=True, **_request())
        return [event async for event in stream]

    events = asyncio.run(run())

    assert events[-1].type == "message_stop"
    _assert_one_final_event(tracker)


def test_create_records_the_response_once(tracker):
    wrapper = _wrapped(tracker)

    message = wrapper.messages.create(**_request())

    assert message.content[0].text == "Hello there"
    _assert_one_final_event(tracker)


def test_async_create_records_the_response_not_the_coroutine(tracker):
    wrapper = _async_wrapped(tracker)

    message = asyncio.run(wrapper.messages.create(**_request()))

    assert message.content[0].text == "Hello there"
    _assert_one_final_event(tracker)
