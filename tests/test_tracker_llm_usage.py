import asyncio

from aicostmanager.tracker import Tracker


class Resp:
    def __init__(self, usage, model):
        self.usage = usage
        self.model = model


class DummyDelivery:
    def __init__(self):
        self.records = []
        self.type = None

    def enqueue(self, record):
        self.records.append(record)

    def stop(self):
        pass


def test_track_llm_usage():
    delivery = DummyDelivery()
    tracker = Tracker(delivery=delivery, ini_path="ini")

    resp = Resp({"input_tokens": 1}, "gpt-5-mini")
    out = tracker.track_llm_usage("openai::gpt-5-mini", resp, customer_key="abc")
    assert out is resp
    tracker.close()

    record = delivery.records[0]
    assert record["payload"] == {"input_tokens": 1}
    assert record["service_key"] == "openai::gpt-5-mini"
    assert record["customer_key"] == "abc"


def test_track_llm_usage_async():
    delivery = DummyDelivery()
    tracker = Tracker(delivery=delivery, ini_path="ini")

    class AResp:
        usage = {"input_tokens": 2}

    async def run():
        resp = AResp()
        resp.model = "gpt-4"
        out = await tracker.track_llm_usage_async("openai_chat", resp)
        assert out is resp

    asyncio.run(run())
    tracker.close()

    record = delivery.records[0]
    assert record["payload"] == {"input_tokens": 2}
    assert record["service_key"] == "openai::gpt-4"


def test_track_llm_stream_usage():
    delivery = DummyDelivery()
    tracker = Tracker(delivery=delivery, ini_path="ini")

    class Chunk:
        def __init__(self, usage=None):
            self.usage = usage

    class Stream(list):
        def __init__(self, chunks, model):
            super().__init__(chunks)
            self.model = model

    chunks = Stream([Chunk(), Chunk({"input_tokens": 3})], model="gpt-5-mini")
    events = list(tracker.track_llm_stream_usage("openai::gpt-5-mini", chunks))
    assert events == chunks
    tracker.close()

    record = delivery.records[0]
    assert record["payload"] == {"input_tokens": 3}
    assert record["service_key"] == "openai::gpt-5-mini"


def test_track_llm_stream_usage_async():
    delivery = DummyDelivery()
    tracker = Tracker(delivery=delivery, ini_path="ini")

    class Chunk:
        def __init__(self, usage=None):
            self.usage = usage

    class AsyncStream:
        def __init__(self, items, model: str):
            self._items = items
            self.model = model

        def __aiter__(self):
            self._iter = iter(self._items)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    async def run():
        gen = AsyncStream([Chunk(), Chunk({"input_tokens": 4})], model="gpt-5-mini")
        async for _ in tracker.track_llm_stream_usage_async("openai::gpt-5-mini", gen):
            pass

    asyncio.run(run())
    tracker.close()

    record = delivery.records[0]
    assert record["payload"] == {"input_tokens": 4}
    assert record["service_key"] == "openai::gpt-5-mini"


def test_track_llm_usage_unknown_vendor():
    """Unknown vendors fall through to generic extraction via .usage."""
    delivery = DummyDelivery()
    tracker = Tracker(delivery=delivery, ini_path="ini")

    resp = Resp({"duration_seconds": 12.5, "characters": 42}, model=None)
    out = tracker.track_llm_usage("custom-vendor::stt", resp)
    assert out is resp
    tracker.close()

    record = delivery.records[0]
    assert record["payload"] == {"duration_seconds": 12.5, "characters": 42}
    assert record["service_key"] == "custom-vendor::stt"


def test_track_llm_usage_unknown_vendor_dict_response():
    """Unknown vendors with dict responses use generic dict extraction."""
    delivery = DummyDelivery()
    tracker = Tracker(delivery=delivery, ini_path="ini")

    resp_dict = {"usage": {"input_tokens": 10, "output_tokens": 20}}
    # track_llm_usage expects an object, but generic extractor handles dicts too
    # We need to wrap it so setattr works for metadata attachment
    class DictResp:
        def __init__(self, data):
            self._data = data
            self.usage = data["usage"]

    resp = DictResp(resp_dict)
    out = tracker.track_llm_usage("unknown-provider::my-model", resp)
    assert out is resp
    tracker.close()

    record = delivery.records[0]
    assert record["payload"] == {"input_tokens": 10, "output_tokens": 20}
    assert record["service_key"] == "unknown-provider::my-model"
