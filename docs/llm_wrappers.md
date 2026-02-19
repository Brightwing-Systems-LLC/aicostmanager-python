# LLM Wrappers

AICostManager provides drop-in wrappers for popular large language model (LLM)
SDK clients. Each wrapper embeds a :class:`~aicostmanager.tracker.Tracker`
so every call automatically records usage without requiring any additional code.
You continue to interact with the native client methods and properties while the
wrapper forwards the usage data in the background.

Wrappers automatically derive the ``service_key`` sent to the tracker from the
model you pass to the underlying client. The corresponding ``api_id`` for the
provider is built in.

The service key format is ``[vendor]::[model]``. For the OpenAI Chat wrapper the
vendor is inferred from the client's ``base_url`` so compatible providers like
Fireworks or xAI are attributed correctly. The other wrappers use fixed vendors:
``openai``, ``anthropic``, ``google``, ``fireworks-ai`` and ``amazon-bedrock``.

## Supported providers

| Wrapper | Native client | ``api_id`` |
| ------- | ------------- | ---------- |
| ``OpenAIChatWrapper`` | ``openai.OpenAI`` | ``openai_chat`` |
| ``OpenAIResponsesWrapper`` | ``openai.OpenAI`` | ``openai_responses`` |
| ``AnthropicWrapper`` | ``anthropic.Anthropic`` | ``anthropic`` |
| ``GeminiWrapper`` | ``google.genai.Client`` | ``gemini`` |
| ``FireworksWrapper`` | ``fireworks.Fireworks`` | ``fireworks-ai`` |
| ``BedrockWrapper`` | ``boto3.client('bedrock-runtime')`` | ``bedrock`` |

## Basic usage

```python
from aicostmanager import OpenAIChatWrapper
from openai import OpenAI

client = OpenAI()
wrapper = OpenAIChatWrapper(client)

resp = wrapper.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello"}],
)
print(resp.choices[0].message.content)

wrapper.close()  # required only for queued delivery
```

The wrapper proxies every attribute on the client, so existing code typically
requires only one extra line to instantiate the wrapper.  A tracker is created
internally using the ``AICM_API_KEY`` environment variable.  Pass ``aicm_api_key``,
``delivery_type`` or ``tracker`` to the wrapper constructor to override this
behaviour.

```python
# Use a queue-based delivery strategy
wrapper = OpenAIChatWrapper(client, delivery_type="PERSISTENT_QUEUE")
```

## Streaming responses

Wrappers handle streaming automatically.  Iterate over the returned stream just
as you would with the native client and usage will be tracked once the final
chunk containing usage arrives.

```python
stream = wrapper.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a short poem"}],
    stream=True,
    stream_options={"include_usage": True},
)

for event in stream:
    if event.type == "message.delta" and event.delta.get("content"):
        print(event.delta["content"], end="")
print()
```

## Inference blocking limits

Wrappers can prevent an inference from running when a matching triggered limit
exists. Enable this behaviour by setting
``AICM_ENABLE_INFERENCE_BLOCKING_LIMITS`` to ``true`` in your ``AICM.INI`` file.
When active, the wrapper checks locally cached triggered limits before making
the LLM call and raises :class:`~aicostmanager.client.exceptions.UsageLimitExceeded`
instead of executing the request.

Normal inferences incur minimal overhead and calls proceed as usual when no
limits are triggered or the setting is disabled.

## Asynchronous clients

If the underlying SDK provides ``async`` methods, the wrappers preserve that
behaviour:

```python
async def run():
    resp = await wrapper.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(resp.choices[0].message.content)
```

Streaming async iterables are also supported and will be tracked in the same
way.

## Other providers

Replace ``OpenAIChatWrapper`` with one of the other provider-specific wrappers
and supply the appropriate native client:

```python
from aicostmanager import (
    OpenAIResponsesWrapper,
    AnthropicWrapper,
    GeminiWrapper,
    FireworksWrapper,
    BedrockWrapper,
)

# OpenAI Responses
resp_wrapper = OpenAIResponsesWrapper(client)
resp_wrapper.responses.create(...)

# Anthropic
import anthropic
anth_wrapper = AnthropicWrapper(anthropic.Anthropic())
anth_wrapper.messages.create(model="claude-sonnet-4-5-20241022", ...)

# Gemini (google-genai SDK)
from google import genai
client = genai.Client()
gem_wrapper = GeminiWrapper(client)
gem_wrapper.models.generate_content(model="gemini-2.5-flash", contents="hello")

# Fireworks
from fireworks import Fireworks
fw_client = Fireworks()
fw_wrapper = FireworksWrapper(fw_client)
fw_wrapper.chat.completions.create(
    model="accounts/fireworks/models/deepseek-r1",
    messages=[{"role": "user", "content": "hi"}],
)

# Bedrock (Converse API)
import boto3
bed_client = boto3.client("bedrock-runtime", region_name="us-east-1")
bed_wrapper = BedrockWrapper(bed_client)
bed_wrapper.converse(
    modelId="anthropic.claude-sonnet-4-5-20241022-v2:0",
    messages=[{"role": "user", "content": [{"text": "hi"}]}],
)
```

Call ``wrapper.close()`` during shutdown when using the queue-based
``PERSISTENT_QUEUE`` delivery to flush buffered tracking data. With the default
immediate delivery, ``close`` is optional.
