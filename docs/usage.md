# Using the Python SDK

Before you begin, sign up for a free account at [AICostManager](https://aicostmanager.com)
and obtain an API key. Export it as `AICM_API_KEY` or pass it directly to
`CostManagerClient` or `Tracker`.

Install with:

```bash
# Using uv (recommended)
uv pip install aicostmanager
# or add to a project
uv add aicostmanager
```

Create a client:

```python
from aicostmanager import CostManagerClient
client = CostManagerClient()

# asynchronous
from aicostmanager import AsyncCostManagerClient
# async with AsyncCostManagerClient() as client:
#     ...
```

You can override defaults when instantiating:

```python
client = CostManagerClient(
    aicm_api_key="sk-api01-...",
    aicm_api_base="https://staging.aicostmanager.com",
    aicm_api_url="/api/v1",
    aicm_ini_path="/path/to/AICM.INI",
    proxies={"http": "http://proxy"},
    headers={"X-Test": "1"},
)
```

Example request:

```python
from datetime import date
from aicostmanager.models import CustomerIn, UsageEventFilters

new_customer = CustomerIn(customer_key="cust1", name="Example")
created = client.create_customer(new_customer)
print(created.uuid)

filters = UsageEventFilters(customer_key="cust1", start_date=date(2024, 1, 1), limit=100)
for event in client.iter_usage_events(filters):
    print(event.event_id)

```

## LLM wrappers

Drop-in wrappers track usage automatically for popular LLM SDK clients:

```python
from aicostmanager import OpenAIChatWrapper
from openai import OpenAI

client = OpenAI()
wrapper = OpenAIChatWrapper(client)
resp = wrapper.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)

wrapper.close()  # required only for queued delivery
```

See [LLM Wrappers](llm_wrappers.md) for the full list of supported providers.

## FastAPI integration

See [Manual Usage Tracking](tracker.md) for a detailed guide on using the `Tracker` class.

When recording custom usage with :class:`Tracker` in a FastAPI application,
create the tracker during application startup so configuration loading doesn't
block individual requests. During shutdown, stop the background delivery using
``Tracker.close``:

```python
from fastapi import FastAPI
from aicostmanager import Tracker

app = FastAPI()


@app.on_event("startup")
async def startup() -> None:
    app.state.tracker = Tracker()


@app.on_event("shutdown")
def shutdown() -> None:
    app.state.tracker.close()


@app.post("/track")
async def track(payload: dict) -> dict:
    app.state.tracker.track("openai::gpt-4o-mini", payload)
    return {"status": "ok"}
```


## Model Catalog

`list_models()` returns every model in the AICM pricing catalog, across all
providers, with its current rates, context window and capabilities. It needs
no API key of its own. The full catalog is over 10,000 models, so to watch for
new models, pass `since` and keep the newest `first_seen_at` for the next call:

```python
from datetime import datetime, timezone

models = client.list_models()                      # everything
anthropic = client.list_models(provider="anthropic")

cursor = datetime(2026, 11, 1, tzinfo=timezone.utc)
new = client.list_models(since=cursor)             # first listed at or after cursor
for m in new:
    print(m.id, m.input_price and m.input_price.cost, m.first_seen_at)
if new:
    cursor = max(m.first_seen_at for m in new)

model = client.get_model("anthropic::claude-fable-5-1")
```

`since` takes a `datetime`, a `date` or an ISO 8601 string; a time without a
zone is UTC. A model's `id` is the service key usage is tracked under.

