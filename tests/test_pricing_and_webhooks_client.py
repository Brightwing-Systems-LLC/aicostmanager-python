"""The client calls the routes the AICM server actually serves.

The catalog is `/pricing/models` (the old `/vendors/`, `/services/` and
`/service-costs/` routes are gone), and webhooks live under the API key's
team as `/teams/{slug}/webhooks/`, updated with PATCH.
"""

import asyncio
from datetime import date, datetime, timezone

import pytest

from aicostmanager.client import AsyncCostManagerClient, CostManagerClient
from aicostmanager.models import PricingModel, WebhookEndpointOut

MODEL = {
    "id": "anthropic::claude-fable-5-1",
    "model_name": "claude-fable-5-1",
    "provider": "Anthropic",
    "provider_id": "anthropic",
    "first_seen_at": "2026-11-02T15:30:00+00:00",
    "input_price": {
        "name": "input",
        "cost": 3.0,
        "per_quantity": 1000000,
        "currency": "USD",
        "unit": "token",
    },
    "output_price": None,
    "context_window": 200000,
    "max_output_tokens": 64000,
    "metadata": {"release_date": "2026-11-01", "supports_reasoning": True},
    "benchmark": None,
    "some_future_field": 1,
}

WEBHOOK = {
    "uuid": "w1",
    "url": "https://example.com/hook",
    "purpose": "limit_notification",
    "active": True,
    "tolerance": 300,
    "created_at": "2026-11-01T00:00:00+00:00",
    "updated_at": "2026-11-01T00:00:00+00:00",
}


class FakeResponse:
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = ""

    def json(self):
        return self._data


class FakeServer:
    """Answers like the AICM server and records each request."""

    def __init__(self, root):
        self.root = root
        self.calls = []

    def answer(self, method, url, **kwargs):
        path = url[len(self.root) :]
        self.calls.append((method, path, kwargs.get("params"), kwargs.get("json")))
        if path == "/triggered-limits":
            return FakeResponse({})
        if path == "/teams/":
            return FakeResponse([{"name": "Acme", "slug": "acme"}])
        if path == "/pricing/models":
            return FakeResponse({"count": 1, "models": [MODEL], "_links": {}})
        if path == f"/pricing/models/{MODEL['id']}":
            return FakeResponse(dict(MODEL, _links={}))
        if path == "/teams/acme/webhooks/":
            if method == "GET":
                return FakeResponse([WEBHOOK, dict(WEBHOOK, uuid="w2", active=False)])
            return FakeResponse(WEBHOOK, 201)
        if path == "/teams/acme/webhooks/w1/":
            return FakeResponse(
                None if method == "DELETE" else WEBHOOK,
                204 if method == "DELETE" else 200,
            )
        return FakeResponse({"detail": "Not Found"}, 404)

    def without_startup(self):
        return [c for c in self.calls if c[1] != "/triggered-limits"]


@pytest.fixture
def sync_client(monkeypatch):
    monkeypatch.setenv("AICM_API_KEY", "sk")
    holder = {}

    def request(session, method, url, **kwargs):
        return holder["server"].answer(method, url, **kwargs)

    monkeypatch.setattr("requests.Session.request", request)
    holder["server"] = FakeServer("")
    client = CostManagerClient()
    holder["server"].root = client.api_root
    holder["server"].calls = []
    client.server = holder["server"]
    return client


@pytest.fixture
def async_client(monkeypatch):
    monkeypatch.setenv("AICM_API_KEY", "sk")
    client = AsyncCostManagerClient()
    server = FakeServer(client.api_root)

    async def request(session, method, url, **kwargs):
        return server.answer(method, url, **kwargs)

    monkeypatch.setattr("httpx.AsyncClient.request", request)
    client.server = server
    return client


class TestListModels:
    def test_lists_every_model(self, sync_client):
        models = sync_client.list_models()
        assert [m.id for m in models] == [MODEL["id"]]
        assert sync_client.server.without_startup() == [
            ("GET", "/pricing/models", {}, None)
        ]

    def test_model_fields(self, sync_client):
        m = sync_client.list_models()[0]
        assert isinstance(m, PricingModel)
        assert m.provider_id == "anthropic"
        assert m.first_seen_at == datetime(2026, 11, 2, 15, 30, tzinfo=timezone.utc)
        assert m.input_price.cost == 3.0
        assert m.output_price is None
        assert m.metadata.release_date == "2026-11-01"
        assert m.metadata.supports_reasoning is True

    @pytest.mark.parametrize(
        "since,sent",
        [
            (
                datetime(2026, 11, 1, 12, tzinfo=timezone.utc),
                "2026-11-01T12:00:00+00:00",
            ),
            (date(2026, 11, 1), "2026-11-01"),
            ("2026-11-01T12:00:00Z", "2026-11-01T12:00:00Z"),
        ],
    )
    def test_since_is_sent_as_iso(self, sync_client, since, sent):
        sync_client.list_models(since=since)
        assert sync_client.server.without_startup()[0][2] == {"since": sent}

    def test_provider_filter(self, sync_client):
        sync_client.list_models(provider="anthropic")
        assert sync_client.server.without_startup()[0][2] == {"provider": "anthropic"}

    def test_get_model(self, sync_client):
        assert sync_client.get_model(MODEL["id"]).model_name == "claude-fable-5-1"

    def test_async_list_models(self, async_client):
        async def run():
            return await async_client.list_models(since=date(2026, 11, 1))

        models = asyncio.run(run())
        assert models[0].provider_id == "anthropic"
        assert async_client.server.calls == [
            ("GET", "/pricing/models", {"since": "2026-11-01"}, None)
        ]

    def test_async_get_model(self, async_client):
        model = asyncio.run(async_client.get_model(MODEL["id"]))
        assert model.id == MODEL["id"]


def test_dead_catalog_methods_are_gone():
    for name in ("list_vendors", "list_vendor_services", "list_service_costs"):
        assert not hasattr(CostManagerClient, name)
        assert not hasattr(AsyncCostManagerClient, name)


class TestWebhooks:
    def test_list_uses_the_keys_team(self, sync_client):
        result = sync_client.list_webhook_endpoints()
        assert [e.uuid for e in result.endpoints] == ["w1", "w2"]
        assert (result.total_count, result.active_count) == (2, 1)
        paths = [c[1] for c in sync_client.server.without_startup()]
        assert paths == ["/teams/", "/teams/acme/webhooks/"]

    def test_team_slug_is_looked_up_once(self, sync_client):
        sync_client.list_webhook_endpoints()
        sync_client.get_webhook_endpoint("w1")
        paths = [c[1] for c in sync_client.server.without_startup()]
        assert paths.count("/teams/") == 1

    def test_active_only(self, sync_client):
        result = sync_client.list_webhook_endpoints(active_only=True)
        assert [e.uuid for e in result.endpoints] == ["w1"]

    def test_create_sends_purpose(self, sync_client):
        out = sync_client.create_webhook_endpoint(
            {"url": "https://example.com/hook", "secret": "s"}
        )
        assert isinstance(out, WebhookEndpointOut)
        assert out.purpose == "limit_notification"
        method, path, _, body = sync_client.server.without_startup()[-1]
        assert (method, path) == ("POST", "/teams/acme/webhooks/")
        assert body["url"] == "https://example.com/hook"

    def test_update_is_patch(self, sync_client):
        sync_client.update_webhook_endpoint("w1", {"active": False})
        method, path, _, body = sync_client.server.without_startup()[-1]
        assert (method, path, body) == (
            "PATCH",
            "/teams/acme/webhooks/w1/",
            {"active": False},
        )

    def test_delete(self, sync_client):
        assert sync_client.delete_webhook_endpoint("w1") is None
        assert sync_client.server.without_startup()[-1][:2] == (
            "DELETE",
            "/teams/acme/webhooks/w1/",
        )

    def test_async_update_is_patch(self, async_client):
        asyncio.run(async_client.update_webhook_endpoint("w1", {"active": False}))
        assert async_client.server.calls[-1][:2] == (
            "PATCH",
            "/teams/acme/webhooks/w1/",
        )

    def test_async_list(self, async_client):
        result = asyncio.run(async_client.list_webhook_endpoints())
        assert result.total_count == 2
