from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import httpx

from ..models import (
    ApiCostEventOut,
    CostEventFilters,
    CostEventsResponse,
    CustomerFilters,
    CustomerIn,
    CustomerOut,
    CustomServiceFilter,
    CustomServiceIn,
    CustomServiceOut,
    CustomServiceSummaryOut,
    PaginatedResponse,
    PricingModel,
    RollupFilters,
    UsageEvent,
    UsageEventFilters,
    UsageLimitIn,
    UsageLimitOut,
    UsageLimitProgressOut,
    UsageRollup,
    WebhookEndpointCreate,
    WebhookEndpointOut,
    WebhookEndpointsResponse,
    WebhookEndpointUpdate,
)
from .base import BaseClient, catalog_params, webhooks_response
from .exceptions import APIRequestError


class AsyncCostManagerClient(BaseClient):
    """Asynchronous variant of :class:`CostManagerClient`."""

    def __init__(
        self,
        *,
        aicm_api_key: Optional[str] = None,
        aicm_api_base: Optional[str] = None,
        aicm_api_url: Optional[str] = None,
        aicm_ini_path: Optional[str] = None,
        session: Optional[httpx.AsyncClient] = None,
        proxies: Optional[dict[str, str]] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(
            aicm_api_key=aicm_api_key,
            aicm_api_base=aicm_api_base,
            aicm_api_url=aicm_api_url,
            aicm_ini_path=aicm_ini_path,
        )
        if session is None:
            proxy = None
            if proxies:
                proxy = next(iter(proxies.values()))
            session = httpx.AsyncClient(proxy=proxy)
        self.session = session
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "aicostmanager-python",
            }
        )
        if headers:
            self.session.headers.update(headers)

    async def close(self) -> None:
        await self.session.aclose()

    async def __aenter__(self) -> "AsyncCostManagerClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = path if path.startswith("http") else self.api_root + path
        resp = await self.session.request(method, url, **kwargs)
        if not resp.status_code or not (200 <= resp.status_code < 300):
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise APIRequestError(resp.status_code, detail)
        if resp.status_code == 204:
            return None
        return resp.json()

    async def _iter_paginated(self, path: str, **params: Any) -> AsyncIterator[dict]:
        while True:
            data = await self._request("GET", path, params=params)
            for item in data.get("results", []):
                yield item
            next_url = data.get("next")
            if not next_url:
                break
            if next_url.startswith(self.api_root):
                path = next_url[len(self.api_root) :]
            else:
                path = next_url
            params = {}

    async def get_triggered_limits(self) -> Dict[str, Any]:
        """Asynchronously fetch triggered limit information."""
        return await self._request("GET", "/triggered-limits")

    async def list_usage_events(
        self,
        filters: UsageEventFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> Any:
        if filters:
            if hasattr(filters, "model_dump"):
                params.update(filters.model_dump(exclude_none=True))
            else:
                params.update({k: v for k, v in filters.items() if v is not None})
        return await self._request("GET", "/usage/events/", params=params)

    async def list_usage_events_typed(
        self,
        filters: UsageEventFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> PaginatedResponse[UsageEvent]:
        """Typed variant of :meth:`list_usage_events`."""
        data = await self.list_usage_events(filters, **params)
        return PaginatedResponse[UsageEvent].model_validate(data)

    async def iter_usage_events(
        self,
        filters: UsageEventFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> AsyncIterator[UsageEvent]:
        if filters:
            if hasattr(filters, "model_dump"):
                params.update(filters.model_dump(exclude_none=True))
            else:
                params.update({k: v for k, v in filters.items() if v is not None})
        async for item in self._iter_paginated("/usage/events/", **params):
            yield UsageEvent.model_validate(item)

    async def get_usage_event(self, event_id: str) -> UsageEvent:
        data = await self._request("GET", f"/usage/event/{event_id}/")
        return UsageEvent.model_validate(data)

    async def list_usage_rollups(
        self,
        filters: RollupFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> Any:
        if filters:
            if hasattr(filters, "model_dump"):
                params.update(filters.model_dump(exclude_none=True))
            else:
                params.update({k: v for k, v in filters.items() if v is not None})
        return await self._request("GET", "/usage/rollups/", params=params)

    async def list_usage_rollups_typed(
        self,
        filters: RollupFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> PaginatedResponse[UsageRollup]:
        """Typed variant of :meth:`list_usage_rollups`."""
        data = await self.list_usage_rollups(filters, **params)
        return PaginatedResponse[UsageRollup].model_validate(data)

    async def iter_usage_rollups(
        self,
        filters: RollupFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> AsyncIterator[UsageRollup]:
        if filters:
            if hasattr(filters, "model_dump"):
                params.update(filters.model_dump(exclude_none=True))
            else:
                params.update({k: v for k, v in filters.items() if v is not None})
        async for item in self._iter_paginated("/usage/rollups/", **params):
            yield UsageRollup.model_validate(item)

    async def list_customers(
        self,
        filters: CustomerFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> Any:
        if filters:
            if hasattr(filters, "model_dump"):
                params.update(filters.model_dump(exclude_none=True))
            else:
                params.update({k: v for k, v in filters.items() if v is not None})
        return await self._request("GET", "/customers/", params=params)

    async def list_customers_typed(
        self,
        filters: CustomerFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> PaginatedResponse[CustomerOut]:
        """Typed variant of :meth:`list_customers`."""
        data = await self.list_customers(filters, **params)
        return PaginatedResponse[CustomerOut].model_validate(data)

    async def iter_customers(self, **params: Any) -> AsyncIterator[CustomerOut]:
        async for item in self._iter_paginated("/customers/", **params):
            yield CustomerOut.model_validate(item)

    async def create_customer(self, data: CustomerIn | Dict[str, Any]) -> CustomerOut:
        payload = data.model_dump(mode="json") if isinstance(data, CustomerIn) else data
        resp = await self._request("POST", "/customers/", json=payload)
        return CustomerOut.model_validate(resp)

    async def get_customer(self, customer_id: str) -> CustomerOut:
        data = await self._request("GET", f"/customers/{customer_id}/")
        return CustomerOut.model_validate(data)

    async def update_customer(
        self, customer_id: str, data: CustomerIn | Dict[str, Any]
    ) -> CustomerOut:
        payload = data.model_dump(mode="json") if isinstance(data, CustomerIn) else data
        resp = await self._request("PUT", f"/customers/{customer_id}/", json=payload)
        return CustomerOut.model_validate(resp)

    async def delete_customer(self, customer_id: str) -> None:
        await self._request("DELETE", f"/customers/{customer_id}/")
        return None

    async def list_usage_limits(self) -> Iterable[UsageLimitOut]:
        data = await self._request("GET", "/usage-limits/")
        return [UsageLimitOut.model_validate(i) for i in data]

    async def create_usage_limit(
        self, data: UsageLimitIn | Dict[str, Any]
    ) -> UsageLimitOut:
        payload = (
            data.model_dump(mode="json") if isinstance(data, UsageLimitIn) else data
        )
        resp = await self._request("POST", "/usage-limits/", json=payload)
        return UsageLimitOut.model_validate(resp)

    async def get_usage_limit(self, limit_id: str) -> UsageLimitOut:
        data = await self._request("GET", f"/usage-limits/{limit_id}/")
        return UsageLimitOut.model_validate(data)

    async def update_usage_limit(
        self, limit_id: str, data: UsageLimitIn | Dict[str, Any]
    ) -> UsageLimitOut:
        payload = (
            data.model_dump(mode="json") if isinstance(data, UsageLimitIn) else data
        )
        resp = await self._request("PUT", f"/usage-limits/{limit_id}/", json=payload)
        return UsageLimitOut.model_validate(resp)

    async def delete_usage_limit(self, limit_id: str) -> None:
        await self._request("DELETE", f"/usage-limits/{limit_id}/")
        return None

    async def list_usage_limit_progress(self) -> Iterable[UsageLimitProgressOut]:
        data = await self._request("GET", "/usage-limits/progress/")
        return [UsageLimitProgressOut.model_validate(i) for i in data]

    # Pricing catalog
    async def list_models(
        self, *, provider: Optional[str] = None, since: Any = None
    ) -> List[PricingModel]:
        """Every model in the AICM pricing catalog, across all providers.

        ``since`` (a datetime, date or ISO 8601 string; naive means UTC) keeps
        only models AICM first listed at or after it. Pass the newest
        ``first_seen_at`` from the previous call to poll for new models.
        ``provider`` takes a provider id such as ``"anthropic"``. The
        unfiltered catalog is over 10,000 models.
        """
        data = await self._request(
            "GET", "/pricing/models", params=catalog_params(provider, since)
        )
        return [PricingModel.model_validate(m) for m in data["models"]]

    async def get_model(self, model_id: str) -> PricingModel:
        """One catalog model by its id, ``<provider_id>::<model_name>``."""
        data = await self._request("GET", f"/pricing/models/{model_id}")
        return PricingModel.model_validate(data)

    # Cost events methods
    async def list_cost_events(
        self,
        filters: CostEventFilters | Dict[str, Any] | None = None,
        **params: Any,
    ) -> CostEventsResponse:
        if filters:
            if hasattr(filters, "model_dump"):
                params.update(filters.model_dump(exclude_none=True))
            else:
                params.update({k: v for k, v in filters.items() if v is not None})
        data = await self._request("GET", "/costs/", params=params)
        return CostEventsResponse.model_validate(data)

    async def list_cost_events_by_response_id(
        self, response_id: str
    ) -> List[ApiCostEventOut]:
        data = await self._request("GET", f"/cost-events/{response_id}")
        return [ApiCostEventOut.model_validate(i) for i in data]

    # Webhook methods
    async def _webhooks_path(self, webhook_uuid: Optional[str] = None) -> str:
        """Webhooks live under the API key's team, which is looked up once."""
        if self._team_slug is None:
            teams = await self._request("GET", "/teams/")
            self._team_slug = teams[0]["slug"]
        path = f"/teams/{self._team_slug}/webhooks/"
        return f"{path}{webhook_uuid}/" if webhook_uuid else path

    async def create_webhook_endpoint(
        self, data: WebhookEndpointCreate | Dict[str, Any]
    ) -> WebhookEndpointOut:
        payload = (
            data.model_dump(mode="json")
            if isinstance(data, WebhookEndpointCreate)
            else data
        )
        resp = await self._request("POST", await self._webhooks_path(), json=payload)
        return WebhookEndpointOut.model_validate(resp)

    async def list_webhook_endpoints(
        self, active_only: bool = False
    ) -> WebhookEndpointsResponse:
        data = await self._request("GET", await self._webhooks_path())
        return WebhookEndpointsResponse.model_validate(
            webhooks_response(data, active_only)
        )

    async def get_webhook_endpoint(self, webhook_uuid: str) -> WebhookEndpointOut:
        data = await self._request("GET", await self._webhooks_path(webhook_uuid))
        return WebhookEndpointOut.model_validate(data)

    async def update_webhook_endpoint(
        self, webhook_uuid: str, data: WebhookEndpointUpdate | Dict[str, Any]
    ) -> WebhookEndpointOut:
        payload = (
            data.model_dump(mode="json", exclude_none=True)
            if isinstance(data, WebhookEndpointUpdate)
            else data
        )
        resp = await self._request(
            "PATCH", await self._webhooks_path(webhook_uuid), json=payload
        )
        return WebhookEndpointOut.model_validate(resp)

    async def delete_webhook_endpoint(self, webhook_uuid: str) -> None:
        await self._request("DELETE", await self._webhooks_path(webhook_uuid))
        return None

    # Custom services methods
    async def list_custom_services(
        self,
        filters: CustomServiceFilter | Dict[str, Any] | None = None,
        **params: Any,
    ) -> List[CustomServiceSummaryOut]:
        if filters:
            if hasattr(filters, "model_dump"):
                params.update(filters.model_dump(exclude_none=True))
            else:
                params.update({k: v for k, v in filters.items() if v is not None})
        data = await self._request("GET", "/custom-services/", params=params)
        return [CustomServiceSummaryOut.model_validate(i) for i in data]

    async def create_custom_service(
        self, data: CustomServiceIn | Dict[str, Any]
    ) -> CustomServiceOut:
        payload = (
            data.model_dump(mode="json") if isinstance(data, CustomServiceIn) else data
        )
        resp = await self._request("POST", "/custom-services/", json=payload)
        return CustomServiceOut.model_validate(resp)

    async def get_custom_service(self, uuid: str) -> CustomServiceOut:
        data = await self._request("GET", f"/custom-services/{uuid}/")
        return CustomServiceOut.model_validate(data)

    async def update_custom_service(
        self, uuid: str, data: CustomServiceIn | Dict[str, Any]
    ) -> CustomServiceOut:
        payload = (
            data.model_dump(mode="json") if isinstance(data, CustomServiceIn) else data
        )
        resp = await self._request("PUT", f"/custom-services/{uuid}/", json=payload)
        return CustomServiceOut.model_validate(resp)

    async def delete_custom_service(self, uuid: str) -> None:
        await self._request("DELETE", f"/custom-services/{uuid}/")
        return None

    async def get_openapi_schema(self) -> Any:
        return await self._request("GET", "/openapi.json")
