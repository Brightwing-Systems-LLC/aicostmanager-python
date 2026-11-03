from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .exceptions import MissingConfiguration
from ..ini_manager import IniManager


class BaseClient:
    """Shared initialization logic for SDK clients.

    The ``api_base`` and ``api_url`` values are resolved in the following order:

    1. Explicit ``aicm_api_base``/``aicm_api_url`` parameters
    2. Values from ``AICM.INI`` (``tracker`` section)
    3. Environment variables ``AICM_API_BASE``/``AICM_API_URL``
    4. Defaults ``https://aicostmanager.com`` and ``/api/v1``
    """

    _team_slug: Optional[str] = None

    def __init__(
        self,
        *,
        aicm_api_key: Optional[str] = None,
        aicm_api_base: Optional[str] = None,
        aicm_api_url: Optional[str] = None,
        aicm_ini_path: Optional[str] = None,
    ) -> None:
        self.ini_manager = IniManager(IniManager.resolve_path(aicm_ini_path))
        self.ini_path = self.ini_manager.ini_path

        self.api_key = aicm_api_key or os.getenv("AICM_API_KEY")

        def _get(option: str, default: str | None = None) -> str | None:
            val = self.ini_manager.get_option("tracker", option)
            if val is not None:
                return val
            return os.getenv(option, default)

        self.api_base = aicm_api_base or _get(
            "AICM_API_BASE", "https://aicostmanager.com"
        )
        self.api_url = aicm_api_url or _get("AICM_API_URL", "/api/v1")
        if not self.api_key:
            raise MissingConfiguration(
                "API key not provided. Set AICM_API_KEY environment variable or pass aicm_api_key"
            )

    @property
    def api_root(self) -> str:
        """Return the combined AICostManager API base URL."""
        return self.api_base.rstrip("/") + self.api_url

    def _store_triggered_limits(self, triggered_limits_response) -> None:
        """Persist triggered limits using the configuration manager."""
        from ..config_manager import ConfigManager

        cfg_mgr = ConfigManager(ini_path=self.ini_path)
        if isinstance(triggered_limits_response, dict):
            tl_data = triggered_limits_response.get(
                "triggered_limits", triggered_limits_response
            )
        else:
            tl_data = triggered_limits_response
        cfg_mgr.write_triggered_limits(tl_data)


def catalog_params(provider: Optional[str], since: Any) -> Dict[str, str]:
    """Query params for ``/pricing/models``; ``since`` may be a date, datetime or string."""
    params: Dict[str, str] = {}
    if provider:
        params["provider"] = provider
    if since is not None:
        params["since"] = since if isinstance(since, str) else since.isoformat()
    return params


def webhooks_response(data: List[dict], active_only: bool) -> Dict[str, Any]:
    """Shape the server's plain list as ``WebhookEndpointsResponse``."""
    endpoints = [e for e in data if e.get("active") or not active_only]
    return {
        "endpoints": endpoints,
        "total_count": len(endpoints),
        "active_count": sum(1 for e in endpoints if e.get("active")),
    }
