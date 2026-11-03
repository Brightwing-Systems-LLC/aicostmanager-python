from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class WebhookEndpointOut(BaseModel):
    """Schema for webhook endpoint response"""

    uuid: str
    url: str
    purpose: str = "limit_notification"
    active: bool = True
    tolerance: int = 300
    created_at: str
    updated_at: str

    model_config = ConfigDict(from_attributes=True, extra="ignore")


class WebhookEndpointCreate(BaseModel):
    """Schema for creating webhook endpoint"""

    url: str
    secret: str
    purpose: str = "limit_notification"
    active: bool = True
    tolerance: int = 300

    model_config = ConfigDict(extra="forbid")


class WebhookEndpointsResponse(BaseModel):
    """The team's webhook endpoints, with counts computed by the client."""

    endpoints: List[WebhookEndpointOut]
    total_count: int
    active_count: int

    model_config = ConfigDict(from_attributes=True)


class WebhookEndpointUpdate(BaseModel):
    """Schema for updating webhook endpoint"""

    url: Optional[str] = None
    secret: Optional[str] = None
    purpose: Optional[str] = None
    active: Optional[bool] = None
    tolerance: Optional[int] = None

    model_config = ConfigDict(extra="forbid")
