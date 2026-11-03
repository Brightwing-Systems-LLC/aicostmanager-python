from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class CostUnitOut(BaseModel):
    """One rate of a model, e.g. ``cost`` USD per ``per_quantity`` input tokens."""

    name: str
    cost: float
    per_quantity: int
    currency: str = "USD"
    unit: str = "token"

    model_config = ConfigDict(extra="ignore")


class PricingModelMetadata(BaseModel):
    release_date: Optional[str] = None
    last_updated_date: Optional[str] = None
    knowledge_cutoff: Optional[str] = None
    model_family: Optional[str] = None
    model_status: Optional[str] = None
    default_temperature: Optional[float] = None
    supports_reasoning: bool = False
    supports_tool_call: bool = False
    supports_structured_output: bool = False
    is_open_weights: bool = False
    input_modalities: List[str] = []
    output_modalities: List[str] = []
    max_input_tokens: Optional[int] = None

    model_config = ConfigDict(extra="ignore")


class PricingModel(BaseModel):
    """A model in the AICM pricing catalog.

    ``id`` is the service key, ``<provider_id>::<model_name>``, the same key
    usage is tracked under. ``first_seen_at`` is when AICM first listed the
    model; keep the newest one as the ``since`` of the next poll.
    """

    id: str
    model_name: str
    provider: str
    provider_id: Optional[str] = None
    first_seen_at: Optional[datetime] = None
    input_price: Optional[CostUnitOut] = None
    output_price: Optional[CostUnitOut] = None
    input_audio_price: Optional[float] = None
    output_audio_price: Optional[float] = None
    context_over_200k_price: Optional[float] = None
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    metadata: Optional[PricingModelMetadata] = None
    benchmark: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="ignore", protected_namespaces=())
