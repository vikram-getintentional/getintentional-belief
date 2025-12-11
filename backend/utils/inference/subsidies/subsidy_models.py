"""
Canonical data structures for subsidies / exogenous triggers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SubsidyType(str, Enum):
    ZMOT = "zmot"
    INTERNAL_MANDATE = "internal_mandate"
    MARKET_SHIFT = "market_shift"
    RELATIONSHIP = "relationship"
    TOOLING = "tooling"


class SubsidyScope(str, Enum):
    GLOBAL = "global"
    SEGMENT = "segment"
    ACCOUNT = "account"
    PERSONA = "persona"


class SubsidyTargetBelief(BaseModel):
    persona_archetype_id: Optional[str] = None
    persona_title_pattern: Optional[str] = None
    from_belief: Optional[str] = None
    to_belief: Optional[str] = None
    belief_edge_tag: Optional[str] = None


class SubsidyIntensity(BaseModel):
    magnitude: float
    starts_at: datetime
    ends_at: Optional[datetime] = None
    decay_half_life_days: Optional[float] = None


class ZMOTPayload(BaseModel):
    observable_signal: str
    trigger_category: str
    inferred_pain_tags: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class SubsidyEvent(BaseModel):
    id: str
    type: SubsidyType
    scope: SubsidyScope

    account_id: Optional[str] = None
    segment_key: Optional[str] = None
    conditions: Dict[str, Any] = Field(default_factory=dict)

    targets: List[SubsidyTargetBelief]
    intensity: SubsidyIntensity

    source: str
    description: str
    created_at: datetime
    last_updated_at: datetime

    zmot_payload: Optional[ZMOTPayload] = None

