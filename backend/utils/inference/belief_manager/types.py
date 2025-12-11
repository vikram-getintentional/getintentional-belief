"""
Shared type definitions for the belief manager pipeline.

These dataclasses represent the information exchanged between
per-engagement processing, SHM persistence, and global aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Literal


# ---------------------------------------------------------------------------
# Global parameter snapshot
# ---------------------------------------------------------------------------


@dataclass
class GlobalParams:
    version: int
    node_weights: Dict[str, Any] = field(default_factory=dict)
    edge_weights: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Persona learning & diagnostics
# ---------------------------------------------------------------------------


@dataclass
class PersonaCandidateStat:
    label: str
    titles: List[str] = field(default_factory=list)
    departments: List[str] = field(default_factory=list)
    account_count: int = 0
    episode_count: int = 0
    occurrence_count: int = 0
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    segments: Dict[str, int] = field(default_factory=dict)


@dataclass
class PersonaImpactMetrics:
    persona_id: str
    wolves_score: float = 0.0
    delta_win_bp: float = 0.0
    centrality: float = 0.0
    involvement_rate: float = 0.0
    blocker_rate: Optional[float] = None
    sample_size: int = 0
    last_updated_at: Optional[datetime] = None
    source: Literal["data_auto", "enrich_user", "initial_seed"] = "data_auto"


# ---------------------------------------------------------------------------
# Prediction + error records
# ---------------------------------------------------------------------------


@dataclass
class PersonaPrediction:
    account_id: str
    timestamp: datetime
    persona_sequence_before: Sequence[str]
    predicted_distribution: Mapping[str, float]
    predicted_topk: Sequence[str]
    rationale: Optional[str] = None
    candidate_paths: Optional[Sequence[Sequence[str]]] = None


@dataclass
class PredictionError:
    account_id: str
    timestamp: datetime
    persona_sequence_before: Sequence[str]
    predicted_distribution: Mapping[str, float]
    predicted_topk: Sequence[str]
    actual_persona: Optional[str]
    hit_at_1: bool
    hit_at_3: bool
    bucket: str
    log_loss: Optional[float] = None
    brier_score: Optional[float] = None
    rationale: Optional[str] = None
    log_likelihood: Optional[float] = None


# ---------------------------------------------------------------------------
# Local adjustments
# ---------------------------------------------------------------------------


@dataclass
class EdgeAdjustment:
    source_id: str
    target_id: str
    delta: float
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeAdjustment:
    node_id: str
    field: str
    delta: float
    band: Optional[str] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalAdjustment:
    account_id: str
    timestamp: datetime
    band: Optional[str]
    edges: List[EdgeAdjustment] = field(default_factory=list)
    nodes: List[NodeAdjustment] = field(default_factory=list)
    error: Optional[PredictionError] = None
    diagnostic: Dict[str, Any] = field(default_factory=dict)
    global_params_version: Optional[int] = None


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    account_id: str
    timestamp: datetime
    meta: Dict[str, Any]
    engagement: Dict[str, Any]
    prediction: Optional[PersonaPrediction] = None
    actual_persona: Optional[str] = None
    error: Optional[PredictionError] = None
    local_adjustment: Optional[LocalAdjustment] = None
    account_meta: Optional[Dict[str, Any]] = None
    global_params_version: Optional[int] = None
    candidate_personas: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Aggregation & evaluation
# ---------------------------------------------------------------------------


@dataclass
class EdgeAdjustmentAggregate:
    source_id: str
    target_id: str
    count: int
    mean_delta: float
    mean_confidence: Optional[float] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeAdjustmentAggregate:
    node_id: str
    field: str
    count: int
    mean_delta: float
    mean_confidence: Optional[float] = None
    band: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdjustmentEvaluation:
    adjustment_id: str
    before_metrics: Dict[str, Any]
    after_metrics: Dict[str, Any]
    improvement_score: float
    rationale: Optional[str] = None


# ---------------------------------------------------------------------------
# Global insights
# ---------------------------------------------------------------------------


@dataclass
class GlobalNodeInsight:
    node_id: str
    label: Optional[str]
    recommendation: str
    confidence: Optional[float]
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalEdgeInsight:
    source_id: str
    target_id: str
    label: Optional[str]
    recommendation: str
    confidence: Optional[float]
    supporting_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalInsightsMeta:
    num_accounts: int
    num_engagements: int
    stability_score: Optional[float] = None
    hit_at_1: Optional[float] = None
    hit_at_3: Optional[float] = None
    log_loss: Optional[float] = None


@dataclass
class GlobalInsights:
    meta: GlobalInsightsMeta
    node_insights: List[GlobalNodeInsight] = field(default_factory=list)
    edge_insights: List[GlobalEdgeInsight] = field(default_factory=list)


__all__ = [
    "PersonaCandidateStat",
    "PersonaImpactMetrics",
    "GlobalParams",
    "PersonaPrediction",
    "PredictionError",
    "EdgeAdjustment",
    "NodeAdjustment",
    "LocalAdjustment",
    "Episode",
    "EdgeAdjustmentAggregate",
    "NodeAdjustmentAggregate",
    "AdjustmentEvaluation",
    "GlobalNodeInsight",
    "GlobalEdgeInsight",
    "GlobalInsightsMeta",
    "GlobalInsights",
]
