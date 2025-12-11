from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SENIORITY_BUCKETS: tuple[str, ...] = ("operator", "manager", "director", "exec")


def _distribution_template() -> Dict[str, float]:
    return {bucket: 0.0 for bucket in SENIORITY_BUCKETS}


DEFAULT_CANONICAL_INFLUENCE_SCALARS: Dict[str, float] = {
    "pain_proximity": 0.6,
    "political_influence": 0.5,
    "veto_power": 0.4,
    "execution_leverage": 0.6,
}

SENIORITY_INFLUENCE_FACTORS: Dict[str, Dict[str, float]] = {
    "operator": {
        "pain_proximity": 1.1,
        "political_influence": 0.6,
        "veto_power": 0.6,
        "execution_leverage": 1.0,
    },
    "manager": {
        "pain_proximity": 0.9,
        "political_influence": 1.2,
        "veto_power": 1.1,
        "execution_leverage": 1.1,
    },
    "director": {
        "pain_proximity": 0.7,
        "political_influence": 1.5,
        "veto_power": 1.4,
        "execution_leverage": 1.3,
    },
    "exec": {
        "pain_proximity": 0.4,
        "political_influence": 2.0,
        "veto_power": 2.0,
        "execution_leverage": 1.5,
    },
}


def clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    """Best-effort clamp of a numeric-ish value."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        num = 0.0
    if num < lo:
        return lo
    if num > hi:
        return hi
    return num


def normalize_distribution(values: Optional[Dict[str, Any]]) -> Dict[str, float]:
    dist = _distribution_template()
    if not values:
        return dist
    for bucket, weight in values.items():
        key = (bucket or "").lower()
        if key in dist:
            dist[key] = clamp(weight)
    return dist


def ensure_meta(meta: Optional[Dict[str, Any]], *, default_source: str = "rule") -> Dict[str, Any]:
    payload = dict(meta or {})
    payload.setdefault("source", default_source)
    payload.setdefault("confidence", 0.0)
    if "examples" not in payload:
        payload["examples"] = []
    return payload


def compute_variant_influence(
    base_scalars: Optional[Dict[str, float]],
    seniority: str,
    overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    scalars = dict(DEFAULT_CANONICAL_INFLUENCE_SCALARS)
    scalars.update(base_scalars or {})
    if overrides:
        scalars.update(overrides)
    factors = SENIORITY_INFLUENCE_FACTORS.get(
        (seniority or "").lower(), {k: 1.0 for k in DEFAULT_CANONICAL_INFLUENCE_SCALARS}
    )
    return {
        key: clamp(scalars.get(key, 0.0) * factors.get(key, 1.0))
        for key in DEFAULT_CANONICAL_INFLUENCE_SCALARS.keys()
    }


@dataclass
class JobNodeAttributes:
    job_id: str = ""
    job_name: str = ""
    job_description: str = ""
    canonical_persona_ids: List[str] = field(default_factory=list)
    pains: List[str] = field(default_factory=list)
    dependencies_upstream: List[str] = field(default_factory=list)
    dependencies_downstream: List[str] = field(default_factory=list)
    typical_titles: List[str] = field(default_factory=list)
    typical_departments: List[str] = field(default_factory=list)
    typical_seniority_distribution: Dict[str, float] = field(
        default_factory=_distribution_template
    )
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_node_attrs(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_name": self.job_name or self.job_description,
            "description": self.job_description or self.job_name,
            "canonical_persona_ids": list(dict.fromkeys(self.canonical_persona_ids)),
            "pains": list(dict.fromkeys(self.pains)),
            "dependencies_upstream": list(dict.fromkeys(self.dependencies_upstream)),
            "dependencies_downstream": list(
                dict.fromkeys(self.dependencies_downstream)
            ),
            "typical_titles": list(dict.fromkeys(self.typical_titles)),
            "typical_departments": list(dict.fromkeys(self.typical_departments)),
            "typical_seniority_distribution": normalize_distribution(
                self.typical_seniority_distribution
            ),
            "meta": ensure_meta(self.meta),
        }


@dataclass
class CanonicalPersonaAttributes:
    canonical_persona_id: str = ""
    label: str = ""
    description: str = ""
    core_jobs: List[str] = field(default_factory=list)
    supporting_jobs: List[str] = field(default_factory=list)
    core_pains: List[str] = field(default_factory=list)
    example_titles: List[str] = field(default_factory=list)
    typical_departments: List[str] = field(default_factory=list)
    typical_seniority_distribution: Dict[str, float] = field(
        default_factory=_distribution_template
    )
    default_influence_scalars: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CANONICAL_INFLUENCE_SCALARS)
    )
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_node_attrs(self) -> Dict[str, Any]:
        label = self.label or "persona"
        return {
            "canonical_persona_id": self.canonical_persona_id or label,
            "label": label,
            "description": self.description,
            "core_jobs": list(dict.fromkeys(self.core_jobs)),
            "supporting_jobs": list(dict.fromkeys(self.supporting_jobs)),
            "core_pains": list(dict.fromkeys(self.core_pains)),
            "example_titles": list(dict.fromkeys(self.example_titles)),
            "typical_departments": list(dict.fromkeys(self.typical_departments)),
            "typical_seniority_distribution": normalize_distribution(
                self.typical_seniority_distribution
            ),
            "default_influence_scalars": {
                key: clamp(value)
                for key, value in (self.default_influence_scalars or {}).items()
            },
            "meta": ensure_meta(self.meta),
        }


@dataclass
class PersonaVariantAttributes:
    persona_variant_id: str = ""
    canonical_persona_id: str = ""
    title: str = ""
    department: str = ""
    seniority: str = ""
    team_context: str = ""
    influence_scalars: Optional[Dict[str, float]] = None
    crm_person_ids: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_node_attrs(
        self, canonical_influence: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        scalars = compute_variant_influence(
            canonical_influence, self.seniority, self.influence_scalars
        )
        return {
            "persona_variant_id": self.persona_variant_id
            or f"{self.canonical_persona_id}:{self.title}".strip(":"),
            "canonical_persona_id": self.canonical_persona_id,
            "title": self.title,
            "department": self.department,
            "seniority": (self.seniority or "").lower(),
            "team_context": self.team_context,
            "influence_scalars": scalars,
            "crm_person_ids": list(dict.fromkeys(self.crm_person_ids)),
            "meta": ensure_meta(self.meta, default_source="crm"),
        }


__all__ = [
    "CanonicalPersonaAttributes",
    "DEFAULT_CANONICAL_INFLUENCE_SCALARS",
    "JobNodeAttributes",
    "PersonaVariantAttributes",
    "SENIORITY_BUCKETS",
    "SENIORITY_INFLUENCE_FACTORS",
    "compute_variant_influence",
    "normalize_distribution",
]
