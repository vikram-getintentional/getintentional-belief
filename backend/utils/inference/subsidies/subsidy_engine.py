"""
In-memory subsidy registry plus helper utilities.

This keeps the rest of the inference stack free of storage concerns so we
can swap in a DB-backed service later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import tanh
import re
from typing import Any, Iterable, List, Mapping, MutableSequence, Optional, Sequence

from .subsidy_models import SubsidyEvent, SubsidyScope

# TODO: replace with persistent store when available.
_SUBSIDY_STORE: MutableSequence[SubsidyEvent] = []


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
def register_subsidy(event: SubsidyEvent) -> None:
    """Best-effort append for now."""
    _SUBSIDY_STORE.append(event)


def clear_subsidies() -> None:
    """Testing hook."""
    _SUBSIDY_STORE.clear()


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------
def _ensure_iterable(value: Optional[Sequence[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v]


def _segment_match(
    event_segment: Optional[str],
    account_segments: Optional[Sequence[str]],
) -> bool:
    if not event_segment:
        return False
    segments = {seg.strip().lower() for seg in _ensure_iterable(account_segments)}
    if not segments:
        return False
    return event_segment.strip().lower() in segments


def _get_persona_attr(persona: Any, key: str) -> Optional[str]:
    if persona is None:
        return None
    if isinstance(persona, Mapping):
        val = persona.get(key)
    else:
        val = getattr(persona, key, None)
    if val is None:
        return None
    return str(val)


def _persona_title(persona: Any) -> Optional[str]:
    for key in ("title", "label", "name"):
        val = _get_persona_attr(persona, key)
        if val:
            return val
    return None


def _persona_canonical_id(persona: Any) -> Optional[str]:
    for key in (
        "canonical_persona_id",
        "canonicalPersonaId",
        "canonical_id",
        "persona_archetype_id",
        "id",
    ):
        val = _get_persona_attr(persona, key)
        if val:
            return val
    return None


def persona_title_matches(persona: Any, pattern: str) -> bool:
    title = _persona_title(persona)
    if not title or not pattern:
        return False
    try:
        return bool(re.search(pattern, title, flags=re.IGNORECASE))
    except re.error:
        return title.lower() == pattern.lower()


def persona_matches_target(persona: Any, target: Any) -> bool:
    persona_id = _persona_canonical_id(persona)
    target_id = getattr(target, "persona_archetype_id", None)
    if target_id and persona_id and target_id == persona_id:
        return True
    target_pattern = getattr(target, "persona_title_pattern", None)
    if target_pattern and persona_title_matches(persona, target_pattern):
        return True
    return False


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------
def get_active_subsidies_for_account(
    account_id: str,
    segment_keys: Optional[Sequence[str]],
    t: Optional[datetime] = None,
) -> List[SubsidyEvent]:
    """
    Return all subsidies that should apply right now.
    """
    if t is None:
        t = datetime.utcnow().replace(tzinfo=timezone.utc)
    active: List[SubsidyEvent] = []
    for subsidy in _SUBSIDY_STORE:
        starts = subsidy.intensity.starts_at
        ends = subsidy.intensity.ends_at
        if starts and starts > t:
            continue
        if ends and ends < t:
            continue

        if subsidy.scope == SubsidyScope.GLOBAL:
            active.append(subsidy)
            continue

        if subsidy.scope == SubsidyScope.ACCOUNT and subsidy.account_id == account_id:
            active.append(subsidy)
            continue

        if (
            subsidy.scope == SubsidyScope.SEGMENT
            and segment_keys
            and _segment_match(subsidy.segment_key, segment_keys)
        ):
            active.append(subsidy)
            continue

        if subsidy.scope == SubsidyScope.PERSONA:
            # Persona scoped subsidies will be filtered in persona matching.
            active.append(subsidy)

    return active


def subsidy_effective_weight(subsidy: SubsidyEvent, t: Optional[datetime] = None) -> float:
    if t is None:
        t = datetime.utcnow().replace(tzinfo=timezone.utc)
    weight = float(subsidy.intensity.magnitude)
    half_life = subsidy.intensity.decay_half_life_days
    if half_life:
        delta_days = (t - subsidy.intensity.starts_at).days
        decay_factor = 0.5 ** (max(0.0, delta_days) / max(half_life, 1e-6))
        weight *= decay_factor
    return max(0.0, weight)


def subsidy_relevance_for_persona(
    persona: Any,
    account_id: str,
    segment_keys: Optional[Sequence[str]],
    t: Optional[datetime] = None,
) -> float:
    active = get_active_subsidies_for_account(account_id, segment_keys, t)
    score = 0.0
    for subsidy in active:
        if subsidy.scope == SubsidyScope.PERSONA:
            # Only count if persona matches at least one target.
            if any(persona_matches_target(persona, target) for target in subsidy.targets):
                score += subsidy_effective_weight(subsidy, t)
            continue
        for target in subsidy.targets:
            if persona_matches_target(persona, target):
                score += subsidy_effective_weight(subsidy, t)
                break
    return score


def wolf_score_dynamic(
    persona: Any,
    base_wolf_score: float,
    account_id: str,
    segment_keys: Optional[Sequence[str]],
    t: Optional[datetime] = None,
) -> float:
    relevance = subsidy_relevance_for_persona(persona, account_id, segment_keys, t)
    multiplier = 1.0 + 0.3 * tanh(relevance)
    return min(1.0, max(0.0, base_wolf_score) * multiplier)


# ---------------------------------------------------------------------------
# Belief edge adjustments
# ---------------------------------------------------------------------------
def _edge_attr(edge: Any, key: str, default: Any = None) -> Any:
    if isinstance(edge, Mapping):
        return edge.get(key, default)
    return getattr(edge, key, default)


def apply_subsidies_to_edge(
    edge: Any,
    account_id: str,
    segment_keys: Optional[Sequence[str]],
    t: Optional[datetime] = None,
) -> float:
    """
    edge: dict-like with:
      - base_transition_prob (float)
      - persona_id / persona_canonical_id (optional)
      - from_belief / to_belief meta
      - tags (list of str)
    """
    if t is None:
        t = datetime.utcnow().replace(tzinfo=timezone.utc)
    base_prob = float(_edge_attr(edge, "base_transition_prob", 0.0) or 0.0)
    persona_id = _edge_attr(edge, "persona_canonical_id") or _edge_attr(edge, "persona_id")
    tags = set(_edge_attr(edge, "tags", []) or [])

    active = get_active_subsidies_for_account(account_id, segment_keys, t)
    total_weight = 0.0
    for subsidy in active:
        for target in subsidy.targets:
            if target.persona_archetype_id and persona_id and target.persona_archetype_id != persona_id:
                continue
            from_belief = _edge_attr(edge, "from_belief")
            to_belief = _edge_attr(edge, "to_belief")
            if target.from_belief and target.from_belief != from_belief:
                continue
            if target.to_belief and target.to_belief != to_belief:
                continue
            if target.belief_edge_tag and target.belief_edge_tag not in tags:
                continue
            total_weight += subsidy_effective_weight(subsidy, t)

    boosted = base_prob * (1.0 + total_weight)
    return min(0.98, max(0.0, boosted))


__all__ = [
    "apply_subsidies_to_edge",
    "clear_subsidies",
    "get_active_subsidies_for_account",
    "persona_matches_target",
    "persona_title_matches",
    "register_subsidy",
    "subsidy_effective_weight",
    "subsidy_relevance_for_persona",
    "wolf_score_dynamic",
]
