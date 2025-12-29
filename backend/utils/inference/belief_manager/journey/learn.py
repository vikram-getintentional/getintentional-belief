# backend/utils/inference/belief_manager/journey/learn.py
from __future__ import annotations

import math
from typing import Dict, Any, Iterable, Optional, Tuple

try:
    from scipy.stats import beta as _beta_dist  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _beta_dist = None


def bucket_channel(channel: Optional[str], source: Optional[str]) -> str:
    """Coarse bucket for the emissions."""
    ch = (channel or "").lower()
    src = (source or "").lower()

    if "email" in ch:
        return "email_" + (src or "generic")
    if "webinar" in ch or "event" in ch:
        return "event_" + (src or "generic")
    if "call" in ch or "meeting" in ch:
        return "meeting_" + (src or "generic")
    if "page" in ch or "website" in ch or "pricing" in ch:
        return "web_" + (src or "generic")
    if "ad" in ch or "display" in ch:
        return "ads_" + (src or "generic")
    return (ch or "unknown") + "_" + (src or "generic")


def _new_transition_entry() -> Dict[str, Any]:
    return {
        "count": 0,
        "support_events": 0,
        "contradict_events": 0,
        "wins": 0,
        "losses": 0,
        "by_state": {},
        "channels": {},
        "assets": {},
    }


def _normalize_state_key(state: Optional[str]) -> str:
    if not state:
        return "unknown"
    normalized = str(state).strip().lower()
    return normalized or "unknown"


def _safe_bucket(bucket: Optional[str]) -> str:
    return (bucket or "unknown").strip() or "unknown"


def beta_interval(
    alpha: float, beta: float, confidence: float = 0.95
) -> Tuple[float, float]:
    """Return a central credible interval for a Beta distribution."""
    if _beta_dist is not None:
        lower = float(_beta_dist.ppf((1 - confidence) / 2, alpha, beta))
        upper = float(
            _beta_dist.ppf(1 - (1 - confidence) / 2, alpha, beta)
        )
        return max(0.0, lower), min(1.0, upper)
    mean = alpha / max(alpha + beta, 1e-9)
    z = 1.96 if confidence >= 0.95 else 1.64
    var = alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1))
    margin = z * math.sqrt(var) if var > 0 else 0.0
    return max(0.0, mean - margin), min(1.0, mean + margin)


def _ensure_transition_entry(
    transition: Dict[str, Any],
    from_pid: str,
    bucket: str,
    to_pid: str,
) -> Dict[str, Any]:
    bucket_map = transition.setdefault(from_pid, {})
    bucket_entries = bucket_map.setdefault(bucket, {})
    existing = bucket_entries.get(to_pid)
    if isinstance(existing, (int, float)):
        bucket_entries[to_pid] = _new_transition_entry()
        bucket_entries[to_pid]["count"] = float(existing)
        existing = bucket_entries[to_pid]
    if existing is None:
        bucket_entries[to_pid] = _new_transition_entry()
        existing = bucket_entries[to_pid]
    return existing


def update_edge_stats(
    stats: Dict[str, Any],
    episode_events: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Update transition & emission counts for a single journey episode.
    """
    events = [e for e in episode_events if e.get("persona_id")]
    if not events:
        return stats

    transition = stats.setdefault("transition", {})
    emission = stats.setdefault("emission", {})

    for e in events:
        pid = e["persona_id"]
        bucket = e.get("bucket") or bucket_channel(e.get("channel"), e.get("source"))
        emission.setdefault(pid, {})
        emission[pid][bucket] = emission[pid].get(bucket, 0) + 1

    support_buckets = {"on_path", "near_path"}
    contradict_buckets = {"off_path", "no_path"}

    for prev, curr in zip(events[:-1], events[1:]):
        from_pid = prev["persona_id"]
        to_pid = curr["persona_id"]
        bucket = _safe_bucket(curr.get("effect_bucket"))
        entry = _ensure_transition_entry(transition, from_pid, bucket, to_pid)
        entry["count"] = float(entry.get("count", 0.0)) + 1.0

        effect_bucket = curr.get("effect_bucket") or ""
        if effect_bucket in support_buckets:
            entry["support_events"] += 1
        if effect_bucket in contradict_buckets:
            entry["contradict_events"] += 1

        meta = curr.get("meta") or {}
        hit1 = bool(meta.get("hit_at_1") or curr.get("hit_at_1"))
        hit3 = bool(meta.get("hit_at_3") or curr.get("hit_at_3"))
        is_win = hit1 or hit3
        if is_win:
            entry["wins"] += 1
        else:
            entry["losses"] += 1

        state_key = _normalize_state_key(curr.get("belief_state"))
        state_entry = entry["by_state"].setdefault(
            state_key, {"wins": 0, "losses": 0, "count": 0}
        )
        state_entry["count"] += 1
        if is_win:
            state_entry["wins"] += 1
        else:
            state_entry["losses"] += 1

        channel_key = (curr.get("channel") or "unknown").lower()
        entry["channels"][channel_key] = entry["channels"].get(channel_key, 0) + 1
        asset_key = curr.get("asset_id") or "unknown"
        entry["assets"][asset_key] = entry["assets"].get(asset_key, 0) + 1

    stats["transition"] = transition
    stats["emission"] = emission
    return stats


def recompute_weights(
    stats: Dict[str, Any],
    base_graph_neighbors: Dict[str, Dict[str, float]],
    alpha: float = 0.5,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    ci_confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Convert updated stats into posterior-weighted transition probabilities.
    """
    transition_stats = stats.get("transition", {})
    weights: Dict[str, Dict[str, Dict[str, Any]]] = {"transition": {}}

    for from_pid, buckets in transition_stats.items():
        weights["transition"].setdefault(from_pid, {})
        priors = base_graph_neighbors.get(from_pid, {})
        total_prior_mass = sum(float(v or 0.0) for v in priors.values()) or 0.0
        for bucket, to_entries in buckets.items():
            weights["transition"][from_pid].setdefault(bucket, {})
            for to_pid, edge in to_entries.items():
                entry = edge or {}
                count = float(entry.get("count", 0.0))
                wins = float(entry.get("wins", 0.0))
                losses = float(entry.get("losses", 0.0))
                prior_weight = float(priors.get(to_pid, 0.0))
                prior_mean = 0.0
                if total_prior_mass > 0:
                    prior_mean = prior_weight / total_prior_mass

                alpha_post = alpha_prior + wins
                beta_post = beta_prior + losses
                posterior_mean = alpha_post / max(alpha_post + beta_post, 1e-9)
                ci_low, ci_high = beta_interval(alpha_post, beta_post, ci_confidence)

                weights["transition"][from_pid][bucket][to_pid] = {
                    "likelihood": posterior_mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "count": count,
                    "support_events": entry.get("support_events", 0),
                    "contradict_events": entry.get("contradict_events", 0),
                    "wins": wins,
                    "losses": losses,
                    "prior_mean": prior_mean,
                }

    return weights
