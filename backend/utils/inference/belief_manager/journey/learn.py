# backend/utils/inference/belief_manager/journey/learn.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
from collections import defaultdict

"""
Events passed here should already be resolved to personas.

Each event dict is expected to have:
{
  "persona_id": "persona:xxxx",
  "channel": "email" | "webinar" | ...,
  "source": "marketing" | "sales",
  # optional:
  "bucket": "email_outbound" | ...
}
"""


def bucket_channel(channel: Optional[str], source: Optional[str]) -> str:
    """Coarse bucket for the emissions.

    Keep this dumb for now; refine later (asset type, stage, etc.).
    """
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
    # fallback
    return (ch or "unknown") + "_" + (src or "generic")


def _ensure_nested(d: Dict, *keys) -> Dict:
    cur = d
    for k in keys:
        if k not in cur:
            cur[k] = {}
        cur = cur[k]
    return cur


def update_edge_stats(
    stats: Dict[str, Any],
    episode_events: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Update transition & emission counts given a single account's episode.

    episode_events: ordered list of engagements that *have persona_id* resolved.
    """
    events = [e for e in episode_events if e.get("persona_id")]
    if not events:
        return stats

    transition = stats.setdefault("transition", {})
    emission = stats.setdefault("emission", {})

    # emissions
    for e in events:
        pid = e["persona_id"]
        bucket = e.get("bucket") or bucket_channel(e.get("channel"), e.get("source"))
        emission.setdefault(pid, {})
        emission[pid][bucket] = emission[pid].get(bucket, 0) + 1

    # transitions (consecutive persona hops)
    for prev, cur in zip(events[:-1], events[1:]):
        from_pid = prev["persona_id"]
        to_pid = cur["persona_id"]
        bucket = bucket_channel(cur.get("channel"), cur.get("source"))
        transition.setdefault(from_pid, {})
        transition[from_pid].setdefault(bucket, {})
        transition[from_pid][bucket][to_pid] = (
            transition[from_pid][bucket].get(to_pid, 0) + 1
        )

    stats["transition"] = transition
    stats["emission"] = emission
    return stats


def recompute_weights(
    stats: Dict[str, Any],
    base_graph_neighbors: Dict[str, Dict[str, float]],
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute P(next_persona | persona, bucket) using Dirichlet smoothing.

    base_graph_neighbors:
        from_persona -> {to_persona: base_weight}

    alpha:
        prior strength; alpha * base_weight becomes pseudocount.
    """
    transition_stats = stats.get("transition", {})
    weights = {"transition": {}}

    for from_pid, buckets in transition_stats.items():
        weights["transition"].setdefault(from_pid, {})
        for bucket, to_counts in buckets.items():
            # collect all possible tos: those we've seen AND neighbors in graph
            all_tos = set(to_counts.keys()) | set(base_graph_neighbors.get(from_pid, {}))
            numerators = {}
            total = 0.0
            for to_pid in all_tos:
                prior = alpha * float(base_graph_neighbors.get(from_pid, {}).get(to_pid, 0.0))
                count = float(to_counts.get(to_pid, 0.0))
                num = prior + count
                numerators[to_pid] = num
                total += num

            if total <= 0:
                # fallback: just normalize base weights
                neigh = base_graph_neighbors.get(from_pid, {})
                tot_base = sum(neigh.values()) or 1.0
                weights["transition"][from_pid][bucket] = {
                    t: w / tot_base for t, w in neigh.items()
                }
            else:
                weights["transition"][from_pid][bucket] = {
                    t: n / total for t, n in numerators.items()
                }

    return weights
