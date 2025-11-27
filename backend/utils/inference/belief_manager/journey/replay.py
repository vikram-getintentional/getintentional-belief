# backend/utils/inference/belief_manager/journey/replay.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional


def next_distribution(
    current_persona_id: str,
    weights: Dict[str, Any],
    channel_bucket: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """
    Return a normalized distribution over next personas for UI 'expected_next'.

    If channel_bucket is provided, prefer that; otherwise average over buckets.
    """
    trans = weights.get("transition", {})
    if current_persona_id not in trans:
        return []

    if channel_bucket and channel_bucket in trans[current_persona_id]:
        dist = trans[current_persona_id][channel_bucket]
        items = list(dist.items())
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    # no specific bucket -> average over all buckets
    buckets = trans[current_persona_id]
    agg = {}
    for b, d in buckets.items():
        for pid, p in d.items():
            agg[pid] = agg.get(pid, 0.0) + p
    total = sum(agg.values()) or 1.0
    items = [(pid, p / total) for pid, p in agg.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items
