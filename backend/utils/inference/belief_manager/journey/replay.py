# backend/utils/inference/belief_manager/journey/replay.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


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

    def _bucket_entries(bucket_data: Dict[str, Any]) -> List[Tuple[str, float]]:
        entries: List[Tuple[str, float]] = []
        for pid, raw in bucket_data.items():
            if isinstance(raw, dict):
                score = float(raw.get("likelihood") or raw.get("prob") or 0.0)
            else:
                score = float(raw or 0.0)
            entries.append((pid, score))
        return entries

    def _normalize(entries: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        filtered = [(pid, score) for pid, score in entries if score > 0.0]
        if not filtered:
            return []
        total_score = sum(score for _, score in filtered) or 1.0
        normalized = [(pid, score / total_score) for pid, score in filtered]
        normalized.sort(key=lambda x: x[1], reverse=True)
        return normalized

    bucket_map = trans[current_persona_id]
    if channel_bucket and channel_bucket in bucket_map:
        return _normalize(_bucket_entries(bucket_map[channel_bucket]))

    aggregate: Dict[str, float] = defaultdict(float)
    for bucket_data in bucket_map.values():
        for pid, score in _bucket_entries(bucket_data):
            aggregate[pid] += score

    return _normalize(list(aggregate.items()))
