from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Sequence


def _clamp_probability(value: float) -> float:
    return max(0.005, min(0.995, value))


def _logit(prob: float) -> float:
    safe = min(0.9999, max(0.0001, prob))
    return math.log(safe / (1.0 - safe))


def _sigmoid(score: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-score))
    except OverflowError:
        return 0.0 if score < 0 else 1.0


def _evidence_delta(entries: Sequence[Dict[str, Any]]) -> float:
    if not entries:
        return 0.0
    signal = 0.0
    for entry in entries:
        confidence = float(entry.get("confidence") or 0.5)
        stage_bonus = 0.1 * float(entry.get("stage_index") or 0.0)
        signal += max(0.02, min(0.15, 0.08 + 0.2 * confidence + stage_bonus))
    return min(0.45, signal)


class EngagementEvidenceModel:
    def update_with_engagements(
        self,
        meta_prediction: Dict[str, Any],
        engagements: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        baseline_p = float(meta_prediction.get("p") or 0.02)
        evidence_count = len(list(engagements or []))
        if evidence_count == 0:
            summary = "Prior-only estimate; no CRM engagements ingested yet."
            return {
                "p": _clamp_probability(baseline_p),
                "ci": {
                    "low": _clamp_probability(baseline_p - 0.04),
                    "high": _clamp_probability(baseline_p + 0.04),
                },
                "evidence": {
                    "engagement_count": 0,
                    "summary": summary,
                },
                "status": "no_engagements",
            }

        delta = _evidence_delta(engagements)
        log_odds = _logit(baseline_p)
        updated_log_odds = log_odds + delta
        probability = _sigmoid(updated_log_odds)
        return {
            "p": _clamp_probability(probability),
            "ci": {
                "low": _clamp_probability(probability - 0.05),
                "high": _clamp_probability(probability + 0.05),
            },
            "evidence": {
                "engagement_count": evidence_count,
                "summary": (
                    f"Updated with {evidence_count} persona engagement signal(s)."
                ),
            },
            "status": "updated",
        }
