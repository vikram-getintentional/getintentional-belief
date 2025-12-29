from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


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


def _confidence_to_engagement(confidence: Optional[float]) -> float:
    conf = 0.5 if confidence is None else float(confidence)
    return min(0.88, max(0.15, 0.25 + 0.6 * conf))


def _historical_delta(entry: Optional[Dict[str, Any]]) -> Optional[float]:
    if not entry:
        return None
    avg_delta = entry.get("avg_delta")
    if isinstance(avg_delta, (int, float)):
        return float(avg_delta)
    total_delta = entry.get("total_delta")
    num_engagements = entry.get("num_engagements")
    if isinstance(total_delta, (int, float)) and isinstance(num_engagements, (int, float)):
        if num_engagements:
            return float(total_delta) / float(num_engagements)
    return None


def _delta_from_bp(delta_bp: Optional[float]) -> float:
    if not delta_bp:
        return 0.0
    return float(delta_bp) / 10000.0


class ArsenalForecaster:
    def forecast(
        self,
        current_win: Dict[str, Any],
        plays: Sequence[Dict[str, Any]],
        *,
        arsenal_impact: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        current_p = float(current_win.get("p") or 0.02)
        base_log_odds = _logit(current_p)
        assumptions: List[Dict[str, Any]] = []
        accumulated_delta = 0.0
        history_lookup: Dict[str, Dict[str, Any]] = {}
        for entry in arsenal_impact or []:
            asset_id = entry.get("asset_id")
            if asset_id:
                history_lookup[str(asset_id)] = entry
        for play in plays or []:
            asset_candidate = (
                play.get("asset_id")
                or (play.get("asset") or {}).get("id")
                or (play.get("asset") or {}).get("asset_id")
            )
            history_entry = (
                history_lookup.get(str(asset_candidate)) if asset_candidate is not None else None
            )
            plan_delta = _delta_from_bp(play.get("expected_delta_bp"))
            history_delta = _historical_delta(history_entry)
            delta_for_filter = history_delta if history_delta is not None else plan_delta
            if abs(delta_for_filter) < 0.0005:
                continue
            delta_log_odds = history_delta if history_delta is not None else plan_delta
            p_engage = _confidence_to_engagement(play.get("confidence"))
            if history_entry:
                history_confidence = history_entry.get("avg_confidence")
                if isinstance(history_confidence, (int, float)):
                    p_engage = max(p_engage, _confidence_to_engagement(history_confidence))
            accumulated_delta += p_engage * delta_log_odds
            assumptions.append(
                {
                    "action_id": play.get("id") or play.get("asset_id") or "unknown",
                    "label": play.get("persona_label") or play.get("description") or "action",
                    "p_engage": round(p_engage, 3),
                    "expected_delta_log_odds": round(delta_log_odds, 4),
                    "history_note": history_entry
                    and (
                        (
                            f"History ({int(history_entry.get('num_engagements') or 0)} signals)"
                            if history_entry.get("num_engagements")
                            else "History evidence"
                        )
                    ),
                    "history_engagements": (
                        int(history_entry.get("num_engagements") or 0) if history_entry else None
                    ),
                }
            )
            if len(assumptions) >= 3:
                break

        if not assumptions:
            assumptions = [
                {
                    "action_id": "generic",
                    "label": "Recommended execution plan",
                    "p_engage": 0.33,
                    "expected_delta_log_odds": 0.01,
                }
            ]
            accumulated_delta += 0.01

        forecast_log_odds = base_log_odds + accumulated_delta
        forecast_p = _sigmoid(forecast_log_odds)
        ci_low = max(0.005, forecast_p - 0.06)
        ci_high = min(0.995, forecast_p + 0.06)
        lift = forecast_p - current_p
        status = (
            "uses_priors_if_no_arsenal_history"
            if not (arsenal_impact or [])
            else "plans_with_arsenal_history"
        )
        return {
            "p": _clamp_probability(forecast_p),
            "ci": {"low": ci_low, "high": ci_high},
            "range": {"low": ci_low, "high": ci_high},
            "assumptions": assumptions,
            "lift_over_current": round(lift, 4),
            "status": status,
        }
