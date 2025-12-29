from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from backend.utils.inference.belief_manager.journey.win_regression import (
    WIN_MODEL_CATEGORICAL,
    WIN_MODEL_NUMERIC,
    WinProbabilityCalibrator,
)


def _clamp_probability(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.005, min(0.995, prob))


def _wilson_interval(wins: float, losses: float, z: float = 1.96) -> Dict[str, float]:
    total = wins + losses
    if total <= 0:
        return {"low": 0.0, "high": 1.0}
    phat = wins / total
    denom = 1.0 + z * z / total
    centre = phat + (z * z) / (2.0 * total)
    margin = z * math.sqrt(
        (phat * (1.0 - phat) / total) + (z * z) / (4.0 * total * total)
    )
    return {
        "low": max(0.0, (centre - margin) / denom),
        "high": min(1.0, (centre + margin) / denom),
    }


def _format_drivers(coefficients: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    drivers: List[Dict[str, Any]] = []
    for entry in coefficients or []:
        feature = entry.get("feature")
        if not feature:
            continue
        weight = entry.get("coefficient") or entry.get("weight") or 0.0
        sign = "+" if weight >= 0 else "-"
        drivers.append(
            {
                "feature": feature,
                "direction": sign,
                "weight": abs(float(weight) if weight is not None else 0.0),
            }
        )
        if len(drivers) >= 3:
            break
    return drivers


def _feature_values_from_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not meta:
        return {}
    attributes = meta.get("attributes") or {}
    combined: Dict[str, Any] = {}
    combined.update(attributes)
    combined.update({k: v for k, v in meta.items() if k != "attributes"})
    result: Dict[str, Any] = {}
    for key in WIN_MODEL_CATEGORICAL + WIN_MODEL_NUMERIC:
        if key in combined:
            result[key] = combined[key]
    return result


class MetaModel:
    def __init__(self, win_summary: Optional[Dict[str, Any]] = None):
        summary = win_summary or {}
        class_balance = summary.get("class_balance") or {}
        self._wins = float(class_balance.get("won") or 0.0)
        self._losses = float(class_balance.get("lost") or 0.0)
        self._sample_size = int(summary.get("sample_size") or 0)
        self._coefficients = summary.get("feature_importances") or []
        self._has_data = self._sample_size >= 8 and (self._wins + self._losses) > 0
        self._calibrator = WinProbabilityCalibrator.from_summary(summary)

    def predict(self, account_meta: Dict[str, Any]) -> Dict[str, Any]:
        if not self._has_data:
            return {
                "p": None,
                "ci": {"low": None, "high": None},
                "coverage": {
                    "similar_deals": int(self._sample_size),
                    "wins": int(self._wins),
                    "losses": int(self._losses),
                },
                "top_drivers": [],
                "status": "no_historical_data",
            }

        prior = (self._wins + 1.0) / (self._wins + self._losses + 2.0)
        interval = _wilson_interval(self._wins, self._losses)
        ci_low = max(0.0, interval.get("low", 0.0))
        ci_high = min(1.0, interval.get("high", 1.0))
        drivers = _format_drivers(self._coefficients)
        probability: Optional[float] = None
        feature_values = _feature_values_from_meta(account_meta)
        if self._calibrator:
            try:
                probability = self._calibrator.predict_probability(feature_values)
            except Exception:
                probability = None
        if probability is None:
            probability = prior
        final_probability = _clamp_probability(probability)
        return {
            "p": final_probability,
            "ci": {"low": ci_low, "high": ci_high},
            "coverage": {
                "similar_deals": int(self._sample_size),
                "wins": int(self._wins),
                "losses": int(self._losses),
            },
            "top_drivers": drivers,
            "status": "ok",
        }
