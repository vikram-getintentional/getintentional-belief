import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.super_models.thesis_builder import ThesisSegment


def canonicalize_category_id(value: str) -> str:
    if not isinstance(value, str):
        value = str(value)
    return value.strip().lower()


def update_dirichlet(
    alphas: Dict[str, float],
    evidence: Dict[str, float],
    epsilon: float = 1e-3,
) -> Dict[str, float]:
    updated = dict(alphas or {})
    for cat, weight in evidence.items():
        updated.setdefault(cat, epsilon)
        updated[cat] += weight
    return updated


def dirichlet_mean(alphas: Dict[str, float]) -> Dict[str, float]:
    if not alphas:
        return {}
    total = sum(alphas.values())
    if total <= 0:
        return {key: 0.0 for key in alphas}
    return {key: value / total for key, value in alphas.items()}


def total_variation_distance(
    p0: Dict[str, float], p1: Dict[str, float]
) -> float:
    keys = set(p0.keys()) | set(p1.keys())
    diff = 0.0
    for key in keys:
        diff += abs(p0.get(key, 0.0) - p1.get(key, 0.0))
    return 0.5 * diff


def drift_level_from_tv(tv: float) -> str:
    if tv >= 0.4:
        return "strong"
    if tv >= 0.2:
        return "moderate"
    return "stable"


def infer_segment_posterior(
    segments: Iterable[ThesisSegment],
    segment_priors: Dict[str, float],
    attribute_bin_values: Dict[str, str],
    attribute_posteriors: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, float]:
    log_scores: Dict[str, float] = {}
    for segment in segments:
        prior = float(segment_priors.get(segment.id, segment.prior_weight or 1.0))
        log_score = math.log(prior + 1e-12)
        for feature_key, bin_value in attribute_bin_values.items():
            alphas = attribute_posteriors.get((segment.id, feature_key))
            if not alphas:
                continue
            mean_probs = dirichlet_mean(alphas)
            probability = mean_probs.get(bin_value, 1e-6)
            log_score += math.log(probability + 1e-12)
        log_scores[segment.id] = log_score
    if not log_scores:
        return {}
    max_log = max(log_scores.values())
    exp_scores = {seg: math.exp(score - max_log) for seg, score in log_scores.items()}
    total = sum(exp_scores.values())
    if total <= 0:
        return {seg: 1.0 / len(exp_scores) for seg in exp_scores}
    return {seg: prob / total for seg, prob in exp_scores.items()}


def assign_segment_by_rules(
    segments: Iterable[ThesisSegment], attribute_bins: Dict[str, str]
) -> List[str]:
    assigned = []
    for segment in segments:
        rules = segment.rules or {}
        conditions = rules.get("conditions", [])
        if all(_evaluate_condition(cond, attribute_bins) for cond in conditions):
            assigned.append(segment.id)
    return assigned


def _evaluate_condition(condition: Dict[str, Any], attribute_bins: Dict[str, str]) -> bool:
    feature_key = condition.get("feature_key")
    op = condition.get("op")
    values = condition.get("values") or []
    if op == "IN":
        target_value = attribute_bins.get(feature_key)
        if target_value is None:
            return False
        return target_value in values
    if op == "ANY":
        return True
    return False
