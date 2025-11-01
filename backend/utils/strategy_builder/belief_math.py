# belief_math.py
from __future__ import annotations
from typing import Dict, Tuple

# Belief = f(Perceptibility, Proximity, Involvement)
# Keep it simple + bounded and monotonic.
def clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))

def org_belief(perc: float, prox: float, inv: float, weights: Tuple[float, float, float] = (0.4, 0.35, 0.25)) -> float:
    w1, w2, w3 = weights
    perc, prox, inv = clamp01(perc), clamp01(prox), clamp01(inv)
    return clamp01(w1 * perc + w2 * prox + w3 * inv)

def expected_belief_delta(asset_fit: float, channel_fit: float, stage_factor: float = 1.0) -> float:
    """Asset fit = solve-the-concern fit; Channel fit = engagement likelihood."""
    asset_fit = clamp01(asset_fit)
    channel_fit = clamp01(channel_fit)
    # small-signal gain; you can tune later
    base = 0.15 * asset_fit * (0.5 + 0.5 * channel_fit)
    return clamp01(stage_factor * base)

def time_decay(days_from_now: int, half_life_days: int = 45) -> float:
    if days_from_now <= 0:
        return 1.0
    # exponential half-life
    return 0.5 ** (days_from_now / max(1, half_life_days))

def stage_gain_coeff(stage: str) -> float:
    stage = (stage or "").lower()
    # discovery-ish stages can move belief more than late implementation
    table = {
        "pre_zmot": 0.8,
        "zmot": 1.0,
        "problem_realization": 1.1,
        "discovery": 1.2,
        "barriers": 1.0,
        "implementation": 0.7,
    }
    return table.get(stage, 1.0)

def lift_label_from_belief_delta(d: float) -> str:
    if d >= 0.12: return "Breakout"
    if d >= 0.08: return "High"
    if d >= 0.04: return "Medium"
    if d > 0.0: return "Low"
    return "None"

def belief_summary_meta(avg_belief: float) -> Dict[str, str]:
    avg_belief = clamp01(avg_belief)
    if avg_belief < 0.25:
        segment = "Cold"
    elif avg_belief < 0.5:
        segment = "Warming"
    elif avg_belief < 0.75:
        segment = "Engaged"
    else:
        segment = "Primed"
    return {"avgBeliefBand": segment}
