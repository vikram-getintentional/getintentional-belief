# ============================
# File: backend/utils/inference/rcs_generators/graphwin.py
# ============================
from __future__ import annotations
from typing import Dict, List, Optional
import math

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _noisy_or(vals):
    prod = 1.0
    for v in vals:
        v = max(0.0, min(1.0, float(v)))
        prod *= (1.0 - v)
    return 1.0 - prod

def compute_graphwin(
    *,
    conv_id: str,
    pain_triggers: List[str],
    dim_weights: Dict[str, float],
    beta0: float = 0.0,
    betas: Optional[Dict[str, float]] = None,
    baseline_prior: float = 0.0,
    hard_on_triggers: Optional[List[str]] = None,
    hard_on_pains: Optional[List[str]] = None,
    hard_on_jobs: Optional[List[str]] = None,
    reach_to_product: Optional[Dict[str, float]] = None,   # trigger -> product
    reach_from_hard: Optional[Dict[str, float]] = None,    # hard-node -> product
) -> float:
    """
    Final win = noisy_or( prior(trigger) * R(trigger→product)  ∪  1.0 * R(hard→product) )
    - priors(trigger) are logistic over attributes (with learnable betas later).
    - If a trigger is explicitly observed (hard_on_triggers), its prior=1.0.
    - Observed pains/jobs don't change trigger priors; they contribute as their own “atoms”
      via reach_from_hard (each with prior=1.0).
    """
    betas = betas or {}
    hard_on_triggers = set(hard_on_triggers or [])
    hard_on_pains = set(hard_on_pains or [])
    hard_on_jobs = set(hard_on_jobs or [])
    reach_to_product = reach_to_product or {}
    reach_from_hard = reach_from_hard or {}

    # (1) priors for triggers
    atoms: List[float] = []
    # logistic over dim_weights (global betas or per trigger future extension)
    score = beta0 + sum(betas.get(k, 0.0) * float(v) for k, v in dim_weights.items())
    p_attr = _sigmoid(score)

    # baseline prior rule: with no attributes, you can force to 0.0 (as per design)
    lam = 1.0 if all(v == 0.0 for v in dim_weights.values()) else 0.0
    p_cal = lam * baseline_prior + (1.0 - lam) * p_attr  # = 0.0 when no attrs

    for t in pain_triggers:
        prior_t = 1.0 if t in hard_on_triggers else p_cal
        reach_t = max(0.0, min(1.0, reach_to_product.get(t, 0.0)))
        atoms.append(prior_t * reach_t)

    # (2) add hard-evidence atoms (pains / jobs / triggers) with prior=1.0
    for h, r in (reach_from_hard or {}).items():
        atoms.append(max(0.0, min(1.0, float(r))))  # prior=1.0 already multiplied

    return float(_noisy_or(atoms))
