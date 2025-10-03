# ============================
# File: backend/utils/inference/rcs_generators/rcs_helpers/graphwin.py
# ============================
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import math

def _sigmoid(x: float) -> float:
    # numerically safe-ish sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def _noisy_or(vals: List[float]) -> float:
    prod = 1.0
    for v in vals:
        v = max(0.0, min(1.0, float(v)))
        prod *= (1.0 - v)
    return 1.0 - prod

def _score_for_trigger(
    t: str,
    dim_weights: Dict[str, float],
    beta0: float,
    betas: Dict,  # can be {(trigger, feature): beta} or {feature: beta}
) -> float:
    s = beta0
    for k, v in dim_weights.items():
        # prefer per-trigger beta; fall back to global beta for that feature
        s += float(v) * float(betas.get((t, k), betas.get(k, 0.0)))
    return s

def compute_graphwin(
    *,
    conv_id: str,
    pain_triggers: List[str],
    dim_weights: Dict[str, float],            # attribute presence/weights (0..1)
    beta0: float = 0.0,                       # baseline logit (learn later)
    betas: Optional[Dict] = None,             # {(t, feat): beta} or {feat: beta}
    # When there is ZERO evidence (no attrs & no direct nodes), baseline_prior is used.
    # Set to 0.0 to enforce "no data => 0".
    baseline_prior: float = 0.0,
    # Direct evidence (these bypass upstream priors entirely):
    hard_on_triggers: Optional[List[str]] = None,
    hard_on_pains: Optional[List[str]] = None,
    hard_on_jobs: Optional[List[str]] = None,
    hard_on_personas: Optional[List[str]] = None,
    # Reach-to-product (backward PPR) for ANY node you might include below:
    reach_to_product: Optional[Dict[str, float]] = None,
) -> float:
    """
    GraphWin = noisyOR(   for each pain_trigger t:  Prior(t) * Reach(t->product),
                          for each observed node e (pain/job/persona): 1.0 * Reach(e->product)  )

    - Prior(t) is logistic(beta0 + sum beta_feat * dim_weight_feat), per trigger.
      If t itself is observed (in hard_on_triggers), Prior(t) := 1.0.

    - Downstream observations (pains/jobs/personas) DO NOT change Prior(t);
      they contribute their own atom (1.0 * Reach(e)) and make upstream priors irrelevant
      along those paths (no double counting is needed; noisyOR handles overlaps).
    """
    betas = betas or {}
    reach = reach_to_product or {}
    hard_on_triggers = set(hard_on_triggers or [])
    hard_on_pains = set(hard_on_pains or [])
    hard_on_jobs = set(hard_on_jobs or [])
    hard_on_personas = set(hard_on_personas or [])

    atoms: List[float] = []

    # --- Bucket A: attribute-seeded priors for pain_triggers ---
    has_any_attr = any(float(v) > 0.0 for v in dim_weights.values())
    for t in pain_triggers:
        if t in hard_on_triggers:
            prior_t = 1.0
        elif has_any_attr:
            score_t = _score_for_trigger(t, dim_weights, beta0, betas)
            prior_t = _sigmoid(score_t)
        else:
            # no attribute evidence at all → use baseline_prior (usually 0.0)
            prior_t = baseline_prior

        Rt = max(0.0, min(1.0, reach.get(t, 0.0)))
        atoms.append(prior_t * Rt)

    # --- Bucket B: direct evidence nodes (downstream) ---
    # pains, jobs, personas each add 1.0 * Reach(e->product)
    for e in (list(hard_on_pains) + list(hard_on_jobs) + list(hard_on_personas)):
        Re = max(0.0, min(1.0, reach.get(e, 0.0)))
        atoms.append(1.0 * Re)

    # If still no atoms at all (empty graph or no reach provided), return 0 safely
    if not atoms:
        return 0.0

    return float(_noisy_or(atoms))
