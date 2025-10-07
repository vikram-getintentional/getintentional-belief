# ============================
# File: backend/utils/inference/rcs_generators/ppr_engine.py
# ============================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Tuple
import math
import networkx as nx

# --- small utils ---
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _noisy_or(vals: Iterable[float]) -> float:
    prod = 1.0
    for v in vals:
        prod *= (1.0 - _clip01(v))
    return 1.0 - prod


@dataclass
class Evidence:
    """
    All the knobs that describe 'what we know right now'.
    - dim_weights : attribute features (already normalized 0..1)
    - hard_*     : evidence nodes that are known (prob=1)
    - betas/beta0: (optional) per-trigger logistic weights (learned later)
    """
    dim_weights: Dict[str, float]
    hard_triggers: List[str]
    hard_pains: List[str]
    hard_jobs: List[str]
    betas: Optional[Dict[Tuple[str, str], float]] = None  # key=(trigger_id, feature_name)
    beta0: float = 0.0


class PPREngine:
    """
    Computes:
      - reverse reach R_t (pain_trigger -> product) via PPR on reversed graph
      - GraphWin = noisyOR_t( Prior(t) * R_t )
      - Involvement via PPR mass under current evidence
      - Activation as ΔGraphWin when forcing node ON
    """
    def __init__(self, G_pruned: nx.DiGraph, *, product_id: str):
        self.G = G_pruned
        self.product_id = product_id
        if product_id not in self.G:
            raise ValueError(f"product_id {product_id} not in graph")

        # cache for pageranks: (seed_id, alpha) -> dict
        self._ppr_cache: Dict[Tuple[str, float], Dict[str, float]] = {}
        # reach cache: trigger -> R_t
        self._reach_cache: Dict[str, float] = {}

    # --------------------------
    # PageRank helpers
    # --------------------------
    def _ppr_single(self, Grev: nx.DiGraph, seed: str, *, alpha: float = 0.85,
                     max_iter: int = 100, tol: float = 1e-8) -> Dict[str, float]:
        key = (seed, alpha)
        if key in self._ppr_cache:
            return self._ppr_cache[key]
        pers = {n: 0.0 for n in Grev.nodes}
        pers[seed] = 1.0
        pr = nx.pagerank(Grev, alpha=alpha, personalization=pers, max_iter=max_iter, tol=tol)
        self._ppr_cache[key] = pr
        return pr

    # --------------------------
    # 1) Reverse reach from trigger to product
    # --------------------------
    def reverse_reach_to_product(
        self,
        pain_triggers: List[str],
        *,
        alpha: float = 0.85
    ) -> Dict[str, float]:
        """
        Forward graph edges are product -> capability -> job -> pain -> pain_trigger.
        Reverse it to walk trigger -> ... -> product; return PR mass at product.
        """
        Grev = self.G.reverse(copy=False)
        out: Dict[str, float] = {}
        for t in pain_triggers:
            if t in self._reach_cache:
                out[t] = self._reach_cache[t]
                continue
            if t not in Grev:
                out[t] = 0.0
                self._reach_cache[t] = 0.0
                continue
            pr = self._ppr_single(Grev, t, alpha=alpha)
            r = float(pr.get(self.product_id, 0.0))
            out[t] = r
            self._reach_cache[t] = r
        return out

    # --------------------------
    # 2) Priors for pain triggers (attributes → trigger prior; allow hard-on override)
    # --------------------------
    def _trigger_priors(
        self,
        pain_triggers: List[str],
        ev: Evidence,
        *,
        baseline_prior: float = 0.0
    ) -> Dict[str, float]:
        """
        If you have learned betas, we score per trigger:
            score_t = beta0 + Σ_k beta_{t,k} * dim_weights[k]
            prior_t = sigmoid(score_t)
        If no betas yet, we reduce to baseline_prior (usually 0) unless trigger is hard-on.
        """
        betas = ev.betas or {}
        dimw = ev.dim_weights or {}
        priors: Dict[str, float] = {}

        has_any_feature = any(abs(v) > 1e-9 for v in dimw.values())

        for t in pain_triggers:
            if t in (ev.hard_triggers or []):
                priors[t] = 1.0
                continue

            if betas:
                score_t = ev.beta0
                for k, v in dimw.items():
                    score_t += betas.get((t, k), 0.0) * float(v)
                priors[t] = _sigmoid(score_t)
            else:
                # no learned betas yet → if no feature evidence, stay at baseline_prior (typically 0)
                priors[t] = baseline_prior if not has_any_feature else 0.0

        return {t: _clip01(p) for t, p in priors.items()}

    # --------------------------
    # 3) GraphWin (current evidence)
    # --------------------------
    def graphwin(
        self,
        pain_triggers: List[str],
        ev: Evidence,
        *,
        baseline_prior: float = 0.0,
        alpha: float = 0.85
    ) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        """
        Returns:
          win: float
          priors: {trigger -> prior}
          reach:  {trigger -> reach-to-product}
        """
        reach = self.reverse_reach_to_product(pain_triggers, alpha=alpha)
        priors = self._trigger_priors(pain_triggers, ev, baseline_prior=baseline_prior)

        # NOTE: downstream evidence (hard pains/jobs) DOES NOT change trigger prior.
        # It only affects viability of paths, captured in reach via PPR on the fixed graph.
        atoms = []
        for t in pain_triggers:
            atoms.append(_clip01(priors.get(t, 0.0)) * _clip01(reach.get(t, 0.0)))
        win = _noisy_or(atoms)
        return float(win), priors, reach

    # --------------------------
    # 4) Involvement (PPR mass under current evidence seeds)
    # --------------------------
    def involvement_scores(
        self,
        *,
        engaged_ids: List[str],
        alpha: float = 0.85
    ) -> Dict[str, float]:
        """
        “How present is this node in flows right now?”
        - Use reversed graph so mass flows toward product direction.
        - Seeds = engaged_ids (if empty, returns zeros).
        - We return PR mass for all nodes; caller can map to persona/jobs/pains subsets.
        """
        if not engaged_ids:
            return {n: 0.0 for n in self.G.nodes}

        Grev = self.G.reverse(copy=False)
        # Build personalization as normalized over engaged seeds present in graph
        seeds = [s for s in engaged_ids if s in Grev]
        if not seeds:
            return {n: 0.0 for n in self.G.nodes}

        pers = {n: 0.0 for n in Grev.nodes}
        share = 1.0 / len(seeds)
        for s in seeds:
            pers[s] = share

        pr = nx.pagerank(Grev, alpha=alpha, personalization=pers)
        return {n: float(pr.get(n, 0.0)) for n in self.G.nodes}

    # --------------------------
    # 5) Activation (marginal Δ in GraphWin when forcing node ON)
    # --------------------------
    def activation_delta(
        self,
        pain_triggers: List[str],
        ev: Evidence,
        *,
        node_id: str,
        node_type: str,
        baseline_prior: float = 0.0,
        alpha: float = 0.85
    ) -> float:
        """
        Δ = GraphWin(ev with node hard-ON) - GraphWin(ev)
        “Force ON” means:
          - if trigger: add to hard_triggers
          - if pain:    add to hard_pains
          - if job:     add to hard_jobs
          - if persona: we infer its primary jobs and add those to hard_jobs (caller can pass jobs)
        """
        # current
        win_now, _, _ = self.graphwin(pain_triggers, ev, baseline_prior=baseline_prior, alpha=alpha)

        # clone evidence
        hardT = set(ev.hard_triggers or [])
        hardP = set(ev.hard_pains or [])
        hardJ = set(ev.hard_jobs or [])

        if node_type == "pain_trigger":
            hardT.add(node_id)
        elif node_type == "pain":
            hardP.add(node_id)
        elif node_type == "job":
            hardJ.add(node_id)
        elif node_type == "persona":
            # noop here; caller should expand persona->jobs and call activation on those
            pass

        ev_forced = Evidence(
            dim_weights=dict(ev.dim_weights),
            hard_triggers=list(hardT),
            hard_pains=list(hardP),
            hard_jobs=list(hardJ),
            betas=ev.betas,
            beta0=ev.beta0
        )
        win_forced, _, _ = self.graphwin(pain_triggers, ev_forced, baseline_prior=baseline_prior, alpha=alpha)
        return _clip01(win_forced - win_now)
