# backend/utils/inference/belief_manager/graph_diff_mapper.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional, Iterable
import math
import networkx as nx
from collections import defaultdict, Counter

# Use your helpers (schema-aware accessors)
from backend.utils.graph_base.network_graph import (
    _set_node_label,
    get_source_nodes_by_target_and_type,
    get_target_nodes_by_source_and_type,
)

# ---- Types for outputs ----
EdgeReco = Dict[str, Any]   # {u, v, rel, delta, reason, band?, confidence?, neighborhood_key?}
NodeReco = Dict[str, Any]   # {node_id, field, delta, reason}
Summary  = Dict[str, Any]

# ---- Canonical + fallback key coalescers ----
NODE_TYPE_KEYS = ("node_type", "type")
EDGE_REL_KEYS  = ("edge_type", "type", "rel")
WEIGHT_KEYS    = ("likelihood", "relevance", "boost", "weight", "prob", "p")

# ---- Learning config (Bayesian + guardrails) ----

SMALL_SAMPLE_CFG: Dict[str, Any] = {
    # make priors lighter so 1–9 events can move the needle
    "prior_strength_start": 1.0,
    "prior_strength_pp": 2.0,

    # shrinkage a bit looser
    "shrink_k_start": 1.5,
    "shrink_k_pp": 1.5,

    # keep absence-of-evidence protection ON
    "only_update_observed": True,
    "allow_negative_when_contradicted": True,  # allow small negative only if a strong alternative observed
    "prior_hi_thresh": 0.40,
    "observed_alt_share": 0.55,

    # let tiny-but-real shifts surface
    "min_transitions_for_pp_update": 1,
    "min_abs_delta_edge": 0.005,

    # temporarily relax credibility gate for tiny samples
    "gate_by_ci": False,
    "min_posterior_shift": 0.02,

    # keep lists tiny
    "max_recos_per_band": 5,
    "max_total_recos": 12,

    # (optional) nudge downstream to be a bit softer
    "band_scale": {"upstream": 1.0, "handoff": 1.0, "downstream": 0.85},
}

DEFAULT_CFG: Dict[str, Any] = dict(
    # Bayesian prior strengths (equivalent sample sizes)
    prior_strength_start=3.0,     # Dirichlet mass for starting persona distribution
    prior_strength_pp=5.0,        # Beta prior strength for pp edges (per-source persona)

    # Shrinkage against tiny samples
    shrink_k_start=4.0,           # scale delta by n/(n+k) ; n for start is count of observed sessions (usually 1)
    shrink_k_pp=3.0,              # n = transitions from a given persona

    # “Absence of evidence” guardrails
    only_update_observed=True,    # if True, never create negative deltas just because we didn’t observe something
    allow_negative_when_contradicted=True,   # allow small negative delta on high-prior edges if contradicted by clear evidence
    prior_hi_thresh=0.45,         # “high prior” threshold for contradiction gating
    observed_alt_share=0.60,      # require that some alternative handoff from the same source captured >= this share

    # Output throttles
    min_transitions_for_pp_update=1,      # require at least this many pa->* transitions before updating
    min_abs_delta_edge=0.02,              # drop tiny deltas
    max_recos_per_band=30,                # hard cap per band to keep summaries usable
    max_total_recos=60,                   # global hard cap

    # Credibility gating (normal approx)
    gate_by_ci=True,
    ci_z=1.64,                    # ~90% interval; increase to 1.96 for ~95%
    min_posterior_shift=0.04,     # require at least this much posterior-prior shift to consider

    # Band scaling (optional gentle re-weighting)
    band_scale=dict(upstream=1.00, handoff=1.00, downstream=0.85),
)

# ------------------ Small helpers ------------------

def _get_first(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _nt(G: nx.DiGraph, n: str) -> Optional[str]:
    """Node type: prefer 'node_type', else 'type'."""
    if n not in G:
        return None
    val = _get_first(G.nodes[n], NODE_TYPE_KEYS, None)
    return str(val).lower() if val is not None else None

def _rel(G: nx.DiGraph, u: str, v: str, default: str = "") -> str:
    """Edge relationship: prefer 'edge_type', else 'type', else 'rel'."""
    if not G.has_edge(u, v):
        return default
    val = _get_first(G[u][v], EDGE_REL_KEYS, default)
    return str(val).lower() if val is not None else default

def _w(
    G: nx.DiGraph,
    u: str,
    v: str,
    fallback_key: str = "likelihood",
    default: float = 0.0,
) -> float:
    """Edge weight: prefer fallback_key, else other common keys."""
    if not G.has_edge(u, v):
        return float(default)
    if fallback_key in G[u][v]:
        try:
            return float(G[u][v][fallback_key])
        except Exception:
            pass
    val = _get_first(G[u][v], WEIGHT_KEYS, default)
    try:
        return float(val)
    except Exception:
        return float(default)

def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _prob_from_expected_next(baseline_expected_next: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    baseline_expected_next: [{"persona": "persona:...", "prob": 0.23}, ...]  OR without prob.
    If 'prob' missing, derive via softmax on rank.
    """
    rows = [r for r in (baseline_expected_next or []) if r.get("persona")]
    if not rows:
        return {}
    if all("prob" in r for r in rows):
        return {r["persona"]: float(r["prob"]) for r in rows}
    logits = list(reversed(range(1, len(rows) + 1)))
    probs  = _softmax([float(z) for z in logits])
    return {r["persona"]: p for r, p in zip(rows, probs)}

def _transition_counts(observed_persona_ids: List[str]) -> Dict[Tuple[str, str], int]:
    C = Counter()
    for a, b in zip(observed_persona_ids, observed_persona_ids[1:]):
        C[(a, b)] += 1
    return dict(C)

#--- Label helpers ---

def _L(G: nx.DiGraph, nid: str) -> str:
    try:
        return _set_node_label(G, nid) or str(nid)
    except Exception:
        return str(nid)

def _safe(G: nx.DiGraph, nid: Optional[str]) -> bool:
    return bool(nid and nid in G)

# ---------- Schema wrappers (canonical directions) ----------
# Canonical schema (directional):
# Product --offers--> Capability --solves--> Pain --felt_in--> Job --performed_by--> Persona
# Job --solves--> Pain
# Pain --triggered_by--> Pain_trigger

def _persona_to_jobs(G: nx.DiGraph, pid: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, pid):
        return out
    for u in get_source_nodes_by_target_and_type(G, pid, "performed_by") or []:
        if _nt(G, u) == "job":
            out.append(u)
    return out

def _job_to_pains_felt(G: nx.DiGraph, jid: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, jid):
        return out
    for u in get_source_nodes_by_target_and_type(G, jid, "felt_in") or []:
        if _nt(G, u) == "pain":
            out.append(u)
    return out

def _pain_to_pain_triggers(G: nx.DiGraph, pain_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, pain_id):
        return out
    for v in get_target_nodes_by_source_and_type(G, pain_id, "triggered_by") or []:
        if _nt(G, v) == "pain_trigger":
            out.append(v)
    return out

def _pain_to_jobs_solves(G: nx.DiGraph, pain_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, pain_id):
        return out
    for u in get_source_nodes_by_target_and_type(G, pain_id, "solves") or []:
        if _nt(G, u) == "job":
            out.append(u)
    return out

def _pain_to_capabilities(G: nx.DiGraph, pain_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, pain_id):
        return out
    for u in get_source_nodes_by_target_and_type(G, pain_id, "solves") or []:
        if _nt(G, u) == "capability":
            out.append(u)
    return out

def _capability__to_products(G: nx.DiGraph, cap_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, cap_id):
        return out
    for u in get_source_nodes_by_target_and_type(G, cap_id, "offers") or []:
        if _nt(G, u) == "product":
            out.append(u)
    return out

def _jobs__to_persona(G: nx.DiGraph, jid: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, jid):
        return out
    for v in get_target_nodes_by_source_and_type(G, jid, "performed_by") or []:
        if _nt(G, v) == "persona":
            out.append(v)
    return out

def _pain_to_jobs_felt(G: nx.DiGraph, pain_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, pain_id):
        return out
    for v in get_target_nodes_by_source_and_type(G, pain_id, "felt_in") or []:
        if _nt(G, v) == "job":
            out.append(v)
    return out

def _pain_trigger_to_pains(G: nx.DiGraph, pain_trigger_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, pain_trigger_id):
        return out
    for u in get_source_nodes_by_target_and_type(G, pain_trigger_id, "triggered_by") or []:
        if _nt(G, u) == "pain":
            out.append(u)
    return out

def _job_to_pains_solves(G: nx.DiGraph, job_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, job_id):
        return out
    for v in get_target_nodes_by_source_and_type(G, job_id, "solves") or []:
        if _nt(G, v) == "pain":
            out.append(v)
    return out

def _capability_to_pains(G: nx.DiGraph, cap_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, cap_id):
        return out
    for v in get_target_nodes_by_source_and_type(G, cap_id, "solves") or []:
        if _nt(G, v) == "pain":
            out.append(v)
    return out

def _product_to_capabilities(G: nx.DiGraph, product_id: str) -> List[str]:
    out: List[str] = []
    if not _safe(G, product_id):
        return out
    for v in get_target_nodes_by_source_and_type(G, product_id, "offers") or []:
        if _nt(G, v) == "capability":
            out.append(v)
    return out

def _persona_to_products(G: nx.DiGraph, pid: str) -> List[str]:
    """
    persona -> jobs (performed_by)
      -> pains felt in those jobs
      -> capabilities solving those pains
      -> products offering those capabilities
    """
    out: List[str] = []
    if not _safe(G, pid):
        return out
    products: List[str] = []
    for j in _persona_to_jobs(G, pid):
        for p in _job_to_pains_felt(G, j):
            for c in _pain_to_capabilities(G, p):
                for pr in _capability__to_products(G, c):
                    products.append(pr)
    # dedupe
    seen = set()
    deduped: List[str] = []
    for pr in products:
        if pr not in seen:
            seen.add(pr)
            deduped.append(pr)
    return deduped

# ------------------ Bayesian helpers ------------------

def _normal_ci_halfwidth(p: float, n_eff: float, z: float) -> float:
    """Cheap approximate CI half-width for a proportion."""
    if n_eff <= 0:
        return 1.0
    p = max(1e-9, min(1.0 - 1e-9, p))
    return z * math.sqrt(p * (1 - p) / n_eff)

def _dirichlet_update_start(
    expected_start: Dict[str, float],
    observed_persona_ids: List[str],
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Bayesian update of starting persona distribution with Dirichlet prior.
    We *only* positively update the actually observed starter unless cfg.only_update_observed=False.
    """
    pri = expected_start or {}
    if not observed_persona_ids:
        return {}, {"n": 0}

    starter = observed_persona_ids[0]
    s0 = float(cfg["prior_strength_start"])
    shrink_k = float(cfg["shrink_k_start"])

    # Construct prior masses
    personas = set(pri) | {starter}
    alpha0: Dict[str, float] = {pid: max(1e-6, pri.get(pid, 0.0)) * s0 for pid in personas}
    n = 1.0  # one observation for start in this session
    counts = {pid: 0.0 for pid in personas}
    counts[starter] = 1.0

    deltas: Dict[str, float] = {}
    for pid in personas:
        a = alpha0[pid] + counts[pid]
        S = sum(alpha0.values()) + n
        p_post = a / S
        p_prior = pri.get(pid, 0.0)
        delta = p_post - p_prior
        # Absence-of-evidence guard: unless explicitly allowed, do not emit negative deltas for unobserved starters
        if cfg["only_update_observed"] and pid != starter and delta < 0:
            delta = 0.0
        # Sample-size shrink
        scaled = delta * (n / (n + shrink_k))
        deltas[pid] = scaled

    # Keep only meaningful deltas
    clean = {pid: dv for pid, dv in deltas.items() if abs(dv) >= cfg["min_abs_delta_edge"]}
    return clean, {"n": n, "starter": starter}

def _group_obs_by_source(obs_counts: Dict[Tuple[str, str], int]) -> Dict[str, Dict[str, int]]:
    by_src: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (pa, pb), c in obs_counts.items():
        by_src[pa][pb] += int(c)
    return by_src

def _contradicted_prior_pairs(
    prior_pp: Dict[Tuple[str, str], float],
    obs_by_src: Dict[str, Dict[str, int]],
    cfg: Dict[str, Any],
) -> Iterable[Tuple[str, str]]:
    """
    Identify (pa,pb_prior) pairs where:
      - prior(pa->pb_prior) is high
      - pb_prior received zero observations
      - some alternative pb* captured majority of observed share
    """
    hi = float(cfg["prior_hi_thresh"])
    alt_share = float(cfg["observed_alt_share"])
    out = []
    for (pa, pb), p0 in prior_pp.items():
        if p0 < hi:
            continue
        obs_row = obs_by_src.get(pa, {})
        n = sum(obs_row.values())
        if n <= 0:
            continue
        if obs_row.get(pb, 0) > 0:
            continue  # not contradicted; it did occur
        # Did some alternative dominate?
        best_pb, best_c = (None, 0)
        if obs_row:
            best_pb, best_c = max(obs_row.items(), key=lambda kv: kv[1])
        if best_c / float(n) >= alt_share:
            out.append((pa, pb))
    return out

def _beta_update_pp(
    G: nx.DiGraph,
    prior_pp: Dict[Tuple[str, str], float],
    obs_counts: Dict[Tuple[str, str], int],
    cfg: Dict[str, Any],
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, Any]]:
    """
    Bayesian update for persona→persona edges using Beta priors per (pa,pb).
    We aggregate by source persona to get the trial count n = sum_b counts(pa->b).
    We return deltas only for:
      - observed (pa,pb) pairs (positive or negative vs prior depending on magnitude), and
      - optionally, high-prior-but-contradicted pairs (negative updates), if enabled.
    """
    s0 = float(cfg["prior_strength_pp"])
    shrink_k = float(cfg["shrink_k_pp"])
    by_src = _group_obs_by_source(obs_counts)

    # Which non-observed high-prior pairs are contradicted by clear alternatives?
    contradicted = set()
    if cfg["allow_negative_when_contradicted"]:
        contradicted = set(_contradicted_prior_pairs(prior_pp, by_src, cfg))

    deltas: Dict[Tuple[str, str], float] = {}
    meta_rows: List[Dict[str, Any]] = []

    # Iterate over personas we actually saw transition from (data-backed)
    for pa, row in by_src.items():
        n = float(sum(row.values()))
        if n < cfg["min_transitions_for_pp_update"]:
            continue

        # Consider all observed pb and (optionally) contradicted-not-seen pb
        candidate_pbs = set(row.keys())
        for (p_pa, p_pb) in contradicted:
            if p_pa == pa:
                candidate_pbs.add(p_pb)

        for pb in candidate_pbs:
            k = float(row.get(pb, 0))
            # Prior probability (fallback: compose from motif if missing)
            p0 = float(prior_pp.get((pa, pb), 0.0))
            alpha0 = p0 * s0
            beta0  = (1.0 - p0) * s0
            # Posterior mean with Beta-Binomial
            p_post = (alpha0 + k) / (alpha0 + beta0 + n)
            raw_delta = p_post - p0

            # Absence-of-evidence guard: if k==0 (unobserved) and it's not explicitly contradicted, suppress negative delta.
            if cfg["only_update_observed"] and k == 0 and (pa, pb) not in contradicted and raw_delta < 0:
                continue

            # Sample-size shrink
            scaled = raw_delta * (n / (n + shrink_k))

            # Credibility gate (normal approx around p_post with n_eff = alpha0+beta0+n)
            if cfg["gate_by_ci"]:
                n_eff = (alpha0 + beta0 + n)
                hw = _normal_ci_halfwidth(p_post, n_eff, float(cfg["ci_z"]))
                # Require the posterior to have shifted by at least both thresholds
                if abs(raw_delta) < cfg["min_posterior_shift"] or abs(raw_delta) < hw:
                    continue

            if abs(scaled) >= cfg["min_abs_delta_edge"]:
                deltas[(pa, pb)] = scaled
                meta_rows.append({
                    "pa": pa, "pb": pb, "n": n, "k": k,
                    "prior": p0, "post": p_post, "raw_delta": raw_delta, "scaled": scaled,
                })

    return deltas, {"rows": meta_rows}

# ---- Fallback prior: compose pp if explicit edges missing ----

def _composed_pp_prior(G: nx.DiGraph, weight_key: str = "likelihood") -> Dict[Tuple[str, str], float]:
    priors: Dict[Tuple[str, str], float] = defaultdict(float)
    personas = [n for n in G.nodes if _nt(G, n) == "persona"]
    for pa in personas:
        for ja in _persona_to_jobs(G, pa):  # ja --performed_by--> pa
            w_ja_pa = _w(G, ja, pa, weight_key, 0.0)
            for p in _job_to_pains_felt(G, ja):  # p --felt_in--> ja
                w_p_ja = _w(G, p, ja, weight_key, 0.0)
                for jb in _pain_to_jobs_solves(G, p):  # jb --solves--> p
                    w_jb_p = _w(G, jb, p, weight_key, 0.0)
                    for pb in _jobs__to_persona(G, jb):  # jb --performed_by--> pb
                        w_jb_pb = _w(G, jb, pb, weight_key, 0.0)
                        score = w_ja_pa * w_p_ja * w_jb_p * w_jb_pb
                        if score > priors[(pa, pb)]:
                            priors[(pa, pb)] = score
    # normalize per pa
    out: Dict[Tuple[str, str], float] = {}
    grouped: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for (pa, pb), val in priors.items():
        grouped[pa].append((pb, val))
    for pa, items in grouped.items():
        Z = sum(v for _, v in items) or 1.0
        for pb, v in items:
            out[(pa, pb)] = v / Z
    return out

def _transition_expectations_from_graph(
    G: nx.DiGraph,
    weight_key: str = "likelihood",
) -> Dict[Tuple[str, str], float]:
    E: Dict[Tuple[str, str], float] = {}
    for u, v in G.edges():
        if _nt(G, u) == "persona" and _nt(G, v) == "persona":
            E[(u, v)] = float(_w(G, u, v, weight_key, 0.0))
    if not E:
        E = _composed_pp_prior(G, weight_key)
    return E

# ---- Confidence heuristic (compact, explainable) ----

def _edge_confidence(
    u: str,
    v: str,
    delta: float,
    obs_counts: Dict[Tuple[str, str], int],
    total_obs: int,
) -> float:
    base = 0.55 + 0.40 * _sigmoid(5.0 * abs(float(delta)))  # 0.55..0.95
    bump = 0.0
    if (u, v) in obs_counts and total_obs > 0:
        bump = min(0.10, 0.02 * min(5, obs_counts[(u, v)]))
    return max(0.55, min(0.99, base + bump))

# ---- Mapping rules: diffs → neighborhoods ----

def _attribute_starting_persona_diff(
    G: nx.DiGraph,
    persona_delta: Dict[str, float],
    weight_key: str = "likelihood",
    allocation: str = "prior_weighted",
    band_scale: Dict[str, float] | None = None,
) -> Tuple[List[EdgeReco], List[NodeReco], List[Dict[str, Any]], Dict[str, Any]]:
    """
    For each starting persona delta:
      1. persona -> jobs it performs
      2. job -> pains it solves (upstream pains)
      3. pain -> pain_triggers
      4. pain -> upstream jobs where pain is felt
      5. job -> upstream personas performing those jobs

    We then attribute the starting-persona delta across this neighborhood,
    and mark perceptibility on the persona itself.
    """
    edge_recos: List[EdgeReco] = []
    node_recos: List[NodeReco] = []
    theses: List[Dict[str, Any]] = []
    neighborhoods: Dict[str, Any] = {}

    bs = band_scale or {}

    for pid, d in persona_delta.items():
        if not _safe(G, pid) or abs(d) < 1e-9:
            continue

        # Node-level perceptibility
        node_recos.append({
            "node_id": pid,
            "node_label": _L(G, pid),
            "field": "perceptibility",
            "delta": d * bs.get("upstream", 1.0),
            "reason": "Starting persona likelihood diff suggests perceptibility misestimated.",
        })

        # 1) Jobs this persona performs
        jobs_lvl1 = _persona_to_jobs(G, pid)

        # 2) Upstream pains those jobs are meant to solve
        pains_solved = {p for j in jobs_lvl1 for p in _job_to_pains_solves(G, j)}

        # 3) Triggers that create those pains
        triggers = {t for p in pains_solved for t in _pain_to_pain_triggers(G, p)}

        # 4) Upstream jobs where these pains are felt
        jobs_felt = {jf for p in pains_solved for jf in _pain_to_jobs_felt(G, p)}

        # 5) Upstream personas doing those jobs
        upstream_personas = {pb for j in jobs_felt for pb in _jobs__to_persona(G, j) if pb != pid}

        candidate_edges: List[Tuple[str, str, str]] = []
        seen_edges: set[Tuple[str, str]] = set()
        motifs: List[List[Tuple[str, str, str]]] = []

        # Neighborhood around starting persona
        for j in jobs_lvl1:
            # job -> persona (performed_by)
            if G.has_edge(j, pid) and (j, pid) not in seen_edges:
                seen_edges.add((j, pid))
                candidate_edges.append((j, pid, _rel(G, j, pid)))
            # job -> pain (solves)
            for p in _job_to_pains_solves(G, j):
                if G.has_edge(j, p) and (j, p) not in seen_edges:
                    seen_edges.add((j, p))
                    candidate_edges.append((j, p, _rel(G, j, p)))
                # pain -> job_felt (felt_in)
                for jf in _pain_to_jobs_felt(G, p):
                    if G.has_edge(p, jf) and (p, jf) not in seen_edges:
                        seen_edges.add((p, jf))
                        candidate_edges.append((p, jf, _rel(G, p, jf)))
                    # job_felt -> upstream persona
                    for up in _jobs__to_persona(G, jf):
                        if G.has_edge(jf, up) and (jf, up) not in seen_edges:
                            seen_edges.add((jf, up))
                            candidate_edges.append((jf, up, _rel(G, jf, up)))
                            motifs.append([
                                (j, "performed_by", pid),
                                (j, "solves", p),
                                (p, "felt_in", jf),
                                (jf, "performed_by", up),
                            ])
                # pain -> trigger
                for t in _pain_to_pain_triggers(G, p):
                    if G.has_edge(p, t) and (p, t) not in seen_edges:
                        seen_edges.add((p, t))
                        candidate_edges.append((p, t, _rel(G, p, t)))

        if not candidate_edges:
            theses.append({
                "persona_id": pid,
                "persona_label": _L(G, pid),
                "thesis": "No upstream neighborhood candidates found; graph may lack job/pain/trigger links.",
            })
            continue

        # Allocate persona-delta across neighborhood
        weights = [float(_w(G, u, v, weight_key, 0.1)) for (u, v, _relname) in candidate_edges]
        S = sum(weights) or 1.0
        for (u, v, rel), w in zip(candidate_edges, weights):
            edge_recos.append({
                "u": u,
                "v": v,
                "rel": rel,
                "u_label": _L(G, u),
                "v_label": _L(G, v),
                "delta": (d * (w / S)) * bs.get("upstream", 1.0),
                "reason": f"Starting-likelihood delta for {pid} propagated through upstream job/pain/trigger chain.",
                "band": "upstream",
                "neighborhood_key": f"start:{pid}",
            })

        neighborhoods[f"start:{pid}"] = {
            "band": "upstream",
            "anchor": pid,
            "anchor_label": _L(G, pid),
            "rationale": (
                "Starting persona diff; upstream jobs, pains, triggers, and upstream personas "
                "around this starting point are likely misweighted."
            ),
            "edges": candidate_edges,
            "edge_labels": [(_L(G, u), _L(G, v), rel) for (u, v, rel) in candidate_edges],
            "motifs": motifs,
            "delta_sum": d,
        }

        theses.append({
            "persona_id": pid,
            "persona_label": _L(G, pid),
            "thesis": (
                "Observed starter is {} than baseline; either this persona should start more/less often, "
                "or an upstream latent persona/job/pain chain is misweighted."
            ).format("more central" if d > 0 else "less central"),
            "jobs": list(jobs_lvl1),
            "pains_solved": list(pains_solved),
            "triggers": list(triggers),
            "upstream_jobs_felt": list(jobs_felt),
            "upstream_personas": list(upstream_personas),
        })

    return edge_recos, node_recos, theses, neighborhoods

def _attribute_pp_diff(
    G: nx.DiGraph,
    pp_delta: Dict[Tuple[str, str], float],
    weight_key: str = "likelihood",
    allocation: str = "prior_weighted",
    band_scale: Dict[str, float] | None = None,
) -> Tuple[List[EdgeReco], List[NodeReco], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Persona→persona handoff delta:
      1. For persona A, get jobs (performed_by).
      2. For each job, get downstream pains this job feels (felt_in).
      3. For each pain, get downstream jobs solving this pain (solves).
      4. For each job, get personas performing this job.

    This uncovers latent or skipped personas that mediate A→B.
    """
    edge_recos: List[EdgeReco] = []
    node_recos: List[NodeReco] = []
    theses: List[Dict[str, Any]] = []
    neighborhoods: Dict[str, Any] = {}

    bs = band_scale or {}

    for (pa, pb), d in pp_delta.items():
        if abs(d) < 1e-9 or not (_safe(G, pa) and _safe(G, pb)):
            continue

        jobs_a = _persona_to_jobs(G, pa)
        jobs_b = _persona_to_jobs(G, pb)

        # pains from A's jobs
        pains_from_a = {p for ja in jobs_a for p in _job_to_pains_felt(G, ja)}
        # jobs that solve those pains
        solver_jobs = {jb for p in pains_from_a for jb in _pain_to_jobs_solves(G, p)}
        # personas performing those jobs
        handoff_personas = {pers for jb in solver_jobs for pers in _jobs__to_persona(G, jb)}

        candidate_edges: List[Tuple[str, str, str]] = []
        seen_edges: set[Tuple[str, str]] = set()
        motifs: List[List[Tuple[str, str, str]]] = []

        # Neighborhood around persona A
        for ja in jobs_a:
            if G.has_edge(ja, pa) and (ja, pa) not in seen_edges:
                seen_edges.add((ja, pa))
                candidate_edges.append((ja, pa, _rel(G, ja, pa)))
            for p in _job_to_pains_felt(G, ja):
                if G.has_edge(p, ja) and (p, ja) not in seen_edges:
                    seen_edges.add((p, ja))
                    candidate_edges.append((p, ja, _rel(G, p, ja)))
                for jb in _pain_to_jobs_solves(G, p):
                    if G.has_edge(jb, p) and (jb, p) not in seen_edges:
                        seen_edges.add((jb, p))
                        candidate_edges.append((jb, p, _rel(G, jb, p)))
                    for pers in _jobs__to_persona(G, jb):
                        if G.has_edge(jb, pers) and (jb, pers) not in seen_edges:
                            seen_edges.add((jb, pers))
                            candidate_edges.append((jb, pers, _rel(G, jb, pers)))
                            motifs.append([
                                (ja, "performed_by", pa),
                                (p, "felt_in", ja),
                                (jb, "solves", p),
                                (jb, "performed_by", pers),
                            ])

        # Neighborhood around persona B (receiver)
        for jb in jobs_b:
            if G.has_edge(jb, pb) and (jb, pb) not in seen_edges:
                seen_edges.add((jb, pb))
                candidate_edges.append((jb, pb, _rel(G, jb, pb)))
            for p in _job_to_pains_felt(G, jb):
                if G.has_edge(p, jb) and (p, jb) not in seen_edges:
                    seen_edges.add((p, jb))
                    candidate_edges.append((p, jb, _rel(G, p, jb)))

        if not candidate_edges:
            theses.append({
                "pair": [pa, pb],
                "pair_labels": [_L(G, pa), _L(G, pb)],
                "thesis": (
                    "No clear job/pain handoff motif found between these personas; "
                    "consider adding felt_in/solves links to explain persona→persona flow."
                ),
            })
            continue

        weights = [float(_w(G, u, v, weight_key, 0.1)) for (u, v, _relname) in candidate_edges]
        S = sum(weights) or 1.0
        for (u, v, rel), w in zip(candidate_edges, weights):
            edge_recos.append({
                "u": u,
                "v": v,
                "rel": rel,
                "delta": (d * (w / S)) * bs.get("handoff", 1.0),
                "u_label": _L(G, u),
                "v_label": _L(G, v),
                "reason": (
                    f"Observed vs expected handoff {pa}->{pb} suggests this job/pain/persona edge is misweighted."
                ),
                "band": "handoff",
                "neighborhood_key": f"handoff:{pa}->{pb}",
            })

        neighborhoods[f"handoff:{pa}->{pb}"] = {
            "band": "handoff",
            "anchor": [pa, pb],
            "anchor_labels": [_L(G, pa), _L(G, pb)],
            "rationale": (
                "Persona→persona handoff diff; jobA→pain→jobB (and latent personas) "
                "around this pair likely need adjustment."
            ),
            "edges": candidate_edges,
            "edge_labels": [(_L(G, u), _L(G, v), rel) for (u, v, rel) in candidate_edges],
            "motifs": motifs,
            "delta_sum": d,
        }

        theses.append({
            "pair": [pa, pb],
            "pair_labels": [_L(G, pa), _L(G, pb)],
            "thesis": (
                "Handoff from {} to {} is {} than assumed; tune job/pain chains and latent personas "
                "that connect them."
            ).format(_L(G, pa), _L(G, pb), "stronger" if d > 0 else "weaker"),
            "jobs_a": list(jobs_a),
            "jobs_b": list(jobs_b),
            "pains_from_a": list(pains_from_a),
            "solver_jobs": list(solver_jobs),
            "handoff_personas": list(handoff_personas),
        })

    return edge_recos, node_recos, theses, neighborhoods

def _attribute_prod_diff(
    G: nx.DiGraph,
    prod_delta: Dict[Tuple[str, str], float],    # (persona, product) -> delta
    weight_key: str = "likelihood",
    allocation: str = "prior_weighted",
    band_scale: Dict[str, float] | None = None,
) -> Tuple[List[EdgeReco], List[NodeReco], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Persona→Product diff:
      1. persona -> jobs
      2. job -> downstream pains (felt_in)
      3. pain -> downstream jobs solving it (solves)
      4. pain -> downstream capabilities solving it (solves)
      5. capability -> product (offers)
      6. job -> personas (up/downstream latent personas)

    This highlights which downstream chain (pains, jobs, capabilities, latent personas)
    should be reweighted when a persona is more/less likely to drive conversion.
    """
    edge_recos: List[EdgeReco] = []
    node_recos: List[NodeReco] = []
    theses: List[Dict[str, Any]] = []
    neighborhoods: Dict[str, Any] = {}

    bs = band_scale or {}

    for (pid, prod), d in prod_delta.items():
        if abs(d) < 1e-9 or not (_safe(G, pid) and _safe(G, prod)):
            continue

        jobs = _persona_to_jobs(G, pid)
        pains_felt = {p for j in jobs for p in _job_to_pains_felt(G, j)}
        solver_jobs = {jb for p in pains_felt for jb in _pain_to_jobs_solves(G, p)}
        caps = {c for p in pains_felt for c in _pain_to_capabilities(G, p)}
        prods = {pr for c in caps for pr in _capability__to_products(G, c)}
        latent_personas = {pers for jb in solver_jobs for pers in _jobs__to_persona(G, jb)}

        candidate_edges: List[Tuple[str, str, str]] = []
        seen_edges: set[Tuple[str, str]] = set()
        motifs: List[List[Tuple[str, str, str]]] = []

        for j in jobs:
            if G.has_edge(j, pid) and (j, pid) not in seen_edges:
                seen_edges.add((j, pid))
                candidate_edges.append((j, pid, _rel(G, j, pid)))
            for p in _job_to_pains_felt(G, j):
                if G.has_edge(p, j) and (p, j) not in seen_edges:
                    seen_edges.add((p, j))
                    candidate_edges.append((p, j, _rel(G, p, j)))
                for jb in _pain_to_jobs_solves(G, p):
                    if G.has_edge(jb, p) and (jb, p) not in seen_edges:
                        seen_edges.add((jb, p))
                        candidate_edges.append((jb, p, _rel(G, jb, p)))
                    for pers in _jobs__to_persona(G, jb):
                        if G.has_edge(jb, pers) and (jb, pers) not in seen_edges:
                            seen_edges.add((jb, pers))
                            candidate_edges.append((jb, pers, _rel(G, jb, pers)))
                    for c in _pain_to_capabilities(G, p):
                        if G.has_edge(c, p) and (c, p) not in seen_edges:
                            seen_edges.add((c, p))
                            candidate_edges.append((c, p, _rel(G, c, p)))
                        for pr in _capability__to_products(G, c):
                            if G.has_edge(pr, c) and (pr, c) not in seen_edges:
                                seen_edges.add((pr, c))
                                candidate_edges.append((pr, c, _rel(G, pr, c)))
                                motifs.append([
                                    (j, "performed_by", pid),
                                    (p, "felt_in", j),
                                    (jb, "solves", p),
                                    (c, "solves", p),
                                    (pr, "offers", c),
                                ])

        if not candidate_edges:
            theses.append({
                "pair": [pid, prod],
                "pair_labels": [_L(G, pid), _L(G, prod)],
                "thesis": (
                    "No downstream product motif found from this persona; "
                    "add pain→capability→product links to explain conversion."
                ),
            })
            continue

        weights = [float(_w(G, u, v, weight_key, 0.1)) for (u, v, _relname) in candidate_edges]
        S = sum(weights) or 1.0
        for (u, v, rel), w in zip(candidate_edges, weights):
            edge_recos.append({
                "u": u,
                "v": v,
                "rel": rel,
                "delta": (d * (w / S)) * bs.get("downstream", 1.0),
                "u_label": _L(G, u),
                "v_label": _L(G, v),
                "reason": (
                    f"Persona→Product delta {pid}->{prod} distributed across downstream job/pain/capability/product chain."
                ),
                "band": "downstream",
                "neighborhood_key": f"prod:{pid}->{prod}",
            })

        neighborhoods[f"prod:{pid}->{prod}"] = {
            "band": "downstream",
            "anchor": [pid, prod],
            "anchor_labels": [_L(G, pid), _L(G, prod)],
            "rationale": (
                "Persona→Product diff; downstream pains, solver jobs, capabilities, latent personas, "
                "and product edges likely need adjustment."
            ),
            "edges": candidate_edges,
            "edge_labels": [(_L(G, u), _L(G, v), rel) for (u, v, rel) in candidate_edges],
            "motifs": motifs,
            "delta_sum": d,
        }

        theses.append({
            "pair": [pid, prod],
            "pair_labels": [_L(G, pid), _L(G, prod)],
            "thesis": (
                "Persona {} driving product {} is {} than assumed; tune downstream pains, "
                "solver jobs, capabilities, and latent personas on the path to conversion."
            ).format(_L(G, pid), _L(G, prod), "more likely" if d > 0 else "less likely"),
            "jobs": list(jobs),
            "pains_felt": list(pains_felt),
            "solver_jobs": list(solver_jobs),
            "capabilities": list(caps),
            "reachable_products": list(prods),
            "latent_personas": list(latent_personas),
        })

    return edge_recos, node_recos, theses, neighborhoods

# ---- Main entry ----

def infer_graph_updates_from_diffs(
    *,
    product_graph: nx.DiGraph,
    baseline_expected_next: List[Dict[str, Any]],
    observed_persona_ids: List[str],
    weight_key: str = "likelihood",
    allocation: str = "prior_weighted",  # "even" or "prior_weighted"
    learn_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Bayesian, guardrailed learner:
      1) Priors:
         - starting persona prior from baseline_expected_next (Dirichlet)
         - persona→persona priors from explicit edges or composed motif (Beta)
      2) Evidence:
         - observed starter (one count for this session)
         - observed persona→persona transitions with counts
      3) Updates:
         - deltas only where we have evidence (or explicit contradiction of a high prior)
         - shrink deltas with sample-size schedule; gate by credibility
         - never punish edges solely due to non-observation (absence-of-evidence guard)
      4) Mapping:
         - convert deltas → neighborhood edge/node suggestions with labels
         - cap outputs per band and overall
    """
    G = product_graph
    cfg = {**DEFAULT_CFG, **(learn_cfg or {})}
    band_scale = cfg.get("band_scale", {})

    # 1a) starting persona Bayesian delta
    expected_start = _prob_from_expected_next(baseline_expected_next)  # pid -> p
    start_deltas, start_meta = _dirichlet_update_start(expected_start, observed_persona_ids, cfg)

    # 1b) persona→persona Bayesian delta
    prior_pp = _transition_expectations_from_graph(G, weight_key=weight_key)    # (pa, pb) -> prior
    obs_counts = _transition_counts(observed_persona_ids)                       # counts
    pp_deltas, pp_meta = _beta_update_pp(G, prior_pp, obs_counts, cfg)

    # 1c) persona→product (hook left intact; no Bayesian layer here yet)
    prior_p2pr: Dict[Tuple[str, str], float] = {}
    for u, v in G.edges():
        if _nt(G, u) == "persona" and _nt(G, v) == "product":
            prior_p2pr[(u, v)] = float(_w(G, u, v, weight_key, 0.0))
    obs_p2pr: Dict[Tuple[str, str], float] = {}
    all_p2pr = set(prior_p2pr) | set(obs_p2pr)
    prod_delta = {pair: (obs_p2pr.get(pair, 0.0) - prior_p2pr.get(pair, 0.0)) for pair in all_p2pr}
    # Guardrails: drop negative “non-observed” downstream deltas entirely
    prod_delta = {k: v for k, v in prod_delta.items() if v > 0}

    # 2) Map deltas → neighborhoods
    e1, n1, t1, nb_start   = _attribute_starting_persona_diff(
        G, start_deltas, weight_key=weight_key, allocation=allocation, band_scale=band_scale
    )
    e2, n2, t2, nb_handoff = _attribute_pp_diff(
        G, pp_deltas, weight_key=weight_key, allocation=allocation, band_scale=band_scale
    )
    e3, n3, t3, nb_prod    = _attribute_prod_diff(
        G, prod_delta, weight_key=weight_key, allocation=allocation, band_scale=band_scale
    )

    edge_recos = (e1 + e2 + e3)
    node_recos = (n1 + n2 + n3)

    neighborhoods = {
        "upstream": nb_start,
        "handoff":  nb_handoff,
        "downstream": nb_prod,
    }

    # Aggregate similar edge recos (preserve band, reasons, and neighborhood_key)
    agg: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in edge_recos:
        key = (r["u"], r["v"], _rel(G, r["u"], r["v"]))
        if key not in agg:
            agg[key] = {
                "u": r["u"], "v": r["v"], "rel": _rel(G, r["u"], r["v"]),
                "delta": 0.0, "reasons": [],
                "band": r.get("band", None),
                "neighborhood_keys": set(),
                "u_label": r.get("u_label") or _L(G, r["u"]),
                "v_label": r.get("v_label") or _L(G, r["v"]),
            }
        agg[key]["delta"] += float(r["delta"])
        if r.get("reason"):
            agg[key]["reasons"].append(r["reason"])
        nk = r.get("neighborhood_key")
        if nk:
            agg[key]["neighborhood_keys"].add(nk)

    merged_edges: List[Dict[str, Any]] = []
    for rec in agg.values():
        rec["neighborhood_keys"] = sorted(list(rec["neighborhood_keys"])) if rec["neighborhood_keys"] else []
        merged_edges.append(rec)

    # Attach confidence (post-merge)
    total = sum(obs_counts.values()) or 0
    for r in merged_edges:
        r["confidence"] = _edge_confidence(r["u"], r["v"], r["delta"], obs_counts, int(total))
        if r.get("band") is None:
            u_type = _nt(G, r["u"]) or ""
            v_type = _nt(G, r["v"]) or ""
            if u_type == "job" and v_type == "persona":
                r["band"] = "upstream"
            elif {u_type, v_type} & {"job", "pain"} and (u_type != v_type):
                r["band"] = "handoff"
            elif (u_type == "product" and v_type == "capability") or (u_type == "capability" and v_type == "pain"):
                r["band"] = "downstream"

    # Prune tiny deltas and throttle per-band + overall
    pruned = [r for r in merged_edges if abs(r["delta"]) >= cfg["min_abs_delta_edge"]]
    had_any_raw = len(merged_edges) > 0

    # If everything got pruned by thresholds but we DO have raw recos,
    # surface the top 3 weakest suggestions as "soft" so the UI has something credible to show.
    if not pruned and had_any_raw:
        fallback_soft = sorted(
            merged_edges, key=lambda x: (abs(x["delta"]), x.get("confidence", 0.0)), reverse=True
        )[:3]
        for r in fallback_soft:
            r["soft"] = True          # UI can render with a lighter style / lower emphasis
            r["confidence"] = min(r.get("confidence", 0.6), 0.7)
            r["reason"] = (r.get("reason") or "") + " (soft: surfaced despite small shift)"
        pruned = fallback_soft

    # sort by |delta| then by confidence
    pruned.sort(key=lambda x: (abs(x["delta"]), x.get("confidence", 0.0)), reverse=True)

    # cap per band
    by_band: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pruned:
        by_band[r.get("band", "misc")].append(r)
    capped: List[Dict[str, Any]] = []
    for band, rows in by_band.items():
        capped.extend(rows[: cfg["max_recos_per_band"]])
    # global cap
    capped = capped[: cfg["max_total_recos"]]

    # Node recos prune & cap similarly
    node_recos = [n for n in node_recos if abs(n["delta"]) >= cfg["min_abs_delta_edge"]]
    node_recos.sort(key=lambda x: abs(x["delta"]), reverse=True)
    node_recos = node_recos[: max(1, cfg["max_total_recos"] // 2)]

    # Human-readable theses (with labels)
    persona_path_expected = [
        pid for pid, _ in sorted(expected_start.items(), key=lambda kv: kv[1], reverse=True)[:3]
    ]
    persona_path_observed = observed_persona_ids[:3]

    A_persona_path_theses = []
    for pid, dv in sorted(start_deltas.items(), key=lambda kv: abs(kv[1]), reverse=True):
        if abs(dv) < cfg["min_abs_delta_edge"]:
            continue
        A_persona_path_theses.append({
            "persona_id": pid,
            "persona_label": _L(G, pid),
            "delta": dv,
            "explanation": (
                "Starting persona likelihood was {} than expected; "
                "adjust perceptibility and upstream motif edges accordingly."
            ).format("higher" if dv > 0 else "lower"),
        })

    # Upstream + handoff
    B_latent_node_theses: List[Dict[str, Any]] = []
    for t in (t1 + t2):
        if "persona_id" in t:
            t["persona_label"] = _L(G, t["persona_id"])
        if "pair" in t and isinstance(t["pair"], list) and len(t["pair"]) == 2:
            a, b = t["pair"]
            t["pair_labels"] = [_L(G, a), _L(G, b)]
        B_latent_node_theses.append(t)

    # Downstream
    C_persona_product_theses: List[Dict[str, Any]] = []
    for t in t3:
        if "pair" in t and isinstance(t["pair"], list) and len(t["pair"]) == 2:
            a, b = t["pair"]
            t["pair_labels"] = [_L(G, a), _L(G, b)]
        C_persona_product_theses.append(t)

    summary: Summary = {
        "persona_path_diff": {
            "expected_top3": persona_path_expected,
            "expected_top3_labels": [_L(G, x) for x in persona_path_expected],
            "observed_top3": persona_path_observed,
            "observed_top3_labels": [_L(G, x) for x in persona_path_observed],
        },
        "learned": {
            "A_persona_path_theses": A_persona_path_theses,
            "B_latent_node_theses": B_latent_node_theses,
            "C_persona_product_theses": C_persona_product_theses,
            "meta": {
                "start": start_meta,
                "pp": pp_meta,
                "cfg_used": cfg,
            }
        },
        "recommendations": {
            "edge_updates_ranked": capped,
            "node_updates_ranked": node_recos,
        },
        "version": 3,  # Bayesian + guardrails
    }

    # Diffs (exposed for debugging)
    diffs_block = {
        "starting_persona_delta": start_deltas,                   # Bayesian-shrunk deltas
        "pp_delta": {f"{a}->{b}": v for (a, b), v in pp_deltas.items()},
        "prod_delta": {f"{a}->{b}": v for (a, b), v in prod_delta.items()},
    }

    return {
        "diffs": diffs_block,
        "summary": summary,
        "neighborhoods": neighborhoods,  # upstream/handoff/downstream
    }
