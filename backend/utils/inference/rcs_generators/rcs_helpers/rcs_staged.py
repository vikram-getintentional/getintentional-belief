# ============================
# File: backend/utils/rcs_v2/rcs_staged.py
# ============================
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx

# graph core
from backend.utils.graph_base.network_graph import (
    get_nodes_list_ids,
    get_product_id_from_subgraph,
)

# PPR engine & overlay helpers (use your graph_base paths)
from backend.utils.inference.rcs_generators.rcs_computations.ppr_builder import build_ppr_engine_from_graph
from backend.utils.inference.rcs_generators.rcs_helpers.overlay_helpers import (
    make_edge_scales_for_concern,
    make_edge_scales_for_persona,
    make_edge_scales_for_attribute,  # available if you need attribute overlays later
)

# Shared math/graph utilities
from backend.utils.inference.rcs_generators.graph_algorithms import (
    # types & pruning
    PRUNE_TYPES,
    compute_activation_from_uplift,
    condition_graph_by_attributes,
    prune_types,
    norm_type,
    # boosts (collect from FULL graph, apply on PRUNED graph)
    collect_zmot_trigger_boosts,
    collect_attribute_trigger_boosts,
    apply_trigger_boosts_inplace,
    # concern staging
    infer_stage_for_concern,
    # persona involvement / activation / care
    persona_involvement_from_jobs,
    compute_activation_care_vector,
    # coalitions & sequences orchestration
    find_top_coalitions,
    build_causal_flows,
    design_sequences_and_campaigns,
)


# ----------------------------
# Local helpers
# ----------------------------
def _clip(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def prune_noise_and_reverse(G: nx.DiGraph) -> nx.DiGraph:
    """
    Convenience helper: prune noisy node types and return a reversed copy.
    Not used internally by rcs_prepare (which needs to apply boosts on the pruned
    forward graph first), but kept exported for any external callers.
    """
    H = prune_types(G)
    return H.reverse(copy=True)

def _noisy_or_update(curr: float, evid: float) -> float:
        # combine multiple evidence hits on the same node
        curr = float(curr or 0.0)
        evid = float(max(0.0, min(1.0, evid)))
        return 1.0 - (1.0 - curr) * (1.0 - evid)

#----------------------------
# Helpers for Attribute Dimension Scoring 
#----------------------------
def _node_type(G: nx.DiGraph, nid: str) -> str:
    return (G.nodes[nid].get("type") or "").lower()

def _get_product_id(G: nx.DiGraph) -> Optional[str]:
    # reuse your existing get_product_id_from_subgraph if available
    try:
        return get_product_id_from_subgraph(G)  # type: ignore
    except Exception:
        # fallback: first node tagged as product
        for n in G.nodes:
            if _node_type(G, n) == "product":
                return n
        return None

def _infer_attr_dimension(G: nx.DiGraph, attr_id: str) -> Optional[str]:
    """
    Try to infer the 'dimension' of an attribute_value node.
    Priority:
      1) node['dimension'] if present
      2) parent 'attribute' node name/key via incoming edge (attribute -> attribute_value)
      3) fallback: None
    """
    ndata = G.nodes[attr_id]
    if "dimension" in ndata and ndata["dimension"]:
        return str(ndata["dimension"]).lower()

    # Try to find parent attribute node
    for u, v in G.in_edges(attr_id):
        if _node_type(G, u) in {"attribute", "attribute_value_group", "attribute_key"}:
            # Use its 'name' or 'id' as dimension surrogate
            dim = G.nodes[u].get("name") or G.nodes[u].get("dimension") or u
            return str(dim).lower()

    # Some graphs store dimension on an 'attribute' edge label
    # If you have that, add a check here.

    return None

def _relevant_for_attr_flow(t: str) -> bool:
    """
    Nodes we keep for dimension PPR runs.
    We want the canonical “product → ... → pain_trigger → attribute_value” corridor,
    plus job/persona if they exist in between. Adjust if your schema differs.
    """
    return t in {
        "product", "capability", "pain", "job",
        "pain_trigger", "attribute", "attribute_value",
        "zmot_event", "observable_moment", "keyword"
    }

def _build_attr_corridor_graph_reversed(G: nx.DiGraph) -> nx.DiGraph:
    """
    Keep only the corridor nodes (so noise doesn’t pollute the signal),
    then reverse so PageRank seeded at attributes can flow *to* the product.
    """
    keep: Set[str] = {n for n in G.nodes if _relevant_for_attr_flow(_node_type(G, n))}
    H = G.subgraph(keep).copy()
    H_rev = H.reverse(copy=True)

    # ensure product has a self-loop with weight 1.0 (helps PR stability)
    product_id = _get_product_id(G)
    if product_id and product_id in H_rev and not H_rev.has_edge(product_id, product_id):
        H_rev.add_edge(product_id, product_id, likelihood=1.0)
    return H_rev

# --- main learner ----------------------------------------------------------

def compute_dimension_weights_from_graph(
    G_full: nx.DiGraph,
    *,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    Learn per-dimension weights using graph structure only:

    For each dimension D:
      1) Gather all attribute_value nodes with dimension == D
      2) Seed PageRank on the *reversed* corridor graph at those nodes (uniform)
      3) Take PR(product) as score(D)
    Normalize scores to sum to 1. If all zero, return uniform over seen dims.
    """
    product_id = _get_product_id(G_full)
    if not product_id:
        return {}

    # Build reversed corridor graph (keeps relevant nodes & reverses edges)
    G_rev = _build_attr_corridor_graph_reversed(G_full)

    # Group attribute_value nodes by dimension
    dims_to_attrs: Dict[str, List[str]] = defaultdict(list)
    for n in G_full.nodes:
        if _node_type(G_full, n) == "attribute_value":
            dim = _infer_attr_dimension(G_full, n)
            if dim:
                if n in G_rev:  # must exist after corridor filter
                    dims_to_attrs[dim].append(n)

    if not dims_to_attrs:
        return {}

    # For each dimension, run PR seeded at its attribute_value nodes
    dim_scores: Dict[str, float] = {}
    for dim, attrs in dims_to_attrs.items():
        if not attrs:
            continue

        # personalization: uniform across this dimension’s attribute values
        Z = float(len(attrs))
        personalization = {nid: (1.0 / Z) for nid in attrs if nid in G_rev}
        if not personalization:
            continue

        # Use edge weights if available; default to 'likelihood'
        try:
            pr = nx.pagerank(
                G_rev,
                alpha=alpha,
                personalization=personalization,
                dangling=personalization,
                weight="likelihood",
                max_iter=max_iter,
                tol=tol,
            )
        except nx.PowerIterationFailedConvergence:
            # Fallback unweighted if weighted PR fails on odd topologies
            pr = nx.pagerank(
                G_rev,
                alpha=alpha,
                personalization=personalization,
                dangling=personalization,
                weight=None,
                max_iter=max_iter,
                tol=tol,
            )

        dim_scores[dim] = float(pr.get(product_id, 0.0))

    # Normalize to weights
    S = sum(dim_scores.values())
    if S <= 0.0:
        # fallback to uniform over dimensions present
        k = float(len(dim_scores)) or 1.0
        return {d: 1.0 / k for d in dim_scores.keys()}

    return {d: (s / S) for d, s in dim_scores.items()}

def compute_dimension_evidence_from_selected_attrs(
    G_full: nx.DiGraph,
    selected_attribute_values: List[str],
) -> Dict[str, float]:
    """
    For each engaged attribute_value, look at incoming edges from pain_triggers:
      evidence(attr) = noisy-or over edges (1 - Π (1 - likelihood(trigger->attr)))
    Then aggregate per dimension with noisy-or again.

    Returns dim -> evidence in [0,1].
    """
    by_dim: Dict[str, float] = defaultdict(float)  # will store 1 - product(1-p)
    # work on FULL graph (edges from triggers to attributes may be pruned otherwise)
    for a in selected_attribute_values:
        if a not in G_full:
            continue
        dim = _infer_attr_dimension(G_full, a)
        if not dim:
            continue

        # Noisy-or over incoming trigger->attr edges
        one_minus_prod = 1.0
        for u, v in G_full.in_edges(a):
            if _node_type(G_full, u) == "pain_trigger":
                eL = float(G_full.edges[u, v].get("likelihood", 0.0))
                eL = max(0.0, min(1.0, eL))
                one_minus_prod *= (1.0 - eL)
        attr_evidence = 1.0 - one_minus_prod

        # Aggregate per dimension via noisy-or
        cur = by_dim[dim]
        by_dim[dim] = 1.0 - (1.0 - cur) * (1.0 - attr_evidence)

    return dict(by_dim)

# ----------------------------
# Context object
# ----------------------------
class RCSPPRContext:
    """
    Holds everything needed by staged RCS routines after rcs_prepare().
    """
    def __init__(
        self,
        *,
        G_pruned: nx.DiGraph,
        G_rev: nx.DiGraph,
        engine,
        conv_id: str,
        seeds: List[str],
        base_win: float,
        baseline_dbg: Dict
    ):
        self.G_pruned = G_pruned            # pruned forward graph (working orientation)
        self.G_rev = G_rev                  # reversed graph used by PPR
        self.engine = engine                # PPREngine
        self.conv_id = conv_id
        self.seeds = seeds
        self.base_win = float(base_win)     # baseline win prob (after boosts)
        self.baseline_dbg = baseline_dbg    # misc baseline debug (occurred, boosts)
        self.seeds_key = "|".join(sorted(seeds or [])) or "uniform"


# ----------------------------
# Stage 0: prepare
# ----------------------------
def rcs_prepare(
    G: nx.DiGraph,
    engaged_nodes: Optional[List[Dict]] = None
) -> Tuple[RCSPPRContext, Dict]:
    product_id = get_product_id_from_subgraph(G)
    if not product_id:
        raise ValueError("no_product_in_graph")

    conv_id = product_id
    print("RCS Prepare received engaged nodes:", engaged_nodes)

    # --- (A) Attribute evidence ---
    selected_attributes = []
    if engaged_nodes:
        for n in engaged_nodes:
            nid = n.get("id") if isinstance(n, dict) else n
            if not nid or nid not in G:
                continue
            if norm_type(G, nid) == "attribute_value":
                selected_attributes.append(nid)

    G_cond = condition_graph_by_attributes(G, selected_attributes)
    dim_weights = compute_dimension_weights_from_graph(G_cond)
    print("[dim-weights]", dim_weights)

    # --- (B) prune noisy ---
    G_pruned = prune_types(G_cond, PRUNE_TYPES)
    if G_pruned.number_of_nodes() == 0:
        return RCSPPRContext(
            G_pruned=G_pruned, G_rev=nx.DiGraph(),
            engine=build_ppr_engine_from_graph(nx.DiGraph()),
            conv_id=conv_id, seeds=[], base_win=0.0,
            baseline_dbg={"occurred": [], "weighted_pr0": {}, "unconditioned_pr0": {}}
        ), {"baseline": {"baseline_win_conditioned": 0.0, "occurred": []}, "cheap": {"top_personas_quick": []}}

    # --- (C) boosts + evidence ---
    trig_boosts: List[Tuple[str, float]] = []
    occurred_ids: List[str] = []

    # knobs so you can tune later
    PERSONA_SEED_SCALE = 0.25   # persona exists ≠ pain is active; keep it weaker than pains/jobs
    JOB_SEED_SCALE     = 1.00   # job engagement is strong evidence
    PAIN_SEED_SCALE    = 1.00   # pain engagement is strong evidence

    
    seed_weights: Dict[str, float] = {}
    if engaged_nodes:
        for n in engaged_nodes:
            nid = n.get("id") if isinstance(n, dict) else n
            if not nid:
                continue

            # normalize an occurrence strength if provided; default 1.0
            occ = float(n.get("occurrence", 1.0)) if isinstance(n, dict) else 1.0
            occ = max(0.0, min(1.0, occ))

            if nid in G_pruned:
                occurred_ids.append(nid)

                t = norm_type(G_pruned, nid)

                if t in {"attribute", "attribute_value"}:
                    # attribute only boosts PTs; do not set node likelihood (it’s categorical)
                    trig_boosts.extend(collect_attribute_trigger_boosts(G, nid))

                elif t == "zmot_event":
                    # zmot boosts the same PT links too
                    trig_boosts.extend(collect_zmot_trigger_boosts(G, nid))

                elif t == "pain":
                    # write evidence onto node likelihood via noisy-or
                    G_pruned.nodes[nid]["likelihood"] = _noisy_or_update(
                        G_pruned.nodes[nid].get("likelihood", 0.0), occ
                    )
                    # seed PR strongly
                    seed_weights[nid] = max(seed_weights.get(nid, 0.0), occ * PAIN_SEED_SCALE)

                elif t == "job":
                    # write evidence onto job node
                    G_pruned.nodes[nid]["likelihood"] = _noisy_or_update(
                        G_pruned.nodes[nid].get("likelihood", 0.0), occ
                    )
                    # seed PR strongly (jobs connect to pains/personas in the reverse graph)
                    seed_weights[nid] = max(seed_weights.get(nid, 0.0), occ * JOB_SEED_SCALE)

                elif t == "persona":
                    # persona exists in the org; set its node likelihood,
                    # but use a *scaled* seed so it doesn’t dominate teleport mass
                    G_pruned.nodes[nid]["likelihood"] = _noisy_or_update(
                        G_pruned.nodes[nid].get("likelihood", 0.0), occ
                    )
                    seed_weights[nid] = max(seed_weights.get(nid, 0.0), occ * PERSONA_SEED_SCALE)

                else:
                    # other node types: mark occurrence (for debugging/UI), but don’t seed
                    pass

    print("Trig Boosts now:", trig_boosts)

    if trig_boosts:
        apply_trigger_boosts_inplace(G_pruned, trig_boosts, boost_edges=True)

    # --- (D) Reverse graph for PPR ---
    G_rev = G_pruned.reverse(copy=True)
    if conv_id in G_rev:
        G_rev.add_edge(conv_id, conv_id, likelihood=1.0)

    # stitch pain->job edges
    for u, v, d in G_pruned.edges(data=True):
        if norm_type(G_pruned, u) == "pain" and norm_type(G_pruned, v) == "job":
            if not G_rev.has_edge(u, v):
                G_rev.add_edge(u, v, likelihood=float(d.get("likelihood", 0.0)))

    # --- (E) seeds ---
    for trig, b in (trig_boosts or []):
        if trig in G_pruned:
            seed_weights[trig] = 1.0 - (1.0 - seed_weights.get(trig, 0.0)) * (1.0 - float(b))
    if not seed_weights and occurred_ids:
        for nid in occurred_ids:
            if norm_type(G_pruned, nid) not in {"zmot_event", "observable_moment", "keyword"}:
                seed_weights[nid] = 1.0

    # --- (F) PR baseline ---
    weighted_pr0: Dict[str, float] = {}
    unconditioned_pr0: Dict[str, float] = {}
    if seed_weights:
        Z = sum(seed_weights.values()) or 1.0
        personalization = {n: (seed_weights.get(n, 0.0) / Z) for n in G_rev.nodes()}
        weighted_pr0 = nx.pagerank(G_rev, alpha=0.85, personalization=personalization,
                                   dangling=personalization, weight="likelihood")
        unconditioned_pr0 = nx.pagerank(G_rev, alpha=0.85, weight="likelihood")
        base_win = float(weighted_pr0.get(conv_id, 0.0))
    else:
        base_win = 0.0
        weighted_pr0, unconditioned_pr0 = {}, {}

    seeds = list(seed_weights.keys())
    engine = build_ppr_engine_from_graph(G_rev)

    baseline_block = {
        "baseline_win_conditioned": base_win,
        "occurred": occurred_ids,
        "zmot_trigger_boosts": trig_boosts,
        "weighted_pr0": weighted_pr0,
        "unconditioned_pr0": unconditioned_pr0,
        "seed_weights": seed_weights,
        "dimension_weights_learned": dim_weights,
    }
    cheap_block = {"top_personas_quick": []}

    ctx = RCSPPRContext(
        G_pruned=G_pruned, G_rev=G_rev,
        engine=engine, conv_id=conv_id,
        seeds=seeds, base_win=base_win,
        baseline_dbg=baseline_block,
    )
    
    return ctx, {"baseline": baseline_block, "cheap": cheap_block}




# ----------------------------
# Stage 1: marginal lifts
# ----------------------------
def rcs_marginal_lifts(ctx: RCSPPRContext, persona_ids: List[str], boost: float = 2.0) -> List[Dict]:
    engine = ctx.engine
    conv_idx = engine.node_index.get(ctx.conv_id)
    if conv_idx is None:
        return [{"id": pid, "marginal_lift": 0.0} for pid in persona_ids]

    out = []
    for pid in persona_ids:
        ov = make_edge_scales_for_persona(ctx.G_pruned, ctx.G_rev, engine, pid, boost)
        pr_u = engine.pr_raw(ctx.seeds, boost_key=f"pers:{pid}", edge_scales=ov, warm_from=("", ctx.seeds_key))
        p1 = float(pr_u[conv_idx])
        out.append({"id": pid, "marginal_lift": max(0.0, p1 - ctx.base_win)})

    out.sort(key=lambda r: r["marginal_lift"], reverse=True)
    return out


# ----------------------------
# Stage 1b: persona activations/involvement/care
# ----------------------------
def rcs_persona_activations(ctx: RCSPPRContext) -> List[Dict]:
    """
    Return rows with id, involvement, activation, care for all personas.
    Uses centralized helpers on the pruned forward graph.
    """
    pr_post = ctx.baseline_dbg.get("weighted_pr0") or {}
    pr_prior = ctx.baseline_dbg.get("unconditioned_pr0") or {}

    persona_ids = get_nodes_list_ids(ctx.G_pruned, "persona", {})

    involvement = {
        pid: persona_involvement_from_jobs(ctx.G_pruned, pr_post, pid)
        for pid in persona_ids
    }

    # Instead of old compute_activation_care_vector, use uplift
    activation = compute_activation_from_uplift(
        ctx.G_pruned, pr_prior, pr_post, focus_types=("persona",), mode="ratio_bounded"
    )


    rows = []
    for pid in persona_ids:
        rows.append({
            "id": pid,
            "involvement": float(involvement.get(pid, 0.0)),
            "activation": float(activation.get(pid, 0.0)),
            "care": float(involvement.get(pid, 0.0)) * float(activation.get(pid, 0.0)) ** 0.5,  # or keep your care calc
        })

    return rows


# ----------------------------
# Stage 2: pair effects (lift interactions)
# ----------------------------
def rcs_pair_effects(ctx: RCSPPRContext, persona_pool: List[str], boost: float = 2.0) -> Dict[Tuple[str, str], Dict[str, float]]:
    engine = ctx.engine
    conv_idx = engine.node_index.get(ctx.conv_id)
    if conv_idx is None:
        return {}

    # Baseline ml for each v
    ml0 = {}
    for v in persona_pool:
        ov_v = make_edge_scales_for_persona(ctx.G_pruned, ctx.G_rev, engine, v, boost)
        pr_v = engine.pr_raw(ctx.seeds, boost_key=f"pers:{v}", edge_scales=ov_v, warm_from=("", ctx.seeds_key))
        p_v = float(pr_v[conv_idx])
        ml0[v] = max(0.0, p_v - ctx.base_win)

    # Cache PR with u boosted
    pr_u_map = {}
    for u in persona_pool:
        ov_u = make_edge_scales_for_persona(ctx.G_pruned, ctx.G_rev, engine, u, boost)
        pr_u = engine.pr_raw(ctx.seeds, boost_key=f"pers:{u}", edge_scales=ov_u, warm_from=("", ctx.seeds_key))
        pr_u_map[u] = pr_u

    results: Dict[Tuple[str, str], Dict[str, float]] = {}
    for u in persona_pool:
        pr_u = pr_u_map[u]
        p_u = float(pr_u[conv_idx])
        for v in persona_pool:
            if v == u:
                continue
            ov_v = make_edge_scales_for_persona(ctx.G_pruned, ctx.G_rev, engine, v, boost)
            pr_uv = engine.pr_raw(ctx.seeds, boost_key=f"pers:{u}+{v}", edge_scales=ov_v, warm_from=(f"pers:{u}", ctx.seeds_key))
            p_uv = float(pr_uv[conv_idx])
            ml_u_v = max(0.0, p_uv - p_u)
            results[(u, v)] = {
                "lift_gain_for_v": max(0.0, ml_u_v - ml0[v])
            }
    return results


# ----------------------------
# Stage 3: concern potentials (CF option C)
# ----------------------------
def rcs_concern_potentials(
    ctx: RCSPPRContext,
    persona_to_concerns: Dict[str, List[str]],
    rel_map: Dict[Tuple[str, str], float],
    I_map: Dict[str, float],
    Care_map: Dict[str, float],
    *,
    boost: float = 2.0,
    top_k: int = 5,
) -> Dict[str, List[Dict]]:
    engine = ctx.engine
    conv_idx = engine.node_index.get(ctx.conv_id)
    if conv_idx is None:
        return {pid: [] for pid in persona_to_concerns.keys()}

    p0 = ctx.base_win
    headroom = max(0.0, 1.0 - p0)

    out: Dict[str, List[Dict]] = {}
    for pid, concerns in persona_to_concerns.items():
        rows = []
        for cid in concerns:
            rel_ic = float(rel_map.get((pid, cid), 0.0))
            ov_c = make_edge_scales_for_concern(ctx.G_pruned, ctx.G_rev, engine, cid, ctx.conv_id, boost)
            pr_c = engine.pr_raw(ctx.seeds, boost_key=f"conc:{cid}", edge_scales=ov_c, warm_from=("", ctx.seeds_key))
            p1 = float(pr_c[conv_idx])
            potential_lift = max(0.0, p1 - p0)

            q_resolve = _clip(I_map.get(pid, 0.0) * Care_map.get(pid, 0.0) * rel_ic)
            exp_gain = min(headroom, potential_lift * q_resolve)

            rows.append({
                "concern_id": cid,
                "potential_lift": potential_lift,
                "rel": rel_ic,
                "score": exp_gain,
                "stage": infer_stage_for_concern(ctx.G_pruned, pid, cid),
            })
        rows.sort(key=lambda r: r["score"], reverse=True)
        out[pid] = rows[:top_k]
    return out


# ----------------------------
# Stage 4: sequences and coalitions (wrappers)
# ----------------------------
def rcs_persona_sequences(
    ctx: RCSPPRContext,
    candidate_personas: List[str],
    concerns_by_persona: Dict[str, List[Dict]],
    assets_by_persona: Dict[str, Dict[str, List[Dict]]],
    pair_effects: Optional[Dict[Tuple[str, str], Dict[str, float]]] = None,
    *,
    boost_factor: float = 2.0,
    alpha_act=(0.0, 1.0, 1.0, 1.0),
    beta_felt=(0.0, 1.0, 1.0, 1.0),
    max_len: int = 5,
    beam_width: int = 5,
    max_coalition_size: int = 3,
    min_synergy: float = 1e-6,
    max_sequences: int = 5
) -> List[Dict]:
    return design_sequences_and_campaigns(
        ctx.G_pruned,
        ctx.seeds,
        ctx.conv_id,
        candidate_personas=candidate_personas,
        concerns_by_persona=concerns_by_persona,
        assets_by_persona=assets_by_persona,
        pair_effects=pair_effects,
        boost_factor=boost_factor,
        alpha_act=alpha_act,
        beta_felt=beta_felt,
        max_len=max_len,
        beam_width=beam_width,
        max_coalition_size=max_coalition_size,
        min_synergy=min_synergy,
        max_sequences=max_sequences
    )


def rcs_persona_coalitions(
    ctx: RCSPPRContext,
    persona_pool: List[str],
    *,
    boost_factor: float = 2.0,
    top_k: int = 8,
    max_coalition_size: int = 3
) -> List[Dict]:
    """
    Thin wrapper over the coalition finder from graph_algorithms.
    """
    return find_top_coalitions(
        ctx.G_pruned,
        ctx.seeds,
        ctx.conv_id,
        persona_pool,
        boost_factor=boost_factor,
        max_coalition_size=max_coalition_size,
        top_k=top_k
    )


# (Optional) If callers need causal flows, you can also export a thin wrapper:
def rcs_causal_flows(
    ctx: RCSPPRContext,
    persona_pool: List[str],
    pair_effects: Dict[Tuple[str, str], Dict[str, float]],
    concerns_by_persona: Dict[str, List[Dict]],
    *,
    top_pairs: int = 8,
    top_chains: int = 5,
    boost_factor: float = 2.0
) -> Dict[str, List[Dict]]:
    return build_causal_flows(
        ctx.G_pruned,
        ctx.seeds,
        ctx.conv_id,
        persona_pool,
        pair_effects,
        concerns_by_persona,
        top_pairs=top_pairs,
        top_chains=top_chains,
        boost_factor=boost_factor
    )
