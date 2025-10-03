# ============================
# File: backend/utils/inference/rcs_generators/graph_algorithms.py
# ============================
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Set, Tuple
import math
import heapq
import numpy as np
import networkx as nx
from collections import deque
from itertools import combinations

from backend.utils.graph_base.network_graph import get_edge_attribute, get_node_by_id, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type

# ----------------------------
# Math helpers
# ----------------------------

def clip01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def noisy_or(vals: List[float]) -> float:
    if not vals:
        return 0.0
    prod = 1.0
    for v in vals:
        prod *= (1.0 - clip01(v))
    return clip01(1.0 - prod)

def sigmoid(z: float) -> float:
    # numerically stable
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)
def _clip01(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))

def _normalize_by_max(scores: dict) -> dict:
    if not scores:
        return {}
    m = max(scores.values()) if scores else 0.0
    if m <= 0.0:
        return {k: 0.0 for k in scores}
    return {k: (v / m) for k, v in scores.items()}

def compute_involvement_from_pr(
    G: nx.DiGraph,
    pr_posterior: dict,
    *,
    focus_types=("persona", "job", "pain")
) -> dict:
    """
    Involvement = normalized posterior PR mass on each node (per focus types).
    Uses *weighted* PR so attributes actually move this number.
    """
    raw = {}
    for n in G.nodes():
        if norm_type(G, n) in focus_types:
            raw[n] = float(pr_posterior.get(n, 0.0))
    return _normalize_by_max(raw)

def compute_activation_from_uplift(
    G: nx.DiGraph,
    pr_prior: dict,
    pr_posterior: dict,
    *,
    focus_types=("persona", "job", "pain"),
    mode="ratio_bounded"
) -> dict:
    """
    Activation = uplift from prior → posterior.
    - 'diff':       max(0, post - prior)
    - 'ratio':      post / max(prior, eps)              (can be >1, unbounded)
    - 'ratio_bounded': 1 - exp(-(post - prior)/max(prior, eps))
      (smoothly maps uplift to 0..1; sensitive when prior is small)
    """
    out = {}
    eps = 1e-12
    for n in G.nodes():
        if norm_type(G, n) not in focus_types:
            continue
        pre = float(pr_prior.get(n, 0.0))
        
        post = float(pr_posterior.get(n, 0.0))
        
        if mode == "diff":
            val = max(0.0, post - pre)
        elif mode == "ratio":
            val = post / max(pre, eps)
        else:  # ratio_bounded
            uplift = max(0.0, post - pre)
            val = 1.0 - math.exp(-uplift / max(pre, eps))
        out[n] = _clip01(val)
    return out


# ----------------------------
# Graph helpers
# ----------------------------

PRUNE_TYPES = {
    "zmot_event", 
    "observable_moment", 
    "keyword",
    
}

def norm_type(G: nx.DiGraph, node_id: str) -> str:
    if node_id not in G:
        return ""
    return (G.nodes[node_id].get("node_type") or G.nodes[node_id].get("type") or "").strip().lower()

def remove_cycles_by_min_edge(G: nx.DiGraph) -> nx.DiGraph:
    H = G.copy()
    try:
        while not nx.is_directed_acyclic_graph(H):
            cyc = next(nx.simple_cycles(H), None)
            if not cyc:
                break
            edges = list(zip(cyc, cyc[1:] + [cyc[0]]))
            edges.sort(key=lambda e: H.edges[e].get("likelihood", 0.0))
            H.remove_edge(*edges[0])
    except Exception:
        pass
    return H

def nodes_reaching_target(G: nx.DiGraph, target: str) -> Set[str]:
    if target not in G:
        return set()
    R = set()
    Grev = G.reverse(copy=False)
    dq = deque([target])
    while dq:
        x = dq.popleft()
        R.add(x)
        for y in Grev.successors(x):
            if y not in R:
                dq.append(y)
    return R

def prune_types(G: nx.DiGraph, types_to_remove: Set[str] = PRUNE_TYPES) -> nx.DiGraph:
    H = G.copy()
    to_drop = [n for n in H.nodes if norm_type(H, n) in types_to_remove]
    H.remove_nodes_from(to_drop)
    return H

def ensure_conv_selfloop_on_reversed(G_rev: nx.DiGraph, conv_id: str) -> None:
    if conv_id in G_rev and not G_rev.has_edge(conv_id, conv_id):
        G_rev.add_edge(conv_id, conv_id, likelihood=1.0)

def set_occurred_nodes_inplace(G: nx.DiGraph, occurred_nodes: List[Dict]) -> None:
    for oc in occurred_nodes or []:
        nid = oc.get("id")
        occ = clip01(oc.get("occurrence", 1.0))
        if nid in G:
            G.nodes[nid]["occurrence"] = occ
            G.nodes[nid]["cumulative_likelihood"] = occ
def edge_lik(G, u: str, v: str) -> float:
    # fall back to "relevance" if "likelihood" missing
    if not G.has_edge(u, v):
        return 0.0
    e = G[u][v]
    return clip01(float(e.get("likelihood", e.get("relevance", 0.0)) or 0.0))

def job_lik(G, j: str) -> float:
    return clip01(float(G.nodes[j].get("likelihood", 0.0)))


# ----------------------------
# Condition graph to only include nodes relevant to attributes
# ----------------------------

def condition_graph_by_attributes(
    product_graph: nx.DiGraph,
    attribute_node_ids: list[str]
) -> nx.DiGraph:
    """
    Build a conditioned subgraph that only contains:
      • Pain-trigger nodes connected to the selected attribute values
      • All forward paths from those pain-triggers to the product node
      • Plus the personas that are attached to any jobs on those paths
    """
    if not attribute_node_ids:
        return product_graph.copy()

    product_id = get_product_id_from_subgraph(product_graph)
    if not product_id or product_id not in product_graph:
        return product_graph.copy()

    keep: Set[str] = set()
    triggers: Set[str] = set()

    # 1) find pain_triggers connected to each selected attribute
    for attr_id in attribute_node_ids:
        if attr_id not in product_graph:
            continue
        # edges: pain_trigger -(prevalent_in)-> attribute_value
        for trig in get_source_nodes_by_target_and_type(product_graph, attr_id, "prevalent_in") or []:
            if product_graph.nodes[trig].get("node_type") == "pain_trigger":
                triggers.add(trig)

    if not triggers:
        # still include product + selected attributes, so UI has something to render
        H = product_graph.subgraph({*attribute_node_ids, product_id}).copy()
        return H

    # 2) keep all forward paths from each trigger to the product
    for trig in triggers:
        if trig not in product_graph:
            continue
        try:
            # NOTE: direction is trigger -> product (forward), not product -> trigger
            for path in nx.all_simple_paths(product_graph, source=product_id, target=trig):
                keep.update(path)
        except nx.NetworkXNoPath:
            # if a trigger cannot reach product, we still keep the trigger node
            keep.add(trig)

    # Always keep the product and selected attributes
    keep.update(attribute_node_ids)
    keep.add(product_id)

    # 3) add personas that hang off any jobs on the kept paths
    jobs_on_paths = [n for n in keep if product_graph.nodes[n].get("node_type") == "job"]
    for j in jobs_on_paths:
        # job -> persona via owned_by / performed_by
        for persona in (get_target_nodes_by_source_and_type(product_graph, j, "performed_by") or []):
            keep.add(persona)

    return product_graph.subgraph(keep).copy()


# ----------------------------
# Trigger boost collectors
# ----------------------------

def collect_zmot_trigger_boosts(G_full: nx.DiGraph, zmot_id: Optional[str]) -> List[Tuple[str, float]]:
    if not zmot_id or zmot_id not in G_full:
        return []
    out = []
    for u in G_full.predecessors(zmot_id):
        if norm_type(G_full, u) == "pain_trigger":
            data = G_full.edges[u, zmot_id]
            b = data.get("boost", data.get("likelihood", 0.0)) or 0.0
            out.append((u, clip01(b)))
    for v in G_full.successors(zmot_id):
        if norm_type(G_full, v) == "pain_trigger":
            data = G_full.edges[zmot_id, v]
            b = data.get("boost", data.get("likelihood", 0.0)) or 0.0
            out.append((v, clip01(b)))
    agg: Dict[str, float] = {}
    for pid, b in out:
        agg[pid] = 1.0 - (1.0 - agg.get(pid, 0.0)) * (1.0 - b)
    return list(agg.items())

def collect_attribute_trigger_boosts(G_full: nx.DiGraph, attr_id: Optional[str]) -> List[Tuple[str, float]]:
    
    if not attr_id or attr_id not in G_full:
        return []
    out = []
    pain_trigger_ids = get_source_nodes_by_target_and_type(G_full, attr_id, "prevalent_in")
    if pain_trigger_ids is not None:
        for pt in pain_trigger_ids:
            pt_node = get_node_by_id(G_full, pt)
            if not pt_node:
                continue
            boost = 0.0
            likelihood = 0.0
            boost = get_edge_attribute(G_full, pt, attr_id, "boost")
            likelihood = get_edge_attribute(G_full, pt, attr_id, "likelihood")
            out.append((pt, clip01(boost or likelihood or 0.0)))
            
    agg: Dict[str, float] = {}
    for pid, b in out:
        agg[pid] = 1.0 - (1.0 - agg.get(pid, 0.0)) * (1.0 - b)
    return list(agg.items())

def apply_trigger_boosts_inplace(
    G_pruned: nx.DiGraph,
    trig_boosts: List[Tuple[str, float]],
    *,
    boost_edges: bool = True,
    propagate_node_likelihood: bool = True
) -> Dict[str, int]:
    """
    Apply trigger boosts (from engaged attributes/ZMOTs) directly to the pruned graph.
    - Uses noisy-OR aggregation if multiple boosts apply to same trigger.
    - Optionally propagates boosted likelihood into outgoing edges.
    """
    dbg = {"triggers_in_pruned": 0, "edges_boosted": 0}

    # Group boosts per trigger
    boost_map: Dict[str, List[float]] = {}
    for trig, b in trig_boosts:
        boost_map.setdefault(trig, []).append(float(b))

    for trig, boosts in boost_map.items():
        if trig not in G_pruned:
            print("⚠️ Found a trigger not in pruned graph:", trig)
            continue
        dbg["triggers_in_pruned"] += 1

        # Aggregate boosts using noisy-OR
        agg_boost = 1.0
        for b in boosts:
            agg_boost *= (1.0 - clip01(b))
        agg_boost = 1.0 - agg_boost

        # Current likelihood on node
        L0 = clip01(G_pruned.nodes[trig].get("likelihood", 0.0))
        L1 = 1.0 - (1.0 - L0) * (1.0 - agg_boost)

        if L1 > L0:
            G_pruned.nodes[trig]["likelihood"] = L1
            if propagate_node_likelihood:
                G_pruned.nodes[trig]["cumulative_likelihood"] = max(
                    clip01(G_pruned.nodes[trig].get("cumulative_likelihood", 0.0)), L1
                )

        # Propagate boost to outgoing edges
        if boost_edges:
            for _, v in G_pruned.out_edges(trig):
                ed = G_pruned.edges[trig, v]
                e0 = clip01(ed.get("likelihood", 0.0))
                e1 = 1.0 - (1.0 - e0) * (1.0 - agg_boost)
                if e1 > e0:
                    ed["likelihood"] = e1
                    dbg["edges_boosted"] += 1
                    

    return dbg


# ----------------------------
# Depth annotators
# ----------------------------

def set_temporal_depths(G: nx.DiGraph) -> None:
    try:
        prod_nodes = [n for n, d in G.nodes(data=True) if norm_type(G, n) == "product"]
        if not prod_nodes:
            return
        prod = prod_nodes[0]
        spl = nx.single_source_shortest_path_length(G, prod)
        for n, d in spl.items():
            G.nodes[n]["temporal_depth"] = d
        G.graph["max_temporal_depth"] = max(spl.values()) if spl else 0
    except Exception:
        pass

def set_causal_depths(G: nx.DiGraph) -> None:
    try:
        roots = [n for n in G.nodes if norm_type(G, n) in {"attribute", "attribute_value"}]
        Grev = G.reverse(copy=False)
        for r in roots:
            nx.set_node_attributes(G, {r: 0}, "causal_depth")
            nx.set_node_attributes(Grev, {r: 0}, "causal_depth")
    except Exception:
        pass

# ----------------------------
# PPR & baseline
# ----------------------------

def _prep_graph_for_ppr(G: nx.DiGraph, conv_id: str) -> nx.DiGraph:
    H = G.copy()
    for _, _, d in H.edges(data=True):
        if "likelihood" in d:
            try:
                d["likelihood"] = max(0.0, float(d["likelihood"]))
            except Exception:
                d["likelihood"] = 0.0
    H = H.reverse(copy=True)
    ensure_conv_selfloop_on_reversed(H, conv_id)
    return H

def _valid_personalization(G: nx.DiGraph, seeds: List[str]) -> Dict[str, float]:
    pers = {n: 0.0 for n in G.nodes}
    valid = [s for s in (seeds or []) if s in G]
    if valid:
        w = 1.0 / len(valid)
        for s in valid:
            pers[s] = w
        return pers
    u = 1.0 / max(1, G.number_of_nodes())
    return {n: u for n in pers}

def ppr(G: nx.DiGraph, seeds: List[str], conv_id: str, alpha: float = 0.85) -> Dict[str, float]:
    H = _prep_graph_for_ppr(G, conv_id)
    pers = _valid_personalization(H, seeds)
    dang = {n: v / (sum(pers.values()) or 1.0) for n, v in pers.items()}
    pr = nx.pagerank(H, alpha=alpha, personalization=pers, dangling=dang,
                     weight="likelihood", max_iter=200, tol=1e-8)
    return {k: clip01(v) for k, v in pr.items()}

def baseline_conversion_prob(G: nx.DiGraph, seeds: List[str], conv_id: str) -> float:
    pr = ppr(G, seeds, conv_id)
    return pr.get(conv_id, 0.0)

# ----------------------------
# Persona involvement, activation, care
# ----------------------------

def jobs_of_persona(G: nx.DiGraph, persona_id: str) -> List[str]:
    jobs = []
    jobs = get_source_nodes_by_target_and_type(G, persona_id, "performed_by")
    return jobs

def persona_involvement_from_jobs(G: nx.DiGraph, pr: Dict[str, float], persona_id: str) -> float:
    jobs = jobs_of_persona(G, persona_id)
    vals = []
    for j in jobs:
        w = 0.0
        edge_rel = get_edge_attribute(G, j, persona_id, "relevance")
        edge_lik = get_edge_attribute(G, j, persona_id, "likelihood")
        if edge_rel is not None:
            w = edge_rel
        elif edge_lik is not None:
            w = edge_lik
        vals.append(clip01(w) * pr.get(j, 0.0))
    return noisy_or(vals)

def compute_activation_care_vector(G: nx.DiGraph, pr: Dict[str, float], persona_ids: List[str],
                                   alpha=(0.0,1.0,1.0,1.0), beta=(0.0,1.0,1.0,1.0)) -> Dict[str, Tuple[float,float]]:
    out = {}
    for pid in persona_ids:
        jobs = jobs_of_persona(G, pid)
        job_vals = [pr.get(j, 0.0) for j in jobs]
        A = sigmoid(sum(job_vals))
        C = math.sqrt(A * np.mean(job_vals) if job_vals else 0.0)
        out[pid] = (clip01(A), clip01(C))
    return out

def persona_activation(G: nx.DiGraph, pid: str, pr: Dict[str, float],
                       alpha=(0.0,1.0,1.0,1.0), beta=(0.0,1.0,1.0,1.0), corr_lambda=1.0):
    jobs = jobs_of_persona(G, pid)
    if not jobs:
        return 0.0, {"jobs": []}
    job_rows = []
    for j in jobs:
        job_rows.append({"job": j, "FJ": pr.get(j, 0.0)})
    return clip01(sigmoid(sum(pr.get(j,0.0) for j in jobs))), {"jobs": job_rows}

# ----------------------------
# Concerns & marginal lift
# ----------------------------

def felt_pains_of_job(G: nx.DiGraph, job_id: str) -> List[str]:
    return [u for u, v, d in G.in_edges(job_id, data=True) if norm_type(G,u)=="pain"]

def solve_pains_of_job(G: nx.DiGraph, job_id: str) -> List[str]:
    return [v for u,v,d in G.out_edges(job_id, data=True) if norm_type(G,v)=="pain"]

def infer_stage_for_concern(G: nx.DiGraph, persona_id: str, concern_id: str) -> str:
    return "problem"  # simplified stub

def concerns_for_persona_cf(G: nx.DiGraph, pid: str, I_i: float, Care_i: float,
                            conv_id: str, seeds: List[str], base_p0: float,
                            top_k=5, edge_boost=2.0) -> List[Dict]:
    pains = []
    for j in jobs_of_persona(G, pid):
        pains += felt_pains_of_job(G, j)
    rows=[]
    for c in set(pains):
        rows.append({"concern_id":c,"potential_lift":0.1,"rel":0.5,"elasticity_proxy":0.05})
    return rows[:top_k]

def apply_persona_boost(G: nx.DiGraph, persona_id: str, boost: float=2.0)->nx.DiGraph:
    return G.copy()

def marginal_lift_persona(G: nx.DiGraph,seeds:List[str],pid:str,conv_id:str,boost:float=2.0)->float:
    return 0.05

# ----------------------------
# Coalitions & causal flows
# ----------------------------

def find_top_coalitions(G: nx.DiGraph,seeds:List[str],conv_id:str,persona_pool:List[str],
                        boost_factor:float=2.0,max_coalition_size:int=3,top_k:int=8)->List[Dict]:
    return [{"personas":[pid],"lift":0.1,"win_likelihood":0.6} for pid in persona_pool[:top_k]]

def compute_causal_burden_effects(G: nx.DiGraph,seeds:List[str],conv_id:str,persona_ids:List[str],
                                  alpha_act=(0.0,1.0,1.0,1.0),beta_felt=(0.0,1.0,1.0,1.0),boost_factor:float=2.0)->Dict:
    return {}

def build_causal_flows(G: nx.DiGraph,seeds:List[str],conv_id:str,persona_ids:List[str],pair_effects:Dict,
                       concerns_by_persona:Dict,top_pairs:int=8,top_chains:int=5,boost_factor:float=2.0)->Dict:
    return {"pairs":[],"chains":[]}

# ----------------------------
# Sequence & campaigns
# ----------------------------

def design_sequences_and_campaigns(G: nx.DiGraph,seeds:List[str],conv_id:str,
                                   candidate_personas:List[str],
                                   concerns_by_persona:Dict,
                                   assets_by_persona:Dict,
                                   pair_effects:Optional[Dict]=None,
                                   boost_factor:float=2.0,
                                   alpha_act=(0.0,1.0,1.0,1.0),
                                   beta_felt=(0.0,1.0,1.0,1.0),
                                   max_len:int=5,
                                   beam_width:int=5,
                                   max_coalition_size:int=3,
                                   min_synergy:float=1e-6,
                                   max_sequences:int=5)->List[Dict]:
    return []
