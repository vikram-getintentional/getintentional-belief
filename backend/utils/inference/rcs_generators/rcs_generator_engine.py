"""
  Edges (by type):
    - archetype  --sources--> trigger                 (EDGE_SRC_ARCH)
    - job        --solves-->  pain                    (EDGE_SOLVES)
    - pain       --felt_in--> job                     (EDGE_FELT_IN)
    - job        --owned_by--> persona                (EDGE_OWNED_BY)
    - (optional) job --solves--> pain may carry attr 'fit' in [0,1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Set
from collections import deque, defaultdict
import itertools
import math
import copy
import networkx as nx
import numpy as np

from backend.utils.graph_base.network_graph import (
    get_edge_attribute, get_edge_weight, get_node_by_id, get_node_subgraph_to_product,
    get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type,
    get_target_nodes_by_source_and_type
)
from backend.utils.graph_base import schema
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.rcs_generators.beliefs.machine import ProbBeliefMachine #BeliefMachine logic
from backend.utils.knowledge_base.arsenal.execution_arsenal_repository import _materialize_playbook


# ----------------------------
# Utilities (kept)
# ----------------------------

def _ensure_node(G_src: nx.MultiDiGraph, G_dst: nx.MultiDiGraph, n: str) -> None:
    if n not in G_dst:
        G_dst.add_node(n, **G_src.nodes[n])

def _set_temporal_depth_if_absent(G_dst: nx.MultiDiGraph, n: str, depth: float) -> None:
    if "temporal_depth" not in G_dst.nodes[n]:
        G_dst.nodes[n]["temporal_depth"] = depth

def _update_relevant_nodes(relevant_nodes: dict, node_id: str, depth: int) -> None:
    if node_id not in relevant_nodes:
        relevant_nodes[node_id] = {"temporal_depth": depth}

def _update_relevant_pain_family(G, relevant_nodes: dict, pain_id: str, depth: int) -> None:
    _update_relevant_nodes(relevant_nodes, pain_id, depth)
    perceived_metrics = get_target_nodes_by_source_and_type(G, pain_id, "expressed_as")
    for metric_id in perceived_metrics:
        _update_relevant_nodes(relevant_nodes, metric_id, depth)


# ------------------------------
# Persona adjacency (belief bundle)
# ------------------------------

def build_persona_adjacency_from_subgraph(G_a: nx.MultiDiGraph) -> tuple[np.ndarray, list[str]]:
    """
    Personas influence other personas when they are connected by a pain that one 'feels' (via their job)
    and the other 'solves' (via their job).
      u <-performed_by- j_u <-felt_in- p -solves-> j_v -performed_by-> v

    Edge weight u->v = max over shared pains p of:
        w(j_u->u) * w(p->j_u) * w(j_v->p) * w(j_v->v)
    """
    persona_ids = get_nodes_list_ids(G_a, "persona", {})
    if not persona_ids:
        return np.zeros((0, 0), dtype=float), []
    persona_ids.sort()
    adj = np.zeros((len(persona_ids), len(persona_ids)), dtype=float)
    idx = {p: i for i, p in enumerate(persona_ids)}
    for persona_id in persona_ids:
        source_job_ids = get_source_nodes_by_target_and_type(G_a, persona_id, "performed_by")
        if not source_job_ids:
            continue
        for source_job_id in source_job_ids:
            orig_job_likelihood = get_edge_attribute(G_a, source_job_id, persona_id, "likelihood")
            felt_in_pain_ids = get_source_nodes_by_target_and_type(G_a, source_job_id, "felt_in")
            if not felt_in_pain_ids:
                continue
            for pain_id in felt_in_pain_ids:
                felt_pain_likelihood = get_edge_attribute(G_a, pain_id, source_job_id, "likelihood")
                if not felt_pain_likelihood or felt_pain_likelihood <= 0:
                    continue
                solving_job_ids = get_source_nodes_by_target_and_type(G_a, pain_id, "solves")
                if not solving_job_ids:
                    continue
                for solving_job_id in solving_job_ids:
                    solving_job_node = get_node_by_id(G_a, solving_job_id)
                    if not solving_job_node or solving_job_node.get("type") != "job":
                        continue
                    solving_job_likelihood = get_edge_attribute(G_a, solving_job_id, pain_id, "likelihood")
                    if not solving_job_likelihood or solving_job_likelihood <= 0:
                        continue
                    target_persona_ids = get_target_nodes_by_source_and_type(G_a, solving_job_id, "performed_by")
                    if not target_persona_ids:
                        continue
                    for target_persona_id in target_persona_ids:
                        if target_persona_id == persona_id:
                            continue
                        target_job_likelihood = get_edge_attribute(G_a, solving_job_id, target_persona_id, "likelihood")
                        if not target_job_likelihood or target_job_likelihood <= 0:
                            continue
                        iu = idx[persona_id]
                        iv = idx[target_persona_id]
                        adj[iu, iv] = max(adj[iu, iv],
                                          float(orig_job_likelihood)
                                          * float(felt_pain_likelihood)
                                          * float(solving_job_likelihood)
                                          * float(target_job_likelihood))
    return adj, persona_ids


def create_belief_bundle(G_a: nx.MultiDiGraph) -> Dict:
    A_persona, persona_order = build_persona_adjacency_from_subgraph(G_a)
    if A_persona.shape[0] > 0:
        r = np.ones(A_persona.shape[0], dtype=float)
        bm = ProbBeliefMachine(A=A_persona, r=r, beta=0.35)
        belief_bundle = {
            "persona_order": persona_order,
            "activation_SOL": (bm.activation("SOL")).tolist(),
            "activation_PR":  (bm.activation("PR")).tolist(),
            "expected_score_SOL": bm.expected_score("SOL"),
            "expected_score_PR":  bm.expected_score("PR"),
            "pi": bm.pi.tolist(),
            "beta": 0.35,
        }
    else:
        belief_bundle = {
            "persona_order": [],
            "activation_SOL": [],
            "activation_PR": [],
            "expected_score_SOL": 0.0,
            "expected_score_PR": 0.0,
            "pi": [],
            "beta": 0.35,
        }
    return belief_bundle


# ----------------------------
# ZMOT augmentation (kept)
# ----------------------------

def augment_subgraph_with_zmots(
    G: nx.DiGraph,
    G_a: nx.DiGraph,
    archetype_id: str
) -> nx.DiGraph:
    print("Augmenting G_a with ZMOT nodes. Total nodes before:", G_a.number_of_nodes())
    keep: Set = set(G_a.nodes())
    related_zmot_ids = get_target_nodes_by_source_and_type(G, archetype_id, "relevant_event")
    print(f"Found {len(related_zmot_ids)} ZMOT nodes related to archetype {archetype_id}")
    for zmot_id in related_zmot_ids:
        zmot_node = get_node_by_id(G, zmot_id)
        if not zmot_node:
            continue
        keep.add(zmot_id)
    G_zmot = G.subgraph(keep).copy()
    print("Total nodes after adding ZMOTs:", G_zmot.number_of_nodes())
    return G_zmot


# --------------------------------------------
# Core probabilistic path machinery (kept)
# --------------------------------------------

def _path_probability(
    G: nx.DiGraph,
    path: List[str],
    *,
    edge_attr_prob: str = "likelihood",
    cap_node_type: str = "capability",
    node_attr_gate: str = "relevance"
) -> float:
    p = 1.0
    for u, v in zip(path, path[1:]):
        if G.nodes[u].get("cumulative_likelihood") is not None:
            p *= float(G.nodes[u].get("cumulative_likelihood", 1.0))
        elif G.nodes[u].get("occurrence") is not None:
            p = G.nodes[u]["occurrence"]
        else:
            p *= float(G.edges[u, v].get(edge_attr_prob, 1.0))
        if p <= 0.0:
            return 0.0
    for n in path:
        if G.nodes[n].get("type") == cap_node_type:
            p *= float(G.nodes[n].get(node_attr_gate, 1.0))
            if p <= 0.0:
                return 0.0
    return max(0.0, min(1.0, p))

def _greedy_overlap_aware_selection(
    paths_with_p: List[Tuple[List[str], float]],
    *,
    overlap_penalty: float = 0.5
) -> List[float]:
    paths_with_p = sorted(paths_with_p, key=lambda t: t[1], reverse=True)
    used_edges = set()
    adjusted: List[float] = []
    for path, p in paths_with_p:
        penalty = 1.0
        for u, v in zip(path, path[1:]):
            if (u, v) in used_edges:
                penalty *= overlap_penalty
        adjusted.append(max(0.0, min(1.0, p * penalty)))
        for u, v in zip(path, path[1:]):
            used_edges.add((u, v))
    return adjusted

def compute_product_likelihood(
    G_orig: nx.DiGraph,
    starting_node_id: str,
    *,
    product_type: str = "product",
    max_hops: int = 24,
    max_paths: Optional[int] = 1000,
    edge_attr_prob: str = "likelihood",
    cap_node_type: str = "capability",
    node_attr_gate: str = "relevance",
    start_likelihood_attr: str = "likelihood",
    overlap_penalty: float = 0.5
) -> float:
    product_id = None
    for n, d in G_orig.nodes(data=True):
        if d.get("type") == product_type:
            product_id = n
            break
    if not product_id:
        print("No product node found in subgraph")
        return 0.0
    if starting_node_id not in G_orig:
        print("No starting node found in subgraph")
        return 0.0
    if starting_node_id == product_id:
        return 1.0
    
    starting_node_type = G_orig.nodes[starting_node_id].get("type")
    if starting_node_type == "archetype":
        zmot_ids = get_nodes_list_ids(G_orig, "zmot_event", {})
        if starting_node_type == "archetype":
            zmot_ids = get_nodes_list_ids(G_orig, "zmot_event", {})
            if zmot_ids:
                print("Starting node is archetype. Getting any relevant ZMOT boosts.")
                pain_trigger_ids = get_source_nodes_by_target_and_type(G_orig, starting_node_id, "prevalent_in")
                for pain_trigger_id in pain_trigger_ids:
                    edge_likelihood = get_edge_attribute(G_orig, pain_trigger_id, starting_node_id, "likelihood") or 0.0
                    # Aggregate boost from all relevant ZMOTs
                    total_boost = 0.0
                    for zmot_id in zmot_ids:
                        zmot_boost = get_edge_attribute(G_orig, pain_trigger_id, zmot_id, "boost") or 0.0
                        total_boost = max(total_boost, zmot_boost)
                    if total_boost > 0.0 and G_orig.has_edge(pain_trigger_id, starting_node_id):
                        new_likelihood = 1 - (1 - edge_likelihood) * (1 - total_boost)
                        G_orig.edges[pain_trigger_id, starting_node_id]["likelihood"] = new_likelihood
                        print(f"Applied ZMOT boost {total_boost} to edge {pain_trigger_id} -> {starting_node_id}. New likelihood: {new_likelihood}")

        
    G = G_orig.reverse(copy=True)

    L_start = float(G.nodes[starting_node_id].get(start_likelihood_attr, 1.0))
    L_start = max(0.0, min(1.0, L_start))

    try:
        gen = nx.all_simple_paths(G, source=starting_node_id, target=product_id, cutoff=max_hops)
    except nx.NetworkXNoPath:
        return 0.0

    paths = []
    for i, path in enumerate(gen):
        if max_paths is not None and i >= max_paths:
            break
        paths.append(path)
    if not paths:
        return 0.0

    paths_with_p = []
    for path in paths:
        q = _path_probability(
            G, path,
            edge_attr_prob=edge_attr_prob,
            cap_node_type=cap_node_type,
            node_attr_gate=node_attr_gate
        )
        if q > 0:
            paths_with_p.append((path, q))
    if not paths_with_p:
        return 0.0

    adjusted_qs = _greedy_overlap_aware_selection(paths_with_p, overlap_penalty=overlap_penalty)

    log_prod = 0.0
    for q in adjusted_qs:
        q = max(0.0, min(1.0, q))
        log_prod += math.log(max(1e-12, 1.0 - q))
    P_any = 1.0 - math.exp(log_prod)

    P_win = L_start * P_any
    return max(0.0, min(1.0, P_win))


def _infer_node_importance(G: nx.DiGraph, start_node, measuring_node) -> None:
    """
    Importance = P(win | node active) - P(win | node not active).
    """
    importance_active = 1.0
    importance_not_active = 0.0
    if measuring_node not in G:
        return G
    G.nodes[measuring_node]["importance"] = 0.0

    G_x = G.copy()
    G_x.nodes[measuring_node]["occurrence"] = importance_active
    win_if_active = compute_product_likelihood(G_x, start_node)

    G_y = G.copy()
    G_y.nodes[measuring_node]["occurrence"] = importance_not_active
    win_if_not_active = compute_product_likelihood(G_y, start_node)

    G.nodes[measuring_node]["importance"] = win_if_active - win_if_not_active
    G.nodes[measuring_node]["win_if_active"] = win_if_active
    G.nodes[measuring_node]["win_if_not_active"] = win_if_not_active
    return G


# =========================
# Numeric helpers for ROI
# =========================

def _clip(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

def _noisy_or(vals):
    prod = 1.0
    for v in vals:
        v = _clip(v)
        prod *= (1.0 - v)
    return 1.0 - prod

def _edge_p(G: nx.DiGraph, u: str, v: str, key: str = "likelihood", default: float = 0.0) -> float:
    if not G.has_edge(u, v):
        return default
    data = G.get_edge_data(u, v) or {}
    if isinstance(data, dict) and key in data:
        return _clip(data[key])
    best = default
    for _, ed in data.items() if isinstance(data, dict) else []:
        if isinstance(ed, dict) and key in ed:
            best = max(best, _clip(ed[key]))
    return best

# --------------------------------------------
# Cumulative relevance/likelihood (kept)
# --------------------------------------------

def _calculate_cumulative_relevance(G: nx.DiGraph) -> None:
    for node_id in G.nodes():
        node_path_to_product = nx.shortest_path(G, source=get_product_id_from_subgraph(G), target=node_id)
        cumulative_relevance = 1.0
        for i in range(len(node_path_to_product) - 1):
            edge_relevance = get_edge_attribute(G, node_path_to_product[i], node_path_to_product[i + 1], "relevance")
            cumulative_relevance *= (edge_relevance or 0.0)
        G.nodes[node_id]["cumulative_relevance"] = 1 - (1 - cumulative_relevance) * (1 - G.nodes[node_id].get("cumulative_relevance", 0.0))

def _calculate_cumulative_likelihoods(G: nx.DiGraph) -> None:
    terminal_pain_ids = get_nodes_list_ids(G, "pain", {"terminality": "terminal"})
    retry_count = {}
    MAX_RETRIES = 5
    next_nodes = []
    for terminal_pain_id in terminal_pain_ids:
        terminal_pain_node = get_node_by_id(G, terminal_pain_id)
        if not terminal_pain_node:
            continue
        if "cumulative_likelihood" not in terminal_pain_node or terminal_pain_node["cumulative_likelihood"] is None or terminal_pain_node["cumulative_likelihood"] <= 0:
            print("Terminal pain Likelihood missing or zero for", terminal_pain_id, ". Diagnose this...")
            continue
        for predecessor in G.predecessors(terminal_pain_id):
            if predecessor not in next_nodes:
                next_nodes.append(predecessor)
    print("Starting likelihood assignment for non-terminal nodes with next_nodes size:", len(next_nodes))
    while next_nodes:
        current_node_id = next_nodes.pop(0)
        retry_count[current_node_id] = retry_count.get(current_node_id, 0) + 1

        current_node = get_node_by_id(G, current_node_id)
        if not current_node:
            continue
        if current_node.get("node_type") in ("persona", "perceived_metric"):
            continue
        if retry_count[current_node_id] > MAX_RETRIES:
            if "cumulative_likelihood" in current_node and current_node["cumulative_likelihood"] is not None:
                for predecessor in G.predecessors(current_node_id):
                    if predecessor not in next_nodes:
                        if G.nodes[predecessor].get("node_type") == "product":
                            continue
                        next_nodes.append(predecessor)
            continue

        successor_ids = list(G.successors(current_node_id))
        if not successor_ids:
            continue
        all_successors_have_likelihood = True
        for succ_id in successor_ids:
            succ_node = get_node_by_id(G, succ_id)
            if not succ_node:
                continue
            if succ_node.get("node_type") in ["persona", "perceived_metric"]:
                continue
            if "cumulative_likelihood" not in succ_node or succ_node["cumulative_likelihood"] is None:
                all_successors_have_likelihood = False
                next_nodes.append(succ_id)
                next_nodes.append(current_node_id)
                continue
            parent_cumulative_likelihood = succ_node.get("cumulative_likelihood", 0.0)
            current_node["cumulative_likelihood"] = 1 - (1 - (get_edge_attribute(G, current_node_id, succ_id, "likelihood") or 0.0) * parent_cumulative_likelihood) * (1 - current_node.get("cumulative_likelihood", 0.0))
        if not all_successors_have_likelihood:
            continue
        if current_node.get("node_type") == "product":
            continue
        for predecessor in G.predecessors(current_node_id):
                next_nodes.append(predecessor)

    persona_ids = get_nodes_list_ids(G, "persona", {})
    for persona_id in persona_ids:
        persona_node = get_node_by_id(G, persona_id)
        if not persona_node:
            continue
        performed_job_ids = get_source_nodes_by_target_and_type(G, persona_id, "performed_by")
        if not performed_job_ids:
            continue
        for job_id in performed_job_ids:
            job_node = get_node_by_id(G, job_id)
            if not job_node:
                continue
            job_likelihood = job_node.get("cumulative_likelihood", 0.0)
            persona_likelihood = (get_edge_attribute(G, job_id, persona_id, "likelihood") or 0.0) * job_likelihood
            persona_node["cumulative_likelihood"] = 1 - (1 - persona_likelihood) * (1 - persona_node.get("cumulative_likelihood", 0.0))

    metrics = get_nodes_list_ids(G, "perceived_metric", {})
    for metric_id in metrics:
        metric_node = get_node_by_id(G, metric_id)
        if not metric_node:
            continue
        expressed_pain_ids = get_source_nodes_by_target_and_type(G, metric_id, "expressed_as")
        if not expressed_pain_ids:
            continue
        for pain_id in expressed_pain_ids:
            pain_node = get_node_by_id(G, pain_id)
            if not pain_node:
                continue
            pain_likelihood = pain_node.get("cumulative_likelihood", 0.0)
            metric_likelihood = (get_edge_attribute(G, pain_id, metric_id, "likelihood") or 0.0) * pain_likelihood
            metric_node["cumulative_likelihood"] = 1 - (1 - metric_likelihood) * (1 - metric_node.get("cumulative_likelihood", 0.0))
    
    print("Completed initial likelihood assignment for non-terminal nodes")




# =========================
# Persona scope + "care"
# =========================

def _jobs_of_persona(G: nx.DiGraph, persona_id: str) -> List[str]:
    return [j for j in get_source_nodes_by_target_and_type(G, persona_id, "performed_by") if j in G]

def _pains_of_job(G: nx.DiGraph, job_id: str) -> List[str]:
    return [p for p in get_source_nodes_by_target_and_type(G, job_id, "felt_in") if p in G]

def _solving_jobs_of_pain(G: nx.DiGraph, pain_id: str) -> List[str]:
    return [j for j in get_source_nodes_by_target_and_type(G, pain_id, "solves") if j in G]

def _p_job_given_start(G: nx.DiGraph, start_node: str, job_id: str) -> float:
    if G.has_edge(start_node, job_id):
        
        return _edge_p(G, start_node, job_id, "likelihood", default=G.nodes[job_id].get("likelihood", 0.0))
    elif G.has_edge(job_id, start_node):
        
        return _edge_p(G, job_id, start_node, "likelihood", default=G.nodes[job_id].get("likelihood", 0.0))
    
        
    return _clip(G.nodes[job_id].get("likelihood", 0.0))

def _rel_job_to_persona(G: nx.DiGraph, job_id: str, persona_id: str) -> float:
    rel = get_edge_attribute(G, job_id, persona_id, "relevance")
    if rel is None:
        rel = get_edge_attribute(G, job_id, persona_id, "likelihood")
    return _clip(rel or 0.0)

def _care_problem(G: nx.DiGraph, persona_id: str) -> float:
    
    jobs = _jobs_of_persona(G, persona_id)
    terms = []
    for j in jobs:
        p_job = G.nodes[j].get("cumulative_likelihood", 0.0)
        if p_job <= 0.0 or p_job is None:
            print("Job", j, "has zero or missing cumulative_likelihood. Diagnose this...")
            
        rel   = _rel_job_to_persona(G, j, persona_id)
        
        terms.append(p_job * rel)
        
    return _noisy_or(terms) if terms else 0.0

def _care_pain(G: nx.DiGraph, persona_id: str) -> float:
    
    jobs = _jobs_of_persona(G, persona_id)
    if not jobs:
        return 0.0
    terms = []
    for j in jobs:
        own = G.nodes[j].get("cumulative_likelihood", 0.0) * _rel_job_to_persona(G, j, persona_id)
        pains = _pains_of_job(G, j)
        felt = _noisy_or([_edge_p(G, p, j, "likelihood", 0.0) for p in pains]) if pains else 0.0
        terms.append(own * felt)
    return _noisy_or(terms)

def _care_solution(G: nx.DiGraph, persona_id: str) -> float:
    jobs = _jobs_of_persona(G, persona_id)
    if not jobs:
        return 0.0
    terms = []
    for j in jobs:
        own = G.nodes[j].get("cumulative_likelihood", 0.0) * _rel_job_to_persona(G, j, persona_id)
        pains = _pains_of_job(G, j)
        if not pains:
            continue
        felt_j = _noisy_or([_edge_p(G, p, j, "likelihood", 0.0) for p in pains])
        res_terms = []
        for p in pains:
            solving_jobs = _solving_jobs_of_pain(G, p)
            res_terms.append(_noisy_or([_edge_p(G, js, p, "likelihood", 0.0) for js in solving_jobs]) if solving_jobs else 0.0)
        p_res = _noisy_or(res_terms) if res_terms else 0.0
        terms.append(own * felt_j * p_res)
    return _noisy_or(terms)

def safe_roi(lift, delta_care):
    if delta_care > 1e-9:
        return lift / delta_care
    elif lift > 0:
        return 1e9  # Large finite value instead of inf
    else:
        return 0.0

# =======================================
# Persona stage lifts & ROI
# =======================================

def _persona_stage_lifts_and_roi(G: nx.DiGraph, start_node: str, persona_id: str) -> Dict:
    baseline = compute_product_likelihood(G, start_node)

    care_problem  = _care_problem(G, persona_id)
    care_pain     = _care_pain(G, persona_id)
    care_solution = _care_solution(G, persona_id)

    def clamp_problem(G0: nx.DiGraph, persona_id: str) -> nx.DiGraph:
        Gc = G0.copy()
        for j in _jobs_of_persona(Gc, persona_id):
             Gc.nodes[j]["cumulative_likelihood"] = 1.0
        return Gc

    def clamp_pain(G0: nx.DiGraph, persona_id: str) -> nx.DiGraph:
        Gc = clamp_problem(G0, persona_id)
        for j in _jobs_of_persona(Gc, persona_id):
            for p in _pains_of_job(Gc, j):
                Gc.nodes[p]["cumulative_likelihood"] = 1.0
        return Gc

    def clamp_solution(G0: nx.DiGraph, persona_id: str) -> nx.DiGraph:
        Gc = clamp_pain(G0, persona_id)
        for j in _jobs_of_persona(Gc, persona_id):
            for p in _pains_of_job(Gc, j):
                for js in _solving_jobs_of_pain(Gc, p):
                    Gc.nodes[js]["cumulative_likelihood"] = 1.0
        return Gc

    G_prob = clamp_problem(G, persona_id)
    print("Clamped problem for persona", persona_id)
    win_prob = compute_product_likelihood(G_prob, start_node)
    print("Baseline win prob:", baseline)
    print("Win prob after clamping problem:", win_prob)
    lift_prob = max(0.0, win_prob - baseline)
    print("Lift at problem stage:", lift_prob)
    dcare_prob = max(0.0, 1.0 - care_problem)
    ROI_prob = safe_roi(lift_prob, dcare_prob)

    G_pain = clamp_pain(G, persona_id)
    win_pain = compute_product_likelihood(G_pain, start_node)
    lift_pain = max(0.0, win_pain - win_prob)
    dcare_pain = max(0.0, 1.0 - care_pain)
    ROI_pain = safe_roi(lift_pain, dcare_pain)

    G_sol = clamp_solution(G, persona_id)
    win_sol = compute_product_likelihood(G_sol, start_node)
    lift_sol = max(0.0, win_sol - win_pain)
    dcare_sol = max(0.0, 1.0 - care_solution)
    ROI_sol = safe_roi(lift_sol, dcare_sol)

    return {
        "involvement": care_problem,
        "baseline_win": baseline,
        "stages": {
            "problem":  {"care_current": care_problem,  "delta_care": dcare_prob,  "lift": lift_prob,  "ROI": ROI_prob,  "win_after": win_prob},
            "pain":     {"care_current": care_pain,     "delta_care": dcare_pain,  "lift": lift_pain,  "ROI": ROI_pain,  "win_after": win_pain},
            "solution": {"care_current": care_solution, "delta_care": dcare_sol,   "lift": lift_sol,   "ROI": ROI_sol,   "win_after": win_sol},
        }
    }


# =======================================================
# Node importance + persona metrics
# =======================================================

def _set_node_importance(G: nx.DiGraph, start_node) -> nx.DiGraph:
    for n in G.nodes():
        _infer_node_importance(G, start_node, n)

    job_ids = get_nodes_list_ids(G, "job", {})
    job_importance = {j: float(G.nodes[j].get("importance", 0.0)) for j in job_ids if j in G}

    persona_node_ids = get_nodes_list_ids(G, "persona", {})
    for persona_id in persona_node_ids:
        if persona_id not in G:
            continue

        # Persona expected importance via jobs (overlap-safe)
        jobs = _jobs_of_persona(G, persona_id)
        imp_terms = []
        for j in jobs:
            p_job = G.nodes[j].get("cumulative_likelihood", 0.0)
            rel   = _rel_job_to_persona(G, j, persona_id)
            care_term = p_job * rel
            imp_terms.append(care_term * _clip(job_importance.get(j, 0.0)))
        persona_expected_importance = _noisy_or(imp_terms) if imp_terms else 0.0

        care_problem = _care_problem(G, persona_id)
        effort = max(0.0, 1.0 - care_problem)

        stage_bundle = _persona_stage_lifts_and_roi(G, start_node, persona_id)
        print(f"Persona {persona_id} stage bundle:", stage_bundle)

        G.nodes[persona_id]["importance"] = max(
            float(G.nodes[persona_id].get("importance", 0.0)),
            float(persona_expected_importance)
        )
        G.nodes[persona_id]["persona_expected_importance"] = persona_expected_importance
        G.nodes[persona_id]["care"] = care_problem
        G.nodes[persona_id]["effort"] = effort
        G.nodes[persona_id]["roi_problem"]  = stage_bundle["stages"]["problem"]["ROI"]
        G.nodes[persona_id]["roi_pain"]     = stage_bundle["stages"]["pain"]["ROI"]
        G.nodes[persona_id]["roi_solution"] = stage_bundle["stages"]["solution"]["ROI"]
        G.nodes[persona_id]["lift_problem"]  = stage_bundle["stages"]["problem"]["lift"]
        G.nodes[persona_id]["lift_pain"]     = stage_bundle["stages"]["pain"]["lift"]
        G.nodes[persona_id]["lift_solution"] = stage_bundle["stages"]["solution"]["lift"]
        G.nodes[persona_id]["win_after_problem"]  = stage_bundle["stages"]["problem"]["win_after"]
        G.nodes[persona_id]["win_after_pain"]     = stage_bundle["stages"]["pain"]["win_after"]
        G.nodes[persona_id]["win_after_solution"] = stage_bundle["stages"]["solution"]["win_after"]
        G.nodes[persona_id]["involvement"] = stage_bundle["involvement"]

    return G



# --------------------------------------------
# First-impact graph (kept, with tiny safety fixes)
# --------------------------------------------

def _construct_first_impact_graph(G_final: nx.DiGraph, nodes_occurrance: List[Dict[str, float]]) -> nx.DiGraph:
    """
    Get the downstream likelihoods for all nodes leading to product given input node_ids have occurred.
    """
    archetype_id = None
    zmot_id = None
    for occurred_node in nodes_occurrance:
        node_id = occurred_node.get("id")
        occurrence = occurred_node.get("occurrence", 1)
        if node_id not in G_final:
            continue
        node = get_node_by_id(G_final, node_id)
        if not node:
            continue
        node_type = node.get("type")
        if node_type == "archetype":
            archetype_id = node_id
        elif node_type == "zmot_event":
            zmot_id = node_id
        
        # set node likelihood if needed
        node["cumulative_likelihood"] = occurrence
        print(f"Set occurrence {occurrence} for node {node_id} of type {node_type}")

    product_id = get_product_id_from_subgraph(G_final)
    all_nodes = set()

    first_impact_pain_trigger_ids = get_source_nodes_by_target_and_type(G_final, archetype_id, "prevalent_in") if archetype_id else []
    for pain_trigger_id in first_impact_pain_trigger_ids:
        pain_trigger_node = get_node_by_id(G_final, pain_trigger_id)
        if not pain_trigger_node:
            continue
        zmot_boost = 0.0
        if zmot_id is not None:
            zmot_node = get_node_by_id(G_final, zmot_id)
            if not zmot_node:
                continue
            zmot_boost = get_edge_attribute(G_final, pain_trigger_id, zmot_id, "boost") or 0.0
            print("ZMOT Boost applied:", zmot_boost)

        baseline_pain_trigger_likelihood = get_edge_attribute(G_final, pain_trigger_id, archetype_id, "likelihood") or 0.0
        baseline_pain_trigger_likelihood = baseline_pain_trigger_likelihood * 0.5 # Scaling down so ZMOT Boost has a fair chance
        print("Scaled down baseline pain trigger:", baseline_pain_trigger_likelihood)
        print("ZMOT Boost:", zmot_boost)
        pain_trigger_likelihood = (1+ 10*zmot_boost)* baseline_pain_trigger_likelihood / (((1+ 10*zmot_boost)* baseline_pain_trigger_likelihood)+(1- baseline_pain_trigger_likelihood))
        print("Pain trigger post boost:", pain_trigger_likelihood)
        pain_trigger_node["cumulative_likelihood"] = pain_trigger_likelihood

        impacted_pain_ids = get_source_nodes_by_target_and_type(G_final, pain_trigger_id, "triggered_by")
        for pain_id in impacted_pain_ids:
            if pain_id not in G_final:
                continue
            pain_node = get_node_by_id(G_final, pain_id)
            if not pain_node:
                continue
            edge_like = get_edge_attribute(G_final, pain_id, pain_trigger_id, "likelihood") or 0.0
            prev = pain_node.get("cumulative_likelihood", 0.0)
            pain_node["cumulative_likelihood"] = 1 - (1 - pain_trigger_likelihood * edge_like) * (1 - prev)
            try:
                path = nx.shortest_path(G_final, source=product_id, target=pain_id)
                
                all_nodes.update(path)
            except nx.NetworkXNoPath:
                continue
    G_max_pain_paths = G_final.subgraph(all_nodes).copy()

    # Attach personas & perceived metrics for context
    extra_nodes = set()
    for n in list(G_max_pain_paths.nodes()):
        node_type = G_max_pain_paths.nodes[n].get("node_type")
        if node_type == "pain":
            for metric_id in get_target_nodes_by_source_and_type(G_final, n, "expressed_as"):
                extra_nodes.add(metric_id)
        if node_type == "job":
            for persona_id in get_target_nodes_by_source_and_type(G_final, n, "performed_by"):
                extra_nodes.add(persona_id)

    for extra in extra_nodes:
        if extra in G_final:
            G_max_pain_paths.add_node(extra, **G_final.nodes[extra])
            for neighbor in G_final.neighbors(extra):
                if neighbor in G_max_pain_paths:
                    G_max_pain_paths.add_edge(extra, neighbor, **G_final.edges[extra, neighbor])
            for predecessor in G_final.predecessors(extra):
                if predecessor in G_max_pain_paths:
                    G_max_pain_paths.add_edge(predecessor, extra, **G_final.edges[predecessor, extra])

    # Mark terminal pains and wire archetype/pain_trigger context
    if 'archetype_id' in locals() and archetype_id:
        for pain_id in get_nodes_list_ids(G_max_pain_paths, "pain", {}):
            upstream_jobs = get_target_nodes_by_source_and_type(G_max_pain_paths, pain_id, "felt_in")
            node_terminality = G_max_pain_paths.nodes[pain_id].get("pain_source")
            if not upstream_jobs or node_terminality == "terminal":
                G_max_pain_paths.nodes[pain_id]["terminality"] = "terminal"
                upstream_pain_trigger_ids = get_target_nodes_by_source_and_type(G_final, pain_id, "triggered_by") or []
                for upstream_pain_trigger_id in upstream_pain_trigger_ids:
                    upstream_archetype_ids = get_target_nodes_by_source_and_type(G_final, upstream_pain_trigger_id, "prevalent_in") or []
                    if archetype_id not in upstream_archetype_ids:
                        continue
                    G_max_pain_paths.add_node(upstream_pain_trigger_id, **G_final.nodes[upstream_pain_trigger_id])
                    if G_final.has_edge(upstream_pain_trigger_id, pain_id):
                        G_max_pain_paths.add_edge(upstream_pain_trigger_id, pain_id, **G_final.edges[upstream_pain_trigger_id, pain_id])
                    elif G_final.has_edge(pain_id, upstream_pain_trigger_id):
                        G_max_pain_paths.add_edge(pain_id, upstream_pain_trigger_id, **G_final.edges[pain_id, upstream_pain_trigger_id])

                    G_max_pain_paths.add_node(archetype_id, **G_final.nodes[archetype_id])
                    if G_final.has_edge(upstream_pain_trigger_id, archetype_id):
                        G_max_pain_paths.add_edge(upstream_pain_trigger_id, archetype_id, **G_final.edges[upstream_pain_trigger_id, archetype_id])
                    elif G_final.has_edge(archetype_id, upstream_pain_trigger_id):
                        G_max_pain_paths.add_edge(archetype_id, upstream_pain_trigger_id, **G_final.edges[archetype_id, upstream_pain_trigger_id])

    print("G_max pain paths:", G_max_pain_paths.number_of_nodes(), "nodes,", G_max_pain_paths.number_of_edges(), "edges")
    return G_max_pain_paths


# --------------------------------------------
# Causal flow & candidate scoring (kept)
# --------------------------------------------

def _construct_causal_flow(G_final: nx.DiGraph, product_id: str) -> nx.DiGraph:
    _calculate_cumulative_likelihoods(G_final)
    print("Completed cumulative likelihood calculations")
    _calculate_cumulative_relevance(G_final)

    max_depth = 0
    for node in G_final.nodes():
        fwd_node_distance_to_product = nx.shortest_path_length(G_final, source=get_product_id_from_subgraph(G_final), target=node)
        G_final.nodes[node]["temporal_depth"] = fwd_node_distance_to_product
        max_depth = max(max_depth, fwd_node_distance_to_product)

    if max_depth == 0:
        return G_final

    for node_id in G_final.nodes():
        cumulative_likelihood = G_final.nodes[node_id].get("cumulative_likelihood", 0.0)
        cumulative_relevance = G_final.nodes[node_id].get("cumulative_relevance", 0.0)
        candidate_score = cumulative_likelihood * cumulative_relevance
        G_final.nodes[node_id]["candidate_score"] = candidate_score

    G_final.graph["max_temporal_depth"] = max_depth
    return G_final


# --------------------------------------------
# REPORT BUILDERS
# --------------------------------------------

def _build_reverse_case_study(G: nx.DiGraph) -> Dict:
    """
    Return a dict with:
      - overview: list of nodes ordered by temporal_depth desc (higher first)
      - personas: list of persona dicts with stage metrics & effort
    """
    # Overview
    rows = []
    for n in G.nodes():
        d = G.nodes[n]
        rows.append({
            "id": n,
            "type": d.get("type") or d.get("node_type"),
            "temporal_depth": d.get("temporal_depth", -1),
            "cumulative_likelihood": d.get("cumulative_likelihood", 0.0),
            "cumulative_relevance": d.get("cumulative_relevance", 0.0),
            "candidate_score": d.get("candidate_score", 0.0)
        })
    rows.sort(key=lambda r: r.get("temporal_depth", -1), reverse=True)

    # Personas with stage metrics
    persona_rows = []
    for pid in get_nodes_list_ids(G, "persona", {}):
        if pid not in G:
            continue
        d = G.nodes[pid]
        persona_rows.append({
            "id": pid,
            "involvement": d.get("involvement", 0.0),
            "effort": d.get("effort", 1.0),
            "roi_problem": d.get("roi_problem", 0.0),
            "roi_pain": d.get("roi_pain", 0.0),
            "roi_solution": d.get("roi_solution", 0.0),
            "lift_problem": d.get("lift_problem", 0.0),
            "lift_pain": d.get("lift_pain", 0.0),
            "lift_solution": d.get("lift_solution", 0.0),
            "win_after_problem": d.get("win_after_problem", 0.0),
            "win_after_pain": d.get("win_after_pain", 0.0),
            "win_after_solution": d.get("win_after_solution", 0.0),
        })
    # Sort personas by best ROI (max across stages), then by involvement desc
    def best_roi(p):
        return max(p.get("roi_problem", 0.0), p.get("roi_pain", 0.0), p.get("roi_solution", 0.0))
    persona_rows.sort(key=lambda r: (best_roi(r), r.get("involvement", 0.0)), reverse=True)

    return {
        "overview": rows,
        "personas": persona_rows
    }

def _build_concerns_summary(G: nx.DiGraph) -> List[Dict]:
    """
    Prioritized list (by ROI) of concerns across personas from this graph.
    Here, a 'concern' is the push to move a persona to a stage (Problem/Pain/Solution).
    We rank each persona-stage by ROI (lift / delta_care) where lift>0.
    """
    out = []
    for pid in get_nodes_list_ids(G, "persona", {}):
        if pid not in G:
            continue
        d = G.nodes[pid]
        persona = pid
        # Stage Descriptives
        # Problem stage details
        jobs = _jobs_of_persona(G, persona)
        upstream_pains = []
        pain_metrics = []
        for j in jobs:
            pains = _pains_of_job(G, j)
            if pains not in upstream_pains:
                upstream_pains.extend(pains)
            for p in pains:
                metrics = get_target_nodes_by_source_and_type(G, p, "expressed_as")
                if metrics not in pain_metrics:
                    pain_metrics.extend(metrics)
        problem_desc = {
            "upstream_pains": upstream_pains,
            "pain_metrics": pain_metrics,
            "jobs": jobs
        }

        # Pain stage details
        downstream_pains = []
        pain_metrics_pain = []
        for j in jobs:
            pains = _pains_of_job(G, j)
            for p in pains:
                downstream_pains.append(p)
                metrics = get_target_nodes_by_source_and_type(G, p, "expressed_as")
                if metrics not in pain_metrics_pain:
                    pain_metrics_pain.extend(metrics)
        pain_desc = {
            "downstream_pains": downstream_pains,
            "pain_metrics": pain_metrics_pain
        }

        # Solution stage details
        solving_jobs = []
        capabilities = []
        for j in jobs:
            pains = _pains_of_job(G, j)
            for p in pains:
                sol_jobs = _solving_jobs_of_pain(G, p)
                if sol_jobs not in solving_jobs:
                    solving_jobs.extend(sol_jobs)
                for sj in sol_jobs:
                    if G.nodes[sj].get("type") == "capability":
                        if sj not in capabilities:
                            capabilities.append(sj)
        solution_desc = {
            "solving_jobs": solving_jobs,
            "capabilities": capabilities
        }
        # Concerns Calculations
        # we don't store delta_care explicitly; recompute quick:
        care_problem  = d.get("care", 0.0)
        care_pain     = d.get("win_after_problem", 0.0)  # not care; so compute again by definition:
        # safer: compute delta care from ROI formula -> but ROI already computed against (1 - care_stage)
        # We'll just provide delta as 1 - care_stage by stage using care proxies:
        # Problem stage:
        dc_prob = max(0.0, 1.0 - d.get("care", 0.0))
        dc_pain = None  # will recompute fresh if needed downstream (left None in summary)
        dc_sol  = None
        out.append({
            "persona": persona, "stage": "problem",
            "ROI": d.get("roi_problem", 0.0), "lift": d.get("lift_problem", 0.0),
            "delta_care": dc_prob,
            "description": problem_desc
        })
        out.append({
            "persona": persona, "stage": "pain",
            "ROI": d.get("roi_pain", 0.0), "lift": d.get("lift_pain", 0.0),
            "delta_care": max(0.0, 1.0 - 0.0),  # unknown care at pain here; kept 1.0 as conservative
            "description": pain_desc
        })
        out.append({
            "persona": persona, "stage": "solution",
            "ROI": d.get("roi_solution", 0.0), "lift": d.get("lift_solution", 0.0),
            "delta_care": max(0.0, 1.0 - 0.0),
            "description": solution_desc
        })

    # Filter out non-positive lifts and sort by ROI desc then lift desc
    out = [r for r in out if r["lift"] > 0]
    out.sort(key=lambda r: (r["ROI"], r["lift"]), reverse=True)
    return out


# -----------------------------
# PLAN BUILDER (uses concerns)
# -----------------------------
def _build_marketing_execution_plan(
    G: nx.DiGraph,
    concerns_summary: List[Dict],
    *,
    product_id: str,
    top_concerns: int = 10,
    plays_per_concern: int = 3,
    max_lead_days: Optional[int] = None,
    budget_per_play: Optional[float] = None
) -> List[Dict]:
    """
    Converts persona-stage concerns into executable plays.
    We weight play scores by (1 + ROI) to reflect graph impact.
    """
    plan: List[Dict] = []
    for item in concerns_summary[:top_concerns]:
        persona_id = item["persona"]
        stage = item["stage"]
        roi = float(item.get("ROI", 0.0))
        lift = float(item.get("lift", 0.0))
        delta_care = float(item.get("delta_care", 0.0))

        # Map persona_id string (e.g., "persona:CFO") → Persona alias (e.g., "CFO")
        # If your IDs are differently shaped, tweak this.
        persona = get_node_by_id(G, persona_id)
        if not persona:
            print("Persona node not found for ID:", persona_id)
            continue
        persona_title = persona.get("title") or persona.get("name") or persona_id
        persona_department = persona.get("department") or "General"
        persona_seniority = persona.get("seniority") or "Manager"

        persona_alias = {
            "title": persona_title,
            "department": persona_department,
            "seniority": persona_seniority
        }

        # Weight plays by impact (1 + ROI); you can also factor lift if you like
        roi_mult = max(0.5, 1.0 + roi)

        recs = _materialize_playbook(
            persona_alias, stage,
            product_id=product_id,
            max_lead_days=max_lead_days, budget=budget_per_play,
            top_k=plays_per_concern, roi_multiplier=roi_mult
        )
        if recs is not None:
            print("Playbook generated for", persona_alias, "at stage", stage, ":", recs)
        else:
            print("No playbook generated for", persona_alias, "at stage", stage)
            

        plan.append({
            "persona": persona_id,
            "stage": stage,
            "roi": roi,
            "lift": lift,
            "delta_care": delta_care,
            "plays": recs
        })
    return plan


# --------------------------------------------
# MAIN: generate_rcs
# --------------------------------------------

def generate_rcs(
    G,
    *,
    archetype_id: Optional[str] = None,
    zmot_id: Optional[str] = None,
    engaged_nodes: Optional[List[str]] = None,
    rel_min: float = 0.0,
    max_nodes: int = 5000,
    # ---- Execution knobs ----
    top_concerns: int = 10,
    plays_per_concern: int = 3,
    max_lead_days: Optional[int] = None,
    budget_per_play: Optional[float] = None
) -> Tuple[nx.DiGraph, Dict]:
    """
    Complete RCS Generator:
      - Build archetype→product subgraph (+ZMOT)
      - Compute belief bundle
      - Compute product likelihood baseline & node importance
      - Build first-impact graph, then causal flow
      - Annotate personas with care/effort and stage ROI/lifts
      - Return (graph, report) where report contains:
          * Reverse Case Study (temporally ordered nodes + persona stage metrics)
          * Graph Concerns Summary (persona-stage items prioritized by ROI)
          * belief_bundle (for reference)
    """
    if archetype_id is None:
        G_final = G.copy()
    else:
        print("Generating RCS for archetype", archetype_id)
        G_a = get_node_subgraph_to_product(G, archetype_id)
        G_zmot = augment_subgraph_with_zmots(G, G_a, archetype_id)
        if G_a.number_of_nodes() > 0:
            if zmot_id is not None and zmot_id in G_zmot:
                print("Pruning to ZMOT node", zmot_id)
                _ = get_node_by_id(G_zmot, zmot_id)
                G_pruned_for_zmot = get_node_subgraph_to_product(G_zmot, zmot_id)
                G_final = G_pruned_for_zmot
            else:
                G_final = G_zmot
        else:
            belief = {"reason": "empty_archetype_subgraph", "archetype_id": archetype_id}
            return G_a, {"belief_bundle": belief, "reverse_case_study": {}, "graph_concerns_summary": []}
        print("----------------------------")

    print("At first order arch+zmot graph:")
    
    

    if archetype_id is not None and archetype_id in G_final:
        graph_win_likelihood = compute_product_likelihood(G_final, archetype_id)
    else:
        graph_win_likelihood = 1.0

    product_id = get_product_id_from_subgraph(G_final)

    occurrance_inputs = []
    if archetype_id is not None:
        archetype_node = get_node_by_id(G_final, archetype_id)
        if archetype_node:
            occurrance_inputs.append({"id": archetype_id, "occurrence": 1})
    if zmot_id is not None:
        zmot_node = get_node_by_id(G_final, zmot_id)
        if zmot_node:
            occurrance_inputs.append({"id": zmot_id, "occurrence": 1})
    if engaged_nodes is not None:
        for n in engaged_nodes:
            n_id = n.get("id")
            n_occurrence = n.get("occurrence", 1)
            if n_id in G_final and get_node_by_id(G_final, n_id):
                occurrance_inputs.append({"id": n_id, "occurrence": n_occurrence})
    print("Creating graph with Occurrance inputs:", occurrance_inputs)
    G_max = _construct_first_impact_graph(G_final, occurrance_inputs)
    print("First impact graph done - constructing causal flow")
    G_causal = _construct_causal_flow(G_max, product_id)


    if any(G_causal.nodes[n].get("cumulative_likelihood") is None or G_causal.nodes[n]["cumulative_likelihood"] <= 0.0 for n in G_causal.nodes()):
        print("Node with zero likelihood in causal graph:", n, " of type: ", G_causal.nodes[n].get("node_type"))
    else:
        print("All nodes have positive likelihood in causal graph")

    

    G_causal.graph["win_likelihood"] = graph_win_likelihood
    _set_node_importance(G_causal, archetype_id)

    # ------- Build the requested report -------
    reverse_case_study = _build_reverse_case_study(G_causal)
    concerns_summary = _build_concerns_summary(G_causal)

    # ------- Build marketing execution plan -------
    marketing_execution_plan = _build_marketing_execution_plan(
        G_causal,concerns_summary,
        product_id=product_id,
        top_concerns=top_concerns,
        plays_per_concern=plays_per_concern,
        max_lead_days=max_lead_days,
        budget_per_play=budget_per_play
    )


    report = {
        "reverse_case_study": reverse_case_study,
        "graph_concerns_summary": concerns_summary,
        "marketing_execution_plan": marketing_execution_plan,

    }

    return G_causal, report



