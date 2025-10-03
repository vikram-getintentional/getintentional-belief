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

def _update_relevant_nodes(relevant_nodes: dict, node_id: str, depth: int) -> None:
    if node_id not in relevant_nodes:
        relevant_nodes[node_id] = {"temporal_depth": depth}

def _update_relevant_pain_family(G, relevant_nodes: dict, pain_id: str, depth: int) -> None:
    _update_relevant_nodes(relevant_nodes, pain_id, depth)
    perceived_metrics = get_target_nodes_by_source_and_type(G, pain_id, "expressed_as")
    for metric_id in perceived_metrics:
        _update_relevant_nodes(relevant_nodes, metric_id, depth)

def remove_cycles(G: nx.DiGraph) -> nx.DiGraph:
    """
    Remove edges that participate in cycles until the graph is a DAG.
    Returns a copy of the input graph with cycles removed.
    """
    G_dag = G.copy()
    try:
        while not nx.is_directed_acyclic_graph(G_dag):
            # Find one cycle
            cycle = list(nx.simple_cycles(G_dag))
            if not cycle:
                break
            # Remove one edge from the first cycle found
            cycle_edges = list(zip(cycle[0], cycle[0][1:] + [cycle[0][0]]))
            # Remove the edge with the lowest likelihood to minimize impact
            cycle_edges.sort(key=lambda e: G_dag.edges[e].get("likelihood", 0.0))
            edge_to_remove = cycle_edges[0]
            G_dag.remove_edge(*edge_to_remove)
    except Exception as e:
        print(f"Error removing cycles: {e}")
    return G_dag
def _set_temporal_depths(G: nx.DiGraph) -> None:
    product_node_id = get_product_id_from_subgraph(G)
    if not product_node_id:
        return
    
    # Set temporal depth for all nodes
    for n in G.nodes():
        temporal_depth = nx.shortest_path_length(G, source=product_node_id, target=n)
        G.nodes[n]["temporal_depth"] = temporal_depth

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
    path: List[str]
) -> float:
    p = 1.0
    for u, v in zip(path, path[1:]):
        if G.nodes[u].get("cumulative_likelihood") is not None:
            p = float(G.nodes[u].get("cumulative_likelihood", 1.0))
        elif G.nodes[u].get("occurrence") is not None:
            p = G.nodes[u]["occurrence"]
        else:
            p *= get_edge_attribute(G, u, v, "likelihood") or 1.0
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

def apply_zmot_boosts_to_pain_trigger_if_applicable(G: nx.DiGraph, occurred_nodes: List[Dict]) -> nx.DiGraph:
    archetype_id = None
    zmot_id = None
    for occurred_node in occurred_nodes:
        node_id = occurred_node.get("id")
        occurrence = occurred_node.get("occurrence", 1)
        if node_id not in G:
            continue
        node = get_node_by_id(G, node_id)
        if not node:
            continue
        node_type = node.get("type")
        if node_type == "archetype":
            archetype_id = node_id
        elif node_type == "zmot_event":
            zmot_id = node_id
        
        # set node likelihood if needed
        node["cumulative_likelihood"] = occurrence

    product_id = get_product_id_from_subgraph(G)
    all_nodes = set()

    if archetype_id is not None:
        first_impact_pain_trigger_ids = get_source_nodes_by_target_and_type(G, archetype_id, "prevalent_in")
    else:
        first_impact_pain_trigger_ids = get_nodes_list_ids(G, "pain_trigger", {})
    for pain_trigger_id in first_impact_pain_trigger_ids:
        pain_trigger_node = get_node_by_id(G, pain_trigger_id)
        if not pain_trigger_node:
            continue
        zmot_boost = 0.0
        if zmot_id is not None:
            zmot_node = get_node_by_id(G, zmot_id)
            if not zmot_node:
                continue
            zmot_boost = get_edge_attribute(G, pain_trigger_id, zmot_id, "boost") or 0.0
            

        baseline_pain_trigger_likelihood = get_edge_attribute(G, pain_trigger_id, archetype_id, "likelihood") or 0.0
        
        pain_trigger_likelihood = 1-(1-baseline_pain_trigger_likelihood)*(1-zmot_boost)
        
        pain_trigger_node["cumulative_likelihood"] = pain_trigger_likelihood
    return G

def compute_product_likelihood(
    G_orig: nx.DiGraph,
    occurred_nodes: list,
    *,
    product_type: str = "product",
    edge_attr_prob: str = "likelihood",
    max_depth: int = 20
) -> float:
    # Find product node
    product_id = None
    for n, d in G_orig.nodes(data=True):
        if d.get("type") == product_type:
            product_id = n
            break
    if not product_id:
        print("No product node found in graph")
        return 0.0
    # Find archetype ID if exists
    archetype_id = None
    for occurred_node in occurred_nodes:
        node_id = occurred_node.get("id")
        node = get_node_by_id(G_orig, node_id)
        G_orig.nodes[node_id]["occurrence"] = occurred_node.get("occurrence", 1.0)
        G_orig.nodes[node_id]["cumulative_likelihood"] = occurred_node.get("occurrence", 1.0)
        node_type = node.get("type") if node else None
        if node_type == "archetype":
            archetype_id = node.get("id")
            break
    if archetype_id is None:
        archetype_nodes = get_nodes_list_ids(G_orig, "archetype", {})
    else:
        archetype_nodes = [archetype_id]
    # Apply ZMOT boosts if applicable
    G_orig = apply_zmot_boosts_to_pain_trigger_if_applicable(G_orig, occurred_nodes)
        
    G_rev = G_orig.reverse(copy=True)
    

    all_path_probs = []
    
    for archetype_node in archetype_nodes:
        if archetype_node not in G_rev:
            continue
        # Enumerate all simple paths from node_id to product_id
        try:
            paths = list(nx.all_simple_paths(G_rev, source=archetype_node, target=product_id, cutoff=max_depth))
        except nx.NetworkXNoPath:
            continue
        # Compute probability for each path
        paths_with_p = []
        for path in paths:
            p = _path_probability(G_rev, path)
            paths_with_p.append((path, p))
        # Adjust for overlap
        adjusted_probs = _greedy_overlap_aware_selection(paths_with_p, overlap_penalty=0.5)
        all_path_probs.extend(adjusted_probs)
    # Aggregate using noisy-or
    max_prob = max(all_path_probs) if all_path_probs else 0.0
    return max_prob


def _infer_node_importance(G: nx.DiGraph, measuring_node, occurred_nodes: List[Dict[str, float]]) -> None:
    """
    Importance = P(win | node active) - P(win | node not active).
    """
    importance_active = 1.0
    importance_not_active = 0.0
    if measuring_node not in G:
        return G
    G.nodes[measuring_node]["importance"] = 0.0
    
    G_x = G.copy()
    active_occurred_nodes = occurred_nodes + [{"id": measuring_node, "occurrence": importance_active}]
    win_if_active = compute_product_likelihood(G_x, active_occurred_nodes)

    G_y = G.copy()
    inactive_occurred_nodes = occurred_nodes + [{"id": measuring_node, "occurrence": importance_not_active}]
    win_if_not_active = compute_product_likelihood(G_y, inactive_occurred_nodes)

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

def _solving_jobs_or_caps_of_pain(G: nx.DiGraph, pain_id: str) -> List[str]:
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
        if not G.nodes[j].get("importance") or G.nodes[j].get("importance") <= 0.0:
            _infer_node_importance(G, j)
            p_job = G.nodes[j].get("importance", 0.0)
        if p_job <= 0.0 or p_job is None:
            p_job = 0.0
        rel = _rel_job_to_persona(G, j, persona_id)
        
        terms.append(p_job * rel)
        
    return max(terms) if terms else 0.0

def _care_pain(G: nx.DiGraph, persona_id: str) -> float:
    """
    Purpose: Measures how much a persona "cares" about the pain stage.
    Calculation: For each job of the persona:
        Multiply job importance × job relevance to persona × max likelihood of feeling any pain in that job.
        Take the maximum across all jobs.
    Interpretation:
        High if the persona has a job that's both important and relevant, and is likely to experience pain in that job.
        Focuses on the pain experience.
    """
    jobs = _jobs_of_persona(G, persona_id)
    if not jobs:
        return 0.0
    terms = []
    for j in jobs:
        if not G.nodes[j].get("importance") or G.nodes[j].get("importance") <= 0.0:
            _infer_node_importance(G, j)
            p_own = G.nodes[j].get("importance", 0.0)
        else:
            p_own = G.nodes[j].get("importance", 0.0)
        if p_own is None or p_own <= 0.0:
            p_own = _infer_node_importance(G, j)
        if p_own <= 0.0:
            p_own = 0.0
        own = p_own * _rel_job_to_persona(G, j, persona_id)
        pains = _pains_of_job(G, j)
        felt = max([_edge_p(G, p, j, "likelihood", 0.0) for p in pains]) if pains else 0.0
        terms.append(own * felt)
    return max(terms) if terms else 0.0

def _care_solution(G: nx.DiGraph, persona_id: str) -> float:
    """
    Purpose: Measures how much a persona "cares" about the solution stage.
    Calculation:
        For each job of the persona:
        Multiply job importance × job relevance × max likelihood of feeling pain × max likelihood that pain is solved by any job/capability.
        Take the maximum across all jobs.
    Interpretation:
        High if the persona has a job that's important, relevant, likely to experience pain, and that pain is likely to be solved.
        Focuses on the pain being solved.
    
    """
    jobs = _jobs_of_persona(G, persona_id)
    if not jobs:
        return 0.0
    terms = []
    for j in jobs:
        if not G.nodes[j].get("importance") or G.nodes[j].get("importance") <= 0.0:
            _infer_node_importance(G, j)
            p_own = G.nodes[j].get("importance", 0.0)
        else:
            p_own = G.nodes[j].get("importance", 0.0)
        if p_own is None or p_own <= 0.0:
            p_own = _infer_node_importance(G, j)
        if p_own <= 0.0:
            p_own = 0.0
        own = p_own * _rel_job_to_persona(G, j, persona_id)
        pains = _pains_of_job(G, j)
        if not pains:
            continue
        felt_j = max([_edge_p(G, p, j, "likelihood", 0.0) for p in pains]) if pains else 0.0
        res_terms = []
        for p in pains:
            solving_jobs = _solving_jobs_or_caps_of_pain(G, p)
            res_terms.append(max([_edge_p(G, js, p, "likelihood", 0.0) for js in solving_jobs]) if solving_jobs else 0.0)
        p_res = max(res_terms) if res_terms else 0.0
        terms.append(own * felt_j * p_res)
    return max(terms) if terms else 0.0

def _persona_effort(G: nx.DiGraph, persona_id: str) -> float:
    """
    Purpose: Measures the min "effort required" to push this persona to the next stage.
    Calculation:
        For each job of the persona:
        Multiply job relevance × job importance × max over pains of (pain importance × max solution fit).
        Take the maximum across all jobs.
    Interpretation:
        High if the persona is connected to jobs that are important and relevant, and those jobs are linked to pains that are important and can be solved.
        Captures the full causal chain: persona → job → pain → solution.
        More holistic than the stage-specific care metrics.
    """
    care_problem = _care_problem(G, persona_id)
    care_pain = _care_pain(G, persona_id)
    care_solution = _care_solution(G, persona_id)
    net_care = max(care_problem, care_pain, care_solution)
    effort = 1.0 - net_care
    return effort

def eigenvector_centrality(G: nx.DiGraph) -> float:
    try:
        ec = nx.eigenvector_centrality(G, max_iter=1000, weight="likelihood")
    except Exception as e:
        print(f"Error computing eigenvector centrality: {e}")
        ec = {}
    return ec

def _persona_involvement(G: nx.DiGraph, persona_id: str, ec: Dict[str, float]) -> float:
    """
    Purpose: Measures overall "involvement"—how deeply a persona is entrenched in the causal graph.
    Calculation:
        Build Eigenvector Centrality (EC) for the graph node.
        Get EC Score for persona Node.
    Interpretation:
        High if the persona is connected to nodes that are in critical paths to resolution.
    """
    """jobs = _jobs_of_persona(G, persona_id)
    persona_involvement_degree = []
    for u,v,data in G.edges(data=True):
        likelihood = data.get("likelihood", 0.0)
        ln_likelihood = -math.log(max(likelihood, 1e-9)) if likelihood > 0 else 1e-9
        data["ln_likelihood"] = ln_likelihood
        
        
    bc = nx.betweenness_centrality(G, weight="ln_likelihood", normalized=True)
    if not jobs:
        return 0.0
    for j in jobs:
        # get betweenness centrality of job j
        persona_likelihood_for_job = _rel_job_to_persona(G, j, persona_id)
        persona_job_betweenness = bc.get(j, 0.0) * persona_likelihood_for_job
        persona_involvement_degree.append(persona_job_betweenness)
    involvement = sum(persona_involvement_degree)
    """
    involvement = ec.get(persona_id, 0.0) if ec else 0.0
    
    return involvement



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

def _persona_stage_lifts_and_roi(G: nx.DiGraph, occurred_nodes: List[Dict[str, float]], persona_id: str) -> Dict:
    baseline = compute_product_likelihood(G, occurred_nodes=occurred_nodes)

    care_problem  = _care_problem(G, persona_id)
    care_pain     = _care_pain(G, persona_id)
    care_solution = _care_solution(G, persona_id)
    ec = eigenvector_centrality(G)
    involvement   = _persona_involvement(G, persona_id, ec)

    def clamp_problem(G0: nx.DiGraph, persona_id: str, occurred_nodes: List[Dict[str, float]]) -> nx.DiGraph:
        Gc = G0.copy()
        oc_prob = occurred_nodes.copy()
        for j in _jobs_of_persona(Gc, persona_id):
            oc_prob.append({"id": j, "occurrence": 1.0})
        return Gc, oc_prob

    def clamp_pain(G0: nx.DiGraph, persona_id: str, occurred_nodes: List[Dict[str, float]]) -> nx.DiGraph:
        Gc, oc_prob = clamp_problem(G0, persona_id, occurred_nodes)
        oc_pain = occurred_nodes.copy()
        for j in _jobs_of_persona(Gc, persona_id):
            for p in _pains_of_job(Gc, j):
                oc_pain.append({"id": p, "occurrence": 1.0})
        return Gc, oc_pain

    def clamp_solution(G0: nx.DiGraph, persona_id: str, occurred_nodes: List[Dict[str, float]]) -> nx.DiGraph:
        Gc, oc_pain = clamp_pain(G0, persona_id, occurred_nodes)
        oc_sol = occurred_nodes.copy()
        for j in _jobs_of_persona(Gc, persona_id):
            for p in _pains_of_job(Gc, j):
                for js in _solving_jobs_or_caps_of_pain(Gc, p):
                    oc_sol.append({"id": js, "occurrence": 1.0})
        
        return Gc, oc_sol
    
    
    print("Baseline win prob:", baseline, "with occurred nodes:", occurred_nodes)
    G_prob, oc_prob = clamp_problem(G, persona_id, occurred_nodes)
    G_pain, oc_pain = clamp_pain(G, persona_id, occurred_nodes)
    G_sol, oc_sol = clamp_solution(G, persona_id, occurred_nodes)

    win_prob = compute_product_likelihood(G_prob, occurred_nodes=oc_prob)
    print("Win prob after problem stage clamp:", win_prob)
    win_pain = compute_product_likelihood(G_pain, occurred_nodes=oc_pain)
    print("Win prob after pain stage clamp:", win_pain)
    win_sol = compute_product_likelihood(G_sol, occurred_nodes=oc_sol)
    print("Win prob after solution stage clamp:", win_sol)
        
    dcare_prob = max(0.0, 1.0 - care_problem)    
    dcare_pain = max(0.0, 1.0 - care_pain)
    dcare_sol = max(0.0, 1.0 - care_solution)
    
    lift_prob = max(0.0, win_prob - baseline)
    lift_pain = max(0.0, win_pain - win_prob)
    lift_sol = max(0.0, win_sol - win_pain)
    

    ROI_prob = safe_roi(lift_prob, dcare_prob)
    ROI_pain = safe_roi(lift_pain, dcare_pain)
    ROI_sol = safe_roi(lift_sol, dcare_sol)
    

    return {
        "involvement": involvement,
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

def _set_node_importance(G: nx.DiGraph, occurrance_nodes: List[Dict[str, float]]) -> nx.DiGraph:
    
    persona_node_ids = get_nodes_list_ids(G, "persona", {})
    for persona_id in persona_node_ids:
        if persona_id not in G:
            continue
        # Set Persona node importance via direct inference
        _infer_node_importance(G, persona_id, occurrance_nodes)


        # Persona expected importance via jobs (overlap-safe)
        jobs = _jobs_of_persona(G, persona_id)
        imp_terms = []
        for j in jobs:
            if not G.nodes[j].get("importance") or G.nodes[j].get("importance") <= 0.0:
                _infer_node_importance(G, j, occurrance_nodes)
            job_importance = G.nodes[j].get("importance", 0.0)
            if not job_importance or job_importance <= 0.0:
                _infer_node_importance(G, j, occurrance_nodes)
            p_job = G.nodes[j].get("cumulative_likelihood", 0.0)
            rel   = _rel_job_to_persona(G, j, persona_id)
            care_term = p_job * rel
            imp_terms.append(care_term * _clip(job_importance.get(j, 0.0)))
        persona_expected_importance = max(imp_terms) if imp_terms else 0.0

        care_problem = _care_problem(G, persona_id)
        effort = _persona_effort(G, persona_id)

        stage_bundle = _persona_stage_lifts_and_roi(G, occurrance_nodes, persona_id)

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
def _construct_first_impact_graph(G_final: nx.DiGraph, occurred_nodes: List[Dict[str, float]]) -> nx.DiGraph:
    """
    Get the downstream likelihoods for all nodes leading to product given input node_ids have occurred.
    """
    archetype_id = None
    zmot_id = None
    for occurred_node in occurred_nodes:
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

    product_id = get_product_id_from_subgraph(G_final)
    all_nodes = set()

    if archetype_id is not None:
        first_impact_pain_trigger_ids = get_source_nodes_by_target_and_type(G_final, archetype_id, "prevalent_in")
    else:
        first_impact_pain_trigger_ids = get_nodes_list_ids(G_final, "pain_trigger", {})
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
            

        baseline_pain_trigger_likelihood = get_edge_attribute(G_final, pain_trigger_id, archetype_id, "likelihood") or 0.0
        
        
        pain_trigger_likelihood = 1-(1-baseline_pain_trigger_likelihood)*(1-zmot_boost)
        
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
    # Sort personas by involvement desc then by best ROI desc
    def best_roi(p):
        return max(p.get("roi_problem", 0.0), p.get("roi_pain", 0.0), p.get("roi_solution", 0.0))
    persona_rows.sort(key=lambda r: (r.get("involvement", 0.0), best_roi(r)), reverse=True)

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
                sol_jobs = _solving_jobs_or_caps_of_pain(G, p)
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
    archetype_node_data: Dict,
    top_concerns: int = 10,
    plays_per_concern: int = 3,
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
            archetype=archetype_node_data,
            top_k=plays_per_concern, roi_multiplier=roi_mult
        )
        if recs is not None:
            print("Playbook generated for", persona_alias, "at stage", stage, ":", recs)
        else:
            print("No playbook generated for", persona_alias, "at stage", stage)
            

        plan.append({
            "roi": roi,
            "lift": lift,
            "delta_care": delta_care,
            "plays": recs
        })
        plan = sorted(plan, key=lambda p: (p["roi"], p["lift"]), reverse=True)
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
    archetype_node_data = {
        "industry": None,
        "revenue_range": None,
        "employee_range": None,
        "geography": None,
    }
    if archetype_id is None:
        G_final = G.copy()
    else:
        print("Generating RCS for archetype", archetype_id)
        archetype_node = get_node_by_id(G, archetype_id)
        if not archetype_node:
            print("Archetype node not found:", archetype_id)
        archetype_node_data = {
            "industry": archetype_node.get("industry"),
            "revenue_range": archetype_node.get("revenue_range"),
            "employee_range": archetype_node.get("employee_range"),
            "geography": archetype_node.get("geography"),
        }
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
    graph_win_likelihood = compute_product_likelihood(G_final, occurred_nodes=occurrance_inputs)
    G_final.graph["win_likelihood"] = graph_win_likelihood
    
    G_max = _construct_first_impact_graph(G_final, occurred_nodes=occurrance_inputs)
    print("First impact graph done - constructing causal flow")
    G_causal = _construct_causal_flow(G_max, product_id)


    zero_likelihood_nodes = [
        n for n in G_causal.nodes()
        if G_causal.nodes[n].get("cumulative_likelihood") is None or G_causal.nodes[n]["cumulative_likelihood"] <= 0.0
    ]
    if zero_likelihood_nodes:
        for n in zero_likelihood_nodes:
            print("Node with zero likelihood in causal graph:", n, " of type: ", G_causal.nodes[n].get("node_type"))
    else:
        print("All nodes have positive likelihood in causal graph")

    

    G_causal.graph["win_likelihood"] = graph_win_likelihood
    g_causal_win = compute_product_likelihood(G_causal, occurred_nodes=occurrance_inputs)
    g_max_win = compute_product_likelihood(G_max, occurred_nodes=occurrance_inputs)
    print("G_causal win likelihood:", g_causal_win,"\nG_max win likelihood:", g_max_win,"\nG_final win likelihood:", graph_win_likelihood)

    if nx.is_directed_acyclic_graph(G_final):
        print("Final graph is a DAG")
    else:
        print("Final graph has cycles - removing cycles")
        G_final = remove_cycles(G_final)
    _set_node_importance(G_final, occurrance_inputs)
    print("Node importance set on full graph")

    _set_temporal_depths(G_final)
    print("Temporal depths set on full graph")

    # ------- Build the requested report -------
    reverse_case_study = _build_reverse_case_study(G_final)
    print("Reverse case study built on G_final")
    concerns_summary = _build_concerns_summary(G_final)

    # ------- Build marketing execution plan -------
    marketing_execution_plan = _build_marketing_execution_plan(
        G_final,concerns_summary,
        product_id=product_id,
        archetype_node_data=archetype_node_data,
        top_concerns=top_concerns,
        plays_per_concern=plays_per_concern,
        
    )

    for plan in marketing_execution_plan:
        if plan.get("plays") is None or len(plan.get("plays")) <= 0:
            plan["plays"] = [{"note": "No existing plays match. Generate a new play."}]


    report = {
        "reverse_case_study": reverse_case_study,
        "graph_concerns_summary": concerns_summary,
        "marketing_execution_plan": marketing_execution_plan,

    }

    return G_final, report



