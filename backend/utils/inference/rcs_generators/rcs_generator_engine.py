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
import networkx as nx

from backend.utils.graph_base.network_graph import get_edge_attribute, get_edge_weight, get_node_by_id, get_node_subgraph_to_product, get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.graph_base import schema
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.rcs_generators.beliefs.machine import ProbBeliefMachine #BeliefMachine logic


# ----------------------------
# Utilities
# ----------------------------

def _ensure_node(G_src: nx.MultiDiGraph, G_dst: nx.MultiDiGraph, n: str) -> None:
    if n not in G_dst:
        G_dst.add_node(n, **G_src.nodes[n])


def _set_temporal_depth_if_absent(G_dst: nx.MultiDiGraph, n: str, depth: float) -> None:
    if "temporal_depth" not in G_dst.nodes[n]:
        G_dst.nodes[n]["temporal_depth"] = depth

def _update_relevant_nodes(relevant_nodes: dict, node_id: str, depth: int) -> None:
    if node_id not in relevant_nodes:
        # Update the node's depth information
        relevant_nodes[node_id] = {"temporal_depth": depth}

def _update_relevant_pain_family(G, relevant_nodes: dict, pain_id: str, depth: int) -> None:
    """
    Update the relevant nodes list with a pain and its family (job, persona).
    """
    _update_relevant_nodes(relevant_nodes, pain_id, depth)
    #pain_trigger_ids = get_target_nodes_by_source_and_type(G, pain_id, "triggered_by")
    #for trigger_id in pain_trigger_ids:
        #_update_relevant_nodes(relevant_nodes, trigger_id, depth)
    perceived_metrics = get_target_nodes_by_source_and_type(G, pain_id, "expressed_as")
    for metric_id in perceived_metrics:
        _update_relevant_nodes(relevant_nodes, metric_id, depth)


#------------------------------
# Util to convert graph to matrix
# ------------------------------
import numpy as np

def build_persona_adjacency_from_subgraph(G_a: nx.MultiDiGraph) -> tuple[np.ndarray, list[str]]:
    """
    Personas influence other personas when they are connected by a pain that one 'feels' (via their job)
    and the other 'solves' (via their job).
      u <-owned_by- j_u <-felt_in- p -solves-> j_v -owned_by-> v

    Edge weight u->v = max over shared pains p of:
        w(j_u->u) * w(p->j_u) * w(j_v->p) * w(j_v->v)

    Returns: (A, personas_in_order)
      A: np.ndarray n x n
      personas_in_order: list[str]
    """
    # personas present in G_a
    persona_ids = get_nodes_list_ids(G_a, "persona", {})
    if not persona_ids:
        return np.zeros((0, 0), dtype=float), []
    persona_ids.sort()  # stable order for reproducibility
    adj = np.zeros((len(persona_ids), len(persona_ids)), dtype=float)
    idx = {p: i for i, p in enumerate(persona_ids)}
    for persona_id in persona_ids:
        source_job_ids = get_source_nodes_by_target_and_type(G_a, persona_id, "performed_by")
        if not source_job_ids:
            print(f"Warning: Persona {persona_id} has no jobs performed_by (Source Jobs - Ln 102), skipping.")
            continue
        for source_job_id in source_job_ids:
            orig_job_likelihood = get_edge_attribute(G_a, source_job_id, persona_id, "likelihood")
            felt_in_pain_ids = get_source_nodes_by_target_and_type(G_a, source_job_id, "felt_in")
            if not felt_in_pain_ids:
                print(f"Warning: Job {source_job_id} performed_by {persona_id} has no pains felt_in, skipping.")
                continue
            for pain_id in felt_in_pain_ids:
                felt_pain_likelihood = get_edge_attribute(G_a, pain_id, source_job_id, "likelihood")
                if felt_pain_likelihood <= 0:
                    print(f"Warning: Pain {pain_id} felt_in by job {source_job_id} performed_by {persona_id} has non-positive likelihood, skipping.")
                    continue
                solving_job_ids = get_source_nodes_by_target_and_type(G_a, pain_id, "solves")
                if not solving_job_ids:
                    print(f"Warning: Pain {pain_id} has no jobs solving it, skipping.")
                    continue
                for solving_job_id in solving_job_ids:
                    solving_job_node = get_node_by_id(G_a, solving_job_id)
                    if not solving_job_node:
                        continue
                    if solving_job_node.get("type") != "job":
                        continue
                    solving_job_likelihood = get_edge_attribute(G_a, solving_job_id, pain_id, "likelihood")
                    if solving_job_likelihood <= 0:
                        print(f"Warning: Job {solving_job_id} solves pain {pain_id} with non-positive likelihood, skipping.")
                        continue
                    target_persona_ids = get_target_nodes_by_source_and_type(G_a, solving_job_id, "performed_by")
                    if not target_persona_ids:
                        print(f"Warning: Job {solving_job_id} has no personas performed_by (Solving Job - ln 127), skipping.")
                        continue
                    for target_persona_id in target_persona_ids:
                        if target_persona_id == persona_id:

                            continue
                        target_job_likelihood = get_edge_attribute(G_a, solving_job_id, target_persona_id, "likelihood")
                        if target_job_likelihood <= 0:
                            print(f"Warning: Persona {target_persona_id} performed_by job {solving_job_id} has non-positive likelihood, skipping.")
                            continue
                        # Update adjacency matrix A[u, v]
                        iu = idx[persona_id]
                        iv = idx[target_persona_id]
                        adj[iu, iv] = orig_job_likelihood * felt_pain_likelihood * solving_job_likelihood * target_job_likelihood

    return adj, persona_ids




# ----------------------------
# 1) Build archetype subgraph with temporal depth
# ----------------------------

def build_archetype_subgraph_with_temporal_depth(
    G: nx.MultiDiGraph,
    archetype_id: str,
    *,
    zmot_id: Optional[str] = None,
    rel_min: float = 0.0,
    max_nodes: int = 5000,
) -> Tuple[nx.MultiDiGraph, Dict]:
    """
    First-encounter temporal depth assignment. If a node is already in the subgraph, do not revisit or increment.

    Depth convention:
      - Archetype/ZMOT/Triggers: depth 0
      - Pains from triggers:     depth 1
      - Jobs solving a pain:     depth of that pain (d)
      - Personas owning a job:   depth d+1
      - Pains felt in a job:     depth d+1 (and enqueued)
    """
    print("Starting RCS generation for archetype")
    if archetype_id not in G:
        raise ValueError(f"archetype_id {archetype_id!r} not in graph")

    product_id = get_product_id_from_subgraph(G)
    G_a = nx.MultiDiGraph()
    audit = {
        "seed_triggers": [],
        "visited_pains": [],
        "visited_jobs": [],
        "added_personas": [],
        "node_count": 0,
        "edge_count": 0,
        "exhausted_cache": False,
        "hit_max_nodes": False,
    }
    relevant_nodes = {}
    d = 0
    print("Seeded archetype and zmot nodes")
    # Find triggers linked to archetype or ZMOT
    seed_triggers: Set[str] = set()

    if not archetype_id:
        print("No archetype ID provided, skipping archetype subgraph build")
        return []

    _update_relevant_nodes(relevant_nodes, archetype_id, d)
    print("Adding archetype node with depth", archetype_id, d)
    

    pain_trigger_ids = get_source_nodes_by_target_and_type(G, archetype_id, "prevalent_in")
    seed_triggers.update(pain_trigger_ids)

    # Seed pains from triggers (depth=1)
    pain_q: deque[Tuple[str, int]] = deque()

    for t in seed_triggers:
        # pains → trigger
        pain_ids = get_source_nodes_by_target_and_type(G, t, "triggered_by")
        for p in pain_ids:
            if p in relevant_nodes:
                continue
            # Ensure pain node is terminal
            felt_in_jobs = get_target_nodes_by_source_and_type(G, p, "felt_in")
            pain_node = get_node_by_id(G, p)
            if not pain_node:
                print(f"Pain Node {p!r} not found. Diagnose this")
                continue
            if not felt_in_jobs:
                print(f"Pain {p!r} is terminal, adding to graph")
                if p not in relevant_nodes:
                    pain_q.append((p, d))
                    print("Queued terminal pain nodes", p, "with pre-depth", d)

                
    # At this point archetype ID is added to relevant_nodes. Only terminal pains are queued with d=0.
    print("Seeded pains from triggers. Starting graph building loop...")
    # Traversal loop (first-encounter rule)
    while pain_q and len(relevant_nodes) < max_nodes:
        
        pain, d = pain_q.popleft()
        d+=1
        # Last pain is dequeued with its original pre-depth. So we increment for analysis. 
        # For fo pain pre-d = 0, now d = 1

        # Skip if pain already captured in relevant nodes
        if pain in relevant_nodes:
            print(f"Pain {pain!r} already in relevant nodes, skipping")
            continue
        _update_relevant_pain_family(G, relevant_nodes, pain, d)
        # Pain is now added with depth d. For fo pain d = 1.
        d+=1
        # For fo pain, d = 2 now.
        cum_relevance = get_cumulative_relevance_data(product_id, pain)

        # Relevance gate (optional)
        if rel_min > 0 and cum_relevance < rel_min:
            print("Pain failed relevance threshold, skipping")
            continue
        
        # Jobs solving this pain: job --solves--> pain (job predecessors)
        job_or_cap_ids = get_source_nodes_by_target_and_type(G, pain, "solves")
        if not job_or_cap_ids:
            print(f"No jobs or caps found solving pain {pain!r}, skipping")
            continue
        for node_id in job_or_cap_ids:
            node = get_node_by_id(G, node_id)
            if node is None:
                print(f"Job or Cap Node {node_id!r} not found. Diagnose this")
                continue
        
            if node_id not in relevant_nodes:
                _update_relevant_nodes(relevant_nodes, node_id, d)  
                # d is already +1 from pain above. So for next order job/ cap -> d=2

            if node.get("type") == "capability":
                product_node_id = get_source_nodes_by_target_and_type(G, node_id, "offers")
                if not product_node_id:
                    print(f"No product node found for capability {node_id!r}, skipping")
                    continue
                if product_node_id[0] != product_id:
                    print("Some error in product id retrieval...")
                    continue
                if product_node_id[0] not in relevant_nodes:
                    _update_relevant_nodes(relevant_nodes, product_node_id[0], d+1)
                    # d is still 2. for product node added at d=3.
                continue  # skip to next pain

            # Personas owning this job: job --owned_by--> persona
            persona_ids = get_target_nodes_by_source_and_type(G, node_id, "performed_by")
            if not persona_ids:
                print(f"No personas found owning job {node_id!r}, skipping")
                continue
            for per in persona_ids:
                if per not in relevant_nodes:
                    _update_relevant_nodes(relevant_nodes, per, d)
                    # Adding persona at same depth as job. For fo = 2. 

            # Additional pains felt in this job: pain2 --felt_in--> job
            felt_in_pain_ids = get_source_nodes_by_target_and_type(G, node_id, "felt_in")
            if not felt_in_pain_ids:
                print(f"No pains felt in job {node_id!r}, skipping")
                continue
            for felt_pain in felt_in_pain_ids:
                if felt_pain not in relevant_nodes:
                    pain_q.append((felt_pain, d))
                    # Queued felt_in pain with d = 2. 
    
    if zmot_id:
        if zmot_id not in G:
            print(f"ZMOT node {zmot_id!r} not found in graph, skipping")
        zmot_node = get_node_by_id(G, zmot_id)
        if not zmot_node:
            print(f"ZMOT node {zmot_id!r} not found in graph, skipping")
        _update_relevant_nodes(relevant_nodes, zmot_id, 0)

    final_nodes_set = set({n for n, d in relevant_nodes.items()})
    G_a = G.subgraph(final_nodes_set)
    for n in G_a.nodes:
        G_a.nodes[n]["depth"] = relevant_nodes[n]["temporal_depth"]

    # Logic to add Belief States
    A_persona, persona_order = build_persona_adjacency_from_subgraph(G_a)
    if A_persona.shape[0] > 0:
        # uniform relevance for now
        r = np.ones(A_persona.shape[0], dtype=float)
        bm = ProbBeliefMachine(A=A_persona, r=r, beta=0.35)

        belief_bundle = {
            "persona_order": persona_order,       # needed to map back to node ids
            "activation_SOL": (bm.activation("SOL")).tolist(),
            "activation_PR":  (bm.activation("PR")).tolist(),
            "expected_score_SOL": bm.expected_score("SOL"),
            "expected_score_PR":  bm.expected_score("PR"),
            "pi": bm.pi.tolist(),                 # state distributions
            # You can persist theta/beta if you plan to reconstruct exactly
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
    G_a.graph["belief_bundle"] = belief_bundle

    # End of belief states logic
    return G_a

def create_belief_bundle(G_a: nx.MultiDiGraph) -> Tuple[ProbBeliefMachine, List[str]]:
    # Logic to add Belief States
    A_persona, persona_order = build_persona_adjacency_from_subgraph(G_a)
    if A_persona.shape[0] > 0:
        # uniform relevance for now
        r = np.ones(A_persona.shape[0], dtype=float)
        bm = ProbBeliefMachine(A=A_persona, r=r, beta=0.35)

        belief_bundle = {
            "persona_order": persona_order,       # needed to map back to node ids
            "activation_SOL": (bm.activation("SOL")).tolist(),
            "activation_PR":  (bm.activation("PR")).tolist(),
            "expected_score_SOL": bm.expected_score("SOL"),
            "expected_score_PR":  bm.expected_score("PR"),
            "pi": bm.pi.tolist(),                 # state distributions
            # You can persist theta/beta if you plan to reconstruct exactly
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


def augment_subgraph_with_zmots(
    G: nx.DiGraph,
    G_a: nx.DiGraph,
    archetype_id: str
) -> nx.DiGraph:
    """
    Return an induced subgraph that equals G_a plus any ZMOT nodes that are
    co-parents of pain_triggers present in G_a. Edges are taken from the FULL graph G.
    """
    # Start from a REAL set of node ids
    print("Augmenting G_a with ZMOT nodes. Total nodes before:", G_a.number_of_nodes())
    keep: Set = set(G_a.nodes())

    # Sanity: product node should already be in G_a; if not, keep it
    # (helps later node->Product traversals on the pruned graph)
    # if "product" in your schema: keep.add(product_id)

    # For each archetype in the archetype subgraph, pull its ZMOT neighbors from FULL graph
    related_zmot_ids = get_target_nodes_by_source_and_type(G, archetype_id, "relevant_event")
    print(f"Found {len(related_zmot_ids)} ZMOT nodes related to archetype {archetype_id}")
    for zmot_id in related_zmot_ids:
        zmot_node = get_node_by_id(G, zmot_id)
        if not zmot_node:
            print(f"ZMOT node {zmot_id!r} not found in graph, skipping")
            continue
        keep.add(zmot_id)
        print(f"Added ZMOT node {zmot_id}")
    G_zmot = G.subgraph(keep).copy()
    print("Total nodes after adding ZMOTs:", G_zmot.number_of_nodes())
    # Build an induced subgraph FROM THE FULL GRAPH (so edges are present)
    return G_zmot





def generate_rcs(G, archetype_id: str, *, zmot_id: Optional[str] = None, rel_min: float = 0.0, max_nodes: int = 5000) -> Tuple[nx.DiGraph, Dict]:
    """
    Complete RCS Generator for an archetype and optional ZMOT.
    """    
    print("Generating RCS for archetype", archetype_id)
    G_a = get_node_subgraph_to_product(G, archetype_id)
    G_zmot = augment_subgraph_with_zmots(G, G_a, archetype_id)
    if G_a.number_of_nodes() > 0:
        if zmot_id is not None and zmot_id in G_zmot:
            print("Pruning to ZMOT node", zmot_id)
            zmot_node = get_node_by_id(G_zmot, zmot_id)
            G_pruned_for_zmot = get_node_subgraph_to_product(G_zmot, zmot_id)
            G_final = G_pruned_for_zmot
        else:
            G_final = G_zmot
    else:
        belief = {"reason": "empty_archetype_subgraph", "archetype_id": archetype_id}
        return G_a, belief  # avoid referencing G_final later 

    if G_final.number_of_nodes() <= 0:
        belief = {"reason": "empty_archetype_subgraph", "archetype_id": archetype_id}
        return G_final, belief
    belief_bundle = create_belief_bundle(G_final)
    G_final.graph["belief_bundle"] = belief_bundle
    graph_win_likelihood = compute_product_likelihood(G_final, archetype_id)
    G_final.graph["win_likelihood"] = graph_win_likelihood



    return G_final, belief_bundle

#--------------------------------------------
# G_final metadata - product likelihood from input graph
#--------------------------------------------

def _path_probability(
    G: nx.DiGraph,
    path: List[str],
    *,
    edge_attr_prob: str = "likelihood",        # edge conditional prob in [0,1]
    cap_node_type: str = "capability",
    node_attr_gate: str = "relevance" # node gate in [0,1] (used e.g., for capability nodes)
) -> float:
    """Multiply edge probabilities; gate by node 'relevance' for capability nodes."""
    p = 1.0
    # multiply edges
    for u, v in zip(path, path[1:]):
        p *= float(G.edges[u, v].get(edge_attr_prob, 1.0))
        if p <= 0.0:
            return 0.0
    # gate by capabilities’ relevance
    for n in path:
        if G.nodes[n].get("type") == cap_node_type:
            p *= float(G.nodes[n].get(node_attr_gate, 1.0))
            if p <= 0.0:
                return 0.0
    return max(0.0, min(1.0, p))


def _greedy_overlap_aware_selection(
    paths_with_p: List[Tuple[List[str], float]],
    *,
    overlap_penalty: float = 0.5  # shrink prob for later paths that overlap already-picked edges
) -> List[float]:
    """
    Sort paths by prob desc. Keep them all, but penalize later ones if they share edges
    with earlier picks (to reduce double-counting). Returns adjusted path probs.
    """
    # sort by p desc
    paths_with_p = sorted(paths_with_p, key=lambda t: t[1], reverse=True)
    used_edges = set()
    adjusted: List[float] = []
    for path, p in paths_with_p:
        # compute penalty if overlapping
        penalty = 1.0
        for u, v in zip(path, path[1:]):
            if (u, v) in used_edges:
                penalty *= overlap_penalty
        adjusted.append(max(0.0, min(1.0, p * penalty)))
        # mark edges as used
        for u, v in zip(path, path[1:]):
            used_edges.add((u, v))
    return adjusted


def compute_product_likelihood(
    G_orig: nx.DiGraph,
    starting_node_id: str,
    *,
    product_type: str = "product",
    max_hops: int = 8,
    max_paths: Optional[int] = 1000,
    edge_attr_prob: str = "likelihood",
    cap_node_type: str = "capability",
    node_attr_gate: str = "relevance",
    start_likelihood_attr: str = "likelihood",
    overlap_penalty: float = 0.5
) -> float:
    """
    P(win | starting_node) ≈ L_start * ( 1 - Π_j (1 - q_j) )

    where each q_j is a path probability from starting_node to product:
      q_j = Π(edge p) * Π(capability node relevance gates)

    Notes:
      - L_start is the prior that the starting evidence is 'active' (node.likelihood, default 1).
      - Uses a noisy-OR to avoid linear double-counting across multiple paths.
      - Penalizes overlapping paths to reduce dependence violations.
    """
    # 0) find product node (assumes one)
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
    
    G = G_orig.reverse(copy=True)


    # 1) prior that the starting evidence is 'on'
    L_start = float(G.nodes[starting_node_id].get(start_likelihood_attr, 1.0))
    L_start = max(0.0, min(1.0, L_start))

    # 2) enumerate paths
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

    # 3) score each path
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

    # 4) reduce double-counting by penalizing overlapping paths
    adjusted_qs = _greedy_overlap_aware_selection(paths_with_p, overlap_penalty=overlap_penalty)

    # 5) noisy-OR aggregation across (approximately) independent routes
    #    P_any = 1 - Π_j (1 - q_j)
    log_prod = 0.0
    for q in adjusted_qs:
        q = max(0.0, min(1.0, q))
        # multiply (1 - q) in log-space for stability
        log_prod += math.log(max(1e-12, 1.0 - q))
    P_any = 1.0 - math.exp(log_prod)

    # 6) final: gate by the starting node's own likelihood
    P_win = L_start * P_any
    return max(0.0, min(1.0, P_win))

    
    
    

#--------------------------------------------
# Experimental code for committee scoring models
#--------------------------------------------

# ---------- Core: Persona scoring ----------

def persona_scores(
    G: nx.DiGraph,
    *,
    persona_type: str = "persona",
    use_belief: bool = True,
    use_influence: bool = True,
    score_cap: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Collect per-persona scores from node attributes.

    Required node attrs (per persona node):
      - likelihood: float in [0,1]
      - relevance:  float in [0,1]

    Optional node attrs:
      - belief:     float in [0,1] (how convinced they currently are)
      - influence:  float >= 0     (org influence weight; default 1.0)

    Returns:
      scores[p] = {
         "L": likelihood,
         "R": relevance,
         "belief": belief,
         "influence": influence,
         "S": effective score used for committee math
      }
    """
    out: Dict[str, Dict[str, float]] = {}
    for n, data in G.nodes(data=True):
        if data.get("type") != persona_type:
            continue
        L = float(data.get("likelihood", 0.0))
        R = float(data.get("relevance", 0.0))
        belief = float(data.get("belief", 1.0)) if use_belief else 1.0
        influence = float(data.get("influence", 1.0)) if use_influence else 1.0

        # Base node score
        base = L * R
        S = base * belief * influence
        if score_cap is not None:
            S = min(S, score_cap)

        out[n] = {
            "L": L,
            "R": R,
            "belief": belief,
            "influence": influence,
            "S": S,
        }
    return out


# ---------- Model A: Democratic committee ----------

def democratic_success(
    scores: Dict[str, Dict[str, float]],
    converted: Iterable[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Democratic model:
      P_win = sum(S over converted) / sum(S over all personas)

    Returns:
      (p_win, breakdown)
      breakdown has {"sum_converted", "sum_total", "coverage"}
    """
    S_total = sum(v["S"] for v in scores.values())
    if S_total <= 0:
        return 0.0, {"sum_converted": 0.0, "sum_total": 0.0, "coverage": 0.0}

    S_conv = 0.0
    for p in converted:
        if p in scores:
            S_conv += scores[p]["S"]

    p_win = S_conv / S_total  # already in [0,1]
    return p_win, {
        "sum_converted": S_conv,
        "sum_total": S_total,
        "coverage": p_win,  # alias
    }


# ---------- Model B: Minimal coalition ----------

def minimal_coalition(
    scores: Dict[str, Dict[str, float]],
    threshold: float,
    *,
    prefer: Optional[Dict[str, float]] = None,
) -> List[str]:
    """
    Find the smallest set of personas whose cumulative S reaches a target
    fraction of total S (threshold in (0,1]).

    Greedy by descending S is optimal for minimizing *count* given a sum target.
    `prefer` is an optional secondary tie-break (higher is better), e.g.,:
      prefer = {"CFO": 1.0, "CTO": 0.8, ...} by node id

    Returns:
      ordered list of persona ids (the coalition)
    """
    assert 0 < threshold <= 1.0, "threshold must be in (0,1]"

    # Sort personas by primary key S desc, secondary preference desc
    items = []
    for p, v in scores.items():
        S = v["S"]
        pref = (prefer or {}).get(p, 0.0)
        items.append((p, S, pref))
    items.sort(key=lambda t: (t[1], t[2]), reverse=True)

    S_total = sum(v["S"] for v in scores.values())
    target = threshold * S_total

    coalition: List[str] = []
    running = 0.0
    for p, S, _ in items:
        if running >= target:
            break
        coalition.append(p)
        running += S

    return coalition


# ---------- Convenience: End-to-end helpers ----------

def recommend_coalitions(
    G: nx.DiGraph,
    *,
    thresholds: Iterable[float] = (0.5, 0.67, 0.8),  # 50%, 2/3rds, 80%
    persona_type: str = "persona",
    use_belief: bool = True,
    use_influence: bool = True,
) -> Dict[float, List[str]]:
    """
    Compute persona scores and produce minimal coalitions at several thresholds.
    """
    scr = persona_scores(
        G,
        persona_type=persona_type,
        use_belief=use_belief,
        use_influence=use_influence,
    )
    out: Dict[float, List[str]] = {}
    for t in thresholds:
        out[t] = minimal_coalition(scr, t)
    return out


def simulate_conversion_run(
    G: nx.DiGraph,
    converted: Iterable[str],
    *,
    persona_type: str = "persona",
    use_belief: bool = True,
    use_influence: bool = True,
) -> Dict[str, float]:
    """
    One-shot simulation for a given converted set under the democratic model.
    """
    scr = persona_scores(
        G,
        persona_type=persona_type,
        use_belief=use_belief,
        use_influence=use_influence,
    )
    p_win, breakdown = democratic_success(scr, converted)
    return {"p_win": p_win, **breakdown}
