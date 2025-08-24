"""
End-to-end RCS generation pipeline starting from a product graph (NetworkX MultiDiGraph).

Pipeline:
  1) Build archetype subgraph with temporal depth (first-encounter rule)
  2) Derive potential committees (structural, belief-independent)
  3) Overlay belief/satisfaction to classify committees (active/latent/blocked)
  4) Deduce persona concerns
  5) Propose engagement plan to move committees over the threshold

Assumptions / Schema (adjust constants below to match your graph):
  Node types:  Archetype, ZMOT (optional), Trigger, Pain, Job, Persona
  Persona attrs (recommended):
    - relevance: float [0,1]
    - role_weight: float >0 (authority multiplier)
    - dept: str (e.g., Finance, IT, Product, Marketing)
    - belief_by_stage: dict like {"purchase":0.6, "impl":0.55, "sustain":0.5}
    - (optional) resolution_fit, job_importance, pain_severity (persona-level defaults)

  Pain attrs (recommended):
    - importance, severity in [0,1] (if present)

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

from backend.utils.graph_base.network_graph import get_edge_attribute, get_edge_weight, get_node_by_id, get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
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
    pain_trigger_ids = get_target_nodes_by_source_and_type(G, pain_id, "triggered_by")
    for trigger_id in pain_trigger_ids:
        _update_relevant_nodes(relevant_nodes, trigger_id, depth)
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
            print(f"Warning: Persona {persona_id} has no jobs performed_by, skipping.")
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
                    solving_job_likelihood = get_edge_attribute(G_a, solving_job_id, pain_id, "likelihood")
                    if solving_job_likelihood <= 0:
                        print(f"Warning: Job {solving_job_id} solves pain {pain_id} with non-positive likelihood, skipping.")
                        continue
                    target_persona_ids = get_target_nodes_by_source_and_type(G_a, solving_job_id, "performed_by")
                    if not target_persona_ids:
                        print(f"Warning: Job {solving_job_id} has no personas performed_by, skipping.")
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