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

from backend.utils.graph_base.network_graph import get_node_by_id, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
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
    personas = [n for n, d in G_a.nodes(data=True) if d.get("type") == "persona"]
    personas.sort()  # stable order for reproducibility
    idx = {p: i for i, p in enumerate(personas)}
    n = len(personas)
    A = np.zeros((n, n), dtype=float)
    if n == 0:
        return A, personas

    # preindex jobs owned_by persona and pains felt_in job, jobs that solve pain
    jobs_by_persona = {p: [u for u, _, _, ed in G_a.in_edges(p, keys=True, data=True)
                           if ed.get("type") == "performed_by" and G_a.nodes[u].get("type") == "job"]
                       for p in personas}

    pains_by_job = {j: [p for p, _, _, ed in G_a.in_edges(j, keys=True, data=True)
                        if ed.get("type") == "felt_in" and G_a.nodes[p].get("type") == "pain"]
                    for j, d in G_a.nodes(data=True) if d.get("type") == "job"}

    jobs_solving_pain = {}
    for p, d in G_a.nodes(data=True):
        if d.get("type") != "pain":
            continue
        jobs_solving_pain[p] = [j for j, _, _, ed in G_a.in_edges(p, keys=True, data=True)
                                if ed.get("type") == "solves" and G_a.nodes[j].get("type") == "job"]

    def w(u, v, etype):
        # get_edge_weight is not imported here; edge attrs carry 'weight'
        # pick max weight over parallel edges of the given type
        mx = 0.0
        for _, _, _, ed in G_a.edges(u, v, keys=True, data=True):
            if ed.get("type") == etype:
                mx = max(mx, float(ed.get("weight", 0.0)))
        return mx

    # build A[u,v]
    for u in personas:
        iu = idx[u]
        for j_u in jobs_by_persona.get(u, []):
            # pains felt in job j_u
            for p in pains_by_job.get(j_u, []):
                w_pju = w(p, j_u, "felt_in")
                if w_pju <= 0:
                    continue
                # jobs that solve p
                for j_v in jobs_solving_pain.get(p, []):
                    w_jvp = w(j_v, p, "solves")
                    if w_jvp <= 0:
                        continue
                    # personas owning j_v
                    for v in [x for x in G_a.successors(j_v)
                              if G_a.nodes[x].get("type") == "persona"
                              and any(ed.get("type") == "performed_by" for _,_,_,ed in G_a.edges(j_v, x, keys=True, data=True))]:
                        if u == v:
                            continue
                        iv = idx[v]
                        w_juu = w(j_u, u, "performed_by")
                        w_jvv = w(j_v, v, "performed_by")
                        if w_juu > 0 and w_jvv > 0:
                            path_w = w_juu * w_pju * w_jvp * w_jvv
                            A[iu, iv] = max(A[iu, iv], path_w)  # max over pains/jobs
    return A, personas




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