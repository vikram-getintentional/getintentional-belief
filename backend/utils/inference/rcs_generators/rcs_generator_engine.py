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

    # Not doing ZMOT work now - remember to bring this in once arche works
    # Also get all ZMOTs for the pain trigger right away

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
                print("Discovered capability node. Adding product node and skipping")
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
                    print(f"Enqueuing higher order pain {felt_pain!r} with depth {d + 2}")
                    pain_q.append((felt_pain, d))
                    # Queued felt_in pain with d = 2. 
    final_nodes_set = set({n for n, d in relevant_nodes.items()})
    G_a = G.subgraph(final_nodes_set)
    for n in G_a.nodes:
        G_a.nodes[n]["depth"] = relevant_nodes[n]["temporal_depth"]

    return G_a


# ----------------------------
# 2) Derive potential committees (structural)
# ----------------------------

def _persona_struct_score(G: nx.MultiDiGraph, n: str) -> float:
    nd = G.nodes[n]
    return float(nd.get("relevance", 0.0)) * float(nd.get("role_weight", 1.0))


def _quorum_ok(members: List[str], persona_depts: Dict[str, str], quorum: Optional[Dict]=None) -> bool:
    if not quorum:
        return True
    # simple policy: require at least one from each required_depts set (any_of logic supported)
    req_any_of: List[List[str]] = quorum.get("departments_any_of", [])
    present = {persona_depts[m] for m in members if m in persona_depts}
    for group in req_any_of:
        if not (set(group) & present):
            return False
    return True


def derive_structural_committees(
    G_a: nx.MultiDiGraph,
    *,
    theta_struct: float = 1.0,            # structural sufficiency threshold (sum of structural scores)
    max_committee_size: int = 5,
    min_relevance: float = 0.1,
    forbid_dictator: bool = True,
    quorum: Optional[Dict] = None,
    top_k: int = 10,
) -> List[Dict]:
    """
    Build minimal winning committees independent of belief.
    - Score = relevance * role_weight
    - Constraint: quorum coverage (optional)
    - Sufficiency: sum(scores) >= theta_struct
    - Minimality: removing any member breaks quorum OR drops score below threshold
    """
    personas = [n for n, d in G_a.nodes(data=True) if d.get("type") == NODE_PERSONA and d.get("relevance", 0) >= min_relevance]
    if not personas:
        return []
    persona_depts = {n: G_a.nodes[n].get("dept", "Unknown") for n in personas}
    scores = {n: _persona_struct_score(G_a, n) for n in personas}
    ordered = sorted(personas, key=lambda n: scores[n], reverse=True)

    committees: List[Dict] = []
    seen_fingerprints: Set[Tuple[str, ...]] = set()

    # Greedy + trim for each seed
    for seed in ordered:
        S: List[str] = [seed]
        if forbid_dictator and scores[seed] >= theta_struct and _quorum_ok(S, persona_depts, quorum):
            # Dictator structurally sufficient; either skip or tag as dictatorial option
            pass

        # Grow until structural threshold reached and quorum covered
        idx = 0
        while sum(scores[m] for m in S) < theta_struct or not _quorum_ok(S, persona_depts, quorum):
            if idx >= len(ordered):
                break
            nxt = ordered[idx]
            idx += 1
            if nxt in S:
                continue
            S.append(nxt)
            if len(S) > max_committee_size:
                break
        if len(S) > max_committee_size:
            continue
        if sum(scores[m] for m in S) < theta_struct or not _quorum_ok(S, persona_depts, quorum):
            continue

        # Trim to minimal
        changed = True
        while changed:
            changed = False
            for m in list(S):
                trial = [x for x in S if x != m]
                if trial and sum(scores[x] for x in trial) >= theta_struct and _quorum_ok(trial, persona_depts, quorum):
                    S = trial
                    changed = True
        fp = tuple(sorted(S))
        if fp in seen_fingerprints:
            continue
        seen_fingerprints.add(fp)

        total = sum(scores[m] for m in S)
        # classify type purely structurally
        dominant_share = max(scores[m] for m in S) / max(total, 1e-9)
        if len(S) == 1:
            ctype = "dictator_structural"
        elif dominant_share >= 0.6 and len(S) <= 3:
            ctype = "hybrid"
        else:
            ctype = "coalition"

        committees.append({
            "members": S,
            "type": ctype,
            "struct_score": total,
            "member_scores": {m: scores[m] for m in S},
        })
        if len(committees) >= top_k:
            break

    return committees


# ----------------------------
# 3) Belief overlay & committee classification
# ----------------------------

def _persona_satisfaction(
    G: nx.MultiDiGraph,
    persona: str,
    *,
    default_importance: float = 0.5,
    default_severity: float = 0.5,
    default_fit: float = 0.5,
) -> float:
    """
    Estimate persona-level satisfaction s in [-1,1] by aggregating pains along their jobs.
    s = (2*fit - 1) * importance * severity
    Aggregation: weighted average over pains with weights (importance*severity).
    Falls back to persona-level defaults when pains/edges lack attributes.
    """
    # gather jobs owned by persona
    jobs = [u for u, v, _, ed in G.in_edges(persona, keys=True, data=True) if ed.get("type") == EDGE_OWNED_BY and G.nodes[u].get("type") == NODE_JOB]
    numer = 0.0
    denom = 0.0
    for j in jobs:
        # pains felt in this job (pain --felt_in--> job)
        pains = [p for p, _, _, ed3 in G.in_edges(j, keys=True, data=True) if ed3.get("type") == EDGE_FELT_IN and G.nodes[p].get("type") == NODE_PAIN]
        if not pains:
            # use persona fallback
            imp = G.nodes[persona].get("job_importance", default_importance)
            sev = G.nodes[persona].get("pain_severity", default_severity)
            fit = G.nodes[persona].get("resolution_fit", default_fit)
            w = imp * sev
            numer += (2*fit - 1.0) * w
            denom += w
            continue
        # compute fit from solving jobs if available
        for p in pains:
            imp = G.nodes[p].get("importance", default_importance)
            sev = G.nodes[p].get("severity", default_severity)
            # find any job solving this pain to derive fit (edge attr 'fit')
            fits: List[float] = []
            for js, _, _, eds in G.in_edges(p, keys=True, data=True):
                if eds.get("type") == EDGE_SOLVES and G.nodes[js].get("type") == NODE_JOB:
                    fits.append(float(eds.get("fit", default_fit)))
            fit = max(fits) if fits else default_fit
            w = imp * sev
            numer += (2*fit - 1.0) * w
            denom += w
    if denom == 0:
        # pure fallback
        imp = G.nodes[persona].get("job_importance", default_importance)
        sev = G.nodes[persona].get("pain_severity", default_severity)
        fit = G.nodes[persona].get("resolution_fit", default_fit)
        return (2*fit - 1.0) * imp * sev
    return max(-1.0, min(1.0, numer / denom))


def _persona_effective_b(
    G: nx.MultiDiGraph,
    persona: str,
    stage: str,
    *,
    satisfaction: Optional[float] = None,
) -> float:
    nd = G.nodes[persona]
    belief = float(nd.get("belief_by_stage", {}).get(stage, nd.get("belief", 0.5)))
    relevance = float(nd.get("relevance", 0.0))
    weight = float(nd.get("role_weight", 1.0))
    s = _persona_satisfaction(G, persona) if satisfaction is None else satisfaction
    return belief * s * relevance * weight


def _logistic(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _committee_conflict_penalty(G: nx.MultiDiGraph, members: List[str], beta: float = 0.0) -> float:
    """Optional simple conflict penalty using pairwise 'conflict' attr on persona-persona edges if present."""
    if beta <= 0 or len(members) <= 1:
        return 0.0
    # Build induced persona subgraph
    sub = G.subgraph(members)
    vals = []
    for u, v, _, d in sub.edges(keys=True, data=True):
        if u in members and v in members and d.get("type") == "conflict":
            vals.append(float(d.get("weight", 0.0)))
    if not vals:
        return 0.0
    avg = sum(vals) / len(vals)
    return beta * avg


def overlay_beliefs_and_classify_committees(
    G_a: nx.MultiDiGraph,
    committees: List[Dict],
    *,
    stages: Iterable[str] = ("purchase", "impl", "sustain"),
    theta_by_stage: Dict[str, float] = None,
    beta_conflict: float = 0.0,
    detractor_eps: float = 0.05,
) -> List[Dict]:
    """
    For each committee, compute stage-wise coalition belief and classify as active/latent/blocked.
    Uses additive-with-logistic model so detractors (negative b_i) subtract.
    """
    if theta_by_stage is None:
        theta_by_stage = {"purchase": 0.70, "impl": 0.65, "sustain": 0.60}

    out: List[Dict] = []
    for c in committees:
        members = c["members"]
        member_b: Dict[str, Dict[str, float]] = {m: {} for m in members}
        detractors: List[Tuple[str, float]] = []
        status_by_stage: Dict[str, str] = {}
        B_by_stage: Dict[str, float] = {}

        for s in stages:
            x = 0.0
            detractors_s: List[Tuple[str, float]] = []
            for m in members:
                b = _persona_effective_b(G_a, m, s)
                member_b[m][s] = b
                x += b
                if b < -detractor_eps:
                    detractors_s.append((m, b))
            x -= _committee_conflict_penalty(G_a, members, beta=beta_conflict)
            B = _logistic(x)
            B_by_stage[s] = B
            theta = theta_by_stage.get(s, 0.7)
            if B >= theta:
                status_by_stage[s] = "active"
            elif B >= max(0.0, theta - 0.05):
                status_by_stage[s] = "latent"
            else:
                status_by_stage[s] = "blocked"
            # collect union of detractors across stages (strongest magnitude kept)
            for name, b in detractors_s:
                detractors.append((name, b))

        # keep strongest detractor instance per member
        strongest: Dict[str, float] = {}
        for name, b in detractors:
            strongest[name] = min(strongest.get(name, 0.0), b)
        detractor_list = sorted(((n, strongest[n]) for n in strongest), key=lambda t: t[1])

        out.append({
            **c,
            "member_b": member_b,
            "B_by_stage": B_by_stage,
            "status_by_stage": status_by_stage,
            "detractors": detractor_list,
        })
    return out


# ----------------------------
# 4) Deduce concerns per persona
# ----------------------------
ASSET_MAP = {
    "security_risk": ["Security brief", "Compliance mapping", "PoC: controls"],
    "integration_fit": ["Integration blueprint", "Solution arch review", "Pilot"],
    "roi_value": ["ROI calculator", "Customer proof", "Exec business case"],
    "governance": ["Governance model", "RACI", "Policy pack"],
}


def _persona_concerns(
    G: nx.MultiDiGraph,
    persona: str,
    stage: str,
    *,
    default_importance: float = 0.5,
    default_severity: float = 0.5,
    default_fit: float = 0.5,
) -> List[Dict]:
    """
    Return a ranked list of concerns for a persona at a stage.
    score = importance * severity * (1 - fit) * (1 - belief)
    concern_type heuristics based on node/edge tags; falls back to generic types.
    """
    belief = float(G.nodes[persona].get("belief_by_stage", {}).get(stage, G.nodes[persona].get("belief", 0.5)))
    jobs = [u for u, v, _, ed in G.in_edges(persona, keys=True, data=True) if ed.get("type") == EDGE_OWNED_BY and G.nodes[u].get("type") == NODE_JOB]
    concerns: List[Tuple[float, str, Dict]] = []
    for j in jobs:
        pains = [p for p, _, _, ed3 in G.in_edges(j, keys=True, data=True) if ed3.get("type") == EDGE_FELT_IN and G.nodes[p].get("type") == NODE_PAIN]
        for p in pains:
            imp = float(G.nodes[p].get("importance", default_importance))
            sev = float(G.nodes[p].get("severity", default_severity))
            fits: List[float] = []
            for js, _, _, eds in G.in_edges(p, keys=True, data=True):
                if eds.get("type") == EDGE_SOLVES and G.nodes[js].get("type") == NODE_JOB:
                    fits.append(float(eds.get("fit", default_fit)))
            fit = max(fits) if fits else default_fit
            score = imp * sev * (1.0 - fit) * (1.0 - belief)
            # infer concern type from tags
            tags = set(str(G.nodes[p].get("tags", "")).lower().split()) | set(str(G.nodes[j].get("tags", "")).lower().split())
            if {"risk", "security", "compliance"} & tags:
                ctype = "security_risk"
            elif {"integration", "api", "migration", "data"} & tags:
                ctype = "integration_fit"
            elif {"roi", "value", "cost", "payback"} & tags:
                ctype = "roi_value"
            elif {"governance", "policy", "control"} & tags:
                ctype = "governance"
            else:
                ctype = "integration_fit" if (1.0 - fit) > 0.4 else "roi_value"
            concerns.append((score, ctype, {"pain": p, "job": j, "fit": fit, "importance": imp, "severity": sev}))
    concerns.sort(key=lambda t: t[0], reverse=True)
    out = []
    for score, ctype, meta in concerns[:3]:
        out.append({
            "persona": persona,
            "stage": stage,
            "concern_type": ctype,
            "score": score,
            "assets": ASSET_MAP.get(ctype, ["Case study", "Demo", "Pilot"]),
            **meta,
        })
    return out


def deduce_concerns(G_a: nx.MultiDiGraph, committees_eval: List[Dict], stages: Iterable[str] = ("purchase", "impl", "sustain")) -> Dict[str, List[Dict]]:
    """
    For each committee member across committees, produce top concerns per stage.
    Returns mapping persona -> list of concern dicts.
    """
    results: Dict[str, List[Dict]] = defaultdict(list)
    for ce in committees_eval:
        for s in stages:
            for m in ce["members"]:
                results[m].extend(_persona_concerns(G_a, m, s))
    # keep top 3 per persona overall
    for m in list(results.keys()):
        results[m] = sorted(results[m], key=lambda d: d["score"], reverse=True)[:3]
    return results


# ----------------------------
# 5) Engagement plan synthesis
# ----------------------------

def plan_engagements(
    committees_eval: List[Dict],
    persona_concerns: Dict[str, List[Dict]],
    *,
    max_actions_per_committee: int = 5,
) -> List[Dict]:
    """
    Suggest a minimal set of engagements to push latent/blocked committees over thresholds.
    Greedy by ΔB per action proxy: prioritize personas with higher concern scores.
    """
    plans: List[Dict] = []
    for ce in committees_eval:
        # Skip fully active committees across all stages
        if all(st == "active" for st in ce["status_by_stage"].values()):
            continue
        actions: List[Dict] = []
        # Rank members by their worst concern
        member_scores = []
        for m in ce["members"]:
            clist = persona_concerns.get(m, [])
            if not clist:
                continue
            worst = clist[0]
            member_scores.append((worst["score"], m, worst))
        member_scores.sort(key=lambda t: t[0], reverse=True)
        for _, m, worst in member_scores[:max_actions_per_committee]:
            actions.append({
                "persona": m,
                "recommend": worst["assets"],
                "target_pain": worst.get("pain"),
                "rationale": f"Address {worst['concern_type']} for {m} to lift fit from {worst.get('fit', 0.5):.2f}",
            })
        plans.append({
            "committee": ce["members"],
            "type": ce["type"],
            "status_by_stage": ce["status_by_stage"],
            "proposed_actions": actions,
        })
    return plans


# ----------------------------
# Orchestration: generate_rcs
# ----------------------------

def generate_rcs(
    G: nx.MultiDiGraph,
    archetype_id: str,
    *,
    zmot_id: Optional[str] = None,
    rel_min: float = 0.0,
    structural_quorum: Optional[Dict] = None,
    theta_struct: float = 1.0,
    theta_by_stage: Dict[str, float] = None,
    beta_conflict: float = 0.0,
) -> Dict:
    """
    One-call RCS generator.
    Returns dict with: subgraph, committees_structural, committees_eval, concerns, engagement_plans, audit
    """
    # 1) Subgraph
    G_a, audit = build_archetype_subgraph_with_temporal_depth(G, archetype_id, zmot_id=zmot_id, rel_min=rel_min)

    # 2) Structural committees
    committees = derive_structural_committees(
        G_a,
        theta_struct=theta_struct,
        quorum=structural_quorum,
    )

    # 3) Belief overlay & classification
    committees_eval = overlay_beliefs_and_classify_committees(
        G_a,
        committees,
        theta_by_stage=theta_by_stage,
        beta_conflict=beta_conflict,
    )

    # 4) Concerns
    concerns = deduce_concerns(G_a, committees_eval)

    # 5) Engagement plan
    plans = plan_engagements(committees_eval, concerns)

    return {
        "graph": G_a,
        "committees_structural": committees,
        "committees_eval": committees_eval,
        "concerns": concerns,
        "engagement_plans": plans,
        "audit": audit,
    }


# ----------------------------
# Example (pseudo) usage
# ----------------------------
if __name__ == "__main__":
    # Suppose you already have a product graph G loaded elsewhere
    G = nx.MultiDiGraph()
    # ... populate G with nodes/edges and types/attrs ...

    # IDs in your graph
    ARCH = "arch:midmarket_saas"
    ZMOT = None

    # Generate RCS
    result = generate_rcs(
        G,
        ARCH,
        zmot_id=ZMOT,
        rel_min=0.1,
        structural_quorum={"departments_any_of": [["Finance"], ["IT", "Security"]]},
        theta_struct=1.0,
        theta_by_stage={"purchase": 0.70, "impl": 0.65, "sustain": 0.60},
        beta_conflict=0.0,
    )

    # You can now inspect:
    # result["committees_eval"], result["engagement_plans"], etc.
