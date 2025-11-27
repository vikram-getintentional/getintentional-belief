# backend/utils/inference/belief_manager/learn/summarize_hypotheses.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import networkx as nx

from backend.utils.inference.belief_manager.graph_diff_mapper import _L

def _fmt_pct(x: float) -> str:
    try:
        return f"{100.0*float(x):.0f}%"
    except Exception:
        return "—"

def _edge_label(G: nx.DiGraph, u: str, v: str) -> str:
    rel = (G[u][v].get("edge_type") or G[u][v].get("type") or G[u][v].get("rel") or "").lower() if G.has_edge(u, v) else ""
    u_label = _L(G, u)
    v_label = _L(G, v)
    return f"{u_label} -[{rel}]-> {v_label}"

def _exists(G: nx.DiGraph, u: str, v: str, rel_lower: str = "") -> bool:
    if not G.has_edge(u, v): 
        return False
    if not rel_lower:
        return True
    r = (G[u][v].get("edge_type") or G[u][v].get("type") or G[u][v].get("rel") or "").lower()
    return r == rel_lower

def summarize_learning_for_humans(G: nx.DiGraph, learning: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert machine diffs + neighborhoods into human, causal hypotheses.
    Returns:
      {
        "persona_graph_inferences": [... str],
        "full_graph_inferences":   [... str],
        "top_edge_updates":        [... str]
      }
    """
    diffs = learning.get("diffs", {})
    nb   = learning.get("neighborhoods", {})
    recs = learning.get("summary", {}).get("recommendations", {})
    edges_ranked = recs.get("edge_updates_ranked", [])
    node_ranked  = recs.get("node_updates_ranked", [])

    # 1) Persona Graph Inferences
    persona_graph_inferences: List[str] = []
    spd = diffs.get("starting_persona_delta", {})
    for pid, delta in sorted(spd.items(), key=lambda kv: abs(kv[1]), reverse=True):
        if abs(delta) < 1e-6:
            continue
        more_less = "more" if delta > 0 else "less"
        p_label = _L(G, pid)
        persona_graph_inferences.append(
            f"{p_label}: showing up {more_less} often at start than expected "
            f"({_fmt_pct(delta)} shift). Likely we misestimated this persona’s perceptibility "
            f"or its upstream job→pain→trigger motif."
        )

    ppd = diffs.get("pp_delta", {})
    # pp_delta keys are "pa->pb"
    for pair_str, delta in sorted(ppd.items(), key=lambda kv: abs(kv[1]), reverse=True):
        if abs(delta) < 1e-6:
            continue
        pa, pb = pair_str.split("->", 1)
        more_less = "more" if delta > 0 else "less"
        pa_label = _L(G, pa)
        pb_label = _L(G, pb)
        persona_graph_inferences.append(
            f"Handoff {pa_label} → {pb_label} occurred {more_less} often than expected "
            f"({_fmt_pct(delta)} shift). Upstream chain around {pa_label}’s job→pain and "
            f"downstream {pb_label} job relevance likely needs adjustment."
        )

    # 2) Full Graph Inferences (causality-flavoured)
    full_graph_inferences: List[str] = []

    # Use neighborhoods to propose counterfactuals:
    # - If start:{persona} is high but we don't see corroborating upstream edges, point at job->persona or pain->job.
    for k, bundle in (nb.get("upstream") or {}).items():
        pid = bundle.get("anchor")
        p_label = bundle.get("anchor_labels")
        delta_sum = bundle.get("delta_sum", 0.0)
        if abs(delta_sum) < 1e-6:
            continue
        edges = bundle.get("edges", [])
        # Evidence counters
        has_job_performed = any(_exists(G,u,v,"performed_by") for (u,v,_) in edges)
        has_pain_felt     = any(_exists(G,u,v,"felt_in") for (u,v,_) in edges)
        has_trigger       = any(_exists(G,u,v,"triggered_by") for (u,v,_) in edges)

        # Hypothesis rules
        if delta_sum > 0:
            # persona shows up earlier than expected
            if not has_trigger and has_pain_felt:
                full_graph_inferences.append(
                    f"{p_label}: earlier-than-expected appearance without supporting pain_trigger→pain evidence. "
                    f"Hypothesis: the job→persona (performed_by) or pain→job (felt_in) weights are too low; "
                    f"triggers might not be the primary driver here."
                )
            elif has_trigger and not has_pain_felt:
                full_graph_inferences.append(
                    f"{p_label}: earlier-than-expected appearance with triggers present but weak pain→job evidence. "
                    f"Hypothesis: increase pain→job (felt_in) and job→persona (performed_by) relevance."
                )
            else:
                full_graph_inferences.append(
                    f"{p_label}: earlier-than-expected start suggests its upstream job/pain neighborhood is underweighted; "
                    f"consider raising performed_by and felt_in edges around this persona."
                )
        else:
            # persona shows up later than expected
            full_graph_inferences.append(
                f"{p_label}: later-than-expected start suggests overestimation of upstream visibility; "
                f"consider lowering performed_by or felt_in edges for jobs/pains adjacent to this persona."
            )

    # Handoff counterfactuals: if pa→pb high but jb→pb missing, point to jb→p or p→ja mismatch
    for k, bundle in (nb.get("handoff") or {}).items():
        pair = bundle.get("anchor", [])
        if len(pair) != 2:
            continue
        pa, pb = pair
        pa_label, pb_label = bundle.get("anchor_labels", ["?", "?"])
        delta_sum = bundle.get("delta_sum", 0.0)
        if abs(delta_sum) < 1e-6:
            continue
        edges = bundle.get("edges", [])
        has_jb_pb = any(_exists(G,u,v,"performed_by") and v==pb for (u,v,_) in edges)  # jb → pb
        has_jb_p  = any(_exists(G,u,v,"solves") and v.startswith("pain:") for (u,v,_) in edges)  # jb → pain
        has_p_ja  = any(_exists(G,u,v,"felt_in") and v.startswith("job:") for (u,v,_) in edges)  # pain → ja

        if delta_sum > 0:
            if not has_jb_pb and has_jb_p:
                full_graph_inferences.append(
                    f"{pa_label}→{pb_label}: stronger-than-expected handoff without jb→pb evidence. "
                    f"Since {pb_label}’s job seems tied to solving the pain, increase job→persona (performed_by) for {pb_label}."
                )
            elif not has_jb_p and has_p_ja:
                full_graph_inferences.append(
                    f"{pa_label}→{pb_label}: stronger-than-expected handoff with pain→ja present but missing jb→pain (solves). "
                    f"Hypothesis: raise {pb_label}’s job→pain (solves) relevance."
                )
            else:
                full_graph_inferences.append(
                    f"{pa_label}→{pb_label}: stronger handoff indicates underweighted chain around pain→job (felt_in/solves) "
                    f"and {pb_label}’s performed_by edges."
                )
        else:
            full_graph_inferences.append(
                f"{pa_label}→{pb_label}: weaker handoff implies we may be overrating either "
                f"{pa_label}’s job→pain link or {pb_label}’s job relevance. Consider reducing solves/felt_in or performed_by nearby."
            )

    # Downstream product counterfactuals
    for k, bundle in (nb.get("downstream") or {}).items():
        anchor = bundle.get("anchor", [])
        if len(anchor) != 2:
            continue
        pid, prod = anchor
        p_lablel, prod_label = bundle.get("anchor_labels", ["?", "?"])
        delta_sum = bundle.get("delta_sum", 0.0)
        if abs(delta_sum) < 1e-6:
            continue
        edges = bundle.get("edges", [])
        has_prod_cap = any(_exists(G,u,v,"offers") for (u,v,_) in edges)   # product→capability
        has_cap_pain = any(_exists(G,u,v,"solves") for (u,v,_) in edges)   # capability→pain
        has_pain_job = any(_exists(G,u,v,"felt_in") for (u,v,_) in edges)  # pain→job
        has_job_per  = any(_exists(G,u,v,"performed_by") for (u,v,_) in edges)  # job→persona

        if delta_sum > 0:
            # persona→product stronger than expected
            if not has_cap_pain and has_prod_cap:
                full_graph_inferences.append(
                    f"{p_label}→{prod_label}: stronger link with product→capability but weak capability→pain. "
                    f"Hypothesis: capabilities tied to this product resolve the relevant pains more than assumed."
                )
            elif has_cap_pain and not has_pain_job:
                full_graph_inferences.append(
                    f"{p_label}→{prod_label}: stronger link with capability→pain present but weak pain→job. "
                    f"Hypothesis: pains are indeed felt in {p_label}’s jobs more frequently; raise pain→job (felt_in)."
                )
            elif has_pain_job and not has_job_per:
                full_graph_inferences.append(
                    f"{p_label}→{prod_label}: stronger link with pain→job present but weak job→persona. "
                    f"Hypothesis: raise job→persona (performed_by) relevance for {p_label}’s roles."
                )
            else:
                full_graph_inferences.append(
                    f"{p_label}→{prod_label}: stronger link suggests the full downstream chain is underweighted; "
                    f"increase offers / solves / felt_in / performed_by along this path."
                )
        else:
            full_graph_inferences.append(
                f"{p_label}→{prod_label}: weaker link suggests we overestimated one or more of "
                f"offers/solves/felt_in/performed_by edges along the chain to {p_label}."
            )

    # 3) Top edge updates (pretty lines)
    top_edge_updates = [
        f"{_edge_label(G, r['u'], r['v'])} | Δ={r['delta']:+.3f} | band={r.get('band','?')} | conf={r.get('confidence',0):.2f}"
        for r in edges_ranked[:10]
    ]

    return {
        "persona_graph_inferences": persona_graph_inferences,
        "full_graph_inferences":   full_graph_inferences,
        "top_edge_updates":        top_edge_updates,
    }
