# ============================
# File: backend/utils/inference/rcs_generators/rcs_generator_engine.py
# ============================
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable, Set
import heapq
import numpy as np
import networkx as nx

from backend.utils.graph_base.network_graph import (
    _set_node_label,
    get_edge_attribute,
    get_node_by_id,
    get_node_subgraph_to_product,
    get_product_id_from_subgraph,
    get_source_nodes_by_target_and_type,
    get_target_nodes_by_source_and_type,
    get_persona_node_ids,
    persona_meta_from_node,
)
from backend.utils.knowledge_base.arsenal.execution_arsenal_repository import get_best_plays_for_concern

# All math/graph helpers live in one place
from backend.utils.inference.rcs_generators.graph_algorithms import (
    # basic math
    clip01 as _clip,
    noisy_or as _noisy_or,
    sigmoid as _sig,
    # structural helpers
    prune_types as _prune_types,
    remove_cycles_by_min_edge as remove_cycles,
    nodes_reaching_target as _nodes_reaching_target,
    norm_type as _norm_type,
    set_occurred_nodes_inplace as _set_occurred_nodes_likelihoods,
    # boosts
    collect_zmot_trigger_boosts as _collect_zmot_trigger_boosts,
    collect_attribute_trigger_boosts as _collect_attribute_trigger_boosts,
    apply_trigger_boosts_inplace as _apply_trigger_boosts_inplace,
    PRUNE_TYPES,
    # depth annotators
    set_temporal_depths,
    set_causal_depths,
    # PPR / baseline
    ppr as _ppr,
    baseline_conversion_prob as _baseline_conversion_prob,
    # persona involvement/activation/care
    jobs_of_persona as _jobs_of_persona,
    persona_involvement_from_jobs as _persona_involvement_from_jobs,
    compute_activation_care_vector as _compute_activation_care_vector,
    persona_activation as _persona_activation,
    # pains/concerns helpers
    felt_pains_of_job as _felt_pains_of_job,
    solve_pains_of_job as _solve_pains_of_job,
    infer_stage_for_concern as _infer_stage_for_concern,
    concerns_for_persona_cf as _concerns_for_persona_cf,
    # marginal lift & boosts
    apply_persona_boost as _apply_persona_boost,
    marginal_lift_persona as _marginal_lift_persona,
    # coalitions / causal flows
    find_top_coalitions as _find_top_coalitions,
    compute_causal_burden_effects as _compute_causal_burden_effects,
    build_causal_flows as _build_causal_flows,
    # sequence / campaign design
    design_sequences_and_campaigns,
)

# =========================================================
# -------------- Main: generate_rcs (no archetypes) -------
# =========================================================

def generate_rcs(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
    rel_min: float = 0.0,
    max_nodes: int = 5000,
    # Execution knobs
    top_concerns_per_persona: int = 5,
    top_personas: int = 20,
    plays_per_concern: int = 3,
    alpha_act=(0.0, 1.0, 1.0, 1.0),
    beta_felt=(0.0, 1.0, 1.0, 1.0),
    boost_factor: float = 2.0
) -> Tuple[nx.DiGraph, Dict]:
    """
    Archetype-free RCS:
      • Conditions entirely on engaged_nodes.
      • Boosts pain_trigger priors/out-edges based on engaged ZMOT and attribute nodes.
      • Preserves the original rich report schema.
    """
    product_id = get_product_id_from_subgraph(G)
    if not product_id:
        return nx.DiGraph(), {"error": "no_product_in_graph"}
    conv_id = product_id

    # --- Work on a pruned copy (remove zmot/observable_moment/keyword) ---
    G_final = _prune_types(G, PRUNE_TYPES)
    if G_final.number_of_nodes() == 0:
        return G_final, {"error": "empty_subgraph"}

    # --- Handle engaged_nodes ------------------------------------------------
    # Collect ZMOT+attribute-trigger boosts on FULL graph; apply to PRUNED graph
    trig_boosts: List[Tuple[str, float]] = []
    boost_dbg = {"triggers_in_pruned": 0, "edges_boosted": 0}
    occurred: List[Dict] = []

    if engaged_nodes:
        for n in engaged_nodes:
            nid = n.get("id") if isinstance(n, dict) else n
            occ = n.get("occurrence", 1.0) if isinstance(n, dict) else 1.0

            # record occurrence if node exists post-prune
            if nid in G_final:
                occurred.append({"id": nid, "occurrence": occ})

            # collect boosts by type from the full graph
            if nid in G:
                t = _norm_type(G, nid)
                if t == "zmot_event":
                    trig_boosts.extend(_collect_zmot_trigger_boosts(G, nid))
                elif t in {"attribute", "attribute_value"}:
                    trig_boosts.extend(_collect_attribute_trigger_boosts(G, nid))

    if trig_boosts:
        boost_dbg = _apply_trigger_boosts_inplace(G_final, trig_boosts, boost_edges=True)

    # --- DAG-ify & annotate depths ------------------------------------------
    if not nx.is_directed_acyclic_graph(G_final):
        G_final = remove_cycles(G_final)
    set_temporal_depths(G_final)
    set_causal_depths(G_final)

    # --- Mark occurrences on nodes (priors/cumulative fields) ---------------
    _set_occurred_nodes_likelihoods(G_final, occurred)
    occurred_ids_all = [o["id"] for o in occurred if o.get("id") in G_final]

    # --- Seeds for PPR = occurred nodes EXCEPT ZMOT/keyword/moment ----------
    seeds_for_ppr = [
        nid for nid in occurred_ids_all
        if _norm_type(G_final, nid) not in {"zmot_event", "observable_moment", "keyword"}
    ]

    # --- Baseline conversion (after boosts) ---------------------------------
    p0_conditioned = _baseline_conversion_prob(G_final, seeds_for_ppr, conv_id)
    G_final.graph["win_likelihood"] = p0_conditioned

    # --- PPR for attention/involvement --------------------------------------
    pr_all = _ppr(G_final, seeds_for_ppr, conv_id=conv_id)

    # --- Personas: involvement, activation, care ----------------------------
    persona_ids = get_persona_node_ids(G_final)
    involvement = {pid: _persona_involvement_from_jobs(G_final, pr_all, pid) for pid in persona_ids}

    # use centralized vector helper for activation/care
    actcare_vec = _compute_activation_care_vector(
        G_final, pr_all, persona_ids, alpha_act, beta_felt
    )

    persona_rows = []
    for pid in persona_ids:
        A_i, C_i = actcare_vec.get(pid, (0.0, 0.0))
        persona_rows.append({
            "id": pid,
            "involvement": _clip(involvement.get(pid, 0.0)),
            "activation": _clip(A_i),
            "care": _clip(C_i),
            "breakdown": {}  # optional: keep light here (heavy calc can live in graph_algorithms if needed)
        })

    topI_all = sorted(persona_rows, key=lambda r: r["involvement"], reverse=True)
    topA_all = sorted(persona_rows, key=lambda r: r["activation"], reverse=True)
    topI = topI_all[:top_personas]
    topA = topA_all[:top_personas]

    # --- Marginal lift -------------------------------------------------------
    ml_rows = []
    for pid in [r["id"] for r in topI]:
        base_row = next((r for r in persona_rows if r["id"] == pid), None)
        delta = _marginal_lift_persona(G_final, seeds_for_ppr, pid, conv_id=conv_id, boost=boost_factor)
        ml_rows.append({
            "id": pid,
            "marginal_lift": float(delta),
            "involvement": float(base_row["involvement"]) if base_row else 0.0,
            "activation": float(base_row["activation"]) if base_row else 0.0,
            "care": float(base_row["care"]) if base_row else 0.0
        })
    top_lift = sorted(ml_rows, key=lambda r: r["marginal_lift"], reverse=True)[:top_personas]

    # --- Concerns + assets (CF) ---------------------------------------------
    concerns_flat: List[Dict] = []
    concerns_by_persona: Dict[str, List[Dict]] = {}
    assets_by_persona: Dict[str, Dict[str, List[Dict]]] = {}

    prod_id = get_product_id_from_subgraph(G_final)
    for row in topI:
        pid = row["id"]
        persona_node = get_node_by_id(G_final, pid) or {}
        display_meta = persona_meta_from_node(persona_node, pid)
        persona_meta = {
            "title": display_meta["title"],
            "department": display_meta["department"],
            "seniority": display_meta["seniority"],
        }

        cons = _concerns_for_persona_cf(
            G_final, pid, row["involvement"], row["care"],
            conv_id=conv_id,
            seeds=seeds_for_ppr,
            base_p0=p0_conditioned,
            top_k=top_concerns_per_persona,
            edge_boost=boost_factor
        )

        concerns_by_persona[pid] = []
        assets_by_persona[pid] = {}

        for c in cons:
            cid = c["concern_id"]
            stage = _infer_stage_for_concern(G_final, pid, cid)

            # Archetype removed: pass empty dict
            plays = get_best_plays_for_concern(
                persona_alias=persona_meta,
                concern_stage=stage,
                product_id=prod_id,
                archetype={},  # ← no archetype in the new graph
                top_k=plays_per_concern,
            )

            if plays:
                top_play = plays[0]
                asset_reco = {
                    "asset_id":   top_play["asset"]["id"],
                    "asset_name": top_play["asset"]["name"],
                    "channel":    top_play.get("channel"),
                    "fitness":    top_play.get("fitness", 0.0),
                    "why":        top_play.get("why", ""),
                }
            else:
                asset_reco = {"note": "No existing plays match. Generate a new play."}

            concerns_flat.append({
                "persona_id": pid,
                "persona_label": _set_node_label(G_final, pid),
                "concern_id": cid,
                "concern_label": _set_node_label(G_final, cid),
                "stage": stage,
                "stage_explainer": _stage_explainer(stage),
                "potential_lift": float(c.get("potential_lift", 0.0)),
                "rel": float(c.get("rel", 0.0)),
                "asset_reco": asset_reco,
            })

            concerns_by_persona[pid].append({
                "concern_id": cid,
                "concern_label": _set_node_label(G_final, cid),
                "potential_lift": float(c.get("score", c.get("potential_lift", 0.0))),
                "cf_lift": float(c.get("potential_lift", 0.0)),
                "elasticity": float(c.get("elasticity_proxy", 0.0)),
                "rel": float(c.get("rel", 0.0)),
                "stage": stage,
                "stage_explainer": _stage_explainer(stage),
            })

            assets_by_persona[pid][cid] = (
                [
                    {
                        "asset_id": p["asset"]["id"],
                        "asset_name": p["asset"]["name"],
                        "channel": p.get("channel"),
                        "fitness": p.get("fitness", 0.0),
                        "why": p.get("why", ""),
                    }
                    for p in plays
                ]
                if plays else
                [asset_reco]
            )

    concerns_flat.sort(key=lambda r: r["potential_lift"], reverse=True)

    baseline_block = {
        "baseline_win_conditioned": float(_baseline_conversion_prob(G_final, seeds_for_ppr, conv_id)),
        "occurred": [o["id"] for o in occurred],
        "zmot_trigger_boosts": [b for b in trig_boosts if b],  # mixed with attributes below
        "attribute_trigger_boosts": [b for b in trig_boosts if b],  # same pool; kept for API parity
        "zmot_boost_debug": boost_dbg,
    }

    top_personas_block = {
        "by_involvement": [{"id": r["id"], "involvement": float(r["involvement"]),
                            "activation": float(r["activation"]), "care": float(r["care"])} for r in topI],
        "by_activation": [{"id": r["id"], "activation": float(r["activation"]),
                           "involvement": float(r["involvement"]), "care": float(r["care"])} for r in topA],
        "by_marginal_lift": top_lift,
    }

    # --- Coalitions & causal flows ------------------------------------------
    coalition_pool_ids = list({*(r["id"] for r in topI[:10]), *(r["id"] for r in topA[:10])})

    coalitions = _find_top_coalitions(
        G_final, seeds_for_ppr, conv_id,
        coalition_pool_ids,
        boost_factor=boost_factor,
        max_coalition_size=3,
        top_k=8
    )
    coalitions_out = []
    for row in coalitions:
        personas = row["personas"]
        merged_concerns = _merge_concerns_for_personas(G_final, personas, concerns_by_persona,
                                                       max_per_persona=3, max_total=8)
        coalitions_out.append({**row, "concerns": merged_concerns})

    pair_effects = _compute_causal_burden_effects(
        G_final, seeds_for_ppr, conv_id, coalition_pool_ids,
        alpha_act=alpha_act, beta_felt=beta_felt, boost_factor=boost_factor
    )
    causal_flows = _build_causal_flows(
        G_final, seeds_for_ppr, conv_id, coalition_pool_ids,
        pair_effects, concerns_by_persona,
        top_pairs=8, top_chains=5, boost_factor=boost_factor
    )

    # --- Sequences & campaign stitching -------------------------------------
    seq_pool_ids = list({*(r["id"] for r in topI[:10]), *(r["id"] for r in topA[:10])})
    sequences_and_campaigns = design_sequences_and_campaigns(
        G_final,
        seeds_for_ppr,
        conv_id,
        candidate_personas=seq_pool_ids,
        concerns_by_persona=concerns_by_persona,
        assets_by_persona=assets_by_persona,
        pair_effects=pair_effects,
        boost_factor=boost_factor,
        alpha_act=alpha_act,
        beta_felt=beta_felt,
        max_len=5,
        beam_width=5,
        max_coalition_size=3,
        min_synergy=1e-6,
        max_sequences=5
    )

    report = {
        "baseline": baseline_block,
        "top_personas": top_personas_block,
        "concerns_by_persona": [
            {"persona": pid, "concerns": clist, "assets": assets_by_persona.get(pid, {})}
            for pid, clist in concerns_by_persona.items()
        ],
        "concerns_flat": concerns_flat,
        "coalitions": coalitions_out,
        "causal_flows": causal_flows,
        "sequences_and_campaigns": sequences_and_campaigns,
    }

    return G_final, report


# =========================================================
# ------------ Tiny local glue (non-math) -----------------
# =========================================================

def _stage_explainer(stage: str) -> str:
    s = (stage or "").strip().lower()
    if s == "problem":
        return "Has not realized the problem"
    if s == "pain":
        return "Feels the pain but not acting on it yet"
    if s == "solution":
        return "Needs a resolution but does not know how"
    return (stage or "").capitalize()


def _merge_concerns_for_personas(
    G: nx.DiGraph,
    personas: List[str],
    per_persona_concerns: Dict[str, List[Dict]],
    *,
    max_per_persona: int = 3,
    max_total: int = 8
) -> List[Dict]:
    # kept here as light glue (no math); safe to leave or move to graph_algorithms if preferred
    seen: Dict[str, Dict] = {}
    for pid in personas:
        for row in (per_persona_concerns.get(pid) or [])[:max_per_persona]:
            cid = row["concern_id"]
            if cid not in seen or row.get("potential_lift", 0.0) > seen[cid].get("potential_lift", 0.0):
                seen[cid] = row
    merged = sorted(seen.values(), key=lambda r: r.get("potential_lift", 0.0), reverse=True)
    return merged[:max_total]
