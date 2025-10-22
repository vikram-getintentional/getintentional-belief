# ============================
# File: backend/utils/inference/rcs_generators/rcs_helpers/strategy_orchestrators.py
# ============================
from __future__ import annotations
from datetime import date, timedelta
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy
import networkx as nx

from backend.utils.crm_management.target_account_manager import get_target_account_ids
from backend.utils.graph_base.network_graph import build_product_graph, get_node_by_id, get_nodes_list_ids, get_product_id_from_subgraph
from backend.utils.inference.rcs_generators.generate_rcs_fast import _nt, generate_rcs
from backend.utils.inference.rcs_generators.graph_algorithms import compute_persona_scores_from_core
from backend.utils.inference.rcs_generators.rcs_helpers.campaign_managers import _campaigns_from_sequences_using_arsenal
from backend.utils.knowledge_base.arsenal.execution_arsenal_repository import get_best_plays_for_concern
from backend.utils.strategy_builder.frozen_stage_simulator import build_multi_quarter_frozen_strategy


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def _to_engaged_payload(ids: List[str]) -> List[Dict]:
    """Convert node IDs to the payload structure used by generate_rcs."""
    return [{"id": nid, "occurrence": 1.0} for nid in sorted(set(ids))]


# ---------------------------------------------------------------------
# BUILD STRATEGY  (Frozen)
# ---------------------------------------------------------------------
def build_account_strategy(
    product_subgraph: nx.DiGraph,
    account_id: Optional[str] = None,
    account_name: Optional[str] = None,
    *,
    attributes: Optional[List[str]] = None,
    zmots: Optional[List[str]] = None,
    initial_engagements: Optional[List[str]] = None,
    boost_factor: float = 2.0,
    scaffold: Optional[Dict] = None
) -> Dict:
    engaged = (attributes or []) + (zmots or []) + (initial_engagements or [])
    print("⚙️ [build_account_strategy] calling generate_rcs with engaged nodes:", engaged)
    G_work, report = generate_rcs(
        product_subgraph,
        engaged_nodes=_to_engaged_payload(engaged),
        boost_factor=boost_factor,
    )
    print("rcs report complete")
    # personas for the execution plan (belief=activation, importance=involvement)
    persona_rows = [
        {"name": pid, "belief": float(r.get("activation", 0.5)), "importance": float(r.get("involvement", 0.5))}
        for pid, r in (report.get("persona_scores") or {}).items()
    ]
    persona_rows.sort(key=lambda x: (x["importance"], x["belief"]), reverse=True)

    print("persona rows complete")
    if not scaffold:
        if attributes:
            attr_nodes = [get_node_by_id(product_subgraph, aid) for aid in attributes if get_node_by_id(product_subgraph, aid)]
            attr_labels = [n.get("label") for n in attr_nodes if n.get("label")]
            
            archetype = {
                "industry": next((n.get("industry") for n in attr_nodes if n.get("industry")), ""),
                "geography": next((n.get("geography") for n in attr_nodes if n.get("geography")), ""),
                "revenue_range": next((n.get("revenue_range") for n in attr_nodes if n.get("revenue_range")), ""),
                "employee_range": next((n.get("employee_range") for n in attr_nodes if n.get("employee_range")), ""),
                "funding_stage": next((n.get("funding_stage") for n in attr_nodes if n.get("funding_stage")), ""),
                "competitors_used": next((n.get("competitors_used") for n in attr_nodes if n.get("competitors_used")), ""),
                "tech_stack": next((n.get("tech_stack") for n in attr_nodes if n.get("tech_stack")), ""),
            }
        else:
            archetype = {}
            
    else:
        a = scaffold.get("archetype") or {}
        archetype = {
            "industry": a.get("industry",""),
            "geography": a.get("geography",""),
            "revenue_range": a.get("revenue_range",""),
            "employee_range": a.get("employee_range",""),
            "funding_stage": a.get("funding_stage",""),
            "competitors_used": a.get("competitors_used",""),
            "tech_stack": a.get("tech_stack",""),
        }
    print("starting campaign sequencing")
    # campaigns from sequences + arsenal
    campaigns = _campaigns_from_sequences_using_arsenal(
        G=G_work,
        account_id=account_id,
        product_id=get_product_id_from_subgraph(product_subgraph),
        report=report,
        start_idx=0,
        plays_per_step=1,  # or 2 if you want 2 rows/step
        archetype = archetype
    )

    print("campaign mapping complete")

    frozen_strategy = build_multi_quarter_frozen_strategy(
        product_subgraph=G_work,
        account_id=account_id,
        account_name=account_name,
        product_id=get_product_id_from_subgraph(product_subgraph),
        rcs_report=report,
        archetype=archetype,
        forced_start_stage=None,        # or "zmot" etc if you know it
        num_phases=4,
        days_per_phase=90,
        simulate_carryover=True,
    )

    print("frozen strategy complete for account:", account_name or account_id or "unknown")

    
    return frozen_strategy, G_work, report




# ---------------------------------------------------------------------
# UPDATE STRATEGY  (Tactical)
# ---------------------------------------------------------------------
def update_tactical_plan(
    product_subgraph: nx.DiGraph,
    frozen_strategy: Dict,
    *,
    attributes: Optional[List[str]] = None,
    zmots: Optional[List[str]] = None,
    engagements_to_date: Optional[List[str]] = None,
    stage: str = "auto",  # "cold" | "warm" | "hot" | "auto"
    boost_factor: float = 2.0,
) -> Dict:
    """
    Tactical refresh that updates priorities within a frozen strategy.

    Inputs:
      - frozen_strategy: the baseline snapshot returned by build_account_strategy()
      - attributes, zmots, engagements_to_date: cumulative engagement so far

    Outputs:
      - Updated portfolio tilt
      - Recommended next concerns/sequences
      - Current stage label ("cold", "warm", "hot")
    """
    engaged = (attributes or []) + (zmots or []) + (engagements_to_date or [])
    print("⚙️ [update_tactical_plan] calling generate_rcs with engaged nodes:", engaged)

    G_work, report = generate_rcs(
        product_subgraph,
        engaged_nodes=_to_engaged_payload(engaged),
        boost_factor=boost_factor,
    )

    # ----------------------------------------
    # Stage inference (auto if not specified)
    # ----------------------------------------
    if stage == "auto":
        has_eng = bool(engagements_to_date)
        win_prob = report.get("baseline", {}).get("win_likelihood", 0.0) or 0.0
        stage = "hot" if win_prob >= 0.35 else ("warm" if has_eng else "cold")

    # ----------------------------------------
    # Adjust portfolio mix
    # ----------------------------------------
    base_policy = frozen_strategy.get("portfolio_policy", {"breadth": 0.6, "depth": 0.2, "mutation": 0.2})
    policy = copy.deepcopy(base_policy)

    if stage == "cold":
        policy.update({"breadth": 0.6, "depth": 0.2, "mutation": 0.2})
    elif stage == "warm":
        policy.update({"breadth": 0.35, "depth": 0.5, "mutation": 0.15})
    elif stage == "hot":
        policy.update({"breadth": 0.2, "depth": 0.7, "mutation": 0.1})

    # ----------------------------------------
    # Restrict to frozen persona pool
    # ----------------------------------------
    frozen_pids = []
    if frozen_strategy.get("frozen_persona_pool"):
        for p in frozen_strategy["frozen_persona_pool"]:
            if isinstance(p, str):
                frozen_pids.append(p)
            elif isinstance(p, dict):
                pid = p.get("id")
                if pid:
                    frozen_pids.append(pid)
    concern_backlog = [
        x for x in report.get("concern_backlog", [])
        if x.get("pid") in frozen_pids
    ][:30]

    # ----------------------------------------
    # Theme re-ranking (dynamic theme lift)
    # ----------------------------------------
    frozen_themes = frozen_strategy.get("themes", [])
    live_sequences = report.get("concern_sequences", [])

    def keyify(s): return (s["persona"], s["concern_id"])
    frozen_keys = [{keyify(s) for s in t.get("sequence", [])} for t in frozen_themes]

    reranked_themes = []
    for idx, theme in enumerate(frozen_themes):
        fkeys = frozen_keys[idx]
        overlap = sum(1 for c in concern_backlog if (c["pid"], c["cid"]) in fkeys)
        lift_now = theme.get("final_win", 0.0) + 0.05 * overlap
        reranked_themes.append((lift_now, theme))

    reranked_themes.sort(key=lambda x: x[0], reverse=True)
    top_theme = reranked_themes[0][1] if reranked_themes else None

    # ----------------------------------------
    # Tactical picks for next sprint
    # ----------------------------------------
    next_concerns = concern_backlog[:8]
    next_sequences = live_sequences[:3] if live_sequences else []

    # Pull IDs for arsenal mapping
    account_id = (
        G_work.graph.get("account_id")
        or frozen_strategy.get("seed_context", {}).get("account_id")
        or G_work.graph.get("account_name")  # fallback to name if that’s all we have
    )
    product_id = get_product_id_from_subgraph(product_subgraph)

    # Optional scaffold → pass archetype if you have one in the frozen strategy
    scaffold = {
        "archetype": (frozen_strategy.get("seed_context", {}).get("archetype") or {})
    }

    archetype = scaffold.get("archetype") or {}

    # Use the real arsenal mapper
    try:
        next_campaigns = _campaigns_from_sequences_using_arsenal(
            G=product_subgraph,
            account_id=account_id,
            product_id=product_id,
            report=report,
            start_idx=0,
            plays_per_step=1,
            archetype=archetype,
        )[:1]  # Take the single best next campaign
    except Exception as e:
        print("[WARN] _campaigns_from_sequences_using_arsenal failed:", e)
        next_campaigns = []

    # ----------------------------------------
    # Final structured output
    # ----------------------------------------
    out = {
        "graph_win_likelihood": report.get("baseline", {}).get("win_likelihood", 0.0),
        "stage": stage,
        "portfolio_policy": policy,
        "top_theme": top_theme,
        # Strategic context (unchanged)
        "frozen_personas": frozen_strategy.get("frozen_persona_pool", []),
        # Tactical focus
        "next_concerns": next_concerns,
        "next_sequences": next_sequences,
        "next_campaigns": next_campaigns,   # <— was missing
        # Pass-through data for UI
        "top_N_personas": report.get("top_personas", {}).get("by_involvement", [])[:10],
        "concerns_by_persona": report.get("concerns_by_persona", []),
        "coalitions": report.get("coalitions", []),
        "concern_coalitions": report.get("concern_coalitions", []),
        "concern_sequences": next_sequences,
        "concern_backlog": concern_backlog,
        "causal_flows": report.get("causal_flows", []),
        "sequences_and_campaigns": report.get("sequences_and_campaigns", []),
    }

    return out, G_work


#--------------------------------
# COMPREHENSIVE ORCHESTRATOR LOGIC
#--------------------------------


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "")).strip("_")[:48]

def _to_timeframe(idx: int) -> Dict[str, str]:
    start = date.today() + timedelta(days=14*idx)
    end   = start + timedelta(days=84)  # ~12 weeks
    return {"startDate": start.isoformat(), "endDate": end.isoformat()}

def _expected_lift(combo: Dict[str, Any]) -> str:
    # Prefer explicit % if provided
    if isinstance(combo.get("expectedLift"), str):
        return combo["expectedLift"]
    # Derive from any numeric fit/engagement you already compute (fallback 8–25%)
    score = float(combo.get("fitScore", combo.get("engagementScore", 0.5)) or 0.5)
    pct = max(8, min(25, int(100*score*0.25)))
    return f"+{pct}%"

def _make_arsenal_rows(combos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for combo in combos or []:
        asset_label = combo.get("asset") or combo.get("assetId") or combo.get("name") or "Asset"
        channel = combo.get("Channel") or combo.get("channel") or combo.get("channel_id") or ""
        rows.append({
            "asset": asset_label,
            "channel": channel,
            "fitment": combo.get("fitment") or ", ".join(combo.get("concernsAddressed", [])) or "High fit",
            "engagement": combo.get("engagement") or combo.get("expectedEngagement", "Medium"),
            "engagementLabel": _engagement_label(float(combo.get("engagementScore", 0.5) or 0.5)),
            "expectedLift": _expected_lift(combo),
            "expectedLiftLabel": _expected_lift_label(combo.get("stage") or "zmot", float(combo.get("fitScore", 0.5) or 0.5)),
            "duration": combo.get("duration") or combo.get("expectedDuration", "2-4 weeks"),
        })
    return rows

def _engagement_label(x: float) -> str:
    return "High" if x >= 0.7 else ("Medium" if x >= 0.4 else "Low")

def _expected_lift_label(stage: str, fitness: float) -> str:
    # coarse mapping; tune later
    base = {
        "pre_zmot": (0.5, 2.0),
        "zmot": (1.0, 4.0),
        "problem_realization": (2.0, 6.0),
        "discovery": (3.0, 8.0),
        "barriers": (2.0, 6.0),
        "implementation": (1.0, 3.0),
    }.get(stage, (1.0, 3.0))
    lo, hi = base
    span = hi - lo
    est = lo + span * max(0.0, min(1.0, fitness))
    return f"+{est:.0f}%"

def _derive_personas_from_campaigns(campaigns: List[Dict[str, Any]]) -> List[str]:
    s = set()
    for c in campaigns or []:
        for p in c.get("personas", []) or []:
            s.add(p)
    return sorted(s)

def _belief_summary(personas_rows: List[Dict[str, Any]], stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    avg = round(sum(p.get("belief", 0.0) for p in personas_rows) / max(1, len(personas_rows)), 4) if personas_rows else 0.0
    if stages:
        dominant = max(stages, key=lambda s: len(s.get("pains", [])) + len(s.get("personas", []))).get("stage", "ZMOT")
        top = max(stages, key=lambda s: len(s.get("pains", [])) + len(s.get("personas", [])))
        ps = ", ".join(top.get("personas", [])[:2]) or "key stakeholders"
        pain = (top.get("pains") or ["core objections"])[0]
        focus = f"Reinforce belief for {ps} by addressing: {pain}."
    else:
        dominant, focus = "New", ""
    return {"average_belief_score": avg, "dominant_stage": dominant, "recommended_focus": focus}

def _normalize_stages(rcs_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tolerant normalizer: ensures stage keys exist; safe if report has no stages."""
    req = ["stage","trigger","belief_before","belief_after","personas","pains"]
    out = []
    for s in (rcs_report or {}).get("stages", []):
        row = {k: s.get(k) for k in req}
        row["stage"] = row.get("stage") or "ZMOT"
        row["trigger"] = row.get("trigger") or ""
        row["belief_before"] = row.get("belief_before") or ""
        row["belief_after"] = row.get("belief_after") or ""
        row["personas"] = list(row.get("personas") or [])
        row["pains"] = list(row.get("pains") or [])
        row["evidence"] = list(s.get("evidence") or [])
        row["recommended_assets"] = list(s.get("recommended_assets") or [])
        out.append(row)
    # If nothing came in, keep a single empty stage (matches your scaffold)
    if not out:
        out = [{
            "stage": "", "trigger": "", "belief_before": "", "belief_after": "",
            "personas": [], "pains": [], "evidence": [], "recommended_assets": []
        }]
    return out

def _pick_zmot_theme(rcs_report: Optional[Dict[str, Any]]) -> str:
    if not rcs_report:
        return ""
    if rcs_report.get("zmot_theme"):
        return rcs_report["zmot_theme"]
    stages = rcs_report.get("stages") or []
    if not stages:
        return ""
    trig = (stages[0].get("trigger") or "").strip()
    return trig[:80]  # heuristic



# ---------------- Scaffold Mapping Internals ----------------

def _attributes_to_node_ids(
    archetype: Dict[str, Any],
    G: nx.DiGraph,
    key_map: Dict[str, str],
    label_fields: Tuple[str, ...],
    node_type_field: str,
) -> List[str]:
    out: List[str] = []
    for k, dim in key_map.items():
        val = (archetype.get(k) or "").strip()
        if not val:
            continue
        nid = _find_attribute_node(G, dim, val, label_fields, node_type_field)
        if nid:
            out.append(nid)
    return _dedup(out)


def _zmot_to_node_ids(
    scaffold: Dict[str, Any],
    G: nx.DiGraph,
    label_fields: Tuple[str, ...],
    node_type_field: str,
) -> List[str]:
    texts: List[str] = []
    # tolerate multiple shapes for zmot_theme: string, dict with label/id, or list
    z = scaffold.get("zmot_theme")
    if isinstance(z, str):
        z_texts = [z.strip()] if z.strip() else []
    elif isinstance(z, dict):
        z_label = z.get("label") or z.get("name") or z.get("id")
        z_texts = [str(z_label).strip()] if z_label else []
    elif isinstance(z, list):
        z_texts = [str(x).strip() for x in z if str(x).strip()]
    else:
        z_texts = []
    texts.extend(z_texts)
    for s in (scaffold.get("reverse_case_study", {}) or {}).get("stages", []) or []:
        trig = (s.get("trigger") or "") if isinstance(s, dict) else str(s)
        trig = (trig or "").strip()
        if trig:
            texts.append(trig)
 

    out: List[str] = []
    for t in texts:
        nid = _find_label_node(G, t, allowed_types=("zmot",), label_fields=label_fields, node_type_field=node_type_field)
        if nid:
            out.append(nid)
    return _dedup(out)


def _initial_engagements_to_node_ids(
    scaffold: Dict[str, Any],
    G: nx.DiGraph,
    label_fields: Tuple[str, ...],
    node_type_field: str,
) -> List[str]:
    """
    Seed engagements from lightweight hints in the scaffold:
    - execution_plan.personas[].name
    - rcs_brief.persona_scores keys
    - campaigns[].personas
    - plays[].target_personas / jobs / pains (if present)
    """
    seeds: List[str] = []

    # execution_plan.personas
    for p in (scaffold.get("execution_plan", {}) or {}).get("personas", []) or []:
        pname = p.get("name")
        if pname:
            nid = _find_label_node(G, pname, ("persona",), label_fields, node_type_field)
            if nid:
                seeds.append(nid)

    # rcs_brief.persona_scores
    ps = (scaffold.get("rcs_brief", {}) or {}).get("persona_scores", {}) or {}
    for pname in ps.keys():
        nid = _find_label_node(G, pname, ("persona",), label_fields, node_type_field)
        if nid:
            seeds.append(nid)

    # campaign target personas
    for c in (scaffold.get("execution_plan", {}) or {}).get("campaigns", []) or []:
        for pname in c.get("personas", []) or []:
            nid = _find_label_node(G, pname, ("persona",), label_fields, node_type_field)
            if nid:
                seeds.append(nid)

    # plays/evidence can carry jobs/pains
    for row in scaffold.get("plays", []) or []:
        seeds.extend(_nodes_from_named_targets(G, row, label_fields, node_type_field))

    for ev in scaffold.get("evidence", []) or []:
        seeds.extend(_nodes_from_named_targets(G, ev, label_fields, node_type_field))

    return _dedup(seeds)


def _engaged_nodes_with_occurrence(
    scaffold: Dict[str, Any],
    G: nx.DiGraph,
    label_fields: Tuple[str, ...],
    node_type_field: str,
    max_occurrence: int = 5
) -> List[Dict[str, Any]]:
    """
    Compose engaged_nodes with integer 'occurrence' that reflect strength.
    Priority sources (highest to lowest):
      1) rcs_brief.persona_scores (use involvement or activation)
      2) execution_plan.personas (belief/importance if present)
      3) campaign target personas
      4) plays/evidence named targets (persona/job/pain)
    """
    accum: Dict[str, float] = {}

    # 1) rcs_brief.persona_scores
    ps = (scaffold.get("rcs_brief", {}) or {}).get("persona_scores", {}) or {}
    for pname, row in ps.items():
        w = float(row.get("involvement", row.get("activation", 0.5)) or 0.5)
        nid = _find_label_node(G, pname, ("persona",), label_fields, node_type_field)
        if nid:
            accum[nid] = max(accum.get(nid, 0.0), w)

    # 2) execution_plan.personas
    for p in (scaffold.get("execution_plan", {}) or {}).get("personas", []) or []:
        pname = p.get("name")
        if not pname:
            continue
        w = float(p.get("importance", p.get("belief", 0.5)) or 0.5)
        nid = _find_label_node(G, pname, ("persona",), label_fields, node_type_field)
        if nid:
            accum[nid] = max(accum.get(nid, 0.0), w)

    # 3) campaigns personas
    for c in (scaffold.get("execution_plan", {}) or {}).get("campaigns", []) or []:
        for pname in c.get("personas", []) or []:
            nid = _find_label_node(G, pname, ("persona",), label_fields, node_type_field)
            if nid:
                accum[nid] = max(accum.get(nid, 0.0), 0.6)

    # 4) plays/evidence named targets → persona/job/pain @ 0.5
    for row in scaffold.get("plays", []) or []:
        for nid in _nodes_from_named_targets(G, row, label_fields, node_type_field):
            accum[nid] = max(accum.get(nid, 0.0), 0.5)
    for ev in scaffold.get("evidence", []) or []:
        for nid in _nodes_from_named_targets(G, ev, label_fields, node_type_field):
            accum[nid] = max(accum.get(nid, 0.0), 0.5)

    # map 0..1 -> 1..max_occurrence
    out = []
    for nid, w in accum.items():
        occ = max(1, min(max_occurrence, int(round(1 + (max_occurrence - 1) * _clamp01(w)))))
        out.append({"id": nid, "occurrence": occ})

    # stable order
    out.sort(key=lambda r: (-r["occurrence"], r["id"]))
    return out


# ---------------- Match helpers ----------------

def _find_attribute_node(
    G: nx.DiGraph,
    dim: str,
    val: str,
    label_fields: Tuple[str, ...],
    node_type_field: str
) -> Optional[str]:
    """
    Expect graph attribute nodes like:
      node_type='attribute', attr_dim in {'industry','geo','funding','revenue','employees'}, attr_val='SaaS' ...
    Fallback to label fuzzy match if attr_val not present.
    """
    best = (None, 0.0)
    target = (val or "").lower()
    all_attribute_nodes = get_nodes_list_ids(G, "attribute_value", {"dimension": dim})
    for attr_node_id in all_attribute_nodes:
        d = G.nodes[attr_node_id]
        cand = (d.get("attr_val") or _first_label(d, label_fields) or "").lower()
        score = _soft_match_score(cand, target)
        if score > best[1]:
            best = (attr_node_id, score)
    # looser acceptance for ranges (EMEA, $50M-$200M, 201-1K, etc.)
    thresh = 0.45 if any(tok in target for tok in ["$", "-", "k", "range"]) else 0.6
    return best[0] if best[1] >= thresh else None


def _find_label_node(
    G: nx.DiGraph,
    label: str,
    allowed_types: Tuple[str, ...],
    label_fields: Tuple[str, ...],
    node_type_field: str
) -> Optional[str]:
    best = (None, 0.0)
    t = (label or "").lower()
    for n, d in G.nodes(data=True):
        if allowed_types and d.get(node_type_field) not in allowed_types:
            continue
        cand = (_first_label(d, label_fields) or "").lower()
        score = _soft_match_score(cand, t)
        if score > best[1]:
            best = (n, score)
    return best[0] if best[1] >= 0.6 else None


def _nodes_from_named_targets(
    G: nx.DiGraph,
    row: Dict[str, Any],
    label_fields: Tuple[str, ...],
    node_type_field: str
) -> List[str]:
    out: List[str] = []
    for key, types in [
        ("target_personas", ("persona",)),
        ("personas", ("persona",)),
        ("jobs", ("job",)),
        ("pains", ("pain",)),
        ("tags", ("persona","job","pain"))
    ]:
        vals = row.get(key)
        if not vals:
            continue
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            for typ in types:
                nid = _find_label_node(G, str(v), (typ,), label_fields, node_type_field)
                if nid:
                    out.append(nid)
                    break
    return _dedup(out)


def _first_label(d: Dict[str, Any], label_fields: Tuple[str, ...]) -> str:
    for f in label_fields:
        if d.get(f):
            return str(d[f])
    return ""


def _soft_match_score(cand: str, target: str) -> float:
    c = (cand or "").strip()
    t = (target or "").strip()
    if not c or not t:
        return 0.0
    if c == t:
        return 1.0
    # token overlap + prefix bonus
    ct = set(re.findall(r"[a-z0-9]+", c.lower()))
    tt = set(re.findall(r"[a-z0-9]+", t.lower()))
    if not ct or not tt:
        return 0.0
    overlap = len(ct & tt) / max(1, len(tt))
    prefix = 1.0 if c.lower().startswith(t.lower()) or t.lower().startswith(c.lower()) else 0.0
    return min(1.0, 0.6*overlap + 0.4*prefix)


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


def _dedup(xs: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

#--- Main Scaffold to Graph Mapping ----
def scaffold_to_orchestrator_inputs(
    *,
    scaffold: Dict[str, Any],
    G: nx.DiGraph,
    attribute_key_map: Optional[Dict[str, str]] = None,
    label_fields: Tuple[str, ...] = ("label", "name", "title"),  # adjust to your graph schema
    node_type_field: str = "node_type",
    max_occurrence: int = 5
) -> Dict[str, Any]:
    """
    Converts the account RCS scaffold to inputs for:
      build_account_strategy(product_subgraph, attributes, zmots, initial_engagements, boost_factor)
      generate_rcs(G, engaged_nodes=[{id, occurrence}, ...], ...)

    Returns:
      {
        "attributes": [node_id, ...],
        "zmots": [node_id, ...],
        "initial_engagements": [node_id, ...],
        "engaged_nodes": [{"id": node_id, "occurrence": int}, ...]
      }
    """
    print("Starting scaffold to orchestrator inputs mapping...")
    attribute_key_map = attribute_key_map or {
        # scaffold.archetype keys -> graph attribute dimension (node attrs)
        "industry": "industry",
        "revenue_range": "revenue_range",
        "employee_range": "employee_range",
        "geography": "geography",
        "funding_stage": "funding_stage",
    }

    # 1) Attribute nodes
    attributes = _attributes_to_node_ids(
        scaffold.get("archetype", {}) or {},
        G,
        attribute_key_map,
        label_fields,
        node_type_field,
    )

    # 2) ZMOT nodes (from zmot_theme + stage triggers)
    zmots = _zmot_to_node_ids(
        scaffold,
        G,
        label_fields,
        node_type_field,
    )

    # 3) Initial engagements (personas/jobs/pains mentioned before strategy runs)
    initial_engagements = _initial_engagements_to_node_ids(
        scaffold,
        G,
        label_fields,
        node_type_field,
    )

    # 4) Engaged nodes with occurrences (weights → small integer counts)
    engaged_nodes = _engaged_nodes_with_occurrence(
        scaffold,
        G,
        label_fields,
        node_type_field,
        max_occurrence=max_occurrence
    )

    return {
        "attributes": attributes,
        "zmots": zmots,
        "initial_engagements": initial_engagements,
        "engaged_nodes": engaged_nodes,
    }

#----Skinny Campaign Builder
def _index_concerns(rcs_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup: cid -> {label, stage?, persona?}
    Consumes rcs_report.concerns_by_persona and concern_backlog.
    """
    idx: Dict[str, Dict[str, Any]] = {}

    # From concerns_by_persona
    for pname, items in (rcs_report.get("concerns_by_persona") or {}).items():
        for it in items or []:
            cid = it.get("cid") or it.get("concern_id")
            if not cid:
                continue
            if cid not in idx:
                idx[cid] = {
                    "label": it.get("concern_label") or it.get("label") or f"Concern {cid}",
                    "stage": it.get("stage"),
                    "persona": pname
                }

    # From concern_backlog (often flat list)
    for it in (rcs_report.get("concern_backlog") or []):
        cid = it.get("cid") or it.get("concern_id")
        if not cid:
            continue
        cur = idx.get(cid, {})
        if "label" not in cur or not cur.get("label"):
            cur["label"] = it.get("label") or it.get("concern_label") or f"Concern {cid}"
        if "stage" not in cur or not cur.get("stage"):
            cur["stage"] = it.get("stage")
        idx[cid] = cur

    return idx





def _synthesize_campaigns_from_sequences(
    rcs_report: Dict[str, Any],
    *,
    start_idx: int = 0
) -> List[Dict[str, Any]]:
    """
    Turn rcs_report.concern_sequences into minimal campaigns:
      - 1 sequence -> 1 campaign
      - personas = unique personas in sequence steps
      - arsenalTable = one row per step (concern), with sensible defaults
    """
    sequences = rcs_report.get("concern_sequences") or []
    if not sequences:
        return []

    concern_idx = _index_concerns(rcs_report)
    campaigns: List[Dict[str, Any]] = []

    for i, seq in enumerate(sequences):
        steps = seq.get("sequence") or []
        personas = []
        arsenal_rows = []

        # Gather personas + build arsenal rows
        for step in steps:
            pname = step.get("persona")
            if pname:
                personas.append(pname)
            cid = step.get("cid") or step.get("concern_id")
            c_info = concern_idx.get(cid or "", {})
            label = c_info.get("label") or f"Concern {cid}" if cid else "Concern"

            # Construct a tiny "combo" so we can reuse _expected_lift()
            # Prefer sequence-level final_win; fallback to step-stage heuristics
            seq_lift = seq.get("final_win")
            combo = {
                "asset": f"[TBD] Resolve “{label}”",
                "recommendedChannels": ["Email Nurture", "LinkedIn Ads", "Sales Assist"],
                "fitment": c_info.get("stage") or "Concern resolution",
                "engagementScore": 0.6 if c_info.get("stage") in ("Problem Realization", "Barriers") else 0.5
            }
            if isinstance(seq_lift, (int, float)) and seq_lift > 0:
                # Map final_win roughly to a % via your existing _expected_lift policy
                combo["fitScore"] = float(seq_lift)  # _expected_lift will scale this

            arsenal_rows.append({
                "asset": combo["asset"],
                "channels": ", ".join(combo["recommendedChannels"]),
                "fitment": combo["fitment"],
                "engagement": "Medium",
                "expectedLift": _expected_lift(combo)
            })

        personas = sorted({p for p in personas if p})

        theme = seq.get("theme")  # optional
        description = f"Resolve prioritized concerns across {' & '.join(personas[:2])}." if personas else "Resolve prioritized concerns."
        camp = {
            "id": f"camp_seq_{i+start_idx:02d}",
            "description": description,
            "timeframe": _to_timeframe(i + start_idx),
            "personas": personas,
            "arsenalTable": arsenal_rows
        }
        campaigns.append(camp)

    return campaigns

def normalize_rcs_stage(rcs):
    if not rcs.get("zmot_theme"):
        rcs["zmot_theme"] = {"label": "pre_zmot", "id": "theme:pre_zmot"}
    return rcs


# ---------- main adapter ----------
def populate_rcs_with_strategy(
    scaffold: Dict[str, Any],
    strategy: Dict[str, Any],
    *,
    rcs_report: Optional[Dict[str, Any]] = None,
    persona_scores_rows: Optional[List[Dict[str, Any]]] = None  # [{name, belief, importance}]
) -> Dict[str, Any]:
    """
    Populates your scaffold (as given in your message) with build_account_strategy outputs.
    Optionally enriches from rcs_report (generate_rcs_fast) + persona score rows.
    Returns the mutated scaffold dict.
    """

    print("Populating scaffold for account:", scaffold.get("account_name") or "unknown")

    # 1) ZMOT theme (prefer rcs_report; else keep scaffold.zmot_theme as-is)
    zmot_theme = _pick_zmot_theme(rcs_report) or scaffold.get("zmot_theme", "")
    scaffold["zmot_theme"] = zmot_theme

    # 2) rcs_brief (pass-through from generate_rcs_fast if available; else keep scaffold defaults)
    if rcs_report:
        brief_keys = [
            "baseline","top_personas","concerns_by_persona","concern_backlog",
            "concern_coalitions","concern_sequences","coalitions",
            "core_scores","persona_scores","sequences_and_campaigns","causal_flows"
        ]
        scaffold["rcs_brief"] = {k: rcs_report.get(k, scaffold["rcs_brief"].get(k, [] if "concern" in k or k in ["coalitions","sequences_and_campaigns","causal_flows"] else {})) for k in brief_keys}
    print("rcs report generated in strat block")
    # 3) reverse_case_study stages (normalize from rcs_report if present)
    stages = _normalize_stages(rcs_report or {})
    scaffold.setdefault("reverse_case_study", {})
    scaffold["reverse_case_study"]["stages"] = stages
    print("stages normalized in strat block - starting campaigns block")
    # 4) execution_plan.campaigns from build_account_strategy
        # 4) execution_plan.campaigns from build_account_strategy OR synthesize from concern_sequences
    campaigns_in = []
    if isinstance(strategy, dict):
        campaigns_in = strategy.get("campaigns") or []
        print("strategy campaigns found in dict:")

    campaigns_out = []
    if campaigns_in:
        # Existing path: reshape provided campaigns
        print("Updating with strategy in campaigns in...")
        for idx, c in enumerate(campaigns_in):
            if c.get("arsenalTable"):
                arsenal_rows = c.get("arsenalTable")
            else:
                combos = c.get("assetCombos") or c.get("assets") or []
                arsenal_rows = _make_arsenal_rows(combos)

            campaigns_out.append({
                "id": c.get("id") or f"camp_{_slug(scaffold.get('account_name') or 'acct')}_{idx:02d}",
                "description": c.get("description") or "Account-specific campaign",
                "timeframe": c.get("timeframe") or _to_timeframe(idx),
                "personas": list(c.get("personas") or []),
                "arsenalTable": arsenal_rows
            })
            print(f"  campaign {idx}: id={campaigns_out[-1]['id']}, personas={campaigns_out[-1]['personas']}, assets={len(arsenal_rows)}")
    else:
        # NEW: synthesize from rcs_report.concern_sequences if strategy had no campaigns
        if rcs_report:
            synthesized = _synthesize_campaigns_from_sequences(rcs_report, start_idx=0)
            campaigns_out.extend(synthesized)
    print("campaigns synthesized in strat block")
    scaffold.setdefault("execution_plan", {})
    scaffold["execution_plan"]["campaigns"] = campaigns_out
    print("campaigns set in strat block")

    # 5) execution_plan.personas
    if persona_scores_rows is not None:
        # Use your math directly (already {name, belief, importance} in [0..1])
        ppl = sorted(persona_scores_rows, key=lambda x: (x.get("importance",0), x.get("belief",0)), reverse=True)
    else:
        # Derive minimal set from campaigns if math not provided
        ppl = [{"name": p, "belief": 0.5, "importance": 0.5} for p in _derive_personas_from_campaigns(campaigns_out)]
    scaffold["execution_plan"]["personas"] = ppl

    # 6) execution_plan.belief_state_summary
    scaffold["execution_plan"]["belief_state_summary"] = _belief_summary(ppl, stages)

    # 7) Optional conveniences for your extra arrays (plays/evidence/nextActions/hypotheses/objectives)
    # Map campaign rows to light-weight 'plays' if you want a quick scaffold:
    if not scaffold.get("plays"):
        plays = []
        for c in campaigns_out:
            for row in c["arsenalTable"]:
                plays.append({
                    "campaign_id": c["id"],
                    "asset": row["asset"],
                    "channel": row["channel"],
                    "target_personas": c["personas"],
                    "fitment": row["fitment"],
                    "expectedLift": row["expectedLift"]
                })
        scaffold["plays"] = plays

    # evidence can mirror stage evidence for quick start if empty
    if not scaffold.get("evidence"):
        ev = []
        for s in stages:
            for e in s.get("evidence", []):
                ev.append({"source": e, "stage": s.get("stage","")})
        scaffold["evidence"] = ev

    # nextActions heuristic if empty: first asset of first two campaigns
    if not scaffold.get("nextActions"):
        na = []
        for c in campaigns_out[:2]:
            if c["arsenalTable"]:
                na.append({"action": f"Launch: {c['arsenalTable'][0]['asset']}", "campaign_id": c["id"]})
        scaffold["nextActions"] = na

    # hypotheses/objectives defaults if empty
    scaffold.setdefault("objectives", scaffold.get("objectives") or {})
    scaffold["objectives"].setdefault("winHypothesis", scaffold["objectives"].get("winHypothesis") or (f"Win via {zmot_theme}" if zmot_theme else "Win via belief lift across economic + tech buyers"))
    scaffold["objectives"].setdefault("kpis", scaffold["objectives"].get("kpis") or [
        {"name": "Sample Persona Belief ≥ 0.XX", "target": "by end of Campaign 1"},
        {"name": "Close cycle time", "target": "-Y% vs baseline"}
    ])

    return scaffold

#-------------------------------------------------------

def load_account_rcs_json(product_id: str, account_id: str) -> Dict[str, Any]:
    from backend.utils.strategy_builder.comprehensive_plan_generator import load_account_rcs_from_disk
    print("Processing account_id:", account_id)
    rcs, _ = load_account_rcs_from_disk(product_id, account_id)
    print("Loaded RCS in strat load_ac cycle", account_id)
    
    return rcs



def rcs_for_target_account(product_id: str, account_id: str) -> Dict[str, Any]:
    # 1) Load your scaffold (as you pasted)
    product_id_actual = get_product_id_from_subgraph(build_product_graph(product_id))
    scaffold = load_account_rcs_json(product_id_actual, account_id)  # or construct it as dict
    if not scaffold:
        raise ValueError(f"No scaffold found for product_id={product_id}, account_id={account_id}")
    # 2) Get your current outputs
    # NOTE: use your actual imports/paths; these are indicative
    product_subgraph = build_product_graph(product_id) 
    print("Setting ac inputs from scaffold")
    account_inputs = scaffold_to_orchestrator_inputs(
        scaffold=scaffold,
        G=product_subgraph)
    account_name = scaffold.get("account_name") or "unknown"
    print("Starting frozen strategy build...")
    frozen_strategy, G_work, rcs_report = build_account_strategy(
        product_subgraph=product_subgraph,
        account_id=account_id,
        account_name=account_name,
        attributes=account_inputs["attributes"],
        zmots=account_inputs["zmots"],
        initial_engagements=account_inputs["initial_engagements"],
        boost_factor=2.0,
        scaffold=scaffold,
    )

    print("Strategy built - starting populate...")

    filled = populate_rcs_with_strategy(
        scaffold=scaffold,
        strategy=frozen_strategy,
        rcs_report=rcs_report,
        persona_scores_rows=frozen_strategy.get("persona_rows")
    )

    print("filling completed")

    return filled

def construct_all_account_rcs(product_id: str) -> Dict[str, Any]:
    from backend.utils.strategy_builder.comprehensive_plan_generator import _save_rcs_json, load_account_rcs_from_disk

    print("Constructing all account RCS for product_id:", product_id)
    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)
    account_ids = get_target_account_ids(product_id_actual, {"status": {"nin": ["Closed-won", "Closed-lost"]}})
    print("Account IDs fetched for product_id", product_id, ":", account_ids)
    rcs_list = []
    for acc in account_ids:
        print("Generating RCS for account_id:", acc)
        rcs = rcs_for_target_account(product_id, acc)
        rcs = normalize_rcs_stage(rcs)
        _save_rcs_json(product_id_actual, acc, rcs)
        rcs_list.append(rcs)
    
    print("Completed RCS generation for all accounts.")
    print("---------------------------------")
    return rcs_list



#----------- Stray Code ---------------
"""

def _synthesize_campaigns_from_sequences_for_strategy(
    account_name: str,
    rcs_report: Dict[str, Any],
    *,
    start_idx: int = 0
) -> List[Dict[str, Any]]:
    sequences = rcs_report.get("concern_sequences") or []
    if not sequences:
        return []
    concern_idx = _index_concerns_for_strategy(rcs_report)
    camps = []
    prefix = _slug(account_name or "acct")
    for i, seq in enumerate(sequences):
        steps = seq.get("sequence") or []
        personas = []
        arsenal_rows = []
        for st in steps:
            pname = st.get("persona")
            if pname:
                personas.append(pname)
            cid = st.get("cid") or st.get("concern_id")
            info = concern_idx.get(cid or "", {"label": f"Concern {cid}" if cid else "Concern"})
            arsenal_rows.append(_lift_from_final_win_or_stage(seq, info))
        personas = sorted({p for p in personas if p})
        camps.append({
            "id": _mk_campaign_id(prefix, i + start_idx),
            "description": f"Resolve prioritized concerns across {', '.join(personas[:2])}." if personas else "Resolve prioritized concerns.",
            "timeframe": _to_timeframe(i + start_idx),
            "personas": personas,
            "arsenalTable": arsenal_rows,
            "milestones": [
                {"name": "Asset #1 live", "due": _to_timeframe(i + start_idx)["startDate"]},
                {"name": "Activation sprint", "due": _to_timeframe(i + start_idx)["startDate"]}
            ],
            "success_gates": [
                {"metric": "Qualified meetings", "target": "≥ 8"},
                {"metric": "Key persona belief", "target": "≥ 0.70"}
            ]
        })
    return camps


def _seq_title(seq_idx: int, theme: Optional[str], personas: List[str]) -> str:
    if theme:
        return f"{theme} — Sequence {seq_idx+1}"
    if personas:
        return f"Campaign {seq_idx+1}: {', '.join(personas[:2])}"
    return f"Campaign {seq_idx+1}"

def _index_concerns_for_strategy(rcs_report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for pname, items in (rcs_report.get("concerns_by_persona") or {}).items():
        for it in items or []:
            cid = it.get("cid") or it.get("concern_id")
            if not cid: 
                continue
            idx.setdefault(cid, {})["label"] = it.get("concern_label") or it.get("label") or f"Concern {cid}"
            idx[cid]["stage"] = idx[cid].get("stage") or it.get("stage")
            idx[cid]["persona"] = idx[cid].get("persona") or pname
    for it in (rcs_report.get("concern_backlog") or []):
        cid = it.get("cid") or it.get("concern_id")
        if not cid: 
            continue
        cur = idx.setdefault(cid, {})
        cur["label"] = cur.get("label") or it.get("label") or it.get("concern_label") or f"Concern {cid}"
        cur["stage"] = cur.get("stage") or it.get("stage")
    return idx

def _mk_campaign_id(prefix: str, i: int) -> str:
    return f"camp_{prefix}_{i:02d}"

def _lift_from_final_win_or_stage(seq_item: Dict[str, Any], step_info: Dict[str, Any]) -> Dict[str, Any]:
    # Build a tiny "combo" so we can reuse _expected_lift()
    combo = {
        "asset": f"Play: Resolve “{step_info.get('label','Concern')}”",
        "recommendedChannels": ["Email Nurture", "LinkedIn Ads", "Sales Assist"],
        "fitment": step_info.get("stage") or "Concern resolution",
        "engagementScore": 0.6 if step_info.get("stage") in ("Problem Realization", "Barriers") else 0.5
    }
    if isinstance(seq_item.get("final_win"), (int, float)) and seq_item["final_win"] > 0:
        combo["fitScore"] = float(seq_item["final_win"])
    return {
        "asset": combo["asset"],
        "channel": combo["channel"],
        "fitment": combo["fitment"],
        "engagement": "Medium",
        "expectedLift": _expected_lift(combo)
    }
"""