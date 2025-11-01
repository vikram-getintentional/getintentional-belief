# ============================
# File: backend/utils/inference/rcs_generators/rcs_helpers/strategy_orchestrators.py
# ============================
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs



# ---------------------------------------------------------------------
# Utility (local)
# ---------------------------------------------------------------------
def _to_engaged_payload(ids: List[Any]) -> List[Dict]:
    """
    Accepts:
      - ['node1','node2', ...]
      - [{'id':'node1', 'occurrence':2}, ...]  (passes through)
    Returns: [{'id': <id>, 'occurrence': 1..N}, ...] with stable order.
    """
    if not ids:
        return []
    if isinstance(ids[0], dict):
        out = []
        seen = set()
        for row in ids:
            nid = row.get("id")
            if not nid or nid in seen:
                continue
            occ = row.get("occurrence", 1.0)
            out.append({"id": nid, "occurrence": float(occ)})
            seen.add(nid)
        return out

    uniq = sorted({str(nid) for nid in ids if nid})
    return [{"id": nid, "occurrence": 1.0} for nid in uniq]


def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "")).strip("_")[:48]


def _to_timeframe(idx: int) -> Dict[str, str]:
    start = date.today() + timedelta(days=14 * idx)
    end = start + timedelta(days=84)
    return {"startDate": start.isoformat(), "endDate": end.isoformat()}


def _engagement_label(x: float) -> str:
    return "High" if x >= 0.7 else ("Medium" if x >= 0.4 else "Low")


def _expected_lift_label(stage: str, fitness: float) -> str:
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


def _expected_lift(combo: Dict[str, Any]) -> str:
    if isinstance(combo.get("expectedLift"), str):
        return combo["expectedLift"]
    score = float(combo.get("fitScore", combo.get("engagementScore", 0.5)) or 0.5)
    pct = max(8, min(25, int(100 * score * 0.25)))
    return f"+{pct}%"


def _make_arsenal_rows(combos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for combo in combos or []:
        asset_label = combo.get("asset") or combo.get("assetId") or combo.get("name") or "Asset"
        channel = combo.get("Channel") or combo.get("channel") or combo.get("channel_id") or ""
        rows.append(
            {
                "asset": asset_label,
                "channel": channel,
                "fitment": combo.get("fitment") or ", ".join(combo.get("concernsAddressed", [])) or "High fit",
                "engagement": combo.get("engagement") or combo.get("expectedEngagement", "Medium"),
                "engagementLabel": _engagement_label(float(combo.get("engagementScore", 0.5) or 0.5)),
                "expectedLift": _expected_lift(combo),
                "expectedLiftLabel": _expected_lift_label(
                    combo.get("stage") or "zmot", float(combo.get("fitScore", 0.5) or 0.5)
                ),
                "duration": combo.get("duration") or combo.get("expectedDuration", "2-4 weeks"),
            }
        )
    return rows


def _derive_personas_from_campaigns(campaigns: List[Dict[str, Any]]) -> List[str]:
    s = set()
    for c in campaigns or []:
        for p in c.get("personas", []) or []:
            s.add(p)
    return sorted(s)


def _belief_summary(personas_rows: List[Dict[str, Any]], stages: List[Dict[str, Any]]) -> Dict[str, Any]:
    import statistics as stats

    avg = round(stats.mean([p.get("belief", 0.0) for p in personas_rows]) if personas_rows else 0.0, 4)
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
    req = ["stage", "trigger", "belief_before", "belief_after", "personas", "pains"]
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
    if not out:
        out = [
            {
                "stage": "",
                "trigger": "",
                "belief_before": "",
                "belief_after": "",
                "personas": [],
                "pains": [],
                "evidence": [],
                "recommended_assets": [],
            }
        ]
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
    return trig[:80]


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
    ct = set(re.findall(r"[a-z0-9]+", c.lower()))
    tt = set(re.findall(r"[a-z0-9]+", t.lower()))
    if not ct or not tt:
        return 0.0
    overlap = len(ct & tt) / max(1, len(tt))
    prefix = 1.0 if c.lower().startswith(t.lower()) or t.lower().startswith(c.lower()) else 0.0
    return min(1.0, 0.6 * overlap + 0.4 * prefix)


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


# ---------------------------------------------------------------------
# BUILD STRATEGY  (Frozen)
# ---------------------------------------------------------------------
def build_account_strategy(
    *,
    G: nx.DiGraph,
    account_id: str,
    product_id: str,
    archetype: Optional[Dict[str, Any]] = None,
    engaged_nodes: Optional[List[Dict[str, Any]]] = None,
    limits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Orchestrates: generate_rcs -> frozen_strategy (phases) with the seeded graph.
    Returns a single dict with phases + a pass-through RCS report (canonical).
    """
    print(f"[build_account_strategy] account={account_id} product={product_id}")

    product_subgraph = G
    engaged_payload = _to_engaged_payload(engaged_nodes or [])

    rcs_report = generate_rcs(
        product_graph=product_subgraph,
        original_graph=G,
        engaged_nodes=engaged_payload,
        max_steps_per_persona=6,
    )

    # ensure sequences aliasing
    if "sequences" not in rcs_report:
        rcs_report["sequences"] = rcs_report.get("concern_sequences") or []
    rcs_report["concern_sequences"] = rcs_report.get("sequences", [])

    # IMPORTANT: downstream builders should see the seeded graph for phases
    seeded_graph = rcs_report.get("graph") or product_subgraph

    phases = build_frozen_strategy(
        product_subgraph=seeded_graph,
        report=rcs_report,
        account_id=account_id,
        product_id=product_id,
        archetype=archetype or {},
        limits=limits or {},
    )

    # Minimal top-N concerns for UI heads-up cards
    concern_backlog = rcs_report.get("concern_backlog", [])
    sequences_for_ui = rcs_report.get("sequences", [])

    top_concerns = []
    seen_c = set()
    for row in concern_backlog:
        lab = row.get("concern_label") or row.get("cid")
        if lab and lab not in seen_c:
            top_concerns.append(lab)
            seen_c.add(lab)
        if len(top_concerns) >= 6:
            break

    # Attach arsenal tables for each campaign/phase
    attach_arsenal_tables_to_phases(
        frozen_strategy={"phases": phases, "report": rcs_report},
        product_id=product_id,
        persona_scores=rcs_report.get("persona_scores", {}),
    )

    return {
        "phases": phases,
        "frozen_personas": rcs_report.get("persona_scores", {}),
        "top_concerns": top_concerns,
        "report": {
            **rcs_report,
            "sequences": sequences_for_ui,
            "concern_sequences": sequences_for_ui,
        },
    }


# ---------------------------------------------------------------------
# UPDATE STRATEGY  (Tactical)
# ---------------------------------------------------------------------
def update_tactical_plan(
    *,
    G: nx.DiGraph,
    account_id: str,
    product_id: str,
    archetype: Optional[Dict[str, Any]] = None,
    engaged_nodes: Optional[List[Dict[str, Any]]] = None,
    limits: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Recompute near-term sequences and campaigns after new engagement signals.
    """
    print(f"[update_tactical_plan] account={account_id} product={product_id}")

    product_subgraph = G
    engaged_payload = _to_engaged_payload(engaged_nodes or [])

    rcs_report = generate_rcs(
        product_graph=product_subgraph,
        original_graph=G,
        engaged_nodes=engaged_payload,
        max_steps_per_persona=6,
    )

    if "sequences" not in rcs_report:
        rcs_report["sequences"] = rcs_report.get("concern_sequences") or []
    rcs_report["concern_sequences"] = rcs_report.get("sequences", [])

    seeded_graph = rcs_report.get("graph") or product_subgraph

    phases = build_frozen_strategy(
        product_subgraph=seeded_graph,
        report=rcs_report,
        account_id=account_id,
        product_id=product_id,
        archetype=archetype or {},
        limits=limits or {},
    )

    next_campaigns = (phases[0].get("campaigns") if phases else []) or []
    next_sequences = rcs_report.get("sequences", [])[:4]
    concern_backlog = rcs_report.get("concern_backlog", [])[:20]

    return {
        "phases": phases,
        "next_sequences": next_sequences,
        "next_campaigns": next_campaigns,
        "concern_backlog": concern_backlog,
        "report": {
            **rcs_report,
            "sequences": rcs_report.get("sequences", []),
            "concern_sequences": rcs_report.get("sequences", []),
        },
    }


# ---------------- Scaffold Mapping Internals ----------------
def _find_attribute_node(
    G: nx.DiGraph,
    dim: str,
    val: str,
    label_fields: Tuple[str, ...],
    node_type_field: str,
) -> Optional[str]:
    """
    Expect graph attribute nodes like:
      node_type='attribute_value', dimension in {'industry','geo','funding','revenue','employees'}, attr_val='SaaS' ...
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
    thresh = 0.45 if any(tok in target for tok in ["$", "-", "k", "range"]) else 0.6
    return best[0] if best[1] >= thresh else None


def _find_label_node(
    G: nx.DiGraph,
    label: str,
    allowed_types: Tuple[str, ...],
    label_fields: Tuple[str, ...],
    node_type_field: str,
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
    node_type_field: str,
) -> List[str]:
    out: List[str] = []
    for key, types in [
        ("target_personas", ("persona",)),
        ("personas", ("persona",)),
        ("jobs", ("job",)),
        ("pains", ("pain",)),
        ("tags", ("persona", "job", "pain")),
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
    max_occurrence: int = 5,
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


# --- Main Scaffold → Orchestrator inputs ----
def scaffold_to_orchestrator_inputs(
    *,
    scaffold: Dict[str, Any],
    G: nx.DiGraph,
    attribute_key_map: Optional[Dict[str, str]] = None,
    label_fields: tuple = ("label", "name", "title"),
    node_type_field: str = "node_type",
    max_occurrence: int = 5,
) -> Dict[str, Any]:
    """
    Converts the account RCS scaffold to inputs for the orchestrator.
    We currently only use engaged_nodes (weights), but keep others for future use.
    """
    print("Starting scaffold to orchestrator inputs mapping...")
    attribute_key_map = attribute_key_map or {
        "industry": "industry",
        "revenue_range": "revenue_range",
        "employee_range": "employee_range",
        "geography": "geography",
        "funding_stage": "funding_stage",
    }

    attributes = _attributes_to_node_ids(
        scaffold.get("archetype", {}) or {}, G, attribute_key_map, label_fields, node_type_field
    )
    zmots = _zmot_to_node_ids(scaffold, G, label_fields, node_type_field)
    initial_engagements = _initial_engagements_to_node_ids(scaffold, G, label_fields, node_type_field)
    engaged_nodes = _engaged_nodes_with_occurrence(
        scaffold, G, label_fields, node_type_field, max_occurrence=max_occurrence
    )

    return {
        "attributes": attributes,
        "zmots": zmots,
        "initial_engagements": initial_engagements,
        "engaged_nodes": engaged_nodes,
    }


# ---- Skinny campaign synthesis (fallback when strategy has none)
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
                    "persona": pname,
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
    start_idx: int = 0,
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

        for step in steps:
            pname = step.get("persona")
            if pname:
                personas.append(pname)
            cid = step.get("cid") or step.get("concern_id")
            c_info = concern_idx.get(cid or "", {})
            label = c_info.get("label") or (f"Concern {cid}" if cid else "Concern")

            seq_lift = seq.get("final_win")
            combo = {
                "asset": f"[TBD] Resolve “{label}”",
                "recommendedChannels": ["Email Nurture", "LinkedIn Ads", "Sales Assist"],
                "fitment": c_info.get("stage") or "Concern resolution",
                "engagementScore": 0.6 if c_info.get("stage") in ("Problem Realization", "Barriers") else 0.5,
            }
            if isinstance(seq_lift, (int, float)) and seq_lift > 0:
                combo["fitScore"] = float(seq_lift)

            arsenal_rows.append(
                {
                    "asset": combo["asset"],
                    "channel": ", ".join(combo["recommendedChannels"]),
                    "fitment": combo["fitment"],
                    "engagement": "Medium",
                    "expectedLift": _expected_lift(combo),
                }
            )

        personas = sorted({p for p in personas if p})
        description = (
            f"Resolve prioritized concerns across {' & '.join(personas[:2])}."
            if personas
            else "Resolve prioritized concerns."
        )
        camp = {
            "id": f"camp_seq_{i + start_idx:02d}",
            "description": description,
            "timeframe": _to_timeframe(i + start_idx),
            "personas": personas,
            "arsenalTable": arsenal_rows,
        }
        campaigns.append(camp)

    return campaigns


def normalize_rcs_stage(rcs: Dict[str, Any]) -> Dict[str, Any]:
    if not rcs.get("zmot_theme"):
        rcs["zmot_theme"] = {"label": "pre_zmot", "id": "theme:pre_zmot"}
    return rcs


# ---------- Public adapter: fill scaffold ----------
def populate_rcs_with_strategy(
    scaffold: Dict[str, Any],
    strategy: Dict[str, Any],
    *,
    rcs_report: Optional[Dict[str, Any]] = None,
    persona_scores_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Populates the scaffold with phases/campaigns.
    If strategy has no campaigns, we synthesize from rcs_report.concern_sequences.
    """
    print("Populating scaffold for account:", scaffold.get("account_name") or "unknown")

    # 1) ZMOT theme
    zmot_theme = _pick_zmot_theme(rcs_report) or scaffold.get("zmot_theme", "")
    scaffold["zmot_theme"] = zmot_theme

    # 2) rcs_brief passthrough
    if rcs_report:
        brief_keys = [
            "baseline",
            "top_personas",
            "concerns_by_persona",
            "concern_backlog",
            "concern_coalitions",
            "concern_sequences",
            "coalitions",
            "core_scores",
            "persona_scores",
            "sequences_and_campaigns",
            "causal_flows",
        ]
        scaffold.setdefault("rcs_brief", {})
        for k in brief_keys:
            if k in rcs_report:
                scaffold["rcs_brief"][k] = rcs_report[k]
    print("rcs report generated in strat block")

    # 3) reverse_case_study stages
    stages = _normalize_stages(rcs_report or {})
    scaffold.setdefault("reverse_case_study", {})
    scaffold["reverse_case_study"]["stages"] = stages
    print("stages normalized in strat block - starting campaigns block")

    # 4) campaigns: prefer strategy-provided, else synthesize from sequences
    campaigns_in = strategy.get("phases", [{}])[0].get("campaigns", []) if isinstance(strategy, dict) else []
    if not campaigns_in:
        campaigns_in = strategy.get("campaigns") or []
    print("strategy campaigns found in dict:" if campaigns_in else "no strategy campaigns; will synthesize")

    campaigns_out: List[Dict[str, Any]] = []
    if campaigns_in:
        for idx, c in enumerate(campaigns_in):
            if c.get("arsenalTable"):
                arsenal_rows = c.get("arsenalTable")
            else:
                combos = c.get("assetCombos") or c.get("assets") or []
                arsenal_rows = _make_arsenal_rows(combos)
            campaigns_out.append(
                {
                    "id": c.get("id") or f"camp_{_slug(scaffold.get('account_name') or 'acct')}_{idx:02d}",
                    "description": c.get("description") or "Account-specific campaign",
                    "timeframe": c.get("timeframe") or _to_timeframe(idx),
                    "personas": list(c.get("personas") or []),
                    "arsenalTable": arsenal_rows,
                }
            )
            print(
                f"  campaign {idx}: id={campaigns_out[-1]['id']}, personas={campaigns_out[-1]['personas']}, assets={len(arsenal_rows)}"
            )
    else:
        if rcs_report:
            campaigns_out.extend(_synthesize_campaigns_from_sequences(rcs_report, start_idx=0))
    print("campaigns synthesized in strat block")

    scaffold.setdefault("execution_plan", {})
    scaffold["execution_plan"]["campaigns"] = campaigns_out
    print("campaigns set in strat block")

    # 5) execution_plan.personas
    if persona_scores_rows is not None:
        ppl = sorted(
            persona_scores_rows, key=lambda x: (x.get("importance", 0), x.get("belief", 0)), reverse=True
        )
    else:
        ppl = [{"name": p, "belief": 0.5, "importance": 0.5} for p in _derive_personas_from_campaigns(campaigns_out)]
    scaffold["execution_plan"]["personas"] = ppl

    # 6) belief summary
    scaffold["execution_plan"]["belief_state_summary"] = _belief_summary(ppl, stages)

    # 7) Convenience scaffolds (plays/evidence/nextActions/objectives)
    if not scaffold.get("plays"):
        plays = []
        for c in campaigns_out:
            for row in c.get("arsenalTable", []):
                plays.append(
                    {
                        "campaign_id": c["id"],
                        "asset": row.get("asset"),
                        "channel": row.get("channel"),
                        "target_personas": c.get("personas", []),
                        "fitment": row.get("fitment"),
                        "expectedLift": row.get("expectedLift"),
                    }
                )
        scaffold["plays"] = plays

    if not scaffold.get("evidence"):
        ev = []
        for s in stages:
            for e in s.get("evidence", []):
                ev.append({"source": e, "stage": s.get("stage", "")})
        scaffold["evidence"] = ev

    if not scaffold.get("nextActions"):
        na = []
        for c in campaigns_out[:2]:
            rows = c.get("arsenalTable") or []
            if rows:
                na.append({"action": f"Launch: {rows[0]['asset']}", "campaign_id": c["id"]})
        scaffold["nextActions"] = na

    scaffold.setdefault("objectives", scaffold.get("objectives") or {})
    scaffold["objectives"].setdefault(
        "winHypothesis",
        scaffold["objectives"].get("winHypothesis")
        or (f"Win via {zmot_theme}" if zmot_theme else "Win via belief lift across economic + tech buyers"),
    )
    scaffold["objectives"].setdefault(
        "kpis",
        scaffold["objectives"].get("kpis")
        or [
            {"name": "Sample Persona Belief ≥ 0.XX", "target": "by end of Campaign 1"},
            {"name": "Close cycle time", "target": "-Y% vs baseline"},
        ],
    )

    return scaffold


# ---------- I/O helpers ----------
def load_account_rcs_json(product_id: str, account_id: str) -> Dict[str, Any]:
    # we import here to avoid a hard dependency at module import time
    try:
        from backend.utils.strategy_builder.comprehensive_plan_generator import load_account_rcs_from_disk
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing comprehensive_plan_generator.load_account_rcs_from_disk") from e

    print("Processing account_id:", account_id)
    rcs, _ = load_account_rcs_from_disk(product_id, account_id)
    print("Loaded RCS in strat load_ac cycle", account_id)
    return rcs


# ---------- End-to-end per account ----------
def rcs_for_target_account(product_id: str, account_id: str) -> Dict[str, Any]:
    """
    End-to-end adapter:
      1) Load scaffold
      2) Build product graph
      3) Map scaffold -> engaged_nodes
      4) Run build_account_strategy
      5) Populate scaffold with phases/campaigns
    """
    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)

    print("Running scaffold for ac")

    scaffold = load_account_rcs_json(product_id_actual, account_id)
    if not scaffold:
        raise ValueError(f"No scaffold found for product_id={product_id}, account_id={account_id}")

    print("Setting ac inputs from scaffold")
    account_inputs = scaffold_to_orchestrator_inputs(scaffold=scaffold, G=product_subgraph)
    print(
        "[orchestrator inputs]",
        "attrs:",
        len(account_inputs.get("attributes", [])),
        "zmots:",
        len(account_inputs.get("zmots", [])),
        "init:",
        len(account_inputs.get("initial_engagements", [])),
        "engaged:",
        len(account_inputs.get("engaged_nodes", [])),
    )

    # Merge attribute/ZMOT/initial seeds into engaged payload
    engaged_payload = list(account_inputs.get("engaged_nodes") or [])
    attr_ids = account_inputs.get("attributes") or []
    zmot_ids = account_inputs.get("zmots") or []
    init_ids = account_inputs.get("initial_engagements") or []

    engaged_payload.extend({"id": nid, "occurrence": 1} for nid in attr_ids)
    engaged_payload.extend({"id": nid, "occurrence": 2} for nid in zmot_ids)
    engaged_payload.extend({"id": nid, "occurrence": 1} for nid in init_ids)

    # dedup by id, keep highest occurrence
    _best = {}
    for r in engaged_payload:
        rid = r["id"]
        _best[rid] = max(_best.get(rid, 0), int(r.get("occurrence", 1)))
    engaged_payload = [{"id": k, "occurrence": v} for k, v in _best.items()]
    engaged_payload.sort(key=lambda r: (-r["occurrence"], r["id"]))
    print("[seed] using engaged nodes:", len(engaged_payload))

    print("Starting frozen strategy build...")
    strat = build_account_strategy(
        G=product_subgraph,
        account_id=account_id,
        product_id=product_id_actual,
        engaged_nodes=engaged_payload,
        archetype=scaffold.get("archetype"),
    )

    print("Strategy built - starting populate...")
    filled = populate_rcs_with_strategy(
        scaffold=scaffold,
        strategy=strat,  # may not include campaigns; synth handles it
        rcs_report=strat.get("report"),  # pass-through from generate_rcs
        persona_scores_rows=None,  # optional: supply if you have your own persona rows
    )
    print("filling completed")
    return filled


# ---------- Batch (generate for many accounts) ----------
def construct_all_account_rcs(product_id: str, account_id: str | None = None) -> List[Dict[str, Any]]:
    # local import to avoid import-time errors when running in isolation
    try:
        from backend.utils.strategy_builder.comprehensive_plan_generator import _save_rcs_json
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing comprehensive_plan_generator._save_rcs_json") from e

    print("Constructing all account RCS for product_id:", product_id)
    product_subgraph = build_product_graph(product_id)
    product_id_actual = get_product_id_from_subgraph(product_subgraph)
    account_ids = get_target_account_ids(
        product_id_actual, {"status": {"nin": ["Closed-won", "Closed-lost"]}}
    )

    print("Account IDs fetched for product_id", product_id, ":", account_ids)

    if account_id:
        print("Specific account_id provided:", account_id)
        if account_id in account_ids:
            account_ids = [account_id]
        else:
            # allow explicit single account even if not in filter list
            account_ids = [account_id]

    rcs_list: List[Dict[str, Any]] = []
    for acc in account_ids:
        print("Generating RCS for account_id:", acc)
        rcs = rcs_for_target_account(product_id, acc)
        rcs = normalize_rcs_stage(rcs)
        _save_rcs_json(product_id_actual, acc, rcs)
        rcs_list.append(rcs)

    print("Completed RCS generation for all accounts.")
    print("---------------------------------")
    return rcs_list
