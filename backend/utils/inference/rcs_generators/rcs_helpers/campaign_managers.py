#---- Helpers for Arsenal Mapping ---
import re
from typing import Any, Dict, List, Optional
import networkx as nx
from datetime import date, datetime, timedelta
from backend.utils.inference.rcs_generators.generate_rcs_fast import _nt

from backend.utils.knowledge_base.arsenal.execution_arsenal_repository import get_best_plays_for_concern

def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "x"


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


def _index_concerns_from_report(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup: cid -> {label, stage, persona}
    Uses concern_backlog; falls back to concerns_by_persona when needed.
    """
    idx: Dict[str, Dict[str, Any]] = {}
    for row in (report.get("concern_backlog") or []):
        cid = row.get("cid") or row.get("concern_id")
        if not cid:
            continue
        idx[cid] = {
            "label": row.get("concern_label") or row.get("label") or f"Concern {cid}",
            "stage": row.get("stage"),
            "persona": row.get("pid")
        }
    if not idx:
        for pname, items in (report.get("concerns_by_persona") or {}).items():
            for it in items or []:
                cid = it.get("cid") or it.get("concern_id")
                if not cid:
                    continue
                idx.setdefault(cid, {
                    "label": it.get("concern_label") or it.get("label") or f"Concern {cid}",
                    "stage": it.get("stage"),
                    "persona": pname
                })
    return idx

def _fallback_asset_row_from_step(step: dict) -> dict:
    """Graceful fallback play when ranker yields no results."""
    label = step.get("concern_label") or step.get("cid") or "Concern"
    return {
        "asset": {"id": "tbd", "name": f"Play: Resolve “{label}”", "format": "email", "evergreen": True},
        "channel": {"id": "email", "name": "Email", "type": "email", "reach_score": 0.6},
        "fitness": 0.4,
        "engagement_score": 0.5,
        "why": "fallback",
    }


def _engagement_bucket(x: float) -> str:
    if x is None: return "Medium"
    if x >= 0.66: return "High"
    if x >= 0.33: return "Medium"
    return "Low"

def _play_to_arsenal_row(play: dict, *, cinfo: dict) -> dict:
    """Convert ranked play → Arsenal table row with concern context."""
    asset = play.get("asset", {}) or {}
    channel = play.get("channel", {}) or {}
    fitness = float(play.get("fitness") or play.get("fitness_score") or 0.0)
    engage = float(play.get("engagement_score") or 0.0)
    stage = (cinfo.get("stage") or "").lower()
    label = cinfo.get("label")

    return {
        "asset": {
            "id": asset.get("id"),
            "name": asset.get("name"),
            "format": asset.get("format"),
            "evergreen": bool(asset.get("evergreen", False)),
        },
        "channel": {
            "id": channel.get("id"),
            "name": channel.get("name") or channel.get("type"),
            "type": channel.get("type"),
            "reach_score": channel.get("reach_score"),
        },
        "stage": stage,
        "fitment": f"{round(fitness * 100)}%",
        "engagement": "High" if engage >= 0.66 else "Medium" if engage >= 0.33 else "Low",
        "expectedLift": f"+{round((0.5 * fitness + 0.2 * engage) * 100)}%",
        "concernsAddressed": [label] if label else [],
        "why": play.get("why"),
    }


def _campaigns_from_sequences_using_arsenal(
    *,
    G: nx.DiGraph,
    account_id: Optional[str],
    product_id: Optional[str],
    report: Dict[str, Any],
    start_idx: int = 0,
    plays_per_step: int = 3,
    archetype: Optional[Dict] = None,
    exclude_pairs: Optional[set[tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """
    For each concern-sequence, fetch best play(s) per step via get_best_plays_for_concern
    and convert into execution_plan.campaigns entries.
    """
    print("starting campaign generation for:", product_id, " with exclude pairs: ", exclude_pairs)
    sequences = report.get("concern_sequences") or []
    if not sequences:
        return []

    concern_idx = _index_concerns_from_report(report)
    prefix = _slug(account_id or "acct")
    campaigns: List[Dict[str, Any]] = []

    # safe helpers
    def _node(G, nid: Optional[str]) -> Dict[str, Any]:
        if nid and nid in G:
            return G.nodes[nid]
        return {}

    def _persona_meta_from_graph(G, pid: Optional[str]) -> Dict[str, str]:
        d = _node(G, pid)
        return {
            "title": d.get("title") or d.get("name") or d.get("label") or (pid or ""),
            "department": d.get("department") or "General",
            "seniority": d.get("seniority") or "Manager",
        }

    def _stage_norm_local(stage: Optional[str]) -> str:
        s = (stage or "").strip().lower()
        if s == "resolution":
            return "solution"
        if s in {"problem", "pain", "execution", "solution"}:
            return s
        # fallbacks for common typos
        if s in {"solve", "soln", "solutions"}:
            return "solution"
        return "execution"


    for i, seq in enumerate(sequences):
        print("starting steps in camp seq")
        
        steps = seq.get("sequence") or []
        personas: List[str] = []
        arsenal_rows: List[Dict[str, Any]] = []

        for st in steps:

            # 1) Extract step fields FIRST (don't touch cid before this)
            cid   = st.get("cid")
            stage = st.get("stage")
            label = st.get("concern_label") or (G.nodes[cid].get("label") if cid in G else cid)
            pid   = st.get("persona") or (concern_idx.get(cid, {}) or {}).get("persona")

            # 2) Persona meta + bookkeeping
            persona_meta  = _persona_meta_from_graph(G, pid)
            if pid:
                st.get("persona") or (concern_idx.get(st.get("cid") or "", {}) or {}).get("persona")
                personas.append(pid)

            # 3) Concern meta used by arsenal row + fallbacks
            concern_type = _nt(G, cid)  # optional, if you want to log it
            cinfo = concern_idx.get(cid) or {
                "concern_id": cid,
                "label":      label,
                "stage":      stage,
            }

            # 4) Archetype safe default
            """archetype = {
                "industry":        scaffold["archetype"].get("industry", ""),
                "geography":       scaffold["archetype"].get("geography", ""),
                "revenue_range":   scaffold["archetype"].get("revenue_range", ""),
                "employee_range":  scaffold["archetype"].get("employee_range", ""),
                "funding_stage":   scaffold["archetype"].get("funding_stage", ""),
                "competitors_used":scaffold["archetype"].get("competitors_used", ""),
                "tech_stack":      scaffold["archetype"].get("tech_stack", "")
            }
            """
            plays = []
            try:
                print("trying get_best_plays block in _campaigns_from_sequences")
                # IMPORTANT: pass concern_id (not concern_type) and include the label
                plays = get_best_plays_for_concern(
                    persona_alias = persona_meta,
                    concern_label = label,
                    concern_stage = stage,
                    product_id    = product_id,
                    archetype     = archetype,
                    top_k         = plays_per_step,
                    exclude_pairs = exclude_pairs,
                ) or []
            except Exception as e:
                print("⚠️ get_best_plays_for_concern failed:", e)
                plays = []

            if not plays:
                # graceful fallback
                print(f"⚠️ No plays found for concern {cid} ({label}) at stage '{stage}' for persona {pid}; using generic fallback.")
                plays = [{
                    "asset":   {"id": "fallback", "name": f"Play: Resolve “{label}”", "format": "email", "evergreen": True},
                    "channel": {"id": "email", "name": "Email", "type": "email", "reach_score": 0.6},
                    "fitness": 0.6,
                    "why":     "generic fallback"
                }]

            for p in plays:
                arsenal_rows.append(_play_to_arsenal_row(p, cinfo=cinfo))
                # remember asset×channel combo to avoid reuse this quarter
                if exclude_pairs is not None:
                    aid = (p.get("asset", {}) or {}).get("id")
                    cid2 = (p.get("channel", {}) or {}).get("id")
                    if aid and cid2:
                        exclude_pairs.add((aid, cid2))


        personas = sorted({p for p in personas if p})
        print("Personas in campaign mgr: ", personas)
        campaigns.append({
            "id": f"camp_{prefix}_{i+start_idx:02d}",
            "description": (
                f"Resolve prioritized concerns across {', '.join(personas[:2])}."
                if personas else "Resolve prioritized concerns."
            ),
            "timeframe": _to_timeframe(i + start_idx),
            "arsenalTable": arsenal_rows,
            "personas": personas,
        })

    print("Campaign generation complete")
    
    return campaigns

