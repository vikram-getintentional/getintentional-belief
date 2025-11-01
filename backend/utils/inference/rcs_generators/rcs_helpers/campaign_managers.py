# backend/utils/inference/rcs_generators/rcs_helpers/campaign_managers.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re

from backend.utils.knowledge_base.arsenal.execution_arsenal_repository import (
    get_best_plays_for_concern,
)

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _slug(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "")).strip("_")[:64]

def _engagement_label(x: float) -> str:
    try:
        x = float(x or 0.0)
    except Exception:
        x = 0.0
    return "High" if x >= 0.7 else ("Medium" if x >= 0.4 else "Low")

def _expected_lift_label(stage: str, fitness: float) -> str:
    # coarse stage-based scale; tune later
    base = {
        "pre_zmot": (0.5, 2.0),
        "zmot": (1.0, 4.0),
        "problem": (2.0, 6.0),
        "problem_realization": (2.0, 6.0),
        "discovery": (3.0, 8.0),
        "barriers": (2.0, 6.0),
        "execution": (2.0, 6.0),
        "implementation": (2.0, 6.0),
        "resolution": (1.0, 3.0),
        "solution": (1.0, 3.0),
    }.get((stage or "").lower(), (1.0, 3.0))
    lo, hi = base
    span = hi - lo
    try:
        f = max(0.0, min(1.0, float(fitness or 0.5)))
    except Exception:
        f = 0.5
    est = lo + span * f
    return f"+{est:.0f}%"

def _persona_alias_from_label(label: str) -> Dict[str, str]:
    # minimal, non-blocking mapping used by arsenal repo
    lab = (label or "").strip()
    dept = ""
    sr   = ""
    m = re.search(r"\b(cfo|finance|revops|sales|marketing|security|engineering|architecture|product|procurement)\b", lab, re.I)
    if m:
        dept = m.group(1).title()
    if re.search(r"\b(cxo|vp|head|director|senior|lead|manager)\b", lab, re.I):
        sr = "senior"
    return {"title": lab, "department": dept, "seniority": sr}

def _safe_seq_list(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Accept either 'sequences' or 'concern_sequences'
    if "sequences" in (report or {}):
        return list(report.get("sequences") or [])
    return list(report.get("concern_sequences") or [])

# ---------------------------------------------------------------------
# PUBLIC: turn sequences into campaigns by pulling plays from the arsenal repo
# ---------------------------------------------------------------------

def _campaigns_from_sequences_using_arsenal(
    *,
    G,  # (currently unused, reserved for future graph-aware enrich)
    account_id: Optional[str],
    product_id: Optional[str],
    report: Dict[str, Any],
    start_idx: int = 0,
    plays_per_step: int = 1,
    archetype: Optional[Dict[str, Any]] = None,
    exclude_pairs: Optional[set[tuple[str, str]]] = None,
    limits: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Build campaigns directly from RCS sequences using the execution_arsenal_repository.
    - No catalogs.
    - Respects exclude_pairs to avoid repeating asset×channel pairs within a plan.
    - Fills essential arsenal row fields; stage_playbooks will attach concerns later.
    """
    sequences = _safe_seq_list(report)
    if not sequences:
        return []

    exclude_pairs = exclude_pairs or set()
    archetype = archetype or {}
    limits = limits or {}
    per_persona_cap = max(1, int(limits.get("per_persona", 99)))
    max_sequences = int(limits.get("max_sequences", len(sequences)))

    seen_per_persona: Dict[str, int] = {}
    campaigns: List[Dict[str, Any]] = []
    seq_count = 0

    for seq_idx, seq in enumerate(sequences):
        if seq_count >= max_sequences:
            break

        steps = list(seq.get("sequence") or [])
        if not steps:
            continue

        # personas present in the sequence
        persona_labels = []
        for st in steps:
            pl = st.get("persona_label") or st.get("persona") or ""
            if pl:
                persona_labels.append(pl)
        personas = sorted({p for p in persona_labels if p})

        key_pid = personas[0] if personas else "unknown"
        if seen_per_persona.get(key_pid, 0) >= per_persona_cap:
            continue

        arsenal_rows: List[Dict[str, Any]] = []

        for step in steps:
            concern_label = step.get("concern_label") or step.get("cid") or ""
            concern_stage = (step.get("stage") or "").lower()
            persona_label = step.get("persona_label") or step.get("persona") or ""
            persona_alias = _persona_alias_from_label(persona_label)

            plays = get_best_plays_for_concern(
                persona_alias=persona_alias,
                concern_label=concern_label,
                concern_stage=concern_stage,
                product_id=product_id,
                archetype=archetype,
                top_k=max(1, plays_per_step),
                exclude_pairs=exclude_pairs,
            ) or []  # defensive

            for p in plays:
                a = p.get("asset", {}) or {}
                c = p.get("channel", {}) or {}
                fitness = float(p.get("fitness", 0.5) or 0.5)
                eng     = float(p.get("engagement_score", 0.5) or 0.5)
                why     = p.get("why", "")

                # Dedup guard across the plan
                pair = (a.get("id"), c.get("id"))
                if pair[0] and pair[1]:
                    if pair in exclude_pairs:
                        continue
                    exclude_pairs.add(pair)

                arsenal_rows.append({
                    "asset":   {"id": a.get("id"), "name": a.get("name"), "format": a.get("format"), "evergreen": a.get("evergreen", False)},
                    "channel": {"id": c.get("id"), "name": c.get("name"), "type": c.get("type"), "reach": c.get("reach_score", 0.0)},
                    "fitment": concern_stage or "Concern resolution",
                    "engagement": _engagement_label(eng),
                    "expectedLift": _expected_lift_label(concern_stage, fitness),
                    "why": why,
                    # concernsAddressed is attached upstream by stage_playbooks
                })

        if not arsenal_rows:
            # Skip empty campaigns (e.g., empty repo)
            continue

        description = "Resolve prioritized concerns"
        if personas:
            description += f" across {', '.join(personas[:2])}"

        camp_id = f"camp_seq_{seq_idx + start_idx:02d}_{_slug(account_id or 'acct')}"
        campaigns.append({
            "id": camp_id,
            "description": description,
            "personas": personas,
            "arsenalTable": arsenal_rows,
            # timeframe is stamped upstream (frozen_stage_simulator or stage_playbooks)
        })

        seen_per_persona[key_pid] = seen_per_persona.get(key_pid, 0) + 1
        seq_count += 1

    return campaigns
