from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from backend.database import SessionLocal
from backend.utils.graph_base.network_graph import (
    _set_node_label,
    build_product_graph,
)
from backend.utils.inference.rcs_generators.generate_rcs_fast import (
    build_persona_coalitions,
    generate_rcs_new,
)
from backend.utils.crm_management.person_models import (
    AccountPersonaMatch,
    AccountPerson,
)
from backend.utils.crm_management.target_account_manager import TargetAccount


GRAPH_BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "graph_base",
)
PERSONA_METRICS_DIR = os.path.join(GRAPH_BASE_DIR, "graph_data", "persona_metrics")

META_KEYS = [
    "industry",
    "geography",
    "revenue_range",
    "employee_range",
    "funding_stage",
]


def _load_persona_wolves_metrics(product_id: str) -> Tuple[Dict[str, Any], Optional[str]]:
    path = os.path.join(PERSONA_METRICS_DIR, f"{product_id}.json")
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}, None
    updated_at = payload.get("updated_at")
    metrics: Dict[str, Any] = {}
    for entry in payload.get("personas", []):
        pid = str(entry.get("persona_id") or entry.get("id") or "").strip()
        if not pid:
            continue
        metrics[pid] = entry
    return metrics, updated_at


def _build_meta_totals(accounts: List[TargetAccount]) -> Dict[str, Counter[str]]:
    totals: Dict[str, Counter[str]] = {key: Counter() for key in META_KEYS}
    for acct in accounts:
        for key in META_KEYS:
            value = getattr(acct, key, None)
            if value:
                totals[key][value] += 1
    return totals


def _format_frequency(count: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return round(count / denom, 4)


def build_persona_insights(product_id: str) -> Dict[str, Any]:
    product_graph = build_product_graph(product_id)
    wolves_metrics, wolves_updated_at = _load_persona_wolves_metrics(product_id)

    rcs_report = generate_rcs_new(
        product_graph=product_graph,
        original_graph=product_graph,
        engaged_nodes=[],
    )
    concerns_by_persona: Dict[str, List[Dict[str, Any]]] = (
        rcs_report.get("concerns_by_persona") or {}
    )
    persona_scores: Dict[str, Dict[str, Any]] = (
        rcs_report.get("persona_scores") or {}
    )
    coalitions_by_persona = build_persona_coalitions(
        product_graph,
        concerns_by_persona,
        sim_threshold=0.6,
        max_group_size=6,
    )

    persona_nodes = {
        pid: data
        for pid, data in product_graph.nodes(data=True)
        if (data.get("node_type") or data.get("type")) == "persona"
    }

    session = SessionLocal()
    try:
        accounts = (
            session.query(TargetAccount)
            .filter(TargetAccount.product_id == product_id)
            .all()
        )
        account_lookup = {
            acct.id: {
                "account_name": acct.account_name,
                "industry": acct.industry,
                "geography": acct.geography,
                "revenue_range": acct.revenue_range,
                "employee_range": acct.employee_range,
                "funding_stage": acct.funding_stage,
            }
            for acct in accounts
        }
        total_accounts = len(accounts)
        meta_totals = _build_meta_totals(accounts)

        matches = (
            session.query(
                AccountPersonaMatch.persona_id,
                AccountPersonaMatch.account_id,
                AccountPersonaMatch.persona_label,
                AccountPerson.name,
                AccountPerson.title,
                AccountPerson.department,
                AccountPerson.seniority,
            )
            .outerjoin(AccountPerson, AccountPersonaMatch.person_id == AccountPerson.id)
            .filter(AccountPersonaMatch.product_id == product_id)
            .all()
        )
    finally:
        session.close()

    accounts_per_persona: Dict[str, set] = defaultdict(set)
    persona_people: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    persona_account_samples: Dict[str, List[str]] = defaultdict(list)
    persona_meta_counts: Dict[str, Counter[Tuple[str, str]]] = defaultdict(Counter)
    account_persona_map: Dict[str, set] = defaultdict(set)

    for row in matches:
        persona_id = row.persona_id
        account_id = row.account_id
        if not persona_id or persona_id not in persona_nodes or not account_id:
            continue
        accounts_per_persona[persona_id].add(account_id)
        account_persona_map[account_id].add(persona_id)
        if len(persona_people[persona_id]) < 5 and row.name:
            persona_people[persona_id].append(
                {
                    "person_name": row.name,
                    "title": row.title,
                    "department": row.department,
                    "seniority": row.seniority,
                    "account_name": account_lookup.get(account_id, {}).get("account_name"),
                }
            )
        if len(persona_account_samples[persona_id]) < 5:
            persona_account_samples[persona_id].append(account_id)
        acct_meta = account_lookup.get(account_id)
        if acct_meta:
            for key in META_KEYS:
                value = acct_meta.get(key)
                if value:
                    persona_meta_counts[persona_id][(key, value)] += 1

    co_occurrence: Dict[str, Counter[str]] = defaultdict(Counter)
    for persona_set in account_persona_map.values():
        for pid in persona_set:
            for other in persona_set:
                if other == pid:
                    continue
                co_occurrence[pid][other] += 1

    personas_payload: List[Dict[str, Any]] = []
    for persona_id, node in persona_nodes.items():
        score_row = persona_scores.get(persona_id) or {}
        label = (
            node.get("label")
            or score_row.get("label")
            or _set_node_label(product_graph, persona_id)
        )
        scores = {
            "perceptibility": float(
                score_row.get("perceptibility")
                or node.get("perceptibility")
                or 0.0
            ),
            "proximity": float(
                score_row.get("proximity") or node.get("proximity") or 0.0
            ),
            "involvement": float(
                score_row.get("involvement") or node.get("involvement") or 0.0
            ),
        }
        if score_row.get("activation") is not None:
            scores["activation"] = float(score_row.get("activation", 0.0))
        if score_row.get("strength") is not None:
            scores["strength"] = float(score_row.get("strength", 0.0))
        wolves = wolves_metrics.get(persona_id) or {}
        scores.update(
            {
                "wolves_score": wolves.get("wolves_score"),
                "wolves_delta_bp": wolves.get("delta_win_bp"),
                "wolves_involvement_rate": wolves.get("involvement_rate"),
                "wolves_blocker_rate": wolves.get("blocker_rate"),
            }
        )

        concerns = []
        for entry in (concerns_by_persona.get(persona_id) or [])[:4]:
            concerns.append(
                {
                    "label": entry.get("concern_label")
                    or entry.get("label")
                    or entry.get("pain_label"),
                    "stage": entry.get("concern_stage") or entry.get("stage"),
                    "phase": entry.get("phase"),
                    "score": entry.get("keyness") or entry.get("lift_proxy"),
                }
            )

        coalition_rows = []
        coalition_counts = co_occurrence.get(persona_id)
        if coalition_counts is None:
            coalition_counts = Counter()
        elif not isinstance(coalition_counts, Counter):
            coalition_counts = Counter(coalition_counts)
        for other_id, count in coalition_counts.most_common(3):
            share = _format_frequency(count, total_accounts)
            coalition_rows.append(
                {
                    "persona_id": other_id,
                    "persona_label": _set_node_label(product_graph, other_id),
                    "frequency": share,
                }
            )

        account_ids = list(accounts_per_persona.get(persona_id, set()))
        overall_freq = _format_frequency(len(account_ids), total_accounts)

        segment_rows: List[Dict[str, Any]] = []
        for (key, value), count in persona_meta_counts.get(persona_id, {}).items():
            total = meta_totals.get(key, {}).get(value, 0)
            if total < 3:
                continue
            share = count / total
            uplift = share - overall_freq
            if uplift < 0.05:
                continue
            segment_rows.append(
                {
                    "label": f"{value} ({key.replace('_', ' ')})",
                    "frequency": round(share, 4),
                }
            )
        segment_rows.sort(key=lambda row: row["frequency"], reverse=True)

        account_samples = []
        for acct_id in persona_account_samples.get(persona_id, [])[:3]:
            acct_meta = account_lookup.get(acct_id)
            if not acct_meta:
                continue
            account_samples.append(
                {
                    "account_id": acct_id,
                    "account_name": acct_meta.get("account_name"),
                    "meta": {k: acct_meta.get(k) for k in META_KEYS},
                }
            )

        personas_payload.append(
            {
                "persona_id": persona_id,
                "label": label,
                "title": node.get("title"),
                "department": node.get("department"),
                "seniority": node.get("seniority"),
                "scores": scores,
                "concerns": concerns,
                "coalitions": coalition_rows,
                "seen_in": {
                    "overall": {
                        "label": "All accounts",
                        "frequency": overall_freq,
                    },
                    "segments": segment_rows[:3],
                },
                "people_samples": persona_people.get(persona_id, [])[:3],
                "account_samples": account_samples,
            }
        )

    personas_payload.sort(
        key=lambda row: (
            row.get("scores", {}).get("wolves_score") or 0.0,
            row.get("scores", {}).get("involvement") or 0.0,
        ),
        reverse=True,
    )

    return {
        "product_id": product_id,
        "wolves_metrics_updated_at": wolves_updated_at,
        "personas": personas_payload,
    }
