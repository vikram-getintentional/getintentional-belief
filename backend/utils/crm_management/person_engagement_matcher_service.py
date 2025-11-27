# backend/utils/crm_management/person_engagement_match_service.py
from __future__ import annotations
from typing import Dict, Any, List
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.target_account_manager import get_account_by_id

# Ensure these models are imported at app startup BEFORE create_all()
from backend.utils.crm_management.person_models import AccountPerson, AccountPersonJob  # noqa: F401
from backend.utils.crm_management.person_service import (
    list_people_for_account,
    match_persona_for_actor_in_graph,   # NEW: graph-aware matcher
    suggest_jobs_for_person,
    upsert_person_from_engagement,
)


def list_account_engagements(product_id: str, account_id: str) -> List[Dict[str, Any]]:
    """
    Read engagements from DB and normalize to the dict shape used by the UI/service.
    """
    db: Session = next(get_db())
    try:
        rows = (
            db.query(TargetAccountEngagement)
              .filter(
                  TargetAccountEngagement.product_id == product_id,
                  TargetAccountEngagement.target_account_id == account_id
              )
              .order_by(TargetAccountEngagement.timestamp_dt.asc())
              .all()
        )
        out: List[Dict[str, Any]] = []
        for r in rows:
            p = r.payload or {}
            p_actor = p.get("actor") or {}
            out.append({
                "account_id": r.target_account_id,
                "timestamp": (r.timestamp or r.timestamp_dt.isoformat()),
                "actor": {
                    "name": r.actor_name,
                    "title": r.actor_title or "",
                    "department": r.actor_department or "",
                    "confidence": (r.actor_confidence or 100) / 100.0,
                    # prefer persisted seniority if present, else payload, else None
                    "seniority": p_actor.get("seniority") or None,
                },
                "channel": r.channel or None,
                "source": (r.source or "other"),
                "raw_activity": r.raw_activity or "",
                "asset_id": r.asset_id or None,
                "inferred": bool(r.inferred),
                "__persisted__": True,
            })
        return out
    finally:
        db.close()


def _aggregate_unique_actors(engagements: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Key: name|title|department (skip rows without a name)
    """
    uniq: Dict[str, Dict[str, Any]] = {}
    for e in engagements:
        actor = (e.get("actor") or {})
        name = (actor.get("name") or "").strip()
        title = (actor.get("title") or "").strip()
        dept = (actor.get("department") or "").strip()
        if not name:
            continue
        key = f"{name}|{title}|{dept}"
        if key not in uniq:
            uniq[key] = {
                "name": name,
                "title": title,
                "department": dept,
                # pass through seniority if present in payload
                "seniority": (actor.get("seniority") or "").strip(),
            }
    return uniq


def get_persona_matches_for_account(product_id: str, account_id: str) -> Dict[str, Any]:
    """
    Orchestrates:
      1) load engagements
      2) aggregate actors
      3) upsert AccountPerson rows
      4) graph-aware persona match + job suggestions
    """
    print("Starting persona matcher")
    if not get_account_by_id(product_id, account_id):
        return {"account_id": account_id, "persons": []}

    engagements = list_account_engagements(product_id, account_id)
    if not engagements:
        return {"account_id": account_id, "persons": []}

    uniq = _aggregate_unique_actors(engagements)

    # Upsert all people first (minimize repeated queries later)
    for _, actor in uniq.items():
        print("Upserting person for actor:", actor)
        pid = upsert_person_from_engagement(product_id, account_id, actor)
        print("Upserted person_id:", pid)

    # Fetch consolidated people list once
    people = list_people_for_account(product_id, account_id)
    print("Fetched people for account:", people)
    people_by_name = {(p["name"] or "").strip(): p for p in people}

    persons_output: List[Dict[str, Any]] = []

    for _, actor in uniq.items():
        p = people_by_name.get(actor["name"])
        if not p:
            continue

        # 🔑 Graph-aware match (exact → fuzzy over graph personas)
        match = match_persona_for_actor_in_graph(product_id, actor)
        print("Graph-aware persona match for actor", actor, "=>", match)

        # ORM fetch for suggest_jobs_for_person
        db: Session = next(get_db())
        try:
            person_obj = db.query(AccountPerson).filter(AccountPerson.id == p["id"]).first()
        finally:
            db.close()

        jobs = suggest_jobs_for_person(product_id, person_obj, k=6) if person_obj else []
        print("Suggested jobs for", p["name"], "=>", jobs)

        persons_output.append({
            "name": actor.get("name"),
            "title": actor.get("title"),
            "department": actor.get("department"),
            "seniority": actor.get("seniority") or None,

            # what to render in UI
            "graph_persona_node_id": match.get("best"),          # ← persona node id in graph
            "graph_persona_score": match.get("score"),           # ← 0..1

            # keep older shape for compatibility (if UI uses these)
            "canonical_persona_best": match.get("best"),
            "canonical_persona_label": match.get("best_label"),
            "canonical_persona_score": match.get("score"),
            "canonical_meta": match.get("canonical_meta"),
            "alternates": match.get("alternates"),

            "jobs": jobs,
        })

    return {"account_id": account_id, "persons": persons_output}
