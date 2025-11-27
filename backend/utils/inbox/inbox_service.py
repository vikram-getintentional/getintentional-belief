# backend/utils/inbox/inbox_service.py
from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional
from hashlib import sha256
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from contextlib import contextmanager

from .inbox_models import (
    Insight, InsightStatus, InsightPriority, InsightType, InsightSource, ActionExecution
)


from backend.database import get_db


@contextmanager
def db_session() -> Iterator[Session]:
    """
    Safe wrapper around FastAPI's get_db() generator.
    - Opens a Session
    - Yields it
    - Commits on success, rolls back on error
    - Ensures the generator is closed so Session is closed
    """
    gen = get_db()
    db: Session = next(gen)  # open
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        # important: close the generator so its `finally: db.close()` runs
        try:
            gen.close()
        except Exception:
            pass

# ---- helpers ----
_DEDUPE_FIELDS = ["source", "type", "account_id", "person_id", "persona_id", "asset_id", "title"]

def _compute_dedupe_key(payload: Dict[str, Any]) -> str:
    s = "|".join(str(payload.get(k, "")) for k in _DEDUPE_FIELDS)
    return sha256(s.encode()).hexdigest()

def _score(j: Dict[str, Any]) -> float:
    p = float(j.get("perceptibility", 0))
    q = float(j.get("proximity", 0))
    i = float(j.get("involvement", 0))
    lift = float(j.get("expected_lift", 0))
    return 0.35*p + 0.35*q + 0.2*i + 0.1*lift

def _priority_from_score(s: float) -> InsightPriority:
    if s >= 0.70: return InsightPriority.now
    if s >= 0.45: return InsightPriority.soon
    return InsightPriority.later



# ---- public API (your requested names) ----
def add_to_inbox_pipeline(payload: Dict[str, Any]) -> Insight:
    data = dict(payload)
    data["dedupe_key"] = data.get("dedupe_key") or _compute_dedupe_key(data)
    j = data.get("justification", {}) or {}
    data["score"] = float(data.get("score") or _score(j))
    if not data.get("priority"):
        data["priority"] = _priority_from_score(data["score"]).value

    with db_session() as db:
        existing = db.query(Insight).filter(Insight.dedupe_key == data["dedupe_key"]).first()
        if existing:
            existing.description = data.get("description", existing.description)
            if j: existing.justification = j
            existing.score = max(existing.score, data["score"])
            if data.get("priority"): existing.priority = InsightPriority(data["priority"])
            existing.status = InsightStatus.new
            existing.action_url = data.get("action_url", existing.action_url)
            if data.get("action_payload"): existing.action_payload = data["action_payload"]
            db.add(existing); db.flush(); db.refresh(existing)
            return existing

        ins = Insight(**data)
        db.add(ins); db.flush(); db.refresh(ins)
        return ins

def display_inbox_items(... ) -> List[Insight]:
    with db_session() as db:
        qs = db.query(Insight)
        # ... filters ...
        return qs.order_by(Insight.status.asc(), Insight.priority.asc(),
                           Insight.score.desc(), Insight.created_at.desc())\
                 .offset(offset).limit(limit).all()

def resolve_inbox(insight_id: str, *, requested_by: str = "user:current",
                  payload: Optional[Dict[str, Any]] = None) -> bool:
    with db_session() as db:
        ins = db.get(Insight, insight_id)               # ← SQLAlchemy 2.0 style
        if not ins: return False
        exec_row = ActionExecution(
            insight_id=insight_id, action=ins.type, requested_by=requested_by,
            payload=payload or {}, status="sent"
        )
        db.add(exec_row)
        ins.status = InsightStatus.done
        return True

def count_open_items(*, account_id: Optional[str] = None) -> int:
    with db_session() as db:
        qs = db.query(Insight).filter(
            Insight.status.in_([InsightStatus.new, InsightStatus.queued,
                                InsightStatus.doing, InsightStatus.snoozed]))
        if account_id:
            qs = qs.filter(Insight.account_id == account_id)
        return qs.count()

def snooze_inbox(insight_id: str, days: int = 7) -> bool:
    with db_session() as db:
        ins = db.get(Insight, insight_id)
        if not ins: return False
        ins.status = InsightStatus.snoozed
        ins.snooze_until = datetime.utcnow() + timedelta(days=days)
        return True

def skip_inbox(insight_id: str) -> bool:
    with db_session() as db:
        ins = db.get(Insight, insight_id)
        if not ins: return False
        ins.status = InsightStatus.skipped
        return True

