from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import os
import uuid
from typing import Optional, Tuple

from sqlalchemy import Column, DateTime, JSON, String, UniqueConstraint, func
from sqlalchemy.orm import Session

from backend.database import Base
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.graph_base.graph_utils.save_and_load_graph_as_json import GRAPH_JSON_DIR

CACHE_TTL = timedelta(days=1)


def _graph_file_path(product_id: str) -> str:
    return os.path.join(GRAPH_JSON_DIR, f"{product_id}_graph.json")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(value: Optional[datetime]) -> Optional[datetime]:
    if not value:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def compute_product_graph_hash(product_id: str) -> str:
    """
    Build a checksum of the serialized graph to detect updates.
    """
    path = _graph_file_path(product_id)
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except Exception:
        return ""


class BeliefThesisCache(Base):
    __tablename__ = "belief_thesis_cache"
    __table_args__ = (
        UniqueConstraint("product_id", "account_id", name="uq_belief_thesis_cache_product_account"),
    )

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String, nullable=False, index=True)
    account_id = Column(String, nullable=False, index=True)
    thesis = Column(JSON, nullable=False)
    graph_hash = Column(String, nullable=True)
    last_engagement_ts = Column(DateTime(timezone=True), nullable=True)
    last_computed_at = Column(DateTime(timezone=True), nullable=False)


def load_belief_thesis_cache(
    db: Session, product_id: str, account_id: str
) -> Optional[BeliefThesisCache]:
    return (
        db.query(BeliefThesisCache)
        .filter_by(product_id=product_id, account_id=account_id)
        .first()
    )


def latest_account_engagement_ts(
    db: Session, product_id: str, account_id: str
) -> Optional[datetime]:
    latest = (
        db.query(func.max(TargetAccountEngagement.timestamp_dt))
        .filter_by(product_id=product_id, target_account_id=account_id)
        .scalar()
    )
    return _ensure_utc(latest)


def evaluate_belief_thesis_cache(
    db: Session, product_id: str, account_id: str
) -> Tuple[bool, Optional[BeliefThesisCache], str, Optional[datetime]]:
    """
    Decide whether the belief thesis needs a rebuild by comparing graph/engagement
    fingerprints and TTLs.
    """
    cache = load_belief_thesis_cache(db, product_id, account_id)
    graph_hash = compute_product_graph_hash(product_id)
    latest_engagement = latest_account_engagement_ts(db, product_id, account_id)
    now = _now_utc()

    if cache is None:
        return True, cache, graph_hash, latest_engagement

    if cache.graph_hash != graph_hash:
        return True, cache, graph_hash, latest_engagement

    if latest_engagement and (
        cache.last_engagement_ts is None or latest_engagement > cache.last_engagement_ts
    ):
        return True, cache, graph_hash, latest_engagement

    if cache.last_computed_at is None or cache.last_computed_at + CACHE_TTL < now:
        return True, cache, graph_hash, latest_engagement

    return False, cache, graph_hash, latest_engagement


def persist_belief_thesis_cache(
    db: Session,
    product_id: str,
    account_id: str,
    thesis: dict,
    graph_hash: str,
    latest_engagement_ts: Optional[datetime],
    existing_cache: Optional[BeliefThesisCache] = None,
) -> BeliefThesisCache:
    cache = existing_cache or load_belief_thesis_cache(db, product_id, account_id)
    if cache is None:
        cache = BeliefThesisCache(
            product_id=product_id,
            account_id=account_id,
            thesis=thesis,
            graph_hash=graph_hash,
            last_engagement_ts=_ensure_utc(latest_engagement_ts),
            last_computed_at=_now_utc(),
        )
        db.add(cache)
        return cache

    cache.thesis = thesis
    cache.graph_hash = graph_hash
    cache.last_engagement_ts = _ensure_utc(latest_engagement_ts)
    cache.last_computed_at = _now_utc()
    return cache
