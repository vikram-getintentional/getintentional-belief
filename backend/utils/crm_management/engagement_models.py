# backend/utils/crm_management/engagement_models.py
from datetime import datetime, timezone
from backend.database import Base
from sqlalchemy import Column, DateTime, String, JSON, Boolean, Integer, Float, Index
from sqlalchemy.orm import Session
import uuid


def now_iso_z():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

class TargetAccountEngagement(Base):
    __tablename__ = "target_account_engagements"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String, index=True)
    target_account_id = Column(String, index=True)

    # canonical/meta
    # Option 1 (typed): DateTime
    timestamp_dt = Column(DateTime(timezone=True), nullable=False)

    # Option 2 (optional raw): keep original string too
    timestamp = Column(String, nullable=False)

    source = Column(String, index=True)              # marketing|sales|cs|product|other
    channel = Column(String, nullable=True, index=True)
    channel_id = Column(String, nullable=True, index=True)
    engagement_verb = Column(String, nullable=True, index=True)
    inferred = Column(Boolean, default=False, index=True)

    # actor quick filters
    actor_name = Column(String, nullable=True)
    actor_title = Column(String, index=True)
    actor_department = Column(String, index=True)
    actor_seniority = Column(String, index=True)
    actor_confidence = Column(Integer, default=100)  # store 0..100

    # activity
    raw_activity = Column(String)
    asset_id = Column(String, nullable=True)
    activity_label = Column(String, nullable=True)
    asset_category = Column(String, nullable=True)
    parser_version = Column(String, nullable=True)
    parser_confidence = Column(Float, nullable=True)

    # original payload for forward-compat flexibility
    payload = Column(JSON)

# helpful composite indexes
Index("idx_tae_prod_acct_ts", TargetAccountEngagement.product_id,
      TargetAccountEngagement.target_account_id, TargetAccountEngagement.timestamp)
Index("idx_tae_title_dept", TargetAccountEngagement.actor_title, TargetAccountEngagement.actor_department)
Index("idx_tae_seniority", TargetAccountEngagement.actor_seniority)
