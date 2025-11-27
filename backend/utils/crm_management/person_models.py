# backend/utils/crm_management/person_models.py
# person_models.py
from datetime import datetime, timezone
from backend.database import Base
from sqlalchemy import Column, DateTime, ForeignKey, String, JSON, Boolean, Integer, Index, UniqueConstraint
from sqlalchemy.orm import Session
import uuid


def _now():
    return datetime.now(timezone.utc)

class AccountPerson(Base):
    """
    A human in a target account (per-product scope so we can keep graphs/products isolated).
    Canonical fields are optional and filled by canonicalizers.
    """
    __tablename__ = "account_people"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String, index=True, nullable=False)
    account_id = Column(String, index=True, nullable=False)

    # raw identity
    name = Column(String, index=True, nullable=False)  # "John Doe"
    title = Column(String, index=True)                 # raw/observed title text
    department = Column(String, index=True)            # raw/observed dept text
    seniority = Column(String, index=True)             # raw/observed seniority text

    # canonicalized/meta
    canonical_persona_id = Column(String, index=True)  # e.g., persona_023
    canonical_department = Column(String, index=True)
    canonical_seniority = Column(String, index=True)

    # bookkeeping
    engagement_count = Column(Integer, default=0)
    last_seen_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        # One human per account/product by name
        UniqueConstraint("product_id", "account_id", "name", name="uq_account_people_identity"),
        Index("ix_account_people_prod_acct", "product_id", "account_id"),
    )


class AccountPersonJob(Base):
    """
    Proposed/confirmed jobs-to-be-done for a person (normalized to canonical job ids when possible).
    """
    __tablename__ = "account_person_jobs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String, index=True, nullable=False)
    account_id = Column(String, index=True, nullable=False)
    person_id = Column(String, ForeignKey("account_people.id"), index=True, nullable=False)

    # suggestion vs confirmed
    source = Column(String, index=True)   # "suggested_graph" | "user_confirmed" | "engagement_inferred"
    status = Column(String, index=True)   # "suggested" | "confirmed" | "rejected"

    # job texts + canonicals
    job_text = Column(String, index=True)           # display
    canonical_job_id = Column(String, index=True)   # e.g., job_041

    # optional extra bag
    meta = Column(JSON)      # e.g., scores, evidence, engagement_ids
    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now)

    __table_args__ = (
        Index("ix_account_person_jobs_identity", "product_id", "account_id", "person_id"),
    )
