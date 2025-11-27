# backend/utils/inbox/inbox_models.py
from __future__ import annotations
from sqlalchemy.orm import relationship, Session
from sqlalchemy import Column, String, Enum, Float, DateTime, JSON, ForeignKey, Boolean
import enum, uuid, datetime as dt

from backend.database import Base

def _uuid() -> str:
    return str(uuid.uuid4())

# --- Enums ---
class InsightSource(str, enum.Enum):
    persona_builder = "persona_builder"
    rcs = "rcs"
    stage_simulator = "stage_simulator"
    plan_builder = "plan_builder"
    arsenal_suggester = "arsenal_suggester"
    zmot_enricher = "zmot_enricher"
    crm_import = "crm_import"
    live_dashboard = "live_dashboard"
    manual = "manual"

class InsightType(str, enum.Enum):
    email = "email"
    asset_share = "asset_share"
    meeting_request = "meeting_request"
    campaign_enroll = "campaign_enroll"
    social_outreach = "social_outreach"
    content_create = "content_create"
    case_study_request = "case_study_request"
    stakeholder_map = "stakeholder_map"
    data_enrich = "data_enrich"
    follow_up = "follow_up"
    assign_owner = "assign_owner"
    note = "note"
    custom = "custom"

class InsightPriority(str, enum.Enum):
    now = "now"
    soon = "soon"
    later = "later"

class InsightStatus(str, enum.Enum):
    new = "new"
    queued = "queued"
    doing = "doing"
    done = "done"
    skipped = "skipped"
    snoozed = "snoozed"
    error = "error"

class ActionStatus(str, enum.Enum):
    queued = "queued"
    sent = "sent"
    scheduled = "scheduled"
    failed = "failed"
    canceled = "canceled"

# --- Models ---
class Insight(Base):
    __tablename__ = "insights"

    id = Column(String, primary_key=True, default=_uuid)
    source = Column(Enum(InsightSource), nullable=False)
    type = Column(Enum(InsightType), nullable=False)

    account_id = Column(String, nullable=False)
    person_id = Column(String, nullable=True)
    persona_id = Column(String, nullable=True)
    asset_id = Column(String, nullable=True)

    # what the user sees
    title = Column(String, nullable=False)
    description = Column(String, nullable=False)

    # “why this?” + model features
    justification = Column(JSON, nullable=False, default=dict)

    # “what to do?” — linkable action
    action_url = Column(String, nullable=True)         # e.g. /actions/send-email?id=...
    action_payload = Column(JSON, nullable=True)       # any args needed

    # prioritization + dedupe
    priority = Column(Enum(InsightPriority), nullable=False, default=InsightPriority.soon)
    score = Column(Float, nullable=False, default=0.0)
    dedupe_key = Column(String, nullable=False)

    # lifecycle
    status = Column(Enum(InsightStatus), nullable=False, default=InsightStatus.new)
    snooze_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    created_by = Column(String, default="system")

    metadata_json = Column("metadata", JSON, default=dict)
    pinned = Column(Boolean, default=False)

    executions = relationship("ActionExecution", back_populates="insight", cascade="all, delete-orphan")

class ActionExecution(Base):
    __tablename__ = "insight_action_executions"

    id = Column(String, primary_key=True, default=_uuid)
    insight_id = Column(String, ForeignKey("insights.id"), nullable=False)
    action = Column(Enum(InsightType), nullable=False)
    requested_by = Column(String, nullable=False)
    requested_at = Column(DateTime, default=dt.datetime.utcnow)
    payload = Column(JSON, default=dict)
    status = Column(Enum(ActionStatus), default=ActionStatus.queued)
    result = Column(JSON, default=dict)

    insight = relationship("Insight", back_populates="executions")