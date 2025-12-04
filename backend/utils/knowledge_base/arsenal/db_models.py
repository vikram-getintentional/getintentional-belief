from __future__ import annotations

import enum
import re
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from backend.database import Base


class AssetCategory(enum.Enum):
    WEBINAR = "webinar"
    EBOOK = "ebook"
    CASE_STUDY = "case_study"
    BLOG_POST = "blog_post"
    LANDING_PAGE = "landing_page"
    NEWS_ARTICLE = "news_article"
    CALCULATOR = "calculator"
    IN_APP_PROMPT = "in_app_prompt"
    EMAIL_SEQUENCE = "email_sequence"
    TOOLKIT = "toolkit"
    PLAYBOOK = "playbook"
    DEMO = "demo"
    EVENT = "event"
    REPORT = "report"
    SALES_CALL = "sales_call"


class ContentType(enum.Enum):
    VIDEO = "video"
    AUDIO = "audio"
    INTERACTIVE = "interactive"
    TEXT = "text"
    TOOL = "tool"


class TimeToConsume(enum.Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class AssetDepth(enum.Enum):
    TEASER = "teaser"
    OVERVIEW = "overview"
    DEEP_DIVE = "deep_dive"
    IMPLEMENTATION_GUIDE = "implementation_guide"
    PLAYBOOK = "playbook"


class ChannelType(enum.Enum):
    EMAIL = "email"
    SALES_OUTREACH = "sales_outreach"
    PAID_SOCIAL = "paid_social"
    PAID_SEARCH = "paid_search"
    DISPLAY = "display"
    WEBINAR_PLATFORM = "webinar_platform"
    COMMUNITY = "community"
    FIELD_EVENT = "field_event"
    IN_APP = "in_app"
    DIRECT_MAIL = "direct_mail"
    PARTNER = "partner"
    WEBSITE = "website"
    ORGANIC_SOCIAL = "organic_social"


class ChannelDelivery(enum.Enum):
    OWNED = "owned"
    PAID = "paid"
    EARNED = "earned"
    IN_PRODUCT = "in_product"


class ApprovalStatus(enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:120] or str(uuid.uuid4())


class ArsenalAsset(Base):
    __tablename__ = "arsenal_assets"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, unique=True, index=True)
    category = Column(Enum(AssetCategory), nullable=True)
    category_text = Column(String, nullable=True)
    content_type = Column(Enum(ContentType), nullable=True)
    content_type_text = Column(String, nullable=True)
    time_to_consume = Column(Enum(TimeToConsume), nullable=True)
    time_to_consume_text = Column(String, nullable=True)
    depth = Column(Enum(AssetDepth), nullable=True)
    depth_text = Column(String, nullable=True)
    description = Column(String, nullable=True)
    metadata_complete = Column(Boolean, default=False, nullable=False)
    created_from_engagement_id = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    call_stage = Column(String, nullable=True)
    target_personas = Column(JSON, nullable=True)
    target_account_segments = Column(JSON, nullable=True)
    target_belief_stages = Column(JSON, nullable=True)
    target_concerns = Column(JSON, nullable=True)
    org_conversion_maturity = Column(String, nullable=True)
    typical_channels = Column(JSON, nullable=True)
    approval_status = Column(
        Enum(ApprovalStatus), default=ApprovalStatus.PENDING, nullable=False, index=True
    )
    derived_metadata = Column(JSON, nullable=True)
    auto_classification_confidence = Column(Float, nullable=True)

    channel_impacts = relationship("AssetChannelImpact", back_populates="asset")

    def update_metadata_status(self) -> None:
        required_fields = [
            self.category or self.category_text,
            self.content_type or self.content_type_text,
            self.time_to_consume or self.time_to_consume_text,
            self.depth or self.depth_text,
        ]
        if self.category == AssetCategory.SALES_CALL:
            required_fields.append(self.call_stage)
        self.metadata_complete = all(field not in (None, "", []) for field in required_fields)

    @staticmethod
    def slug_for(name: str) -> str:
        return _slugify(name)

    def mark_approved(self) -> None:
        self.approval_status = ApprovalStatus.APPROVED
        self.metadata_complete = bool(self.metadata_complete)


class ArsenalChannel(Base):
    __tablename__ = "arsenal_channels"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    slug = Column(String, nullable=False, unique=True, index=True)
    channel_type = Column(Enum(ChannelType), nullable=True)
    channel_type_text = Column(String, nullable=True)
    delivery_mode = Column(Enum(ChannelDelivery), nullable=True)
    delivery_mode_text = Column(String, nullable=True)
    reach_score_estimate = Column(Float, nullable=True)
    metadata_complete = Column(Boolean, default=False, nullable=False)
    created_from_engagement_id = Column(String, nullable=True)
    notes = Column(String, nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    target_personas = Column(JSON, nullable=True)
    target_account_segments = Column(JSON, nullable=True)
    target_belief_stages = Column(JSON, nullable=True)
    target_concerns = Column(JSON, nullable=True)
    org_conversion_maturity = Column(String, nullable=True)
    typical_assets = Column(JSON, nullable=True)
    approval_status = Column(
        Enum(ApprovalStatus), default=ApprovalStatus.PENDING, nullable=False, index=True
    )
    derived_metadata = Column(JSON, nullable=True)
    auto_classification_confidence = Column(Float, nullable=True)

    asset_impacts = relationship("AssetChannelImpact", back_populates="channel")

    def update_metadata_status(self) -> None:
        required_fields = [
            self.channel_type or self.channel_type_text,
            self.delivery_mode or self.delivery_mode_text,
        ]
        self.metadata_complete = all(field not in (None, "", []) for field in required_fields)

    @staticmethod
    def slug_for(name: str) -> str:
        return _slugify(name)

    def mark_approved(self) -> None:
        self.approval_status = ApprovalStatus.APPROVED
        self.metadata_complete = bool(self.metadata_complete)


class AssetChannelImpact(Base):
    __tablename__ = "asset_channel_impacts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    asset_id = Column(String, ForeignKey("arsenal_assets.id"), nullable=False, index=True)
    channel_id = Column(String, ForeignKey("arsenal_channels.id"), nullable=False, index=True)
    product_id = Column(String, index=True, nullable=False)
    persona_id = Column(String, nullable=False, index=True)
    belief_transition_id = Column(String, nullable=True, index=True)
    impact_strength = Column(Float, nullable=True)
    evidence_count = Column(Integer, default=0, nullable=False)
    last_evidence_at = Column(DateTime, nullable=True)
    evidence_details = Column(JSON, nullable=True)

    asset = relationship("ArsenalAsset", back_populates="channel_impacts")
    channel = relationship("ArsenalChannel", back_populates="asset_impacts")

    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "channel_id",
            "persona_id",
            "belief_transition_id",
            name="uq_asset_channel_belief",
        ),
    )

    def register_evidence(self, delta_strength: float, metadata: Optional[dict] = None) -> None:
        """Update running impact strength and evidence stats."""
        previous = self.impact_strength or delta_strength
        alpha = 0.3
        self.impact_strength = (1 - alpha) * previous + alpha * delta_strength
        self.evidence_count = (self.evidence_count or 0) + 1
        self.last_evidence_at = datetime.utcnow()
        if metadata:
            details = list(self.evidence_details or [])
            details.append(metadata)
            # keep last 20 to avoid unbounded growth
            self.evidence_details = details[-20:]
