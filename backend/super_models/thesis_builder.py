import datetime
import enum

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    String,
    Text,
)

from backend.database import Base


class AttributeType(str, enum.Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class ThesisSegmentFeatureDefinition(Base):
    __tablename__ = "thesis_segment_features"

    id = Column(String, primary_key=True)
    product_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    key = Column(String, nullable=False)
    type = Column(Enum(AttributeType), nullable=False)
    bins = Column(JSON, nullable=True)
    categories = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ThesisSegment(Base):
    __tablename__ = "thesis_segments"

    id = Column(String, primary_key=True)
    product_id = Column(String, index=True, nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    rules = Column(JSON, nullable=False)
    prior_weight = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ThesisComponentType(str, enum.Enum):
    PAIN = "pain"
    ZMOT = "zmot"
    ASPIRATION = "aspiration"
    BELIEF = "belief"
    ATTRIBUTE = "attribute"


class ThesisComponentPrior(Base):
    __tablename__ = "thesis_component_priors"

    id = Column(String, primary_key=True)
    product_id = Column(String, index=True, nullable=False)
    segment_id = Column(String, ForeignKey("thesis_segments.id"), nullable=False)
    component_type = Column(Enum(ThesisComponentType), nullable=False)
    parent_pain_id = Column(String, nullable=True)
    parent_aspiration_id = Column(String, nullable=True)
    feature_key = Column(String, nullable=True)
    alphas = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ThesisComponentPosterior(Base):
    __tablename__ = "thesis_component_posteriors"

    id = Column(String, primary_key=True)
    product_id = Column(String, index=True, nullable=False)
    segment_id = Column(String, ForeignKey("thesis_segments.id"), nullable=False)
    component_type = Column(Enum(ThesisComponentType), nullable=False)
    parent_pain_id = Column(String, nullable=True)
    parent_aspiration_id = Column(String, nullable=True)
    feature_key = Column(String, nullable=True)
    alphas = Column(JSON, nullable=False)
    last_updated_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    baseline_alphas = Column(JSON, nullable=True)
    baseline_at = Column(DateTime, nullable=True)


class ThesisCall(Base):
    __tablename__ = "thesis_calls"

    id = Column(String, primary_key=True)
    product_id = Column(String, index=True, nullable=False)
    account_id = Column(String, nullable=True)
    external_call_id = Column(String, nullable=True)
    attribute_bins = Column(JSON, nullable=False)
    stakeholders = Column(JSON, nullable=True)
    raw_summary = Column(JSON, nullable=True)
    segment_id = Column(String, ForeignKey("thesis_segments.id"), nullable=True)
    transcript_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ThesisCallEvidence(Base):
    __tablename__ = "thesis_call_evidence"

    id = Column(String, primary_key=True)
    call_id = Column(String, ForeignKey("thesis_calls.id"), nullable=False)
    product_id = Column(String, index=True, nullable=False)
    segment_id = Column(String, ForeignKey("thesis_segments.id"), nullable=True)
    component_type = Column(Enum(ThesisComponentType), nullable=False)
    category_id = Column(String, nullable=False)
    parent_pain_id = Column(String, nullable=True)
    parent_aspiration_id = Column(String, nullable=True)
    feature_key = Column(String, nullable=True)
    intensity = Column(Float, nullable=False, default=1.0)
