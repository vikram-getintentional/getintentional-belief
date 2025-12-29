import datetime
import enum

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text

from backend.database import Base


class LabelSource(str, enum.Enum):
    TEACHER_LLM = "teacher_llm"
    STUDENT = "student"
    HUMAN = "human"


class CallSegment(Base):
    __tablename__ = "call_segments"

    id = Column(String, primary_key=True)
    call_id = Column(String, ForeignKey("thesis_calls.id"), index=True, nullable=False)
    idx = Column(Integer, nullable=False, default=0)
    speaker_role = Column(String, nullable=True)
    phase = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class SegmentInterpretation(Base):
    __tablename__ = "segment_interpretations"

    id = Column(String, primary_key=True)
    segment_id = Column(String, ForeignKey("call_segments.id"), index=True, nullable=False)
    payload = Column(JSON, nullable=False)
    source = Column(Enum(LabelSource), nullable=False)
    model_version = Column(String, nullable=True)
    confidence = Column(Float, nullable=False)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
