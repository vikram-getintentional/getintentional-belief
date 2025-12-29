from typing import Dict, List, Optional

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.routes.thesis_builder import (
    CallCanonicalSummary,
    CallIngestionPayload,
    StakeholderRecord,
    ingest_thesis_call,
)
from backend.super_models.call_interpretation import CallSegment
from backend.super_models.thesis_builder import ThesisCall
from backend.utils.interpreters.sales_call_interpreter import (
    aggregate_call_interpretations,
    interpret_segment_with_student_then_teacher,
    load_canonical_vocab,
    StudentModel,
    TeacherClient,
)

router = APIRouter(prefix="/api/sales-call")


class SegmentUpload(BaseModel):
    text: str
    speaker_role: Optional[str] = None
    phase: Optional[str] = None
    idx: Optional[int] = None


class SalesCallInterpretationRequest(BaseModel):
    product_id: str
    account_id: Optional[str] = None
    external_call_id: Optional[str] = None
    transcript_text: str
    attribute_bins: Dict[str, str] = Field(default_factory=dict)
    stakeholders: List[StakeholderRecord] = Field(default_factory=list)
    segments: Optional[List[SegmentUpload]] = None


@router.post("/interpret")
def interpret_sales_call(payload: SalesCallInterpretationRequest, db: Session = Depends(get_db)):
    if not payload.transcript_text.strip():
        raise HTTPException(status_code=400, detail="Transcript text required.")

    call = ThesisCall(
        id=uuid.uuid4().hex,
        product_id=payload.product_id,
        account_id=payload.account_id,
        external_call_id=payload.external_call_id,
        attribute_bins=payload.attribute_bins,
        stakeholders=[stakeholder.dict(exclude_none=True) for stakeholder in payload.stakeholders],
        raw_summary={"source": "sales_call_interpreter"},
        transcript_text=payload.transcript_text,
    )
    db.add(call)
    db.flush()

    segments_data = payload.segments or [
        SegmentUpload(text=payload.transcript_text, idx=0),
    ]
    created_segments = []
    for idx, segment_payload in enumerate(segments_data):
        segment = CallSegment(
            id=uuid.uuid4().hex,
            call_id=call.id,
            idx=segment_payload.idx or idx,
            speaker_role=segment_payload.speaker_role,
            phase=segment_payload.phase,
            text=segment_payload.text,
        )
        db.add(segment)
        created_segments.append(segment)
    db.flush()

    student_model = StudentModel()
    teacher_client = TeacherClient()
    canonical_vocab = load_canonical_vocab()

    for segment in created_segments:
        interpret_segment_with_student_then_teacher(
            db=db,
            segment=segment,
            student_model=student_model,
            teacher_client=teacher_client,
            canonical_vocab=canonical_vocab,
        )

    canonical_summary = aggregate_call_interpretations(db, call.id)
    canonical_model = CallCanonicalSummary(**{
        "pains": canonical_summary.get("pains", []),
        "jobs": canonical_summary.get("jobs", []),
        "zmots": canonical_summary.get("zmots", []),
        "aspirations": canonical_summary.get("aspirations", []),
        "beliefs": canonical_summary.get("beliefs", []),
        "emotions": canonical_summary.get("emotions", []),
    })

    ingestion_payload = CallIngestionPayload(
        account_id=payload.account_id,
        external_call_id=payload.external_call_id,
        attribute_bins=payload.attribute_bins,
        stakeholders=payload.stakeholders,
        canonical_summary=canonical_model,
        raw_summary={"source": "sales_call_interpreter"},
        transcript_text=payload.transcript_text,
    )

    ingest_thesis_call(payload.product_id, ingestion_payload, db)
    return {"call_id": call.id, "canonical_summary": canonical_summary}
