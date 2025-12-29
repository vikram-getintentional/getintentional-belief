import uuid
from typing import Optional

from sqlalchemy.orm import Session

from backend.super_models.call_interpretation import CallSegment, LabelSource, SegmentInterpretation
from backend.utils.interpreters.sales_call_interpreter.student_model import StudentModel
from backend.utils.interpreters.sales_call_interpreter.teacher_client import TeacherClient
from backend.utils.interpreters.sales_call_interpreter.vocab import (
    CanonicalVocabulary,
    load_canonical_vocab,
)


def interpret_segment_with_student_then_teacher(
    *,
    db: Session,
    segment: CallSegment,
    student_model: Optional[StudentModel] = None,
    teacher_client: Optional[TeacherClient] = None,
    canonical_vocab: Optional[CanonicalVocabulary] = None,
    confidence_threshold: float = 0.7,
) -> SegmentInterpretation:
    student_model = student_model or StudentModel()
    teacher_client = teacher_client or TeacherClient()
    canonical_vocab = canonical_vocab or load_canonical_vocab()

    prediction = student_model.predict(
        segment_text=segment.text,
        speaker_role=segment.speaker_role,
        phase=segment.phase,
        canonical_vocab=canonical_vocab,
    )
    payload = prediction.to_payload()
    explanation: Optional[str] = None
    model_version = student_model.version
    confidence = prediction.confidence
    source = LabelSource.STUDENT
    if prediction.confidence < confidence_threshold:
        teacher_result = teacher_client.interpret_segment(
            segment_text=segment.text,
            speaker_role=segment.speaker_role,
            phase=segment.phase,
            canonical_vocab=canonical_vocab,
        )
        payload = teacher_result.payload
        explanation = teacher_result.explanation
        model_version = None
        confidence = 1.0
        source = LabelSource.TEACHER_LLM

    interpretation = SegmentInterpretation(
        id=uuid.uuid4().hex,
        segment_id=segment.id,
        payload=payload,
        source=source,
        model_version=model_version,
        confidence=confidence,
        explanation=explanation,
    )
    db.add(interpretation)
    db.flush()
    return interpretation
