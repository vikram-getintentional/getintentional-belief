from dataclasses import dataclass
from typing import Dict, Optional

from backend.utils.interpreters.sales_call_interpreter.student_model import StudentModel, StudentPredictionResult
from backend.utils.interpreters.sales_call_interpreter.vocab import CanonicalVocabulary


@dataclass
class TeacherResult:
    payload: Dict
    explanation: Optional[str] = None


class TeacherClient:
    def __init__(self):
        self._student = StudentModel()

    def interpret_segment(
        self,
        segment_text: str,
        speaker_role: Optional[str],
        phase: Optional[str],
        canonical_vocab: CanonicalVocabulary,
    ) -> TeacherResult:
        prediction: StudentPredictionResult = self._student.predict(
            segment_text=segment_text,
            speaker_role=speaker_role,
            phase=phase,
            canonical_vocab=canonical_vocab,
        )
        payload = prediction.to_payload()
        explanation = "Teacher fallback using keyword heuristics and expanded intensity."
        return TeacherResult(payload=payload, explanation=explanation)
