from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from backend.utils.interpreters.sales_call_interpreter.vocab import CanonicalEntry, CanonicalVocabulary


@dataclass
class LabelIntensity:
    id: str
    intensity: float


@dataclass
class StudentPredictionResult:
    pains: List[LabelIntensity]
    jobs: List[LabelIntensity]
    zmots: List[LabelIntensity]
    aspirations: List[LabelIntensity]
    emotions: List[LabelIntensity]
    belief_level: Optional[str]
    belief_score: float
    confidence: float

    def to_payload(self) -> Dict[str, List[Dict]]:
        return {
            "pains": [{"pain_id": p.id, "intensity": p.intensity} for p in self.pains],
            "jobs": [{"job_id": j.id, "intensity": j.intensity} for j in self.jobs],
            "zmots": [
                {"pain_id": z.id.split("|")[0], "zmot_id": z.id.split("|")[1], "intensity": z.intensity}
                if "|" in z.id
                else {"pain_id": "unknown", "zmot_id": z.id, "intensity": z.intensity}
                for z in self.zmots
            ],
            "aspirations": [
                {"pain_id": a.id.split("|")[0], "aspiration_id": a.id.split("|")[1], "intensity": a.intensity}
                if "|" in a.id
                else {"pain_id": "unknown", "aspiration_id": a.id, "intensity": a.intensity}
                for a in self.aspirations
            ],
            "beliefs": [
                {
                    "pain_id": b.id.split("|")[0],
                    "aspiration_id": b.id.split("|")[1],
                    "belief_level": self.belief_level or "medium",
                    "intensity": b.intensity,
                }
                if "|" in b.id
                else {
                    "pain_id": "unknown",
                    "aspiration_id": "unknown",
                    "belief_level": self.belief_level or "medium",
                    "intensity": b.intensity,
                }
            ]
            if self.belief_level
            else [],
            "emotions": [{"label": e.id, "intensity": e.intensity} for e in self.emotions],
        }


class StudentModel:
    version = "student-v0.1"

    def predict(
        self,
        segment_text: str,
        speaker_role: Optional[str],
        phase: Optional[str],
        canonical_vocab: CanonicalVocabulary,
    ) -> StudentPredictionResult:
        tokens = self._prepare_text(segment_text, speaker_role, phase)
        pains = self._match(tokens, canonical_vocab.pains)
        jobs = self._match(tokens, canonical_vocab.jobs)
        zmots = self._match(tokens, canonical_vocab.zmots)
        aspirations = self._match(tokens, canonical_vocab.aspirations)
        emotions = self._match(tokens, canonical_vocab.emotions)
        belief_level, belief_score = self._predict_belief(tokens, canonical_vocab.belief_levels)
        confidence = self._compute_confidence(pains, jobs, zmots, aspirations, belief_score, emotions)
        return StudentPredictionResult(
            pains=pains,
            jobs=jobs,
            zmots=zmots,
            aspirations=aspirations,
            emotions=emotions,
            belief_level=belief_level,
            belief_score=belief_score,
            confidence=confidence,
        )

    def _prepare_text(self, segment_text: str, speaker_role: Optional[str], phase: Optional[str]) -> str:
        parts = []
        if speaker_role:
            parts.append(f"[ROLE={speaker_role}]")
        if phase:
            parts.append(f"[PHASE={phase}]")
        parts.append(segment_text.lower())
        return " ".join(parts)

    def _match(self, text: str, entries: List[CanonicalEntry]) -> List[LabelIntensity]:
        matches: List[LabelIntensity] = []
        for entry in entries:
            hit = any(keyword in text for keyword in entry.keywords)
            if hit:
                matches.append(LabelIntensity(id=entry.id, intensity=min(1.0, entry.base_score)))
        return matches

    def _predict_belief(self, text: str, belief_levels: List[str]) -> Tuple[Optional[str], float]:
        for level in belief_levels:
            if level in text:
                return level, 0.9
        return None, 0.0

    def _compute_confidence(
        self,
        pains: List[LabelIntensity],
        jobs: List[LabelIntensity],
        zmots: List[LabelIntensity],
        aspirations: List[LabelIntensity],
        belief_score: float,
        emotions: List[LabelIntensity],
    ) -> float:
        def max_intensities(items: List[LabelIntensity]) -> float:
            if not items:
                return 0.0
            return max(item.intensity for item in items)

        conf_pains = max_intensities(pains)
        conf_jobs = max_intensities(jobs)
        conf_zmots = max_intensities(zmots)
        conf_asp = max_intensities(aspirations)
        conf_emotions = max_intensities(emotions)
        # belief_score already 0-1
        aggregated = (
            0.25 * conf_pains
            + 0.15 * conf_jobs
            + 0.2 * conf_zmots
            + 0.2 * conf_asp
            + 0.1 * belief_score
            + 0.1 * conf_emotions
        )
        return max(0.0, min(1.0, aggregated))
