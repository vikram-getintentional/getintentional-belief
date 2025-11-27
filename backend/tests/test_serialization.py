import json
from datetime import datetime, timezone

from backend.utils.inference.belief_manager.belief_manager import _json_safe
from backend.utils.inference.belief_manager.types import (
    EdgeAdjustment,
    LocalAdjustment,
    NodeAdjustment,
    PersonaPrediction,
    PredictionError,
)
from backend.utils.inference.belief_manager.journey.shm_service import _json_clean


def _sample_adjustment(ts: datetime) -> LocalAdjustment:
    edge = EdgeAdjustment(
        source_id="persona:u",
        target_id="persona:v",
        delta=0.42,
        rationale="edge rationale",
        confidence=0.8,
        diagnostics={"band": "handoff"},
    )
    node = NodeAdjustment(
        node_id="persona:u",
        field="perceptibility",
        delta=-0.13,
        band="upstream",
        rationale="node rationale",
        confidence=0.65,
        diagnostics={"note": "test"},
    )
    prediction = PersonaPrediction(
        account_id="acct-1",
        timestamp=ts,
        persona_sequence_before=("persona:x",),
        predicted_distribution={"persona:u": 0.7, "persona:v": 0.3},
        predicted_topk=("persona:u", "persona:v"),
        rationale=None,
        candidate_paths=(("persona:x", "persona:u"),),
    )
    error = PredictionError(
        account_id="acct-1",
        timestamp=ts,
        persona_sequence_before=("persona:x",),
        predicted_distribution={"persona:u": 0.7, "persona:v": 0.3},
        predicted_topk=("persona:u", "persona:v"),
        actual_persona="persona:v",
        hit_at_1=False,
        hit_at_3=True,
        bucket="jump_ahead",
        log_loss=1.23,
        brier_score=0.45,
        rationale="mis-weighted path",
    )
    return LocalAdjustment(
        account_id="acct-1",
        timestamp=ts,
        band="handoff",
        edges=[edge],
        nodes=[node],
        error=error,
        diagnostic={"metrics": {"surprise": 0.8}},
        global_params_version=3,
    )


def test_json_safe_handles_nested_dataclasses_and_datetimes():
    ts = datetime(2024, 10, 2, 12, 30, tzinfo=timezone.utc)
    adjustment = _sample_adjustment(ts)

    payload = {
        "timestamp": ts,
        "local_adjustment": adjustment,
        "prediction": adjustment.error,
        "edges": adjustment.edges,
    }

    safe_payload = _json_safe(payload)

    # Should be JSON encodable without TypeErrors
    json.dumps(safe_payload)

    assert safe_payload["timestamp"].startswith("2024-10-02T12:30:00")
    assert safe_payload["local_adjustment"]["timestamp"].startswith("2024-10-02T12:30:00")
    assert safe_payload["prediction"]["bucket"] == "jump_ahead"
    assert safe_payload["edges"][0]["source_id"] == "persona:u"


def test_json_clean_matches_json_safe_for_problematic_values():
    ts = datetime(2024, 5, 17, 9, 15, tzinfo=timezone.utc)
    adjustment = _sample_adjustment(ts)
    payload = {
        "metrics": {"error": adjustment.error, "local_adjustment": adjustment},
        "predicted_topK": adjustment.error.predicted_topk,
        "walk_paths": adjustment.error.persona_sequence_before,
    }

    cleaned = _json_clean(payload)

    # cleaned payload should also be JSON serializable
    json.dumps(cleaned)

    assert cleaned["metrics"]["error"]["actual_persona"] == "persona:v"
    assert cleaned["predicted_topK"][0] == "persona:u"
