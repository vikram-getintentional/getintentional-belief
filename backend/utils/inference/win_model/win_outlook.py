from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from backend.utils.inference.win_model.arsenal_forecaster import ArsenalForecaster
from backend.utils.inference.win_model.engagement_model import EngagementEvidenceModel
from backend.utils.inference.win_model.meta_model import MetaModel


def build_win_outlook(
    *,
    account_meta: Dict[str, Any],
    win_regression: Optional[Dict[str, Any]],
    observed_engagements: Optional[Sequence[Dict[str, Any]]],
    projected_engagements: Optional[Sequence[Dict[str, Any]]],
    plays: Optional[Sequence[Dict[str, Any]]],
    arsenal_impact: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    meta_model = MetaModel(win_regression)
    right_to_win = meta_model.predict(account_meta)
    engagement_model = EngagementEvidenceModel()
    observed_list = observed_engagements or []
    baseline_win = engagement_model.update_with_engagements(right_to_win, observed_list)
    forecaster = ArsenalForecaster()
    predicted_win = forecaster.forecast(
        baseline_win,
        plays or [],
        arsenal_impact=arsenal_impact,
    )
    projected_list = projected_engagements or []
    projected_count = len(projected_list)
    observed_count = len(observed_list)
    forecast_evidence_count = observed_count + projected_count
    if projected_count:
        forecast_summary = (
            f"Includes {projected_count} planned engagement signal(s) on top of the baseline."
        )
    else:
        forecast_summary = baseline_win.get("evidence", {}).get(
            "summary", "Forecast follows the baseline."
        )
    predicted_win["evidence"] = {
        "engagement_count": forecast_evidence_count,
        "observed_signals": observed_count,
        "planned_signals": projected_count,
        "summary": forecast_summary,
    }
    baseline_win["evidence"]["observed_signals"] = observed_count
    diagnostics = {
        "graph_walk_reachability": 1.0,
        "note": "Diagnostic only; not shown in UI",
    }
    return {
        "diagnostics": diagnostics,
        "right_to_win": right_to_win,
        "baseline_win": baseline_win,
        "predicted_win": predicted_win,
    }
