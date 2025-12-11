from __future__ import annotations

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from backend.utils.strategy_builder.execution_models import ExecutionInterventionDecision


def _find_record(
    db: Session,
    *,
    product_id: str,
    account_id: Optional[str],
    intervention_id: str,
) -> Optional[ExecutionInterventionDecision]:
    return (
        db.query(ExecutionInterventionDecision)
        .filter(
            ExecutionInterventionDecision.product_id == product_id,
            ExecutionInterventionDecision.account_id == account_id,
            ExecutionInterventionDecision.intervention_id == intervention_id,
        )
        .one_or_none()
    )


def upsert_decision(
    db: Session,
    *,
    product_id: str,
    scope: str,
    account_id: Optional[str],
    intervention_id: str,
    status: str,
    persona: Optional[str],
    concern: Optional[str],
    asset_type: Optional[str],
    channel: Optional[str],
    recommended_asset_id: Optional[str],
    recommended_asset_name: Optional[str],
    selected_asset_id: Optional[str],
    selected_asset_name: Optional[str],
    selected_asset_type: Optional[str],
    selected_channel: Optional[str],
    notes: Optional[str],
    metadata: Optional[Dict],
    user_id: Optional[str],
) -> ExecutionInterventionDecision:
    record = _find_record(
        db,
        product_id=product_id,
        account_id=account_id,
        intervention_id=intervention_id,
    )
    if not record:
        record = ExecutionInterventionDecision(
            product_id=product_id,
            account_id=account_id,
            intervention_id=intervention_id,
        )
        db.add(record)

    record.scope = scope
    record.status = status
    record.persona = persona
    record.concern = concern
    record.asset_type = asset_type
    record.channel = channel
    record.recommended_asset_id = recommended_asset_id
    record.recommended_asset_name = recommended_asset_name
    record.selected_asset_id = selected_asset_id
    record.selected_asset_name = selected_asset_name
    record.selected_asset_type = selected_asset_type
    record.selected_channel = selected_channel
    record.notes = notes
    record.metadata_json = metadata or {}
    if user_id:
        record.updated_by = user_id
        if not record.created_by:
            record.created_by = user_id

    db.commit()
    db.refresh(record)
    return record


def list_decisions(
    db: Session,
    *,
    product_id: str,
    account_id: Optional[str] = None,
) -> List[ExecutionInterventionDecision]:
    query = db.query(ExecutionInterventionDecision).filter(
        ExecutionInterventionDecision.product_id == product_id,
    )
    if account_id is None:
        query = query.filter(ExecutionInterventionDecision.account_id.is_(None))
    else:
        query = query.filter(ExecutionInterventionDecision.account_id == account_id)
    query = query.order_by(ExecutionInterventionDecision.updated_at.desc())
    return list(query.all())
