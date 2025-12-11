from __future__ import annotations

import datetime as dt
import uuid
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    JSON,
    String,
    UniqueConstraint,
)

from backend.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class ExecutionInterventionDecision(Base):
    __tablename__ = "execution_intervention_decisions"

    id = Column(String, primary_key=True, default=_uuid)
    product_id = Column(String, nullable=False, index=True)
    account_id = Column(String, nullable=True, index=True)
    scope = Column(String, nullable=False)  # "portfolio" or "account"
    intervention_id = Column(String, nullable=False, index=True)

    persona = Column(String, nullable=True)
    concern = Column(String, nullable=True)
    asset_type = Column(String, nullable=True)
    channel = Column(String, nullable=True)

    recommended_asset_id = Column(String, nullable=True)
    recommended_asset_name = Column(String, nullable=True)

    selected_asset_id = Column(String, nullable=True)
    selected_asset_name = Column(String, nullable=True)
    selected_asset_type = Column(String, nullable=True)
    selected_channel = Column(String, nullable=True)

    status = Column(String, nullable=False, default="pending")  # accepted | overridden
    notes = Column(String, nullable=True)

    metadata_json = Column("metadata", JSON, default=dict)

    created_by = Column(String, nullable=True)
    updated_by = Column(String, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=dt.datetime.utcnow,
        onupdate=dt.datetime.utcnow,
    )

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "account_id",
            "intervention_id",
            name="uq_execution_intervention_decision",
        ),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "product_id": self.product_id,
            "account_id": self.account_id,
            "scope": self.scope,
            "intervention_id": self.intervention_id,
            "persona": self.persona,
            "concern": self.concern,
            "asset_type": self.asset_type,
            "channel": self.channel,
            "recommended_asset_id": self.recommended_asset_id,
            "recommended_asset_name": self.recommended_asset_name,
            "selected_asset_id": self.selected_asset_id,
            "selected_asset_name": self.selected_asset_name,
            "selected_asset_type": self.selected_asset_type,
            "selected_channel": self.selected_channel,
            "status": self.status,
            "notes": self.notes,
            "metadata": self.metadata_json or {},
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "created_at": (
                self.created_at.isoformat() if isinstance(self.created_at, dt.datetime) else None
            ),
            "updated_at": (
                self.updated_at.isoformat() if isinstance(self.updated_at, dt.datetime) else None
            ),
        }
