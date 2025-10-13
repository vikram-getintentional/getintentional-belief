from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, DateTime, Boolean, JSON, Index
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class GraphMeta(Base):
    """
    One row per active product graph build. You can keep multiple for history,
    but only one should have is_current=True per company/product.
    """
    __tablename__ = "graph_meta"

    id = Column(Integer, primary_key=True)
    company_id = Column(String, index=True, nullable=False)
    product_id = Column(String, index=True, nullable=False)

    graph_version = Column(Integer, nullable=False)           # monotonic
    graph_hash = Column(String, nullable=False)               # sha256 hex
    is_current = Column(Boolean, default=True, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # helpful for debugging
    notes = Column(String, nullable=True)

    __table_args__ = (
        Index("ix_graphmeta_current", "company_id", "product_id", "is_current"),
    )


class Strategy(Base):
    """
    Cached strategy outputs. Two 'types':
      - frozen: payload for initial_nodes only (engaged_hash == "")
      - tactical: payload for initial_nodes + engaged_nodes (engaged_hash != "")
    We mark the latest valid one as is_current=True (overwrite semantics).
    """
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)

    company_id = Column(String, index=True, nullable=False)
    product_id = Column(String, index=True, nullable=False)

    type = Column(String, nullable=False)  # "frozen" | "tactical"

    # Content-addressed keys
    inputs_hash = Column(String, index=True, nullable=False)   # sha256 hex
    engaged_hash = Column(String, index=True, nullable=False, default="")  # "" for frozen
    graph_hash = Column(String, index=True, nullable=False)
    graph_version = Column(Integer, nullable=False)
    algo_hash = Column(String, index=True, nullable=False)

    # State
    is_current = Column(Boolean, default=True, nullable=False)
    superseded_by = Column(Integer, nullable=True)  # FK to strategies.id (optional)

    # Payload + telemetry
    payload = Column(JSON, nullable=False)
    cost_ms = Column(Integer, nullable=True)
    cost_tokens = Column(Integer, nullable=True)
    confidence_score = Column(Integer, nullable=True)  # optional 0-100

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        # Enforce single “current” row per logical key.
        # SQLite can’t do partial unique indexes; we include is_current in the key.
        Index(
            "uq_strategy_current_key",
            "company_id", "product_id", "type",
            "inputs_hash", "engaged_hash", "graph_hash", "algo_hash", "is_current",
            unique=True,
        ),
        Index("ix_strategy_lookup", "company_id", "product_id", "type", "inputs_hash", "engaged_hash"),
    )
