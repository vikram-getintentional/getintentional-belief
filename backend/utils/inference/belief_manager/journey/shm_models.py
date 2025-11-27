# backend/utils/inference/belief_manager/journey/shm_models.py
from __future__ import annotations
from typing import Any, Dict, Optional
from sqlalchemy import Column, String, DateTime, Integer, JSON, Index
import uuid
import datetime as dt

from backend.database import Base

def _uuid() -> str:
    return str(uuid.uuid4())

class SHMEpisode(Base):
    """
    Sequential Hippocampal Memory (SHM) episode.

    Each row is one *step* in an episode:
    - An episode is a sequence of observations for (product, account).
    - Episode is identified by (product_id, account_id, episode_id).
    """

    __tablename__ = "shm_episodes"

    id = Column(String, primary_key=True, default=_uuid)

    # identity / context
    product_id = Column(String, index=True, nullable=False)
    account_id = Column(String, index=True, nullable=False)

    # episode identity
    episode_id = Column(String, index=True, nullable=False)
    step_index = Column(Integer, nullable=False)  # position within episode

    # time
    timestamp = Column(DateTime, default=dt.datetime.utcnow, index=True)

    # belief/journey context
    persona_id = Column(String, index=True, nullable=False)
    belief_state = Column(String, nullable=False)  # e.g. "unaware", "zmot", "discovery", "evaluation", ...
    belief_score = Column(Integer, nullable=True)  # optional 0–100, or None if not computed

    # engagement
    engagement_type = Column(String, nullable=False)  # "email", "webinar", "demo_call", ...
    channel = Column(String, nullable=True)          # "marketing", "sales", "cs", ...
    asset_id = Column(String, nullable=True)         # if you have asset ids from arsenal

    # learning label for this step
    effect_bucket = Column(
        String,
        nullable=True,
        # e.g. "positive", "neutral", "negative", "off_path", "on_path"
    )

    # raw / debug payload
    meta = Column(JSON, default=dict)  # free-form, store journey step json, scores, etc.

    __table_args__ = (
        Index(
            "idx_shm_episode_seq",
            "product_id",
            "account_id",
            "episode_id",
            "step_index",
        ),
    )
