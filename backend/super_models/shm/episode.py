# backend/super_models/shm/episode.py

from __future__ import annotations
import enum
import uuid
import datetime as dt

from sqlalchemy import (
    Column,
    String,
    Enum,
    DateTime,
    Integer,
    Float,
    JSON,
    ForeignKey,
    Boolean,
    Index,
)
from sqlalchemy.orm import relationship

from backend.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def utcnow():
    return dt.datetime.now(dt.timezone.utc)


class EpisodeOutcome(str, enum.Enum):
    unknown = "unknown"  # Outcome of this episode is unknown
    open = "open"        # Is an active opp, has not closed yet
    won = "won"
    lost = "lost"


class StepBucket(str, enum.Enum):
    on_path = "on_path"
    early = "early"
    late = "late"
    off_path = "off_path"


class ShmUpdateType(str, enum.Enum):
    bayes_param_update = "bayes_param_update"
    structure_add_edge = "structure_add_edge"
    structure_remove_edge = "structure_remove_edge"
    structure_weaken_edge = "structure_weaken_edge"
    structure_strengthen_edge = "structure_strengthen_edge"
    path_delta = "path_delta"
    graph_level_thesis = "graph_level_thesis"
    neighborhood = "neighborhood"
    handoff_neighborhood = "handoff_neighborhood"


class ShmEpisode(Base):
    """
    One belief journey for (product_id, account_id).

    - Groups a time-ordered set of ShmEpisodeSteps
    - Outcome is win / loss / open to handle survivorship bias
    """

    __tablename__ = "shm_episodes_meta"

    id = Column(String, primary_key=True, default=_uuid)

    product_id = Column(String, index=True, nullable=False)
    account_id = Column(String, index=True, nullable=False)

    # optional: pointer to a particular thesis / model version used
    journey_model_version = Column(String, nullable=True)

    outcome = Column(Enum(EpisodeOutcome), default=EpisodeOutcome.unknown, index=True)

    # Was the episode “complete” from our POV?
    is_censored = Column(Boolean, default=False)  # e.g. churned, no closure event

    started_at = Column(DateTime, default=utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)

    # simple meta/tags to segment later (e.g. ICP type, segment, region...)
    account_meta = Column(JSON, nullable=True)

    # roll-up metrics and learning summaries (opaque JSON blobs)
    metrics = Column(JSON, nullable=True)
    learning_summary = Column(JSON, nullable=True)
    candidate_personas = Column(JSON, nullable=True)

    # convenience count of steps in this journey
    num_steps = Column(Integer, nullable=True)

    # relationships
    steps = relationship(
        "ShmEpisodeStep",
        back_populates="episode",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    learning_updates = relationship(
        "ShmLearningUpdate",
        back_populates="episode",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


# useful composite index for querying episodes
Index(
    "ix_shm_episodes_meta_product_account_started",
    ShmEpisode.product_id,
    ShmEpisode.account_id,
    ShmEpisode.started_at,
)


class ShmEpisodeStep(Base):
    """
    A single time step within an episode's belief journey.

    Mirrors entries in journey["steps"] from belief_manager.build_belief_thesis.
    """

    __tablename__ = "shm_episode_steps"

    id = Column(String, primary_key=True, default=_uuid)

    episode_id = Column(
        String,
        ForeignKey("shm_episodes_meta.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # t from journey["steps"][i]["t"]
    t_index = Column(Integer, nullable=False)

    # coarse bucket for analysis (on_path / early / late / off_path)
    bucket = Column(Enum(StepBucket), nullable=True, index=True)

    # observed persona at this step (if any)
    observed_persona_id = Column(String, nullable=True)

    # top predicted persona id at this step (if any)
    predicted_top_persona_id = Column(String, nullable=True)

    # raw prediction + walk data as JSON (opaque; mirrors journey["steps"][i])
    predicted_topK = Column(JSON, nullable=True)
    walk_paths = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)

    # convenient flags
    hit_at_1 = Column(Boolean, nullable=True)
    hit_at_3 = Column(Boolean, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)

    episode = relationship("ShmEpisode", back_populates="steps")


Index(
    "ix_shm_episode_steps_episode_t",
    ShmEpisodeStep.episode_id,
    ShmEpisodeStep.t_index,
)


class ShmLearningUpdate(Base):
    """
    A single learning update emitted during/after an episode.

    Typically corresponds to a ranked edge or node recommendation from the
    graph_diff_mapper / Bayesian learner.
    """

    __tablename__ = "shm_learning_updates"

    id = Column(String, primary_key=True, default=_uuid)

    episode_id = Column(
        String,
        ForeignKey("shm_episodes_meta.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    product_id = Column(String, index=True, nullable=False)
    account_id = Column(String, index=True, nullable=False)

    update_type = Column(Enum(ShmUpdateType), nullable=False, index=True)

    # band / neighborhood band like "upstream" | "handoff" | "downstream"
    band = Column(String, nullable=True)

    # edge-local info (for parameter/structural updates)
    u_node_id = Column(String, nullable=True)
    v_node_id = Column(String, nullable=True)

    # node-local info (for perceptibility / node proposals, etc.)
    node_id = Column(String, nullable=True)
    field = Column(String, nullable=True)  # e.g. "perceptibility"

    delta = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

    # opaque details: reason, labels, neighborhood keys, etc.
    payload = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)

    episode = relationship("ShmEpisode", back_populates="learning_updates")


Index(
    "ix_shm_learning_updates_prod_acc",
    ShmLearningUpdate.product_id,
    ShmLearningUpdate.account_id,
)
Index(
    "ix_shm_learning_updates_episode_type",
    ShmLearningUpdate.episode_id,
    ShmLearningUpdate.update_type,
)
