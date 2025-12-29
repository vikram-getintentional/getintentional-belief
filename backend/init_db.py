# backend/init_db.py

from backend.database import Base, engine
from backend.super_models import company_analysis, company_value_prop
from backend.super_models.shm import episode
from backend.utils.crm_management import (
    engagement_models,
    person_models,
    target_account_manager,
)
from backend.utils.inference.belief_manager import cache as belief_cache  # noqa: F401
from backend.utils.inference.belief_manager.journey import shm_models as journey_shm_models
from backend.utils.knowledge_base.arsenal import db_models as arsenal_db_models  # noqa: F401
from backend.utils.strategy_builder import execution_models  # noqa: F401


def ensure_schema() -> None:
    """
    Create all registered tables and make sure dependent columns exist.
    """
    Base.metadata.create_all(bind=engine)
    try:
        from backend.utils.crm_management.engagement_service import ensure_engagement_columns

        ensure_engagement_columns()
    except ImportError:
        # best-effort: if the module cannot be imported, just skip the migration step
        pass
    try:
        from backend.utils.inference.belief_manager.journey.shm_service import (
            ensure_shm_meta_columns,
        )

        ensure_shm_meta_columns()
    except ImportError:
        pass


def init_db():
    ensure_schema()
