# backend/init_db.py

from backend.database import Base, engine
from backend.super_models import company_analysis, company_value_prop


def init_db():
    Base.metadata.create_all(bind=engine)
