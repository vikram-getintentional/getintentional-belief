# backend/supermodels/company_value_prop.py

from sqlalchemy import Column, String
from backend.database import Base

class CompanyValueProp(Base):
    __tablename__ = "company_value_props"

    company_id = Column(String, primary_key=True, index=True)
    url = Column(String)
    summary = Column(String)
    capabilities = Column(String)  # stored as JSON string
    capability_pains = Column(String)  # NEW: stores JSON blob of capability-pain-confidence
