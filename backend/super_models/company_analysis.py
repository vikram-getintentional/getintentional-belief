# backend/models/company_analysis.py

from sqlalchemy import Column, String
from backend.database import Base

class CompanyAnalysis(Base):
    __tablename__ = "saved_analyses"

    url = Column(String, primary_key=True, index=True)
    summary = Column(String)
    capabilities = Column(String)  # stored as JSON
    capability_pains = Column(String)  # NEW: stores JSON blob of capability-pain-confidence
