# backend/schema_templates/save_value_prop.py

import json
from sqlalchemy import Column, String
from sqlalchemy.orm import Session
from backend.database import Base
from backend.super_models.company_value_prop import CompanyValueProp

# ✅ Save or update value prop for a company
def save_company_value_prop(
    db: Session,
    company_id: str,
    url: str,
    summary: str,
    capabilities: list[dict]
):
    capabilities_json = json.dumps(capabilities)

    existing = db.query(CompanyValueProp).filter_by(company_id=company_id).first()
    if existing:
        existing.url = url
        existing.summary = summary
        existing.capabilities = capabilities_json
    else:
        new_entry = CompanyValueProp(
            company_id=company_id,
            url=url,
            summary=summary,
            capabilities=capabilities_json
        )
        db.add(new_entry)

    db.commit()


# ✅ Fetch value prop for a company
def get_company_value_prop(db: Session, company_id: str):
    record = db.query(CompanyValueProp).filter_by(company_id=company_id).first()
    if not record:
        return None

    return {
        "url": record.url,
        "summary": record.summary,
        "capabilities": json.loads(record.capabilities)
    }
