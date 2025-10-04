# crud/analysis.py

from sqlalchemy.orm import Session
from backend.super_models.company_analysis import CompanyAnalysis  # We'll define this next
import json
from backend.db.schema_templates.save_company_value_prop import CompanyValueProp



def save_analysis_by_url(db: Session, url: str, summary: str, capabilities: list[dict]):
    """
    For backwards compatibility — save analysis by URL (instead of company_id).
    """
    capabilities_json = json.dumps(capabilities)
    existing = db.query(CompanyValueProp).filter_by(url=url).first()

    if existing:
        existing.summary = summary
        existing.capabilities = capabilities_json
    else:
        new_entry = CompanyValueProp(
            company_id=url,  # TEMP fallback if no real company_id
            url=url,
            summary=summary,
            capabilities=capabilities_json
        )
        db.add(new_entry)

    db.commit()


def get_analysis_by_url(db: Session, url: str):
    """
    Fetch a record by URL.
    """
    normalized_url = (
        url.replace("http://", "")
           .replace("https://", "")
           .replace("www.", "")
           .strip("/")
    )

    record = db.query(CompanyValueProp).filter(CompanyValueProp.url.contains(normalized_url)).first()

    if not record:
        return None

    return {
        "summary": record.summary,
        "capabilities": json.loads(record.capabilities)
    }

def save_analysis(db: Session, url: str, summary: str, capabilities: list[dict]):
    capabilities_json = json.dumps(capabilities)
    existing = db.query(CompanyAnalysis).filter(CompanyAnalysis.url == url).first()

    if existing:
        existing.summary = summary
        existing.capabilities = capabilities_json
    else:
        new_record = CompanyAnalysis(
            url=url,
            summary=summary,
            capabilities=capabilities_json
        )
        db.add(new_record)

    db.commit()