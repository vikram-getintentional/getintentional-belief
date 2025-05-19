from sqlalchemy import MetaData, Table, Column, String, Float
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from backend.database import engine
from backend.db.schema_templates.persona_table import create_company_personas_table

def save_company_personas(company_id: str, persona_entries: list[dict]):
    """
    Save selected personas and pains into the {company_id}_personas table.

    Each entry in persona_entries should be:
    {
        "persona": "Head of RevOps",
        "job_title": "RevOps",
        "pains": ["Data is siloed", "CRM doesn't sync"],
        "stage": "ZMOT",
        "belief_score": 0.7,
        "importance_score": 0.5,
        "source": "manual"
    }
    """
    metadata = MetaData()
    table = create_company_personas_table(metadata, company_id)
    metadata.create_all(bind=engine, tables=[table])  # Ensures table exists

    conn = engine.connect()
    try:
        # Clear existing records (optional)
        conn.execute(table.delete())

        flattened_rows = []
        for entry in persona_entries:
            for pain in entry["pains"]:
                flattened_rows.append({
                    "persona": entry["persona"],
                    "job_title": entry["job_title"],
                    "pain": pain,  # Flattened single pain per row
                    "stage": entry["stage"],
                    "belief_score": entry["belief_score"],
                    "importance_score": entry["importance_score"],
                    "source": entry["source"]
                })

        # Insert all entries
        conn.execute(table.insert(), persona_entries)
        conn.commit()
        print(f"✅ Saved {len(persona_entries)} entries to {company_id}_personas")
    except SQLAlchemyError as e:
        print("❌ Error saving company personas:", e)
    finally:
        conn.close()
