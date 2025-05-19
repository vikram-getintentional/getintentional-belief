from sqlalchemy import Table, Column, String, Float, MetaData

def create_company_personas_table(metadata: MetaData, company_id: str) -> Table:
    table_name = f"{company_id}_personas"
    
    return Table(
        table_name,
        metadata,
        Column("persona", String, nullable=False, primary_key=True),
        Column("job_title", String, nullable=False, primary_key=True),
        Column("pain", String, nullable=False, primary_key=True),
        Column("stage", String, nullable=False, default="ZMOT"),
        Column("belief_score", Float, nullable=False, default=0.5),
        Column("importance_score", Float, nullable=False, default=0.5),
        Column("source", String, nullable=False, default="manual"),
    )
