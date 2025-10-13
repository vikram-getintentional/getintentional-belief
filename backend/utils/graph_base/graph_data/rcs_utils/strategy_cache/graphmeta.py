# backend/utils/strategy_cache/graphmeta.py
from sqlalchemy.orm import Session
from backend.utils.graph_base.graph_data.rcs_utils.strategy_model import GraphMeta

def get_current_graphmeta(db: Session, company_id: str, product_id: str) -> GraphMeta | None:
    return (
        db.query(GraphMeta)
        .filter(GraphMeta.company_id == company_id,
                GraphMeta.product_id == product_id,
                GraphMeta.is_current == True)
        .order_by(GraphMeta.graph_version.desc())
        .first()
    )

def upsert_graphmeta(db: Session, company_id: str, product_id: str, graph_hash: str, graph_version: int) -> GraphMeta:
    # mark older currents as not current
    db.query(GraphMeta).filter(
        GraphMeta.company_id == company_id,
        GraphMeta.product_id == product_id,
        GraphMeta.is_current == True
    ).update({"is_current": False})

    gm = GraphMeta(
        company_id=company_id,
        product_id=product_id,
        graph_hash=graph_hash,
        graph_version=graph_version,
        is_current=True,
    )
    db.add(gm)
    db.commit()
    db.refresh(gm)
    return gm
