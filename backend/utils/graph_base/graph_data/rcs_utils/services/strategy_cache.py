# backend/services/strategy_cache.py
from typing import Dict, Any, Iterable, Optional
from sqlalchemy.orm import Session
from backend.db.models.strategy import Strategy
from backend.utils.strategy_cache.keys import (
    inputs_hash_from_initial_nodes, engaged_hash_from_nodes,
    algo_hash_from_params,
)
from backend.utils.strategy_cache.graphmeta import get_current_graphmeta

# --- Pluggable hooks you already have ---
def generate_rcs(product_graph, initial_nodes: list[str], algo_params: dict) -> dict:
    """TODO: Your existing expensive baseline RCS generator (frozen)."""
    raise NotImplementedError

def generate_rcs_with_engagements(product_graph, initial_nodes: list[str], engaged_nodes: list[str], algo_params: dict) -> dict:
    """TODO: Your existing generator when engagements are included (full plan)."""
    raise NotImplementedError

def diff_tactical_vs_frozen(full_plan: dict, frozen_plan: dict) -> dict:
    """TODO: Compute actionable deltas (new priorities, next best actions, etc.)."""
    return {
        "delta": {},  # summarize whatever you need
        "full_plan": full_plan,
        "baseline": frozen_plan,
    }

# --- Core cache helpers ---
def _mark_old_current_false(db: Session, row: Strategy):
    db.query(Strategy).filter(
        Strategy.company_id == row.company_id,
        Strategy.product_id == row.product_id,
        Strategy.type == row.type,
        Strategy.inputs_hash == row.inputs_hash,
        Strategy.engaged_hash == row.engaged_hash,
        Strategy.graph_hash == row.graph_hash,
        Strategy.algo_hash == row.algo_hash,
        Strategy.is_current == True,
    ).update({"is_current": False})

def _insert_current(db: Session, payload: dict, row: Strategy) -> Strategy:
    db.add(row)
    db.commit()
    db.refresh(row)
    return row

def _lookup_current(db: Session, **key_fields) -> Strategy | None:
    return (
        db.query(Strategy)
        .filter_by(**key_fields, is_current=True)
        .order_by(Strategy.id.desc())
        .first()
    )

# --- Public API ---
def get_or_build_frozen(
    db: Session,
    *,
    company_id: str,
    product_id: str,
    product_graph,
    initial_nodes: Iterable[str],
    algo_version: str,
    algo_params: Dict[str, Any],
    attributes: Optional[Dict[str, Any]] = None,
    zmot: Optional[Any] = None,
) -> dict:
    gm = get_current_graphmeta(db, company_id, product_id)
    if not gm:
        # If you don’t persist GraphMeta, compute from graph here and set version=1
        raise RuntimeError("No current GraphMeta found. Build/record the graph first.")

    i_hash = inputs_hash_from_initial_nodes(initial_nodes, attributes, zmot)
    a_hash = algo_hash_from_params(algo_version, algo_params)

    hit = _lookup_current(
        db,
        company_id=company_id,
        product_id=product_id,
        type="frozen",
        inputs_hash=i_hash,
        engaged_hash="",  # frozen
        graph_hash=gm.graph_hash,
        algo_hash=a_hash,
    )
    if hit:
        return hit.payload

    # MISS → compute & store
    frozen_payload = generate_rcs(product_graph, list(set(initial_nodes)), algo_params)

    row = Strategy(
        company_id=company_id,
        product_id=product_id,
        type="frozen",
        inputs_hash=i_hash,
        engaged_hash="",
        graph_hash=gm.graph_hash,
        graph_version=gm.graph_version,
        algo_hash=a_hash,
        is_current=True,
        payload=frozen_payload,
    )
    # overwrite semantics: ensure no duplicate “current”
    _mark_old_current_false(db, row)
    _insert_current(db, frozen_payload, row)
    return frozen_payload


def get_or_build_tactical(
    db: Session,
    *,
    company_id: str,
    product_id: str,
    product_graph,
    initial_nodes: Iterable[str],
    engaged_nodes: Iterable[str],
    algo_version: str,
    algo_params: Dict[str, Any],
    attributes: Optional[Dict[str, Any]] = None,
    zmot: Optional[Any] = None,
) -> dict:
    gm = get_current_graphmeta(db, company_id, product_id)
    if not gm:
        raise RuntimeError("No current GraphMeta found. Build/record the graph first.")

    i_hash = inputs_hash_from_initial_nodes(initial_nodes, attributes, zmot)
    e_hash = engaged_hash_from_nodes(engaged_nodes)
    a_hash = algo_hash_from_params(algo_version, algo_params)

    hit = _lookup_current(
        db,
        company_id=company_id,
        product_id=product_id,
        type="tactical",
        inputs_hash=i_hash,
        engaged_hash=e_hash,
        graph_hash=gm.graph_hash,
        algo_hash=a_hash,
    )
    if hit:
        return hit.payload

    # Need baseline (frozen) for the diff
    frozen = get_or_build_frozen(
        db,
        company_id=company_id,
        product_id=product_id,
        product_graph=product_graph,
        initial_nodes=initial_nodes,
        algo_version=algo_version,
        algo_params=algo_params,
        attributes=attributes,
        zmot=zmot,
    )

    # Compute full plan with engagements, then diff vs frozen
    full_plan = generate_rcs_with_engagements(
        product_graph, list(set(initial_nodes)), list(set(engaged_nodes)), algo_params
    )
    tactical_payload = diff_tactical_vs_frozen(full_plan, frozen)

    row = Strategy(
        company_id=company_id,
        product_id=product_id,
        type="tactical",
        inputs_hash=i_hash,
        engaged_hash=e_hash,
        graph_hash=gm.graph_hash,
        graph_version=gm.graph_version,
        algo_hash=a_hash,
        is_current=True,
        payload=tactical_payload,
    )
    _mark_old_current_false(db, row)
    _insert_current(db, tactical_payload, row)
    return tactical_payload
