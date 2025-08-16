import uuid
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

COMPANY_PATH = "backend/utils/graph_base/graph_data/company_nodes.json"
ARCHETYPE_PATH = "backend/utils/graph_base/graph_data/archetype_nodes.json"

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def get_or_create_archetype_node(
    industry=None, revenue_range=None, employee_range=None, funding_stage=None, geography=None, *,
    node_id=None, return_created=False
):
    """
    True get-or-create on the 5-tuple. All fields normalized.
    Optionally update by node_id.
    """
    data = load_json(ARCHETYPE_PATH)

    # Update by id
    if node_id:
        for node in data:
            if node["id"] == node_id:
                # update fields if provided (keep original if None)
                if industry is not None:       node["industry"] = _norm(industry)
                if revenue_range is not None:  node["revenue_range"] = _norm(revenue_range)
                if employee_range is not None: node["employee_range"] = _norm(employee_range)
                if funding_stage is not None:  node["funding_stage"] = _norm(funding_stage)
                if geography is not None:      node["geography"] = _norm(geography)
                save_json(ARCHETYPE_PATH, data)
                return (node, False) if return_created else node
        raise ValueError(f"Archetype with node_id {node_id} not found.")

    i = _norm(industry)
    r = _norm(revenue_range)
    e = _norm(employee_range)
    f = _norm(funding_stage)
    g = _norm(geography)

    # Deduplicate on the tuple
    for node in data:
        if (_norm(node.get("industry")) == i and
            _norm(node.get("revenue_range")) == r and
            _norm(node.get("employee_range")) == e and
            _norm(node.get("funding_stage")) == f and
            _norm(node.get("geography")) == g):
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "industry": i,
        "revenue_range": r,
        "employee_range": e,
        "funding_stage": f,
        "geography": g
    }
    data.append(new_node)
    save_json(ARCHETYPE_PATH, data)
    return (new_node, True) if return_created else new_node
