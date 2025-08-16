import uuid
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

JOB_PATH = "backend/utils/graph_base/graph_data/job_nodes.json"

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def get_or_create_job_node(description=None, *, importance=None, node_id=None, return_created=False):
    """
    Create/update a Job node.
    - description: str (primary label)
    - importance: optional float (how important this job is)
    - node_id: optional, to update an existing node
    """
    data = load_json(JOB_PATH)

    # Update by id
    if node_id:
        for node in data:
            if node["id"] == node_id:
                if description:
                    node["description"] = description
                if isinstance(importance, (int, float)):
                    node["importance"] = float(importance)
                save_json(JOB_PATH, data)
                return (node, False) if return_created else node
        raise ValueError(f"Job with node_id {node_id} not found.")

    desc_norm = _norm(description)
    for node in data:
        if _norm(node.get("description")) == desc_norm:
            # merge importance
            if isinstance(importance, (int, float)):
                node["importance"] = float(importance)
            save_json(JOB_PATH, data)
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "description": description,
    }
    if isinstance(importance, (int, float)):
        new_node["importance"] = float(importance)

    data.append(new_node)
    save_json(JOB_PATH, data)
    return (new_node, True) if return_created else new_node
