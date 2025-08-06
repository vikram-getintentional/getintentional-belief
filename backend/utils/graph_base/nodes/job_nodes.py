import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

JOB_PATH = "backend/utils/graph_base/graph_data/job_nodes.json"

def get_or_create_job_node(description=None, node_id=None, pain_source=None, return_created=False):
    data = load_json(JOB_PATH)

     # If node_id is provided, update the existing node
    if node_id:
        for node in data:
            if node["id"] == node_id:
                node["description"] = description or node["description"]
                node["pain_source"] = pain_source or node["pain_source"]
                save_json(JOB_PATH, data)
                return (node, False) if return_created else node

        raise ValueError(f"Job with node_id {node_id} not found.")

    for node in data:
        if node["description"] == description:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "description": description,
        "pain_source": pain_source or "unknown"
    }
    data.append(new_node)
    save_json(JOB_PATH, data)
    return (new_node, True) if return_created else new_node
