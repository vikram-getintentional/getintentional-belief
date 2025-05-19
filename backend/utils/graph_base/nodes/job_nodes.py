import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

JOB_PATH = "backend/utils/graph_base/graph_data/job_nodes.json"

def get_or_create_job_node(description, return_created=False):
    data = load_json(JOB_PATH)

    for node in data:
        if node["description"] == description:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "description": description
    }
    data.append(new_node)
    save_json(JOB_PATH, data)
    return (new_node, True) if return_created else new_node
