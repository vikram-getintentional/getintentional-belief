import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

PERSONA_PATH = "backend/utils/graph_base/graph_data/persona_nodes.json"

def get_or_create_persona_node(title, seniority, department, return_created=False):
    data = load_json(PERSONA_PATH)

    for node in data:
        if node["title"] == title and node["seniority"] == seniority and node["department"] == department:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "title": title,
        "seniority": seniority,
        "department": department
    }
    data.append(new_node)
    save_json(PERSONA_PATH, data)
    return (new_node, True) if return_created else new_node
