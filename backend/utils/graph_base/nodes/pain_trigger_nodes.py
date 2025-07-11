import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

PAIN_TRIGGER_PATH = "backend/utils/graph_base/graph_data/pain_trigger_nodes.json"

def get_or_create_pain_trigger_node(pain_trigger: dict, return_created=False):
    """
    pain_trigger: dict with keys 'attribute', 'dimension', 'direction'
    """
    data = load_json(PAIN_TRIGGER_PATH)

    # Normalize for comparison
    attr = pain_trigger.get("attribute", "").strip().lower()
    dim = pain_trigger.get("dimension", "").strip().lower()
    dirn = pain_trigger.get("direction", "").strip().lower()

    for node in data:
        if (
            node.get("attribute", "").strip().lower() == attr and
            node.get("dimension", "").strip().lower() == dim and
            node.get("direction", "").strip().lower() == dirn
        ):
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "attribute": attr,
        "dimension": dim,
        "direction": dirn
    }
    data.append(new_node)
    save_json(PAIN_TRIGGER_PATH, data)
    return (new_node, True) if return_created else new_node
