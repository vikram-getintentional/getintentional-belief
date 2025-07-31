import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

PAIN_TRIGGER_PATH = "backend/utils/graph_base/graph_data/pain_trigger_nodes.json"

def get_or_create_pain_trigger_node(attribute, dimension, direction, return_created=False):
    data = load_json(PAIN_TRIGGER_PATH)

    # Normalize for matching
    attribute = attribute.strip().lower()
    dimension = dimension.strip().lower()
    direction = direction.strip().lower()

    for node in data:
        if (
            node.get("attribute", "").strip().lower() == attribute and
            node.get("dimension", "").strip().lower() == dimension and
            node.get("direction", "").strip().lower() == direction
        ):
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "attribute": attribute,
        "dimension": dimension,
        "direction": direction
    }
    data.append(new_node)
    save_json(PAIN_TRIGGER_PATH, data)
    return (new_node, True) if return_created else new_node
