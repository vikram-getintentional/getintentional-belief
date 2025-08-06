import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

PAIN_PATH = "backend/utils/graph_base/graph_data/pain_nodes.json"
METRIC_PATH = "backend/utils/graph_base/graph_data/perceived_metrics.json"

def get_or_create_pain_node(text, return_created=False):
    data = load_json(PAIN_PATH)

    for node in data:
        if node["text"] == text:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "text": text
    }
    data.append(new_node)
    save_json(PAIN_PATH, data)
    return (new_node, True) if return_created else new_node


def get_or_create_metric_node(metric, return_created=False):
    data = load_json(METRIC_PATH)

    # Normalize for matching
    metric = metric.strip().lower()

    for node in data:
        if node["text"] == metric:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "text": metric
    }
    data.append(new_node)
    save_json(METRIC_PATH, data)
    return (new_node, True) if return_created else new_node
