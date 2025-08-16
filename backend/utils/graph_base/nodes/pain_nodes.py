import uuid
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

PAIN_PATH = "backend/utils/graph_base/graph_data/pain_nodes.json"
METRIC_PATH = "backend/utils/graph_base/graph_data/perceived_metrics.json"

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def get_or_create_pain_node(text, *, pain_source=None, relevance=None, node_id=None, return_created=False):
    """
    Create/update a Pain node.
    - text: str (primary label)
    - pain_source: "internal" | "external" | None
    - relevance: optional float
    - node_id: optional, to update an existing node
    """
    data = load_json(PAIN_PATH)

    # Update by id
    if node_id:
        for node in data:
            if node["id"] == node_id:
                if text:
                    node["text"] = text
                if pain_source in {"internal", "external"}:
                    node["pain_source"] = pain_source
                if isinstance(relevance, (int, float)):
                    node["relevance"] = float(relevance)
                save_json(PAIN_PATH, data)
                return (node, False) if return_created else node
        raise ValueError(f"Pain with node_id {node_id} not found.")

    # Lookup by normalized text
    text_norm = _norm(text)
    for node in data:
        if _norm(node.get("text", "")) == text_norm:
            # merge any new attributes
            if pain_source in {"internal", "external"}:
                node["pain_source"] = pain_source
            if isinstance(relevance, (int, float)):
                node["relevance"] = float(relevance)
            save_json(PAIN_PATH, data)
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "text": text,
        "pain_source": pain_source or "unknown"
    }
    if isinstance(relevance, (int, float)):
        new_node["relevance"] = float(relevance)

    data.append(new_node)
    save_json(PAIN_PATH, data)
    return (new_node, True) if return_created else new_node


def get_or_create_metric_node(metric, return_created=False):
    data = load_json(METRIC_PATH)
    metric_norm = (metric or "").strip().lower()

    for node in data:
        if (node.get("text") or "").strip().lower() == metric_norm:
            return (node, False) if return_created else node

    new_node = {
        "id": str(uuid.uuid4()),
        "text": metric_norm
    }
    data.append(new_node)
    save_json(METRIC_PATH, data)
    return (new_node, True) if return_created else new_node
