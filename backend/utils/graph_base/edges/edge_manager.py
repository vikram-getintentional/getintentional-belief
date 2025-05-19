import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

EDGE_PATH = "backend/utils/graph_base/graph_data/graph_edges.json"

def add_edge(source_id, target_id, edge_type, weight = 1, return_created=False):
    data = load_json(EDGE_PATH)

    for edge in data:
        if edge["source"] == source_id and edge["target"] == target_id and edge["type"] == edge_type:
            return (edge, False) if return_created else edge

    new_edge = {
        "id": str(uuid.uuid4()),
        "source": source_id,
        "target": target_id,
        "type": edge_type,
        "weight":  weight
    }
    data.append(new_edge)
    save_json(EDGE_PATH, data)
    
    return (new_edge, True) if return_created else new_edge


def calculate_edge_weight(edge_type, source=None, similarity=None):
    """
    Calculates the weight for an edge based on its type, source, and similarity.

    Args:
        edge_type (str): The type of the edge (e.g., "performed_by", "addresses", "solves").
        source (str): The source of the edge (e.g., "openai").
        similarity (float): The similarity score for the edge (used for "solves").

    Returns:
        float: The calculated weight for the edge.
    """
    if edge_type == "performed_by":  # Persona-Job
        return 1.0  # Default weight for persona-job edges
    elif edge_type == "addresses":  # Job-Pain
        return 1.0  # Default weight for job-pain edges
    elif edge_type == "solves":  # Pain-Capability
        if similarity is not None:
            if source == "openai":
                return max(similarity, 0.75)  # OpenAI responses get a minimum weight of 0.75
            else:
                return similarity  # Use the similarity score directly for non-OpenAI sources
        else:
            raise ValueError("Similarity must be provided for 'solves' edges.")
    else:
        raise ValueError(f"Unknown edge type: {edge_type}")