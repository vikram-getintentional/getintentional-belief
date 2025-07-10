import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json
from datetime import datetime, timezone

CUMULATIVE_RELEVANCE_PATH = "backend/utils/graph_base/graph_data/cumulative_relevance.json"
def add_or_update_cumulative_relevance_data(product_id: str, cumulative_relevance: dict) -> dict:
    """
    Updates the cumulative_data dict in-memory with the given product_id and cumulative_relevance.
    Returns the updated cumulative_data dict.
    """
    # Use load_json to load existing cumulative data if needed
    print("Starting add/update cumulative relevance to json")
    print("Input cumulative_relevance:", cumulative_relevance)
    cumulative_data = load_json(CUMULATIVE_RELEVANCE_PATH)
    print("Loaded cumulative data with:", cumulative_data)
    if product_id not in cumulative_data:
        cumulative_data[product_id] = []
    for node_id, relevance in cumulative_relevance.items():
        existing_entry = next((entry for entry in cumulative_data[product_id] if entry["node_id"] == node_id), None)
        if existing_entry:
            existing_entry["cumulative_relevance"] = relevance
        else:
            cumulative_data[product_id].append({
                "node_id": node_id,
                "cumulative_relevance": relevance
            })
    
    # Save the updated cumulative data back to the file
    save_json(CUMULATIVE_RELEVANCE_PATH, cumulative_data)
    return cumulative_data



