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
    
    cumulative_data = load_json(CUMULATIVE_RELEVANCE_PATH)
    if cumulative_data:
        print("Loaded cumulative data")
    else:
        print("No existing cumulative data found, initializing new data structure.")
        save_json(CUMULATIVE_RELEVANCE_PATH, {})  # Ensure the file exists
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


def get_cumulative_relevance_data(product_id: str, node_id: str) -> dict:
    """
    Retrieves cumulative relevance data for a specific node_id from the cumulative relevance JSON file.
    """
    cumulative_data = load_json(CUMULATIVE_RELEVANCE_PATH)
    if product_id not in cumulative_data:
        return 0.0
    for entry in cumulative_data[product_id]:
        if entry["node_id"] == node_id:
            return entry["cumulative_relevance"]
    return 0.0
