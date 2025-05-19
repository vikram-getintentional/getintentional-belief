import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

CAPABILITY_PATH = "backend/utils/graph_base/graph_data/capability_nodes.json"

def get_or_create_capability_node(name, description=None, return_created=False):
    """
    Creates or retrieves a capability node.

    Args:
        name (str): The name of the capability.
        description (str): A description of the capability (optional).
        return_created (bool): Whether to return a tuple indicating if the node was newly created.

    Returns:
        dict or tuple: The capability node, and optionally a boolean indicating if it was newly created.
    """
    data = load_json(CAPABILITY_PATH)
    
    # Check if the capability node already exists
    for node in data:
        if node["name"] == name:
            return (node, False) if return_created else node

    # Create a new capability node
    new_node = {
        "id": str(uuid.uuid4()),
        "name": name,
        "description": description or ""
    }
    data.append(new_node)
    save_json(CAPABILITY_PATH, data)
    

    return (new_node, True) if return_created else new_node