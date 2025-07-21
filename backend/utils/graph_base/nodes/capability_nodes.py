import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json
from backend.utils.graph_base.edges.edge_manager import add_edge

CAPABILITY_PATH = "backend/utils/graph_base/graph_data/capability_nodes.json"

def get_or_create_capability_node(name, description=None, capability_coreness=0.0, capability_centrality=0.0, node_id=None, return_created=False):
    """
    Creates, retrieves or updates a capability node.

    Args:
        name (str): The name of the capability.
        description (str): A description of the capability (optional).
        node_id (str): The ID of the capability node to update (optional).
        return_created (bool): Whether to return a tuple indicating if the node was newly created.

    Returns:
        dict or tuple: The capability node, and optionally a boolean indicating if it was newly created.
    """
    data = load_json(CAPABILITY_PATH)

    # If node_id is provided, update the existing node
    if node_id:
        for node in data:
            if node["id"] == node_id:
                node["name"] = name
                node["description"] = description or node["description"]
                node["coreness"] = capability_coreness or node["coreness"]
                node["centrality"] = capability_centrality or node["centrality"]
                save_json(CAPABILITY_PATH, data)
                return (node, False) if return_created else node

        raise ValueError(f"Capability with node_id {node_id} not found.")
    
    # Check if the capability node already exists
    for node in data:
        if node["name"] == name:
            return (node, False) if return_created else node

    # Create a new capability node
    new_node = {
        "id": str(uuid.uuid4()),
        "name": name,
        "description": description or "",
        "coreness": capability_coreness or 0.0,
        "centrality": capability_centrality or 0.0
    }
    data.append(new_node)
    save_json(CAPABILITY_PATH, data)
    

    return (new_node, True) if return_created else new_node



