import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

PRODUCT_PATH = "backend/utils/graph_base/graph_data/product_nodes.json"

import uuid
import json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json

PRODUCT_PATH = "backend/utils/graph_base/graph_data/product_nodes.json"

def get_or_create_product_node(summary=None, domain=None, industry=None, company_id=None, url=None, product_id=None, plg_flag=None, return_created=False):
    """
    Retrieves a Product node if it exists, otherwise creates a new one.

    Args:
        summary (str): The value proposition or summary of the Product node.
        company_id (str): The unique ID of the company.
        url (str): The URL associated with the Product node.
        plg_flag (bool): Whether the product has PLG (Product-Led Growth) enabled.
        return_created (bool): Whether to return a tuple (node, created).

    Returns:
        dict or tuple: The Product node (existing or newly created). If return_created=True, returns (node, created).
    """
    data = load_json(PRODUCT_PATH)

    # Check if the node already exists
    for node in data:
        if node["company_id"] == company_id and node["id"] == product_id:
            # Update the existing node with new values if provided
            if summary:
                node["summary"] = summary
            if domain:
                node["domain"] = domain
            if industry:
                node["industry"] = industry
            if plg_flag is not None:
                node["plg_flag"] = plg_flag
            save_json(PRODUCT_PATH, data)  # Save updated data
            return (node, False) if return_created else node

    # Create a new Product node
    new_node = {
        "id": str(uuid.uuid4()),
        "summary": summary,
        "domain": domain,
        "industry": industry,
        "company_id": company_id,
        "url": url,
        "plg_flag": plg_flag
    }
    data.append(new_node)
    save_json(PRODUCT_PATH, data)
    return (new_node, True) if return_created else new_node
