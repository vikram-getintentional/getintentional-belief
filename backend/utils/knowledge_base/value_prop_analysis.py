
import uuid
from backend.utils.graph_base.agent_graph_builder import _product, _capability
from backend.utils.graph_base.agent_graph_builder import _upsert_edge
from backend.utils.graph_base.graph_utils.save_and_load_graph_as_json import save_graph_as_json, load_graph_from_json
import networkx as nx
import os
import json

from backend.utils.graph_base.network_graph import get_node_by_id, get_node_id, get_product_id_from_subgraph, get_target_nodes_by_source_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.discovery_engine.openai_helper import extract_summary_and_capabilities

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOOKUP_PATH = os.path.join(BASE_DIR, "utils", "graph_base", "graph_data", "company_product_lookup.json")


def get_product_id_from_company_id(company_id: str):
    if os.path.exists(LOOKUP_PATH):
        with open(LOOKUP_PATH, "r") as f:
            lookup = json.load(f)
        print("Lookup loaded successfully.")
        return lookup.get(company_id)
    print("Lookup file does not exist. Returning None.")
    return None

def update_company_product_lookup(company_id: str, product_id: str):
    lookup = {}
    if os.path.exists(LOOKUP_PATH):
        with open(LOOKUP_PATH, "r") as f:
            lookup = json.load(f)
    lookup[company_id] = product_id
    with open(LOOKUP_PATH, "w") as f:
        json.dump(lookup, f, indent=2)
        print(f"Updated lookup for company {company_id} with product ID {product_id}.")

def generate_product_value_prop(company_id: str, url: str, text: str, plg_cta: bool, footer_features: list):
    # Step 1: Lookup product_id from company_id
    if company_id is None:
        print("No company_id provided, cannot lookup product_id.")
        return None
    product_lookup_id = get_product_id_from_company_id(company_id)
    if product_lookup_id:
        print("Product ID found in lookup:", product_lookup_id)
        G = load_graph_from_json(product_lookup_id)
        if G:
            product_id = get_product_id_from_subgraph(G)
            if product_id:
                product_node = G.nodes[product_id]
                summary = product_node.get("summary")
                domain = product_node.get("domain")
                industry = product_node.get("industry")
                plg_flag = product_node.get("plg_flag")
                capabilities = [
                    G.nodes[cap_id] for cap_id in G.successors(product_id)
                    if G[product_id][cap_id].get("type") == "offers"
                ]
                if capabilities:
                    capability_nodes = [
                        {
                            "id": cap.get("id"),
                            "name": cap.get("name"),
                            "description": cap.get("description"),
                            "coreness": cap.get("coreness", 0.0),
                            "centrality": cap.get("centrality", 0.0),
                            "node_type": cap.get("node_type", "capability")
                        }
                        for cap in capabilities
                    ]
                    print("Data found in graph!")
                    return {
                        "product_node_id": product_id,
                        "summary": summary,
                        "domain": domain,
                        "industry": industry,
                        "plg_flag": plg_flag,
                        "capabilities": capability_nodes
                    }
    if not product_lookup_id or not G or not capabilities or not product_id:
        # Step 2: If product graph does not exist, extract summary/capabilities and create graph
        print("No existing product graph found, extracting summary and capabilities.")
        result = extract_summary_and_capabilities(text, plg_cta, footer_features)
        summary = result.get("summary", "")
        domain = result.get("domain", "")
        industry = result.get("industry", "")
        capabilities = result.get("capabilities", [])
        print("Extraction complete. Creating graph nodes..")
        # Generate new product_id if no product_id found
        if not product_lookup_id:
            print("No product_id found in lookup.")
            product_lookup_id = str(uuid.uuid4())
            print("Generated new Product lookup id:", product_lookup_id, ". Now creating lookup entry.")
            update_company_product_lookup(company_id, product_lookup_id)

        # Create new graph and upsert product node
        print("Creating new product graph with ID:", product_lookup_id)
        # Create a new directed graph
        G = nx.DiGraph()
        G.graph["product_lookup_id"] = product_lookup_id
        product_node_id = _product(G, product_lookup_id, {
            "company_id": company_id,
            "url": url,
            "summary": summary,
            "domain": domain,
            "industry": industry,
            "plg_flag": plg_cta,
            "node_type": "product"
        })
        print("Upserting capability nodes")
        # Upsert capability nodes and edges
        capability_nodes = []
        for cap in capabilities:
            name = cap.get("name")
            description = cap.get("description")
            if not name or not description:
                print(f"Skipping capability with missing name/description: {cap}")
                continue
            cap_id = _capability(
                G,
                name=name,
                description=description,
                attrs={
                    "coreness": cap.get("coreness", 0.0),
                    "node_type": "capability"
                }
            )
            _upsert_edge(G, product_node_id, "offers", cap_id)
            capability_nodes.append({
                "id": cap_id,
                "name": name,
                "description": description,
                "coreness": cap.get("coreness", 0.0),
                "node_type": "capability"
            })
        print("Capability nodes upserted. Now saving graph.")
        # Save graph
        save_graph_as_json(G, product_lookup_id)
        print("Graph saved successfully. Returning product data.")
        return {
            "product_node_id": product_node_id,
            "summary": summary,
            "domain": domain,
            "industry": industry,
            "plg_flag": plg_cta,
            "capabilities": capability_nodes
        }




def get_product_value_prop_capabilities(sub_graph: nx.DiGraph):
    """
    Retrieves the product value proposition and capabilities for a given company ID.
    If no product node or capabilities exist, it extracts the summary and capabilities.

    Args:
        company_id (str): The unique ID of the company.
        base_graph (Graph): The graph object containing nodes and edges.
        url (str): The URL associated with the product.
        text (str): The text input for extracting summary and capabilities.
        plg_cta (bool): Whether the product has PLG (Product-Led Growth) enabled.
        footer_features (list): Additional features extracted from the footer.

    Returns:
        dict: A dictionary containing the value proposition summary and capabilities.
    """
    # Discover Product Node
    product_id = get_product_id_from_subgraph(sub_graph)
    print("Discovering Product Node with ID:", product_id)
    if not product_id:
        print("No Product node found for the given criteria.")
        return None
    product_node = get_node_by_id(sub_graph, product_id)
    if not product_node:
        print("No Product node found with ID:", product_id)
        return None
    product_summary = product_node.get("summary", "")
    product_domain = product_node.get("domain", "")
    product_industry = product_node.get("industry", "")
    product_plg_flag = product_node.get("plg_flag", False)

    capabilities_list = get_target_nodes_by_source_and_type(sub_graph, product_id, "offers")
    print("Discovered Capabilities list size:", len(capabilities_list))
    capabilities = []
    for cap_id in capabilities_list:
        capability_node = get_node_by_id(sub_graph, cap_id)
        if capability_node:
            relevance = get_cumulative_relevance_data(product_id, cap_id)
            capabilities.append({
                "name": capability_node.get("name", ""),
                "description": capability_node.get("description", ""),
                "coreness": capability_node.get("coreness", 0.0),
                "centrality": capability_node.get("centrality", 0.0),
                "relevance": relevance
            })
        else:
            print("Capability node not found:", cap_id)
    
    product_data = {
        "product_node_id": product_id,
        "summary": product_summary,
        "domain": product_domain,
        "industry": product_industry,
        "plg_flag": product_plg_flag,
        "capabilities": capabilities
    }
        
    return product_data

