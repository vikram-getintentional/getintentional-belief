from backend.utils.dev_environment.graph_loader import load_product_graph_from_folder, load_product_nodes_from_folder
from backend.utils.graph_base.network_graph import build_product_graph, get_node_by_id, get_node_id, get_target_nodes_by_source_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.discovery_engine.openai_helper import extract_summary_and_capabilities
from backend.utils.graph_base.nodes.product_nodes import get_or_create_product_node
from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node
from backend.utils.graph_base.edges.edge_manager import add_edge
import networkx as nx

def generate_product_value_prop(company_id: str, url: str, text: str, plg_cta: bool, footer_features: list):
    """
    This is only hit when creating the original value prop and capabilities for a new product graph in the onboarding cycle. 

    Args:
        company_id (str): The unique ID of the company.
        url (str): The URL associated with the product.
        text (str): The text input for extracting summary and capabilities.
        plg_cta (bool): Whether the product has PLG (Product-Led Growth) enabled.
        footer_features (list): Additional features extracted from the footer.

    Returns:
        dict: A dictionary containing the value proposition summary and capabilities.
    """
    # Initialize variables
    product_node = None
    capability_nodes = []
    extract_summary = False

    # Step 1: Check if product node already exists using get_product_id_from_company_id
    product_id = get_product_id_from_company_id(company_id)
    if product_id:
        product_subgraph = load_product_graph_from_folder(product_id=product_id)
        print("Product node already exists for company_id:", company_id)
        product_id = get_node_id(product_subgraph, "product", {})
        product_node = get_node_by_id(product_subgraph, product_id)
        summary = product_node.get("summary")
        if not summary:
            extract_summary = True
        capabilities = get_target_nodes_by_source_and_type(product_subgraph, product_id, "offered_by")
        if not capabilities:
            extract_summary = True
        for cap_id in capabilities:
            capability_node = get_node_by_id(product_subgraph, cap_id)
            print("Capability node found:", capability_node)
            if capability_node:
                capability_nodes.append({
                    "id": capability_node.get("id"),
                    "name": capability_node.get("name"),
                    "description": capability_node.get("description"),
                    "coreness": capability_node.get("coreness", 0.0),
                    "centrality": capability_node.get("centrality", 0.0),
                    "node_type": capability_node.get("node_type")
                    })
            else:
                print("Capability node not found:", cap_id)
    else:
        print("No product subgraph found for company_id:", company_id)
        extract_summary = True
    # Step 2: If product subgraph does not exist, scrape website and summarize
    if product_id is None or extract_summary:
        print("Starting site extraction for ", company_id,"..")
        result = []
        result = extract_summary_and_capabilities(text, plg_cta, footer_features)
        print("Result from OpenAI:", result)
        
        if "summary" in result:
            summary = result["summary"]
            domain = ""
            industry = ""
            if "domain" in result:
                domain = result["domain"]
            if "industry" in result:
                industry = result["industry"]
                
            capabilities = result.get("capabilities", [])

            # Create Product node
            product_node = get_or_create_product_node(
                summary=summary,
                domain=domain,
                industry=industry,
                company_id=company_id,
                url=url,
                plg_flag=plg_cta
            )
        product_subgraph = build_product_graph(product_id)
        if not product_subgraph:
            print("Something is wrong at the end of get_product_value_prop. No product subgraph found.")
        # Process capabilities
        if product_node:
            capability_nodes = process_capabilities(product_subgraph, product_node["id"], capabilities)
    print("Last line before returning")
    return {
        "product_node_id": product_node.get("id"),
        "summary": summary,
        "capabilities": capability_nodes
    }

def get_product_id_from_company_id(company_id: str):
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
    # Initialize variables
    all_product_nodes = load_product_nodes_from_folder()
    if all_product_nodes is not None:
        print("Product nodes found in the graph data.")
        print("All product nodes:", all_product_nodes)
        for product_node in all_product_nodes.values():
            if product_node.get("company_id") == company_id:
                print("Product node found for company_id:", company_id)
                selected_product_id = product_node.get("id")
                return selected_product_id
    return None
    


def process_capabilities(base_graph: nx.DiGraph, product_node_id: str, capabilities: list):
    """
    Processes a list of capabilities by checking if they exist, updating them, or creating new nodes.
    Adds edges between capability nodes and the product node.

    Args:
        base_graph (Graph): The graph object containing nodes and edges.
        product_node_id (str): The ID of the product node.
        capabilities (list): A list of capabilities with {name, description}.

    Returns:
        None
    """
    print("Processing capabilities for product node ID:", product_node_id)
    processed_capabilities = []
    for capability in capabilities:
        capability_name = capability.get("name", "").strip()
        capability_description = capability.get("description", "").strip()
        capability_coreness = capability.get("coreness", 0.0)

        # Check if Capability node exists using get_node_id()
        capability_node_id = get_node_id(base_graph, "Capability", capability_name)
        if capability_node_id:
            # Update existing Capability node
            capability_node = get_node_by_id(base_graph, capability_node_id)
            capability_node["description"] = capability_description
            print(f"Updated existing Capability node: {capability_node}")

        else:
            # Create new Capability node
            capability_node = get_or_create_capability_node(
                name=capability_name,
                description=capability_description,
                capability_coreness=capability_coreness
            )
            print(f"Created new Capability node: {capability_node}")
            capability_node_id = capability_node["id"]

        # Add edge between Capability node and Product node
        add_edge(
            product_id=product_node_id,
            source_id=product_node_id,
            target_id=capability_node_id,
            edge_type="offered_by",
            weight=1.0
        )
        print(f"Added edge from Capability node {capability_node_id} to Product node {product_node_id}")

        # Append processed capability to the list
        processed_capabilities.append({
            "node_id": capability_node_id,
            "name": capability_name,
            "description": capability_description,
            "coreness": capability_coreness,
            "node_type": "capability"
        })

    return processed_capabilities


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
    product_id = get_node_id(sub_graph, "product", {})
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

    capabilities_list = get_target_nodes_by_source_and_type(sub_graph, product_id, "offered_by")
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

