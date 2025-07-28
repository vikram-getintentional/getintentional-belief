from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.discovery_engine.openai_helper import extract_summary_and_capabilities
from backend.utils.graph_base.nodes.product_nodes import get_or_create_product_node
from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node
from backend.utils.graph_base.edges.edge_manager import add_edge
from backend.utils.graph_base.graph import Graph

def get_product_value_prop(company_id: str, base_graph, url: str, text: str, plg_cta: bool, footer_features: list):
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
    product_node = None
    capabilities = []

    # Check if Product node exists for company_id
    print("Checking for existing Product node... with company_id:", company_id, "and url:", url)
    product_node_id = base_graph.get_node_id("product", {"company_id": company_id, "url": url})
    print("Product node ID:", product_node_id)
    if product_node_id:
        product_node = base_graph.get_node_by_id(product_node_id)
        print("Product node found:", product_node)
        product_node_id = product_node["id"]
        value_prop = product_node["summary"]
        

        # Find edges where target=product_node_id and edge_type="offered_by"
        capability_nodes_list = base_graph.get_target_nodes_by_source_and_type(product_node_id, "offered_by")
        print("Capabilities list size:", len(capability_nodes_list))
        print("Capability nodes found:", capability_nodes_list)
        for cap_id in capability_nodes_list:
           

            capability_node = base_graph.get_node_by_id(cap_id)
            if capability_node:
                print("Capability node found:", capability_node)
                capabilities.append({
                    "node_id": capability_node.get("id", ""),
                    "name": capability_node.get("name", ""),
                    "description": capability_node.get("description", "")
                })
            else:
                print("Capability node not found:", cap_id)
        

    # If no product node or capabilities found, extract summary and capabilities
    if not product_node or not value_prop or len(capabilities) == 0:
        print(f"No Product node or capabilities found for company_id: {company_id}. Extracting data from GPT...")
        result = []
        result = extract_summary_and_capabilities(text, plg_cta, footer_features)
        
        if "summary" in result:
            summary = result["summary"]
            capabilities = result.get("capabilities", [])

            # Create Product node
            product_node = get_or_create_product_node(
                summary=summary,
                company_id=company_id,
                url=url,
                plg_flag=plg_cta
            )

    # Process capabilities
    if product_node:
        process_capabilities(base_graph, product_node["id"], capabilities)
    
    return {
        "product_node_id": product_node["id"],
        "summary": product_node["summary"],
        "capabilities": capabilities
    }


def process_capabilities(base_graph, product_node_id: str, capabilities: list):
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
        capability_node_id = base_graph.get_node_id("Capability", capability_name)
        if capability_node_id:
            # Update existing Capability node
            capability_node = base_graph.get_node_by_id(capability_node_id)
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

        # Add edge between Capability node and Product node
        add_edge(
            source_id=product_node_id,
            target_id=capability_node["id"],
            edge_type="offered_by",
            weight=1.0
        )
        print(f"Added edge from Capability node {capability_node['id']} to Product node {product_node_id}")

        # Append processed capability to the list
        processed_capabilities.append({
            "node_id": capability_node_id,
            "name": capability_name,
            "description": capability_description
        })

    return processed_capabilities


def get_product_value_prop_capabilities(sub_graph: Graph):
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
    product_id = sub_graph.get_node_id("product", {})
    print("Discovering Product Node with ID:", product_id)
    if not product_id:
        print("No Product node found for the given criteria.")
        return None
    product_node = sub_graph.get_node_by_id(product_id)
    if not product_node:
        print("No Product node found with ID:", product_id)
        return None
    product_summary = product_node.get("summary", "")
    product_plg_flag = product_node.get("plg_flag", False)

    capabilities_list = sub_graph.get_target_nodes_by_source_and_type(product_id, "offered_by")
    print("Discovered Capabilities list size:", len(capabilities_list))
    capabilities = []
    for cap_id in capabilities_list:
        capability_node = sub_graph.get_node_by_id(cap_id)
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
        "plg_flag": product_plg_flag,
        "capabilities": capabilities
    }
        
    return product_data

