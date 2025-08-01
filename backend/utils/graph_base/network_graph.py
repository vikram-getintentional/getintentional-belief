import networkx as nx
import os

from backend.utils.dev_environment.graph_loader import load_product_graph_from_folder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DATA_PATH = os.path.join(BASE_DIR,"backend", "utils", "graph_base", "graph_data")


def build_product_graph(product_id):
    """
    Build a NetworkX DiGraph for the given product_id using your internal graph data.
    """
    product_graph = nx.DiGraph()
    # Load raw data from graph_loader
    node_registry, graph_edges, edge_weights = load_product_graph_from_folder(product_id)

    # Add nodes
    for node_id, node_data in node_registry.items():
        product_graph.add_node(node_id, **node_data)

    # Add edges
    for edge in graph_edges:
        source = edge["source"]
        target = edge["target"]
        attrs = {k: v for k, v in edge.items() if k not in ["source", "target", "weight"]}
        weight = edge_weights.get((source, target), attrs.get("weight", 1.0))
        product_graph.add_edge(source, target, weight=weight, **attrs)

    return product_graph

def get_node_by_id(G, node_id: str) -> dict:
    """
    Retrieves a node from the node registry by its ID.

    Args:
        node_id (str): The unique ID of the node.

    Returns:
        dict: The node details, or None if the node does not exist.
    """
    return G.nodes[node_id] if node_id in G.nodes else None

def get_node_id(G, node_type, properties):
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == node_type and all(data.get(k) == v for k, v in properties.items()):
            return node_id
    return None

def get_nodes_list(G, node_type: str, properties: dict) -> list:
    """
    Returns a list of (node_id, node_data) tuples for all nodes of a given type matching the provided properties.
    """
    return [
        (node_id, node_data)
        for node_id, node_data in G.nodes(data=True)
        if node_data.get("node_type") == node_type and all(node_data.get(k) == v for k, v in properties.items())
    ]

def get_nodes_list_ids(G, node_type: str, properties: dict) -> list:
    """
    Returns a list of (node_ids) for all nodes of a given type matching the provided properties.
    """
    return [
        node_id
        for node_id, node_data in G.nodes(data=True)
        if node_data.get("node_type") == node_type and all(node_data.get(k) == v for k, v in properties.items())
    ]

def get_edge_weight(G, source_id, target_id) -> float:
    return G.edge_weights.get((source_id, target_id), 0.0)

def get_source_nodes_by_target_and_type(G, target_id: str, edge_type: str) -> list:
    """
    Retrieves all source node IDs from the graph where the target matches the given ID and the edge type matches.

    Args:
        target_id (str): The target node ID to filter edges.
        edge_type (str): The type of the edge to filter.

    Returns:
        list: A list of source node IDs matching the criteria.
    """
    return [
        u for u, v, d in G.in_edges(target_id, data=True)
        if d.get("type") == edge_type
    ]
def get_target_nodes_by_source_and_type(G, source_id, edge_type):
    return [
        v for u, v, d in G.out_edges(source_id, data=True)
        if d.get("type") == edge_type
    ]

def get_all_target_nodes(G, node_id):
    return [G.nodes[v] for _, v in G.out_edges(node_id)]

def get_all_source_nodes(G, node_id):
    return [G.nodes[u] for u, _ in G.in_edges(node_id)]

def get_product_id_from_subgraph(G) -> str:
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "product":
            return node_id
    raise ValueError("No product node found in subgraph.")

def update_capability_centralities(G):
    updated_caps_list = []
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "capability":
            pains = G.get_target_nodes_by_source_and_type(G, node_id, "solves")
            centrality = sum(G.get_edge_weight(G, node_id, pain_id) or 0 for pain_id in pains)
            normalized = centrality / len(pains) if pains else 0.0
            G.nodes[node_id]["centrality"] = normalized
            updated_caps_list.append(G.nodes[node_id])
    return updated_caps_list

def update_capabilities_by_nodes_list(G, capability_nodes):
    """
    Updates existing capabilities in the NetworkX graph by node_id.
    """
    print("Updating capabilities by nodes list with input:", capability_nodes)
    updated_nodes = []
    for capability_node in capability_nodes:
        capability_id = capability_node.get("id")
        if not capability_id:
            raise ValueError("Capability ID is required for capability updates.") 
        if capability_id not in G.nodes:
            raise ValueError(f"Capability node {capability_id} not found in graph.")
        # Update attributes
        for attr in ["name", "description", "coreness", "centrality"]:
            value = capability_node.get(attr)
            if value is not None:
                G.nodes[capability_id][attr] = value
        updated_nodes.append(G.nodes[capability_id])
    return updated_nodes

def set_capabilities_relevance(G, capability_threshold=0.5, coreness_threshold=0.4):
    product_id = get_node_id(G, "product", {})
    capability_node_ids = get_target_nodes_by_source_and_type(G, product_id, "offered_by")
    functional_capabilities_ids = []
    blocker_capabilities_ids = []
    for capability_id in capability_node_ids:
        node = G.nodes[capability_id]
        centrality = node.get("centrality", 0)
        coreness = node.get("coreness", 0)
        if centrality < capability_threshold and coreness < coreness_threshold:
            node["importance"] = "blocker"
            blocker_capabilities_ids.append(capability_id)
        elif centrality > capability_threshold and coreness >= coreness_threshold:
            node["importance"] = "critical"
            functional_capabilities_ids.append(capability_id)
    return functional_capabilities_ids, blocker_capabilities_ids

def add_capabilities_to_product(G, product_id, capabilities):
    added_capabilities = []
    for capability in capabilities:
        node_id = capability.get("id")
        G.add_node(node_id, **capability)
        G.add_edge(node_id, product_id, type="offered_by")
        added_capabilities.append(G.nodes[node_id])
    return added_capabilities

def get_cumulative_relevance(G):
    """
    Calculate cumulative relevance for all nodes in the graph.
    """
    relevance_data = {}
    for node_id, data in G.nodes(data=True):
        if "relevance" in data:
            relevance = data["relevance"]
            if node_id not in relevance_data:
                relevance_data[node_id] = 0.0
            relevance_data[node_id] += relevance
    return relevance_data

def calculate_cumulative_relevance(G):
    """
    Calculate cumulative relevance for all nodes in the graph.
    """
    relevance_data = {}
    for node_id, data in G.nodes(data=True):
        if "relevance" in data:
            relevance = data["relevance"]
            if node_id not in relevance_data:
                relevance_data[node_id] = 0.0
            relevance_data[node_id] += relevance
    return relevance_data

def calculate_eigenvector_centrality(G):
    """
    Returns eigenvector centrality for all nodes in the graph.
    """
    return nx.eigenvector_centrality_numpy(G)

def calculate_pagerank_centrality(G, alpha=0.85):
    """
    Returns PageRank centrality for all nodes in the graph.
    """
    return nx.pagerank(G, alpha=alpha)

def calculate_personalized_pagerank_centrality(G, input_node, alpha=0.85):
    """
    Returns personalized PageRank centrality for all nodes in the graph.
    """
    personalization = {node: 0 for node in G.nodes()}
    personalization[input_node] = 1

    return nx.pagerank(G, alpha=alpha, personalization=personalization)
