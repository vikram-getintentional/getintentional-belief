import math
import networkx as nx
import os
from collections import defaultdict, deque
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
            pains = get_target_nodes_by_source_and_type(G, node_id, "solves")
            centrality = sum(get_edge_weight(G, node_id, pain_id) or 0 for pain_id in pains)
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

def relevance(G, start_node, root_node, visited=None):
    """
    Calculate relevance of a node based on its connections to the root node,
    using edge weights.
    """
    if visited is None:
        visited = set()
    if start_node in visited:
        return 0.0  # Already visited, avoid cycle
    visited.add(start_node)

    if start_node not in G.nodes:
        return 0.0
    if start_node == root_node:
        return 1.0

    total_relevance = 0.0
    predecessors = list(G.predecessors(start_node))
    for predecessor in predecessors:
        # Get the edge weight (default to 1.0 if not set)
        weight = G.get_edge_data(predecessor, start_node).get("weight", 1.0)
        total_relevance += weight * relevance(G, predecessor, root_node, visited)

    return total_relevance / len(predecessors) if predecessors else 0.0


def reverse_belief_weight(G, start_node, visited=None):
    """
    For a given start_node, returns the likelihood of reaching product_node,
    using edge weights as transition probabilities.
    """
    product_node = get_node_by_id(G, "product", {})
    if visited is None:
        visited = set()
    if start_node == product_node:
        return 1.0
    if start_node in visited:
        return 0.0  # Avoid cycles
    visited.add(start_node)

    neighbors = list(G.neighbors(start_node))
    if not neighbors:
        return 0.0

    total_weight = sum(G.get_edge_data(start_node, neighbor).get("weight", 1.0) for neighbor in neighbors)
    likelihood = 0.0
    for neighbor in neighbors:
        edge_weight = G.get_edge_data(start_node, neighbor).get("weight", 1.0)
        transition_prob = edge_weight / total_weight if total_weight > 0 else 0
        likelihood += transition_prob * reverse_belief_weight(G, neighbor, visited.copy())

    return likelihood

def max_flow(G):
    """
    Calculate the maximum flow in the graph using the Edmonds-Karp algorithm.
    Calculates the maximum flow for every node in the graph assuming the source is the  product node.

    """
    product_id = get_node_id(G, "product", {})
    flow_results = {}
    print("Starting max flow")
    for node_id, _ in G.nodes(data=True):
        print("Processing node:", node_id)
        if node_id == product_id:
            continue
        flow_value, flow_dict = nx.maximum_flow(G, product_id, node_id, capacity='weight')
        print("Flow value for node: ", flow_value)
        total_inbound_weight = sum(G.get_edge_weight(G, n, node_id) for n in G.predecessors(node_id))
        if total_inbound_weight > 0:
            normalized_flow_value = flow_value / total_inbound_weight
        flow_results[node_id] = {"flow_value": flow_value, "normalized_flow_value": normalized_flow_value}

    return flow_results

def v1_calculate_soft_or_relevance(G, max_path_length=15):
    """
    Calculates soft-or relevance from observed_node to product_node.
    """
    net_relevance = []
    product_node = get_node_id(G, "product", {})
    for observed_node, _ in G.nodes(data=True):
        if observed_node == product_node:
            continue
        paths = nx.all_simple_paths(G, product_node, observed_node, cutoff=max_path_length)
        path_scores = []
        for path in paths:
            score = 1.0
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i+1])
                score *= edge_data.get("weight", 1.0)
            path_scores.append(score)
        if not path_scores:
            node_relevance = 0.0
        node_relevance = 1 - math.prod(1 - p for p in path_scores)
        net_relevance.append({
            "node_id": observed_node,
            "relevance": node_relevance
        })
    return net_relevance


def calculate_soft_or_relevance(G, damping=0.85, max_iter=50, tol=1e-5):
    """
    This is an iterative logic with damping. Just using same fn names because Im lazy
    """
    # Initialize all node relevance
    product_node = get_node_id(G, "product", {})
    relevance = {node: 0.0 for node in G.nodes()}
    relevance[product_node] = 1.0  # Product node is the source of relevance

    for iteration in range(max_iter):
        delta = 0
        new_relevance = {}

        for node in G.nodes():
            if node == product_node:
                new_relevance[node] = 1.0
                continue

            incoming = G.in_edges(node, data='weight')
            belief_inputs = [relevance[u] * w for u, _, w in incoming]
            updated_value = 1 - product(1 - b for b in belief_inputs)

            # Apply damping
            damped_value = damping * updated_value + (1 - damping) * relevance[node]

            # Track max change for convergence
            delta = max(delta, abs(damped_value - relevance[node]))
            new_relevance[node] = damped_value


        relevance = new_relevance

        if delta < tol:
            print(f"Converged in {iteration+1} iterations.")
            break

    # Convert to desired output format
    net_relevance = [
        {"node_id": node, "relevance": rel}
        for node, rel in relevance.items()
        if node != product_node  # optionally exclude product node
    ]
    return net_relevance

from functools import reduce
def product(lst):
    return reduce(lambda x, y: x * y, lst, 1) if lst else 1


