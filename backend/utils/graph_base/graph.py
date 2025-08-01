from collections import defaultdict, deque
from typing import Dict, List, Set
from math import exp

from backend.utils.graph_base.edges.edge_manager import add_edge
from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data

class Graph:
    def __init__(
        self,
        node_registry=None,
        graph_edges=None,
        edge_weights=None
    ):
        self.pain_to_upstream_jobs: Dict[str, List[Dict]] = {}
        self.node_registry: Dict[str, dict] = node_registry or {}
        self.edge_weights: Dict[tuple, float] = edge_weights or {}
        self.graph_edges: List[Dict] = graph_edges or []

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
            for node_id, node_data in self.node_registry.items()
            if node_data.get("node_type") == node_type and all(node_data.get(k) == v for k, v in properties.items())
        ]

    def get_nodes_list_ids(G, node_type: str, properties: dict) -> list:
        """
        Returns a list of (node_ids) for all nodes of a given type matching the provided properties.
        """
        return [
            node_id
            for node_id, node_data in self.node_registry.items()
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


    """
    NOW DEPRECATED
    def extract_product_subgraph(self, product_id: str) -> "Graph":
        
        visited = set()
        to_visit = [product_id]
        subgraph_nodes = {}
        subgraph_edges = []
        subgraph_weights = {}

        while to_visit:
            node_id = to_visit.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self.get_node_by_id(node_id)
            if node:
                subgraph_nodes[node_id] = node

            SHARED_NODE_TYPES = {"icp_industry", "icp_revenue", "icp_employees", "icp_funding_stage","icp_geography", "zmot_keywords", "zmot_observablemoments"}

            for edge in self.graph_edges:
                if edge.get("source") == node_id or edge.get("target") == node_id:
                    # Add edge if not already added
                    if edge not in subgraph_edges:
                        subgraph_edges.append(edge)
                        subgraph_weights[(edge.get("source"), edge.get("target"))] = self.get_edge_weight(edge.get("source"), edge.get("target"))
                    # Add the other node to to_visit if not visited
                    other_id = edge.get("target") if edge.get("source") == node_id else edge.get("source")
                    if other_id in visited or other_id in to_visit:
                        continue
                    other_node = self.get_node_by_id(other_id)
                    # If the current node is a shared node type, do NOT traverse out from it
                    this_node = self.get_node_by_id(node_id)
                    if this_node and this_node.get("node_type") in SHARED_NODE_TYPES:
                        continue
                    if other_node:
                        to_visit.append(other_id)
        return Graph(
            node_registry=subgraph_nodes,
            graph_edges=subgraph_edges,
            edge_weights=subgraph_weights
        )

    """
    
    
    """
    NOW DEPRECATED
    
    def calculate_cumulative_relevance(self):
        product_id = self.get_node_id("product", {})
        print("Starting cum relevance calc for product ID:", product_id)
        if not product_id:
            print("No product ID found in subgraph.")
            return {}
        cumulative_relevance = {}
        cumulative_relevance[product_id] = 1.0

        stack = []
        # Add all direct targets of the product node to the stack
        for edge in self.graph_edges:
            if edge["source"] == product_id:
                stack.append(edge["target"])
        print("Filled stack - Current stack length:", len(stack))

        while stack:
            node_id = stack.pop()
            if node_id in cumulative_relevance:
                continue  # Already calculated

            # Get all source nodes for this node
            sources = [edge["source"] for edge in self.graph_edges if edge["target"] == node_id]
            # Only proceed if all sources have their cumulative relevance set
            if all(source in cumulative_relevance for source in sources):
                print(f"Calculating cumulative relevance for node {node_id} with sources {sources}")
                total = 0.0
                for source in sources:
                    weight = self.get_edge_weight(source, node_id) or 1.0
                    total += cumulative_relevance[source] * weight
                # Normalize by in-degree (number of sources)
                if sources:
                    cumulative_relevance[node_id] = total / len(sources)
                else:
                    cumulative_relevance[node_id] = total
                # Push all target nodes of this node to the stack for further processing
                for edge in self.graph_edges:
                    if edge["source"] == node_id and edge["target"] not in cumulative_relevance:
                        stack.append(edge["target"])
            else:
                # Not all sources are ready, push this node back and push missing sources
                print(f"Source nodes for node {node_id} not yet calculated. Adding back to stack")
                stack.append(node_id)
                for source in sources:
                    if source not in cumulative_relevance:
                        stack.append(source)

        return cumulative_relevance
    """

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

# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.add_upstream_job("Inaccurate sales forecasts", {"job": "Oversee sales strategy"})
    print(g.get_upstream_jobs("Inaccurate sales forecasts"))
