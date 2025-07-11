from collections import defaultdict, deque
from typing import Dict, List, Set

from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data

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

    def get_node_by_id(self, node_id: str) -> dict:
        """
        Retrieves a node from the node registry by its ID.

        Args:
            node_id (str): The unique ID of the node.

        Returns:
            dict: The node details, or None if the node does not exist.
        """
        return self.node_registry.get(node_id)

    def get_node_id(self, node_type: str, properties: dict) -> str:
        for node_id, node_data in self.node_registry.items():
            if node_data.get("node_type") == node_type:
                
                for k, v in properties.items():
                    print(f"  {k}: node has '{node_data.get(k)}', looking for '{v}'")
                    print(f"    Equal? {node_data.get(k) == v}")
                if all(node_data.get(k) == v for k, v in properties.items()):
                    
                    return node_id
        print("No match found.")
        return None
    
    def get_nodes_list(self, node_type: str, properties: dict) -> list:
        """
        Returns a list of (node_id, node_data) tuples for all nodes of a given type matching the provided properties.
        """
        return [
            (node_id, node_data)
            for node_id, node_data in self.node_registry.items()
            if node_data.get("node_type") == node_type and all(node_data.get(k) == v for k, v in properties.items())
        ]
    
    def get_edge_weight(self, source_id: int, target_id: int) -> float:
        return self.edge_weights.get((source_id, target_id))

    def add_upstream_job(self, pain: str, job_info: Dict):
        if pain not in self.pain_to_upstream_jobs:
            self.pain_to_upstream_jobs[pain] = []
        self.pain_to_upstream_jobs[pain].append(job_info)

    def get_upstream_jobs(self, pain: str) -> List[Dict]:
        return self.pain_to_upstream_jobs.get(pain, [])
    
    def get_upstream_jobs_by_id(self, pain_id: int) -> List[Dict]:
        """
        Return all jobs where pain_id is the source and edge_type is 'addresses'.
        Output: List of dicts: {'job_id': ..., 'impact': ...}
        """
        results = []
        for edge in self.graph_edges:
            if (
                edge.get("source_id") == pain_id
                and edge.get("edge_type") == "addresses"
            ):
                results.append({
                    "job_id": edge.get("target_id"),
                    "impact": edge.get("weight", 1.0)
                })
        return results
    
    def get_source_nodes_by_target_and_type(self, target_id: str, edge_type: str) -> list:
        """
        Retrieves all source node IDs from the graph where the target matches the given ID and the edge type matches.

        Args:
            target_id (str): The target node ID to filter edges.
            edge_type (str): The type of the edge to filter.

        Returns:
            list: A list of source node IDs matching the criteria.
        """
        return [
            edge["source"] for edge in self.graph_edges
            if edge["target"] == target_id and edge["type"] == edge_type
        ]
    def get_target_nodes_by_source_and_type(self, source_id: str, edge_type: str) -> list:
        """
        Retrieves all source node IDs from the graph where the target matches the given ID and the edge type matches.

        Args:
            source_id (str): The source node ID to filter edges.
            edge_type (str): The type of the edge to filter.

        Returns:
            list: A list of target node IDs matching the criteria.
        """
        return [
            edge["target"] for edge in self.graph_edges
            if edge["source"] == source_id and edge["type"] == edge_type
        ]
    
    def extract_subgraph(self, start_node_id: str) -> "Graph":
        visited = set()
        to_visit = [start_node_id]
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

            # Traverse outgoing edges
            for edge in self.graph_edges:
                if edge.get("source") == node_id:
                    target_id = edge.get("target")
                    if target_id not in visited:
                        subgraph_edges.append(edge)
                        subgraph_weights[(edge.get("source"), edge.get("target"))] = self.get_edge_weight(edge.get("source"), edge.get("target"))
                        to_visit.append(target_id)
                elif edge.get("target") == node_id:
                    source_id = edge.get("source")
                    if source_id not in visited:
                        subgraph_edges.append(edge)
                        subgraph_weights[(edge.get("source"), edge.get("target"))] = self.get_edge_weight(edge.get("source"), edge.get("target"))
                        to_visit.append(source_id)

        return Graph(
            node_registry=subgraph_nodes,
            graph_edges=subgraph_edges,
            edge_weights=subgraph_weights
        )
    def get_all_source_nodes(self, node) -> Set[str]:
        """
        Returns all source nodes for a given node by traversing incoming edges.
        """
        source_list = []
        node_id = node["id"]
        for edge in self.graph_edges:
            if edge["target"] == node_id:
                source_id = edge["source"]
                source = self.get_node_by_id(source_id)
                source_list.append(source)
        return source_list

    def get_all_target_node_ids(self, node) -> Set[str]:
        """
        Returns all target nodes for a given node by traversing outgoing edges.
        """
        target_list = []
        node_id = node["id"]
        for edge in self.graph_edges:
            if edge["source"] == node_id:
                target_id = edge["target"]
                target = self.get_node_by_id(target_id)
                target_list.append(target)
        return target_list
    def extract_product_subgraph(self, product_id: str) -> "Graph":
        """
        Extracts a subgraph containing all nodes and edges connected to the given product_id,
        by traversing both incoming and outgoing edges (undirected traversal).
        Additionally, it collects the weights of edges and calculates cumulative_relevance.
        It then checks cumulative_relevance.json. If product_id exists, and node_id exists, it updates the cumulative_relevance value. 
        If product_id exists and node_id does not exist, it appends the node_id and cumulative relevance inside product_id.
        If product_id does not exist it creates product_id and node_id & cumulative_relevance in the json.
        cumulative_relevance.json is structured as follows:
        [product_id: {
            {
            "node_id": node_id,
            "cumulative_relevance": node_cumulative_relevance
            },
            {
            "node_id": node_id,
            "cumulative_relevance": node_cumulative_relevance
            }...
        }]
        """
        # Initialize the subgraph
        if not product_id:
            raise ValueError("Product ID cannot be empty.")
        if product_id not in self.node_registry:
            raise ValueError(f"Product ID {product_id} does not exist in the graph.")
        # Initialize visited set and queue for BFS
        visited = set()
        to_visit = [product_id]
        subgraph_nodes = {}
        subgraph_edges = []
        subgraph_weights = {}
    
        # Now we have the subgraph with all nodes and edges connected to the product_id
        while to_visit:
            node_id = to_visit.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self.get_node_by_id(node_id)
            if node:
                subgraph_nodes[node_id] = node

            # Traverse outgoing edges
            for edge in self.graph_edges:
                if edge.get("source") == node_id:
                    target_id = edge.get("target")
                    if target_id not in visited:
                        subgraph_edges.append(edge)
                        subgraph_weights[(edge.get("source"), edge.get("target"))] = self.get_edge_weight(edge.get("source"), edge.get("target"))
                        to_visit.append(target_id)
                elif edge.get("target") == node_id:
                    source_id = edge.get("source")
                    if source_id not in visited:
                        subgraph_edges.append(edge)
                        subgraph_weights[(edge.get("source"), edge.get("target"))] = self.get_edge_weight(edge.get("source"), edge.get("target"))
                        to_visit.append(source_id)
        # Calculate cumulative relevance for all nodes in the subgraph
        cumulative_relevance = self.calculate_cumulative_relevance(product_id)
        if not cumulative_relevance:
            raise ValueError(f"No cumulative relevance calculated for product ID {product_id}.")
        print("Cumulative relevance calculated for graph")
        add_or_update_cumulative_relevance_data(product_id, cumulative_relevance)

        return Graph(
            node_registry=subgraph_nodes,
            graph_edges=subgraph_edges,
            edge_weights=subgraph_weights
        )
        

    def calculate_cumulative_relevance(self, product_id: str) -> dict:
        """
        Calculates and normalizes cumulative relevance for all nodes connected to the given product_id.
        Returns a dict: node_id -> normalized cumulative_relevance (0-1).
        """
        print("Starting cumulative relevance calculation")
        cumulative_relevance = defaultdict(float)
        cumulative_relevance[product_id] = 1.0
        visited = set()
        to_visit = [product_id]

        while to_visit:
            node_id = to_visit.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            

            # Traverse outgoing edges
            for edge in self.graph_edges:
                if edge.get("source") == node_id:
                    target_id = edge.get("target")
                    if target_id not in visited:
                        to_visit.append(target_id)
                    weight = edge.get("weight", 1.0)
                    cumulative_relevance[target_id] += cumulative_relevance[node_id] * weight

            # Traverse incoming edges
            for edge in self.graph_edges:
                if edge.get("target") == node_id:
                    source_id = edge.get("source")
                    if source_id not in visited:
                        to_visit.append(source_id)
                    weight = edge.get("weight", 1.0)
                    cumulative_relevance[source_id] += cumulative_relevance[node_id] * weight
        print("Cumulative relevance calculation complete. Now normalizing")
        # Normalize all scores to 0-1
        if cumulative_relevance:
            max_relevance = max(cumulative_relevance.values())
            if max_relevance > 0:
                cumulative_relevance = {
                    node_id: score / max_relevance
                    for node_id, score in cumulative_relevance.items()
                }
            else:
                cumulative_relevance = dict(cumulative_relevance)
        else:
            cumulative_relevance = {}

        return dict(cumulative_relevance)
    
    def get_all_pain_paths_to_base_by_id(self, pain_id, base_pain_ids, visited=None):
        """
        Returns all upstream paths from a pain node to any base pain node.
        Each path is a list of (pain_id, job_id, impact) tuples.
        """
        if visited is None:
            visited = set()
        if pain_id in visited:
            return []
        visited.add(pain_id)
        if pain_id in base_pain_ids:
            return [[(pain_id, None, 1.0)]]
        all_paths = []
        upstream_jobs = self.get_upstream_jobs_by_id(pain_id)
        for upstream in upstream_jobs:
            dep_job_id = upstream["job_id"]
            dep_pain_id = upstream["pain_id"]
            impact = upstream.get("impact") or self.get_edge_weight(pain_id, dep_job_id) or 0.5
            sub_paths = self.get_all_pain_paths_to_base_by_id(dep_pain_id, base_pain_ids, visited.copy())
            for path in sub_paths:
                all_paths.append([(pain_id, dep_job_id, impact)] + path)
        return all_paths
        

    def get_product_id_from_subgraph(self) -> str:
        # Assumes there is only one product node in the subgraph
        for node in self.node_registry.values():
            if node.get("type") == "product":
                return node["id"]
        raise ValueError("No product node found in subgraph.")
    
    def update_capability_centralities(self):
        """
        Updates each capability node in the graph with its degree centrality score.
        """
        # Assuming self.node_registry or similar holds all nodes
        capability_list = self.get_nodes_list("capability", {})
        for cap in capability_list:
            cap_node = cap[1]
            cap_id = cap_node["id"]
            
            connected_pains = self.get_target_nodes_by_source_and_type(cap_id, "solves")
            
            cap_centrality = 0.0
            for pain_id in connected_pains:
                
                edge_weight = self.get_edge_weight(cap_id, pain_id)
                
                cap_centrality += edge_weight
                
            
            # Normalize the centrality score
            normalized_centrality = cap_centrality/len(connected_pains) if connected_pains else 0.0
            
            # Update the node's centrality
            cap_node["centrality"] = normalized_centrality
            # Persist the update to capability_nodes.json
            get_or_create_capability_node(
                name=cap_node["name"],
                description=cap_node.get("description", ""),
                capability_coreness=cap_node.get("coreness", 0.0),
                capability_centrality=normalized_centrality,
                node_id=cap_id
            )
        
        

# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.add_upstream_job("Inaccurate sales forecasts", {"job": "Oversee sales strategy"})
    print(g.get_upstream_jobs("Inaccurate sales forecasts"))
