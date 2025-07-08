from typing import Dict, List, Set

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
        """
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

            for edge in self.graph_edges:
                # If this node is source or target, add the edge and the other node
                if edge.get("source") == node_id or edge.get("target") == node_id:
                    # Add edge if not already added
                    if edge not in subgraph_edges:
                        subgraph_edges.append(edge)
                        subgraph_weights[(edge.get("source"), edge.get("target"))] = self.get_edge_weight(edge.get("source"), edge.get("target"))
                    # Add the other node to to_visit if not visited
                    other_id = edge.get("target") if edge.get("source") == node_id else edge.get("source")
                    if other_id not in visited and other_id not in to_visit:
                        to_visit.append(other_id)

        return Graph(
            node_registry=subgraph_nodes,
            graph_edges=subgraph_edges,
            edge_weights=subgraph_weights
        )
    
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
    
    def get_all_paths_between_nodes(self, start_node_id, end_node_id, path=None, visited=None, edge_types=None):
        """
        Finds all paths from start_node_id to end_node_id in a directed graph,
        traversing both incoming and outgoing edges, but never revisiting a node in the same path.
        Each path is a list of (source_id, target_id, weight) tuples.
        edge_types: set of edge types to follow (if None, follow all).
        """
        if path is None:
            path = []
        if visited is None:
            visited = set()
        if edge_types is None:
            edge_types = set(edge["type"] for edge in self.graph_edges)

        if start_node_id == end_node_id:
            return [path.copy()]

        visited.add(start_node_id)
        all_paths = []

        for edge in self.graph_edges:
            # Outgoing edges
            if edge["source"] == start_node_id and edge["type"] in edge_types:
                target = edge["target"]
                if target not in visited:
                    new_path = path + [(start_node_id, target, self.get_edge_weight(start_node_id, target) or 1.0)]
                    all_paths.extend(
                        self.get_all_paths_between_nodes(target, end_node_id, new_path, visited.copy(), edge_types)
                    )
            # Incoming edges
            elif edge["target"] == start_node_id and edge["type"] in edge_types:
                source = edge["source"]
                if source not in visited:
                    new_path = path + [(source, start_node_id, self.get_edge_weight(source, start_node_id) or 1.0)]
                    all_paths.extend(
                        self.get_all_paths_between_nodes(source, end_node_id, new_path, visited.copy(), edge_types)
                    )
        return all_paths

    def compute_cumulative_relevance_from_node(self, start_id, sink_id):
        """
        For a given node_id, computes cumulative relevance by summing product of impacts along all downstream paths leading to the base node
        Starting from the node_id, runs a DFS to find all paths to base_node_ids along with edge weights.
        Cumulative relevance is calculated as the sum of (product of edge weights along each path).

        """
        
        """all_paths = self.get_all_paths_between_nodes(start_id, sink_id)
        cumulative_relevance = 0.0
        for path in all_paths:
           path_impact = 1.0
           path_length = len(path)
           for (_, _, weight) in path:
               path_impact *= weight
           cumulative_relevance += path_impact
           normalized_cumulative_relevance = cumulative_relevance / len(path) if path else 0.0
           """
        all_paths =[]
        cumulative_relevance = 1.0
        normalized_cumulative_relevance = 1.0
        return cumulative_relevance, normalized_cumulative_relevance, all_paths

    def percolate_cumulative_relevance(self):
        """
        Starting from product node - percolates to each connected node and updates cumulative relevance for each node.
        This method assumes that the product node is the root of the graph and all other nodes are connected to it.
        Logic is:
        From self - find product node
        For product node id - find all connected nodes and edge weight. 
        For each connected node - Set cum_relevance of each node as edge weight.
        For each connected node - traverse up to find next level node
        Set cum relevance as previous level cum_relevance * current edge weight.
        Recurse to find cumulative relevance until no more new connections exist.
        """
        

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
            print("Starting update centrality for capability:", cap_id, "name:", cap_node["name"])
            connected_pains = self.get_target_nodes_by_source_and_type(cap_id, "solves")
            print(f"Capability {cap_id} is connected to {len(connected_pains)} pains.")
            cap_centrality = 0.0
            for pain_id in connected_pains:
                print("Pulling centrality for pain:", pain_id)
                edge_weight = self.get_edge_weight(cap_id, pain_id)
                print(f"Edge from {cap_id} to {pain_id} has weight {edge_weight}")
                cap_centrality += edge_weight
                print(f"Current centrality for capability: {cap_centrality}")
            print(f"Calculated total centrality for capability {cap_id}: {cap_centrality}")
            # Normalize the centrality score
            normalized_centrality = cap_centrality/len(connected_pains) if connected_pains else 0.0
            print(f"Normalized centrality for capability {cap_id}: {normalized_centrality}")
            # Update the node's centrality
            cap_node["centrality"] = normalized_centrality


# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.add_upstream_job("Inaccurate sales forecasts", {"job": "Oversee sales strategy"})
    print(g.get_upstream_jobs("Inaccurate sales forecasts"))
