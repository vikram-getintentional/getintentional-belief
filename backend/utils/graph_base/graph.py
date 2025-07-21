from collections import defaultdict, deque
from typing import Dict, List, Set
from math import exp

from backend.utils.graph_base.edges.edge_manager import add_edge
from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data

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
    
    def get_all_target_nodes(self, node) -> Set[str]:
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
                    # Add the other node to to_visit if not visited
                    other_id = edge.get("target") if edge.get("source") == node_id else edge.get("source")
                    if other_id not in visited and other_id not in to_visit:
                        to_visit.append(other_id)
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
    
    
    

    def calculate_cumulative_relevance(self, epsilon = 1e-6) -> dict:
        """
        Going back to a simpler calculation of cumulative relevance. 
        We calculate the relevance of each node as 
        """
        product_id = self.get_node_id("product", {})
        if not product_id:
            print("No product ID found in subgraph.")
            return {}
        cumulative_relevance = defaultdict(float)
        cumulative_relevance[product_id] = 1.0
        max_iter = 10
        print("Starting cumulative relevance calculation for product ID:", product_id)
        # Build reverse graph (downstream map)
        node_targets = {}
        for node in self.node_registry.values():
            node_id = node["id"]
            node_targets_list = self.get_all_target_nodes(node)
            node_targets[node_id] = [n["id"] for n in node_targets_list]

        # Initialize relevance values
        relevance = {}
        relevance = {
            node_id: get_cumulative_relevance_data(product_id, node_id)
            for node_id in self.node_registry.keys()
        }


        # This line seeds the prod relevance to 1.0 for when relevance data doesnt exist
        relevance[product_id] = 1.0

        frontier = deque(
            node_id for node_id, rel in relevance.items()
            if rel > 0.0
        )
        print("Initial frontier:", list(frontier))
        visited = set()
        for _ in range(max_iter):
            if not frontier:
                break

            next_frontier = set()

            while frontier:
                node_id = frontier.popleft()
                # Checks for bug testing - kill this
                node_data = self.get_node_by_id(node_id)
                node_type = node_data.get("node_type", "unknown")
                if node_type == "persona":
                    node_text = node_data.get("title")
                elif node_type == "pain":
                    node_text = node_data.get("text")
                elif node_type == "job":
                    node_text = node_data.get("description", node_data.get("text"))
                elif node_type == "pain_trigger":
                    node_text = node_data.get("attribute", node_data.get("text"))
                else:
                    node_text = "Unknown Node Type"
                
                
                old_value = relevance.get(node_id, 0.0)
                this_node = self.get_node_by_id(node_id)
                sources = self.get_all_source_nodes(this_node)
                # Check for empty sources - mainly for product node
                if not sources:
                    print(f"No sources found for node {node_id}. Treating as root node.")
                    # This is a root node (e.g., product node)
                    # Propagate its relevance to its targets
                    current_node = self.get_node_by_id(node_id)
                    target_nodes = self.get_all_target_nodes(current_node)
                    for target in target_nodes:
                        target_id = target.get("id")
                        # Calculate new relevance for the target
                        w = self.get_edge_weight(node_id, target_id) or 0.0
                        new_value = relevance[node_id] * w
                        old_value = relevance.get(target_id, 0.0)
                        if abs(new_value - old_value) > epsilon:
                            relevance[target_id] = new_value
                            next_frontier.add(target_id)
                    continue
                irrelevance = 1.0
                for source in sources:
                    source_id = source.get("id")
                    w = self.get_edge_weight(source_id, node_id) or 0.0
                    # Check for if edge_weight is None
                    if w is None:
                        w = 0.0
                    source_rel = relevance.get(source_id, 0.0)
                    irrelevance += (1 - source_rel * w)
                    
                
                new_value = 1 - exp(-1*irrelevance)
                
                
                if abs(new_value - old_value) > epsilon or node_id not in visited:
                    relevance[node_id] = new_value
                    current_node = self.get_node_by_id(node_id)
                    target_nodes = self.get_all_target_nodes(current_node)
                    # Check for empty target nodes
                    if not target_nodes:
                        print(f"No target nodes found for node {node_id}. Skipping.")
                        continue
                    for target in target_nodes:
                        target_id = target.get("id")
                        next_frontier.add(target_id)
                visited.add(node_id)

            frontier = deque(next_frontier)
        # Ensure relevance is a dict of {node_id: cumulative relevance}
        for node_id, rel in relevance.items():
            if rel > 0.0:
                cumulative_relevance[node_id] = rel
            else:
                cumulative_relevance[node_id] = 0.0
        return dict(cumulative_relevance)


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
        updated_caps_list = []
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
            updated_caps_list.append(cap_node)
        self.update_capabilities_by_nodes_list(updated_caps_list)

    def update_capabilities_by_nodes_list(self, capability_nodes):
        """
        Updates existing capabilities by node_id.
        """
        updated_nodes = []
        for capability_node in capability_nodes:
            capability_id = capability_node.get("id")
            if not capability_id:
                raise ValueError("Capability ID is required for capability updates.")
            capability = self.get_node_by_id(capability_id)
            name = capability.get("name", "")
            description = capability.get("description", "")
            coreness = capability.get("coreness", 0.0)
            centrality = capability.get("centrality", 0.0)
            
            updated_node = get_or_create_capability_node(
                name=name,
                description=description,
                node_id=capability_id,
                capability_coreness=coreness,
                capability_centrality=centrality,
                return_created=True
            )
            updated_nodes.append(updated_node)
        return updated_nodes

    def set_capabilities_relevance(self, capability_threshold = 0.5, coreness_threshold = 0.4) -> None:
        """
        Sets the relevance for all capabilities in a product subgraph based on centrality and coreness as functional or blockers.
        """
        # Logic to determine importance of capabilities based on centrality and coreness and add to a reduced set of "functional capabilities"
        product_id = self.get_node_id("product", {})
        print("Starting product persona discovery for product ID:", product_id)
        capability_node_ids = self.get_target_nodes_by_source_and_type(product_id, "offered_by")
        if not capability_node_ids:
            print(f"No capabilities found for product ID {product_id}.")
            return []
        functional_capabilities_ids = []
        blocker_capabilities_ids = []
        for capability_id in capability_node_ids:
            capability_node = self.get_node_by_id(capability_id)
            capability_centrality = capability_node.get("centrality", 0)
            coreness = capability_node.get("coreness", 0)
            if capability_centrality < capability_threshold and coreness < coreness_threshold:
                capability_node["importance"]= "blocker"
                blocker_capabilities_ids.append(capability_id)
                continue
            elif capability_centrality > capability_threshold and coreness >= coreness_threshold:
                capability_node["importance"] = "critical"
                functional_capabilities_ids.append(capability_id)
            else:
                print("Detected anomaly in coreness vs cap centrality - please verify \n")
                print(capability_node.get("name"), "has coreness", coreness, "and centrality", capability_centrality)

        return functional_capabilities_ids, blocker_capabilities_ids

    def add_capabilities_to_product(self, capabilities):
        """
        Adds new capabilities to a product node and creates edges.
        """
        product_id = self.get_product_id_from_subgraph()
        added_capabilities = []
        for capability in capabilities:
            capability_name = capability.get("name", "").strip()
            capability_description = capability.get("description", "").strip()
            capability_node = get_or_create_capability_node(
                name=capability_name,
                description=capability_description,
                coreness=0.9
            )
            capability_node_id = capability_node["id"]
            add_edge(
                source_id=capability_node_id,
                target_id=product_id,
                edge_type="offered_by"
            )
            added_capabilities.append(capability_node)
        return added_capabilities

# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.add_upstream_job("Inaccurate sales forecasts", {"job": "Oversee sales strategy"})
    print(g.get_upstream_jobs("Inaccurate sales forecasts"))
