from backend.utils.graph_base.network_graph import get_edge_weight, get_node_by_id, get_node_id, get_nodes_list_ids, get_target_nodes_by_source_and_type
import networkx as nx

from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data


def get_node_label(product_subgraph:nx.DiGraph, node_id):
    """
    Get the label of a node.
    """
    node = get_node_by_id(product_subgraph, node_id)
    if node["node_type"] == "job":
        label = node.get("description", "").strip()
    elif node["node_type"] == "pain":
        label = node.get("text", "").strip()
    elif node["node_type"] == "capability":
        label = node.get("name", "").strip()
    elif node["node_type"] == "persona":
        title = node.get("title", "").strip()
        department = node.get("department", "").strip()
        seniority = node.get("seniority", "").strip()
        label = f"{title} | {department} | {seniority}"
    elif node["node_type"] == "zmot_triggerevents":
        label = node.get("trigger_event", "").strip()
    elif node["node_type"] == "zmot_observablemoments":
        label = node.get("observable_moment", "").strip()
    elif node["node_type"] == "zmot_keyword":
        label = node.get("keyword", "").strip()
    elif node["node_type"] == "pain_trigger":
        attribute = node.get("attribute", "").strip()
        dimension = node.get("dimension", "").strip()
        direction = node.get("direction", "").strip()
        label = f"{attribute} | {dimension} | {direction}"
    elif node["node_type"] == "perceived_metric":
        label = node.get("metric", "").strip()
    else:
        label = node.get("id", "").strip()
    return label


def assign_temporal_depths(product_subgraph: nx.DiGraph, allowed_edge_types=None):
    """
    Assigns temporal depth to each node using shortest path from product_node,
    filtering only on allowed edge types.
    """
    print("Assigning temporal depths...")
    product_id = get_node_id(product_subgraph, "product", {})
    product_node = get_node_by_id(product_subgraph, product_id)
    if allowed_edge_types is None:
        # Default set of causal edge types (can be adjusted)
        # This should set a depth for all nodes except archetype nodes
        allowed_edge_types = {
            "triggered_by",
            "addresses",
            "solves",
            "offered_by",
            "performed_by",
            "offered_by"
        }

    # Build a filtered, reversed graph using only causal edges
    reversed_G = nx.DiGraph()

    for u, v, data in product_subgraph.edges(data=True):
        edge_type = data.get("type")
        if edge_type in allowed_edge_types:
            reversed_G.add_edge(v, u)  # reverse the edge for upstream traversal
    print("Reversed graph with allowed edges")
    temporal_depths = {}
    for node in reversed_G.nodes:
        try:
            depth = nx.shortest_path_length(reversed_G, source=node, target=product_id)
            temporal_depths[node] = -depth
        except nx.NetworkXNoPath:
            temporal_depths[node] = None  # unreachable
            print("Unreachable  node:", node)
    print("Depths calculated")
    # Save to original graph
    nx.set_node_attributes(product_subgraph, temporal_depths, 'temporal_depth')
    return temporal_depths


def get_top_archetypes(product_subgraph, threshold = 0.2, top_n = 2):
	"""
	Get top archetypes based on relevance threshold.
	"""
	product_id = get_node_id(product_subgraph, "product", {})
	product_node = get_node_by_id(product_subgraph, product_id)
	if not product_node:
		raise ValueError("Product node not found.")

	archetypes = []
	archetype_ids = get_nodes_list_ids(product_subgraph, "archetype", {})
	for archetype_id in archetype_ids:
		archetype = get_node_by_id(product_subgraph, archetype_id)
		archetype["relevance"] = get_cumulative_relevance_data(product_id, archetype_id)
		archetypes.append(archetype)
	sorted_archetypes = sorted(archetypes, key=lambda x: x["relevance"], reverse=True)
	top_archetypes = [archetype for archetype in sorted_archetypes if archetype["relevance"] >= threshold]
	return top_archetypes[:top_n]

def get_best_match_archetypes(product_subgraph, attribute_combo: dict, threshold=0.2, top_n=5):
    
    product_id = get_node_id(product_subgraph,"product", {})
    archetype_ids = []
    naive_archetype_ids = get_nodes_list_ids(product_subgraph, "archetype", {})
    for naive_archetype_id in naive_archetype_ids:
         related_zmot_ids = get_target_nodes_by_source_and_type(product_subgraph, naive_archetype_id, "responds_to")
         if not related_zmot_ids or len(related_zmot_ids) == 0:
             continue # We want to only consider archetypes that have a zmot downstream
         archetype_ids.append(naive_archetype_id)
         
    matches = []
    
    # Only keep keys that are in priorities
    relevant_keys = {"industry", "revenue_range", "employee_range", "funding_stage", "geography"}
    attribute_combo = {k: v for k, v in attribute_combo.items() if k in relevant_keys}
    # Helper: Related industries (expand as needed)
    related_industries = {
        "automotive": ["transportation", "mobility", "manufacturing"],
        # Add more mappings as needed
    }

    # Helper: Bracket ordering (expand as needed)
    revenue_brackets = ["0-1m", "1-10m", "10-100m", "100-500m", "500m-1b", "1b+"]
    employee_brackets = ["1-50", "51-200", "201-500", "501-1000", "1000-5000", "5000-10000", "10000+"]
    funding_stages = ["seed", "series a", "series b", "series c+", "public"]

    def get_next_bracket(brackets, value, direction="up"):
        try:
            idx = brackets.index(value.lower())
            if direction == "up" and idx < len(brackets) - 1:
                return brackets[idx + 1]
            elif direction == "down" and idx > 0:
                return brackets[idx - 1]
        except ValueError:
            return None
        return None

    # 1. Try exact match
    for archetype_id in archetype_ids:
        archetype = get_node_by_id(product_subgraph, archetype_id)
        match_score = 0
        # Priority order
        priorities = [
            ("industry", 5),
            ("revenue_range", 4),
            ("geography", 3),
            ("employee_range", 2),
            ("funding_stage", 1)
        ]
        exact = True
        for key, weight in priorities:
            val = attribute_combo.get(key)
            arch_val = str(archetype.get(key, "")).lower()
            if val:
                if arch_val == str(val).lower():
                    match_score += weight
                else:
                    exact = False
        if exact:
            archetype["relevance"] = get_cumulative_relevance_data(product_id, archetype_id)
            archetype["match_score"] = match_score
            matches.append(archetype)

    # 2. If no exact match, try softer matching
    if not matches:
        print("No exact matches found, trying softer matching...")
        for archetype_id in archetype_ids:
            archetype = get_node_by_id(product_subgraph, archetype_id)
            match_score = 0
            for key, weight in priorities:
                val = attribute_combo.get(key)
                arch_val = str(archetype.get(key, "")).lower()
                print(f"Matching {key}: input={val}, node={arch_val}")
                if val:
                    # Industry: try related
                    if key == "industry":
                        if arch_val == str(val).lower():
                            match_score += weight
                        elif val.lower() in related_industries and arch_val in related_industries[val.lower()]:
                            match_score += weight - 1
                    # Revenue/Employee/Funding: try next higher, then lower
                    elif key in ["revenue_range", "employee_range", "funding_stage"]:
                        brackets = revenue_brackets if key == "revenue_range" else employee_brackets if key == "employee_range" else funding_stages
                        if arch_val == str(val).lower():
                            match_score += weight
                        else:
                            up = get_next_bracket(brackets, str(val).lower(), "up")
                            down = get_next_bracket(brackets, str(val).lower(), "down")
                            if arch_val == up:
                                match_score += weight - 1
                            elif arch_val == down:
                                match_score += weight - 2
                    # Geography: exact, then skip
                    elif key == "geography":
                        if arch_val == str(val).lower():
                            match_score += weight
            if match_score > 0:
                archetype["relevance"] = get_cumulative_relevance_data(product_id, archetype_id)
                archetype["match_score"] = match_score
                matches.append(archetype)

    # Sort by match_score (priority), then relevance
    sorted_matches = sorted(matches, key=lambda x: (x["match_score"], x["relevance"]), reverse=True)
    return [a for a in sorted_matches if a["relevance"] >= threshold][:top_n]

def get_best_match_zmots_for_archetype(product_subgraph, archetype_id, threshold=0.2, top_n=5):
	product_id = get_node_id(product_subgraph, "product", {})
	related_zmot_ids = get_target_nodes_by_source_and_type(product_subgraph, archetype_id, "responds_to")
	matches = []

	for zmot_id in related_zmot_ids:
		zmot_node = get_node_by_id(product_subgraph, zmot_id)
		zmot_relevance = get_cumulative_relevance_data(product_id, zmot_id)
		if zmot_relevance < threshold:
			continue
		zmot_node["relevance"] = zmot_relevance
		zmot_node["org_relevance"] = get_edge_weight(product_subgraph, archetype_id, zmot_id)
		matches.append(zmot_node)

	sorted_matches = sorted(matches, key=lambda x: x["relevance"], reverse=True)
	return sorted_matches[:top_n]