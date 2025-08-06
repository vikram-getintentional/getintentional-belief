


"""
we now have a beautiful loop in our graph like: 
source - node type - target
product node - is_icp - archetype node
archetype node - responds_to - zmot trigger event

And then we have
job_id - triggered_by - zmot  node
zmot node - observed_in - observable moment node
zmot node - associated_with - keyword node

And we already have
product node - offered_by - capability node
capability node - solves - pain node
pain node - adresses - job node
job node - performed_by - pain node

job node - solves - pain node

pain node - scales_with - pain_trigger node
pain node - expressed_as - perceived_metric node


"""

from backend.utils.graph_base.network_graph import get_node_by_id, get_node_id


def get_top_archetypes(product_subgraph, threshold = 0.2):
    """
	Get top archetypes based on relevance threshold.
	"""
	product_id = get_node_id(product_subgraph, "product",{})
    product_node = get_node_by_id(product_subgraph, product_id)
    if not product_node:
        raise ValueError("Product node not found.")
	
	archetypes = []
	archetype_ids = get_nodes_list_ids(sub_graph, "archetype", {})
	for archetype_id in archetype_ids:
		relevance = get_cumulative_relevance_data(product_id, archetype_id)
		if relevance > threshold:
			archetype_node = get_node_by_id(sub_graph, archetype_id)
			industry = archetype_node.get("industry", "").strip().lower()
			revenue = archetype_node.get("revenue_range", "").strip().lower()
			employees = archetype_node.get("employee_range", "").strip().lower()
			funding_stage = archetype_node.get("funding_stage", "").strip().lower()
			geography = archetype_node.get("geography", "").strip().lower()
			archetypes.append({
				"industry": industry,
				"revenue": revenue,
				"employees": employees,
				"funding_stage": funding_stage,
				"geography": geography,
				"relevance": relevance
			})
	print("Archetypes found:", archetypes)
	return archetypes