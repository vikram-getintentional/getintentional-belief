from typing import Dict, List, Any
from backend.utils.graph_base.graph import Graph
from backend.utils.graph_base.graph_utils.aggregate_persona_cards import aggregate_persona_cards


# Function def get_company_products
# This function retrieves all products associated with a given company ID.
# Input is the base graph object and the company ID.
# The code looks through the graph and returns a list of product dictionaries that have company_id matching the input.


def get_company_products(base_graph: Graph, company_id: str) -> List[Dict[str, Any]]:
    # This assumes get_nodes_list returns a list of (node_id, node_data) tuples
    return [
        {"id": node_id, "url": node_data.get("url")}
        for node_id, node_data in base_graph.get_nodes_list("product", {"company_id": company_id})
    ]

#Function def get_product_personas
# This function retrieves all personas associated with a given product ID.
# Input is the base graph object and the product ID.
# 1. The code looks through the graph starting from the product ID to identify mapped capability nodes (edge_type is "offered_by"). 
# 2. Then it traverses to each capability to identify pains solved by capability (edge type is "adresses").
# 3. Then it traverses each pain to identify jobs where this pain (edge type is "solves")
# 4. Then it traverses each job to identify personas that are associated with the job (edge type is "performed_by").
# 5. Each persona - job - pain - capability is added to a dict. 
# 6. Then it traverses each job to identify upstream pains caused by this job (edge type is "impacts").
# 7. For each pain it has to recurse through steps 3-6 to get all hop 0, upstream, and downstream personas.
# The code looks through the graph and returns a list of persona dictionaries that have product_id matching the input.

def get_product_personas(base_graph: Graph, product_id: str) -> List[Dict[str, Any]]:
    print("Recalculating capability centralities")
    base_graph.update_capability_centralities()
    print("Starting product persona discovery for product ID:", product_id)
    personas = {}
    base_pain_ids = set()

    def traverse(job_id, pain_id, capability_id, visited):
        
        if (job_id, pain_id) in visited:
            return
        visited.add((job_id, pain_id))

        job_personas = base_graph.get_target_nodes_by_source_and_type(job_id, "performed_by")
        
        for persona_id in job_personas:
            if persona_id not in personas:
                # --- Look up node data ---
                persona_node = base_graph.get_node_by_id(persona_id) 
                job_node = base_graph.get_node_by_id(job_id)
                pain_node = base_graph.get_node_by_id(pain_id)
                capability_node = base_graph.get_node_by_id(capability_id)

                # Extract persona fields
                persona_title = persona_node.get("title")
                persona_seniority = persona_node.get("seniority")
                persona_department = persona_node.get("department")

                job_desc = job_node.get("description", job_node.get("text"))
                pain_desc = pain_node.get("description", pain_node.get("text"))

                capability_name = capability_node.get("name", capability_node.get("text"))
                # Optionally, add a relevance score (default to 1.0)
                relevance = pain_node.get("weight", 1.0)

                # Compute cumulative_criticality for this persona's pain
                cumulative_criticality, _ = base_graph.compute_cumulative_criticality_for_pain_id(
                    pain_id, base_pain_ids
                )

                personas[persona_id] = {
                    "persona": {
                        "title": persona_title,
                        "department": persona_department,
                        "seniority": persona_seniority
                    },
                    "job": job_desc,
                    "pain": pain_desc,
                    "capability": capability_name,
                    "relevance": cumulative_criticality,
                    "persona_id": persona_id,
                    "job_id": job_id,
                    "pain_id": pain_id,
                    "capability_id": capability_id,
                    "product_id": product_id
                }
                print("Discovered persona data:", persona_title, "with criticality: ", cumulative_criticality, "\n")

        # Upstream and downstream traversal (unchanged)
        upstream_pains = base_graph.get_source_nodes_by_target_and_type(job_id, "impacts")
        for up_pain in upstream_pains:
            upstream_jobs = base_graph.get_source_nodes_by_target_and_type(up_pain, "solves")
            for up_job in upstream_jobs:
                traverse(up_job, up_pain, capability_id, visited)
        downstream_pains = base_graph.get_target_nodes_by_source_and_type(job_id, "impacts")
        for down_pain in downstream_pains:
            downstream_jobs = base_graph.get_target_nodes_by_source_and_type(down_pain, "solves")
            for down_job in downstream_jobs:
                traverse(down_job, down_pain, capability_id, visited)

    print("Checking Capabilities for node...")
    capabilities = base_graph.get_target_nodes_by_source_and_type(product_id, "offered_by")
    if not capabilities:
        print(f"No capabilities found for product ID {product_id}.")
    for capability in capabilities:
        pains = base_graph.get_target_nodes_by_source_and_type(capability, "solves")
        base_pain_ids.update(pains)
        for pain in pains:
            jobs = base_graph.get_target_nodes_by_source_and_type(pain, "addresses")
            for job in jobs:
                print("Traversing persona discovery for job, pain capability")
                traverse(job, pain, capability, set())

    aggregated_personas = aggregate_persona_cards(list(personas.values()))
    print("Aggregated personas:", aggregated_personas)
    return aggregated_personas

# Function to get persona relevance for all personas in a graph.
# Input is a graph object. 
# Logic is to traverse each persona and compute cumulative relevance downwards. 
# Returns a list of aggregated_personas using the aggregate_persona_cards function.
def get_persona_relevance(sub_graph: Graph) -> list[dict]:
    print("Recalculating capability centralities")
    sub_graph.update_capability_centralities()
    print("Starting persona relevance computation")
    personas = []
    product_id = sub_graph.get_node_id("product",{})

    for persona_id, _ in sub_graph.get_nodes_list("persona",{}):
        _, normalized_relevance, _ = sub_graph.compute_cumulative_relevance_from_node(
            persona_id, product_id
        )
        print("Cumulative relevance for persona ID:", persona_id, "is", normalized_relevance)
        persona_node = sub_graph.get_node_by_id(persona_id)
        jobs = []
        # For each job performed by this persona
        persona_jobs = sub_graph.get_source_nodes_by_target_and_type(
            persona_id, "performed_by"
        )
        for job_id in persona_jobs:
            job_node = sub_graph.get_node_by_id(job_id)
            if not job_node:
                print(f"Job node not found for ID: {job_id}")
                continue
            # For each pain solved by this job
            pain_ids = sub_graph.get_target_nodes_by_source_and_type(
                job_id, "solves"
            )
            pains = []
            for pain_id in pain_ids:
                pain_node = sub_graph.get_node_by_id(pain_id)
                if pain_node:
                    pains.append(pain_node.get("description") or pain_node.get("text") or "")
                else:
                    print(f"Pain node not found for ID: {pain_id}")
            jobs.append({
                "description": job_node.get("description") or job_node.get("text") or "",
                "pains": pains
            })

        persona = {
            "persona": {
                "title": persona_node.get("title"),
                "department": persona_node.get("department"),
                "seniority": persona_node.get("seniority"),
            },
            "relevance": normalized_relevance,
            "jobs": jobs
        }
        personas.append(persona)
        

    
    return personas


def rebuild_graph_with_relevance(sub_graph: Graph, capability_threshold = 0.4, relevance_threshold = 0.5) -> List[Dict[str, Any]]:
    """
    Goal is - to get a list of terminal nodes where cumulative relevance is above threshold so we can consider a GPT call for them. 
    Finally this function should return an empty list indicating that there are no terminal nodes that have high relevance but haven't been processed yet.
    Structuring relevance and graph building logic for all nodes.
    - We want to start with the product id, and get capabilities list.
    - set_capabilities_relevance will set the relevance for all capabilities in a product subgraph based on centrality and coreness as functional or blockers.
    - Then we will traverse the subgraph from the given parent node and discovering source and target nodes that have useful relevance.
    - We re-feed the output of this traversal to get the next order relevant nodes. 
    

    """
    sub_graph.update_capability_centralities()
    product_id = sub_graph.get_node_id("product",{})
    print("Starting product persona discovery for product ID:", product_id)
    capability_node_ids = sub_graph.get_target_nodes_by_source_and_type(product_id, "offered_by")
    if not capability_node_ids:
        print(f"No capabilities found for product ID {product_id}.")
        return []
    functional_cap_ids, blocker_cap_ids = set_capabilities_relevance(sub_graph, capability_threshold)
    traversed_nodes = set()
    traversed_nodes.add(product_id)
    relevant_next_hop_nodes = []

    if not functional_cap_ids:
        print(f"No functional capabilities found for product ID {product_id}.")
        return []
    for cap_id in functional_cap_ids:
        if cap_id in traversed_nodes:
            continue
        traversed_nodes.add(cap_id)
    next_nodes = get_relevant_neighbor_nodes(sub_graph, functional_cap_ids, relevance_threshold)


    
    
    # Pass the flat list of persona-job-pain-capability dicts to aggregate_persona_cards
    aggregated_personas = aggregate_persona_cards(base_graph, persona_entries)
    print("Aggregated personas - Product Personas:")
    for data in aggregated_personas:
        print(data,"\n")
    return aggregated_personas

def set_capabilities_relevance(sub_graph: Graph, capability_threshold = 0.5, coreness_threshold = 0.4) -> None:
    """
    Sets the relevance for all capabilities in a product subgraph based on centrality and coreness as functional or blockers.
    """
    # Logic to determine importance of capabilities based on centrality and coreness and add to a reduced set of "functional capabilities"
    product_id = sub_graph.get_node_id("product",{})
    print("Starting product persona discovery for product ID:", product_id)
    capability_node_ids = sub_graph.get_target_nodes_by_source_and_type(product_id, "offered_by")
    if not capability_node_ids:
        print(f"No capabilities found for product ID {product_id}.")
        return []
    functional_capabilities_ids = []
    blocker_capabilities_ids = []
    for capability_id in capability_node_ids:
        capability_node = sub_graph.get_node_by_id(capability_id)
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


def get_relevant_neighbor_nodes(sub_graph, parent_node_ids, relevance_threshold= 0.5):
        """
        Traverses the subgraph from the given parent node and discovering source and target nodes (neighbors). 
        Calculates relevance for each neighbor, and runs traversal only if relevance is above threshold.
        Returns a list of terminal nodes where cumulative relevance is above threshold. 
        If all terminal nodes are below threshold, it returns an empty list.
        """
        visited = set()
        product_id = sub_graph.get_node_id("product",{})
        neighbor_nodes = []
        for parent_node_id in parent_node_ids:
            source_node_ids = sub_graph.get_all_source_nodes(parent_node_id)
            target_node_ids = sub_graph.get_all_target_nodes(parent_node_id)
            for source_node_id in source_node_ids:
                if source_node_id in visited:
                    continue
                visited.add(source_node_id)
                node_relevance = sub_graph.get_cumulative_relevance(product_id, source_node_id)
                if node_relevance < relevance_threshold:
                    continue
                neighbor_nodes.append(source_node_id)
            for target_node_id in target_node_ids:
                if target_node_id in visited:
                    continue
                visited.add(target_node_id)
                node_relevance = sub_graph.get_cumulative_relevance(product_id, target_node_id)
                if node_relevance < relevance_threshold:
                    continue
                neighbor_nodes.append(target_node_id)
        return neighbor_nodes
