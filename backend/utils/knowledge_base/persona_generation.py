from typing import Dict, List, Any
from backend.utils.graph_base.graph import Graph
from backend.utils.graph_base.graph_utils.aggregate_persona_cards import aggregate_persona_cards
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data, add_or_update_cumulative_relevance_data


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



def get_product_personas(base_graph: Graph, product_id: str) -> List[Dict[str, Any]]:
    """
    This is a pretty cool traversal function that starts from the product ID, gets capabilities, and bubbles all the way up. 
    Only thing is - its kind of redundant since we're doing this bubble up thingy quite a bit in other places (cum_Rel calculations, product_graph creation, etc.)
    Also get_persona_relevance kind of does the same thing but better... 
    Leaving this here as a breadcrumb for the future but for now, it's just a sad little orphan.
    """
    print("Recalculating capability centralities")
    base_graph.update_capability_centralities()
    print("Starting product persona discovery for product ID:", product_id)
    persona_entries = []
    base_pain_ids = set()

    def traverse(job_id, pain_id, capability_id, visited):
        if (job_id, pain_id) in visited:
            return
        visited.add((job_id, pain_id))

        job_personas = base_graph.get_target_nodes_by_source_and_type(job_id, "performed_by")
        for persona_id in job_personas:
            # Compute cumulative_relevance for this persona's pain
            cumulative_relevance = get_cumulative_relevance_data(product_id, persona_id)
            persona_entry = {
                "persona_id": persona_id,
                "relevance": cumulative_relevance,
                "job_id": job_id,
                "pain_id": pain_id,
                "capability_id": capability_id,
                "product_id": product_id
            }
            persona_entries.append(persona_entry)
            

        # Upstream and downstream traversal
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

    capabilities = base_graph.get_target_nodes_by_source_and_type(product_id, "offered_by")
    if not capabilities:
        print(f"No capabilities found for product ID {product_id}.")
    for capability in capabilities:
        pains = base_graph.get_target_nodes_by_source_and_type(capability, "solves")
        base_pain_ids.update(pains)
        for pain in pains:
            jobs = base_graph.get_target_nodes_by_source_and_type(pain, "addresses")
            for job in jobs:
                traverse(job, pain, capability, set())

    # Pass the flat list of persona-job-pain-capability dicts to aggregate_persona_cards
    aggregated_personas = aggregate_persona_cards(base_graph, persona_entries)
    print("Aggregated personas - Product Personas:")
    for data in aggregated_personas:
        print(data,"\n")
    return aggregated_personas


def get_persona_relevance(sub_graph: Graph) -> list[dict]:
    print("Recalculating capability centralities")
    sub_graph.update_capability_centralities()
    print("Starting persona relevance computation")
    personas = []
    product_id = sub_graph.get_node_id("product",{})

    for persona_id, _ in sub_graph.get_nodes_list("persona",{}):
        # Get cumulative_relevance value for this persona from cumulative_relevance.json
        cumulative_relevance = get_cumulative_relevance_data(product_id, persona_id)
        
        print("Cumulative relevance for persona ID:", persona_id, "is", cumulative_relevance)
        # For each job performed by this persona
        persona_jobs = sub_graph.get_source_nodes_by_target_and_type(
            persona_id, "performed_by"
        )
        for job_id in persona_jobs:
            
            # For each pain solved by this job
            pain_ids = sub_graph.get_source_nodes_by_target_and_type(
                job_id, "addresses"
            )
            
            for pain_id in pain_ids:
                
                persona = {
                    "persona_id": persona_id,
                    "relevance": cumulative_relevance,
                    "job_id": job_id,
                    "pain_id": pain_id,
                    "product_id": product_id,
                }
                print("Persona data:", persona)
                personas.append(persona)
                
    aggregated_personas = aggregate_persona_cards(sub_graph, personas)
    print("Aggregated personas - Persona Relevance:")
    for data in aggregated_personas:
        print(data, "\n")
    
    return aggregated_personas
