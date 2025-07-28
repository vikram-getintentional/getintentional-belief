from typing import Dict, List, Any
import json
from collections import defaultdict
from backend.utils.graph_base.graph import Graph
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data, get_cumulative_relevance_data


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
    aggregated_personas = list(personas.values())
    print("Aggregated personas:", aggregated_personas)
    return aggregated_personas



def get_persona_relevance(sub_graph: Graph) -> list[dict]:
    print("Starting relevance computation - at this point centrality & cum relevance should be set")
    personas = []
    product_id = sub_graph.get_node_id("product",{})


    relevance_nodes = sub_graph.calculate_cumulative_relevance()
    add_or_update_cumulative_relevance_data(product_id, relevance_nodes)
    print("Cumulative relevance data updated for product ID:", product_id)

    for persona_id, _ in sub_graph.get_nodes_list("persona",{}):
        normalized_relevance = get_cumulative_relevance_data(product_id, persona_id)
        persona_node = sub_graph.get_node_by_id(persona_id)
        jobs = []
        pains = []
        # For each job performed by this persona
        persona_jobs = sub_graph.get_source_nodes_by_target_and_type(
            persona_id, "performed_by"
        )
        for job_id in persona_jobs:

            job_node = sub_graph.get_node_by_id(job_id)
            if not job_node:
                print(f"Job node not found for ID: {job_id}")
                continue
            job_relevance = get_cumulative_relevance_data(product_id, job_id)
            jobs.append({
                "description": job_node.get("description") or job_node.get("text") or "",
                "relevance": job_relevance
            })
            # For each pain solved by this job
            pain_ids = sub_graph.get_source_nodes_by_target_and_type(
                job_id, "addresses"
            )
            for pain_id in pain_ids:
                pain_node = sub_graph.get_node_by_id(pain_id)
                if not pain_node:
                    print(f"Pain node not found for ID: {pain_id}")
                    continue
                pain_relevance = get_cumulative_relevance_data(product_id, pain_id)
                pains.append({
                    "description": pain_node.get("description") or pain_node.get("text") or "",
                    "relevance": pain_relevance
                })

        persona = {
            "persona_id": persona_id,
            "persona": {
                "title": persona_node.get("title"),
                "department": persona_node.get("department"),
                "seniority": persona_node.get("seniority"),
            },
            "relevance": normalized_relevance,
            "jobs": jobs,
            "pains": pains
        }
        personas.append(persona)
    
    print("Persona relevance computed, total personas found:", personas)
    aggregated_personas = aggregated_personas_map(sub_graph, personas, threshold=0.2)

    return aggregated_personas


def aggregated_personas_map(sub_graph:Graph, match_results: list[dict], threshold: float = 0.0) -> list[dict]:
    final = []
    if not match_results:
        print("No match results found, returning empty list.")
        return final
    persona_map = defaultdict(lambda: {
        "persona_title": "",
        "persona_departments": set(),
        "persona_seniority": set(),
        "persona_ids": set(),
        "max_relevance": 0.0,
        "jobs": set(),
        "pains": set()
    })

    for entry in match_results:
        persona = entry.get("persona", {})
        relevance = entry.get("relevance", 0.0)
        if relevance < threshold:
            continue

        title = persona.get("title", "")
        department = persona.get("department", "")
        seniority = persona.get("seniority", "")
        persona_id = entry.get("persona_id", "")

        persona_map[title]["persona_title"] = title
        persona_map[title]["persona_departments"].add(department)
        persona_map[title]["persona_seniority"].add(seniority)
        persona_map[title]["persona_ids"].add(persona_id)
        persona_map[title]["max_relevance"] = max(persona_map[title]["max_relevance"], relevance)

        # Aggregate jobs and pains
        jobs = entry.get("jobs", [])
        pains = entry.get("pains", [])
        for job in jobs:
            persona_map[title]["jobs"].add(json.dumps(job))
        for pain in pains:
            persona_map[title]["pains"].add(json.dumps(pain))

    # Convert sets to lists for output
    for card in persona_map.values():
        card["persona_departments"] = list(card["persona_departments"])
        card["persona_seniority"] = list(card["persona_seniority"])
        card["persona_ids"] = list(card["persona_ids"])
        #Sort each job and pain in card by relevance
        card["jobs"] = sorted(list(card["jobs"]), key=lambda x: json.loads(x)["relevance"], reverse=True)
        card["pains"] = sorted(list(card["pains"]), key=lambda x: json.loads(x)["relevance"], reverse=True)
        final.append(card)
    # Sort the final list by max_relevance in descending order
    final.sort(key=lambda x: x["max_relevance"], reverse=True)
    print("Aggregate - final output:", final)
    return final


