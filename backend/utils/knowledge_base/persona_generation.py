from typing import Dict, List, Any
import json
from collections import defaultdict
from backend.utils.graph_base.network_graph import calculate_eigenvector_centrality, calculate_pagerank_centrality, calculate_personalized_pagerank_centrality, calculate_soft_or_relevance, get_cumulative_relevance, get_node_by_id, get_node_id, get_nodes_list, get_source_nodes_by_target_and_type, max_flow, relevance, reverse_belief_weight, update_capability_centralities
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data, get_cumulative_relevance_data

import networkx as nx

# Function def get_company_products
# This function retrieves all products associated with a given company ID.
# Input is the base graph object and the company ID.
# The code looks through the graph and returns a list of product dictionaries that have company_id matching the input.


def get_company_products(base_graph: nx.DiGraph, company_id: str) -> List[Dict[str, Any]]:
    # This assumes get_nodes_list returns a list of (node_id, node_data) tuples
    return [
        {"id": node_id, "url": node_data.get("url")}
        for node_id, node_data in get_nodes_list(base_graph,"product", {"company_id": company_id})
    ]


def get_persona_relevance(sub_graph: nx.DiGraph) -> list[dict]:
    print("Starting relevance computation - at this point centrality & cum relevance should be set")
    personas = []

    product_id = get_node_id(sub_graph, "product",{})


    relevance_nodes = calculate_soft_or_relevance(sub_graph)
    cumulative_relevance = {item["node_id"]: item["relevance"] for item in relevance_nodes}
    for node_id, relevance in cumulative_relevance.items():
        node_data = get_node_by_id(sub_graph, node_id)
        if node_data and node_data.get("node_type") == "persona":
            text = node_data.get("title", "")
        elif node_data and node_data.get("node_type") == "job":
            text = node_data.get("description", "") or node_data.get("text", "")
        elif node_data and node_data.get("node_type") == "pain":
            text = node_data.get("description", "") or node_data.get("text", "")
        elif node_data and node_data.get("node_type") == "capability":
            text = node_data.get("name", "") or node_data.get("description", "")
        elif node_data and node_data.get("node_type") == "product":
            text = node_data.get("url", "")
        else:
            text = node_data.get("name", "") or node_data.get("description", "") or node_data.get("text", "") or "XXXX"
        print(f"Node ID: {node_id}, Node Type: {node_data.get('node_type')}, Text: {text}, Relevance: {relevance}, ")

    add_or_update_cumulative_relevance_data(product_id, cumulative_relevance)

    print("Starting persona traversal")
    print("----------------------------------")
    for persona_id, _ in get_nodes_list(sub_graph, "persona",{}):
        persona_node = get_node_by_id(sub_graph, persona_id)
        print("Processing persona:", persona_id, "with title:", persona_node.get("title", "Unknown"))
        normalized_relevance = get_cumulative_relevance_data(product_id, persona_id)
        persona_node = get_node_by_id(sub_graph, persona_id)
        print("Persona node data:", persona_node)
        jobs = []
        pains = []
        # For each job performed by this persona
        persona_jobs = get_source_nodes_by_target_and_type(sub_graph, persona_id, "performed_by")
        for job_id in persona_jobs:

            job_node = get_node_by_id(sub_graph, job_id)
            if not job_node:
                print(f"Job node not found for ID: {job_id}")
                continue
            job_relevance = get_cumulative_relevance_data(product_id, job_id)
            jobs.append({
                "description": job_node.get("description") or job_node.get("text") or "",
                "relevance": job_relevance
            })
            # For each pain solved by this job
            pain_ids = get_source_nodes_by_target_and_type(sub_graph, job_id, "addresses")
            for pain_id in pain_ids:
                pain_node = get_node_by_id(sub_graph, pain_id)
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
        print("Persona:", persona)
        print("-----------------------------------")
        personas.append(persona)
    
    
    print("Persona relevance computed, total personas found:", personas)



    aggregated_personas = aggregated_personas_map(sub_graph, personas, threshold=0.2)
    return aggregated_personas


def aggregated_personas_map(sub_graph: nx.DiGraph, match_results: list[dict], threshold: float = 0.0) -> list[dict]:
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


