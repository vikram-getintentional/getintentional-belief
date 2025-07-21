from collections import defaultdict
from backend.utils.graph_base.graph import Graph
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data

def aggregate_persona_cards(sub_graph: Graph, match_results: list[dict], threshold: float = 0.0):
    final = []
    """
    Input is a list of match results - as a list of personas of the format:
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
    
    This function will do - for each persona in entry - check if relevance is above threshold. 
    Only add to personaCard if relevance is above threshold.
    It will aggregate all personas with the same title into one bundle with the list of departments, seniorities, jobs, pains.
    It will output the max relevance of all persona_ids as the aggregated relevance. 
    The expected output will be of the format:
    personaCard = {
            "persona_title": persona title,
            "persona_departments": [departments],
            "persona_seniority": [seniority],
            "persona_ids": [persona_ids],
            "max_relevance": max_relevance(relevance(personas)),
            "jobs": jobs,
            "pains": pains
        }
    """
    product_id = sub_graph.get_node_id("product", {})
    print("Starting aggregation of persona cards")
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

    print("Input match results:", match_results)
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
        for job in jobs:
            persona_map[title]["jobs"].add(str(job))
        pains = entry.get("pains", [])
        for pain in pains:
            persona_map[title]["pains"].add(str(pain))

    # Convert sets to lists for output
    for card in persona_map.values():
        card["persona_departments"] = list(card["persona_departments"])
        card["persona_seniority"] = list(card["persona_seniority"])
        card["persona_ids"] = list(card["persona_ids"])
        card["jobs"] = list(card["jobs"])
        card["pains"] = list(card["pains"])
        final.append(card)
    print("Aggregate - final output:", final)
    return final

    