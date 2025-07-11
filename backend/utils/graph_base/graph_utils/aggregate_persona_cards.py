from collections import defaultdict

from backend.utils.graph_base.graph import Graph

def aggregate_persona_cards(product_graph: Graph, persona_list: list[dict], threshold: float = 0.05):
    print("Aggregating persona cards...")
    
    grouped = {}
    for entry in persona_list:
        
        relevance = entry.get("relevance", 0)
        if relevance < threshold:
            continue
        persona_id = entry["persona_id"]
        job_id = entry.get("job_id")
        pain_id = entry.get("pain_id")
        

        persona_node = product_graph.get_node_by_id(persona_id) or {}
        job_node = product_graph.get_node_by_id(job_id) or {}
        pain_node = product_graph.get_node_by_id(pain_id) or {}
        

        persona_name = persona_node.get("title", "Unknown Persona")
        dept = persona_node.get("department", "Unknown Department")
        seniority = persona_node.get("seniority", "Unknown Seniority")
        job_desc = job_node.get("description", "Unknown Job")
        pain_desc = pain_node.get("text", "Unknown Pain")
        

        if persona_id not in grouped:
            grouped[persona_id] = {
                "persona": {
                    "title": persona_name,
                    "department": dept,
                    "seniority": seniority
                },
                "total_relevance": 0,
                "jobs": defaultdict(set),  # job_desc -> set of pain_desc
                
            }

        grouped[persona_id]["total_relevance"] += relevance
        grouped[persona_id]["jobs"][job_desc].add(pain_desc)
        

    # Build the final output
    final = []
    for obj in grouped.values():
        persona = obj["persona"]
        relevance = round(obj["total_relevance"], 3)
        jobs = [
            {
                "description": job,
                "pains": list(pains)
            }
            for job, pains in obj["jobs"].items()
        ]
        

        final.append({
            "persona": persona,
            "relevance": relevance,
            "jobs": jobs,
            
        })

    return final