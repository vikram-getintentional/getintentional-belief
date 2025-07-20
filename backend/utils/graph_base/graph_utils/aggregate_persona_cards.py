from collections import defaultdict
from backend.utils.graph_base.graph import Graph

def aggregate_persona_cards(sub_graph: Graph, match_results: list[dict], threshold: float = 0.0):
    final = []
    """
    From a list of match results, does the following: 
    1. Aggregates the results by personas > jobs > pains > capabilities
    2. Calculates relevance scores for each persona
    3. Only returns personas with relevance scores above a certain threshold
    3. Returns a list of dictionaries of the following format:
    - Persona ID
    - Persona relevance
        - Jobs:
            - Job description
            - Pains:
                - Pain description
                - Capabilities:
                    - Capability name
                    - Capability description
    """
    print("Starting aggregation of persona cards")
    if not match_results:
        print("No match results found, returning empty list.")
        return final
    for entry in match_results:
        persona_id = entry.get("persona_id")
        if not persona_id:
            print("No persona ID found in entry, skipping.")
            continue
        # Initialize persona if not already done
        
        
        # Aggregate relevance
        for persona in final:
            if persona["persona_id"] == persona_id:
                persona["relevance"] += entry.get("relevance", 0.0)
                job_id = entry.get("job_id")
                if job_id:
                    job = persona["jobs"][job_id]
                    job["description"] = entry.get("job_description", "")
                    pain_id = entry.get("pain_id")
                    if pain_id:
                        pain = job["pains"][pain_id]
                        pain["description"] = entry.get("pain_description", "")
                        capability = {
                            "name": entry.get("capability_name", ""),
                            "description": entry.get("capability_description", "")
                        }
                        pain["capabilities"].append(capability)

    # Aggregate results by persona
    persona_map = defaultdict(lambda: {
        "persona": "",
        "persona_id": "",
        "relevance": 0.0,
        "jobs": defaultdict(lambda: {
            "description": "",
            "pains": defaultdict(lambda: {
                "description": "",
                "capabilities": []
            })
        })
    })
    for entry in match_results:
        persona_id = entry.get("persona_id")
        if persona_id not in persona_map:
            persona_map[persona_id] = {
                "persona": entry.get("persona"),
                "persona_id": persona_id,
                "relevance": 0.0,
                "jobs": defaultdict(lambda: {
                    "description": "",
                    "pains": defaultdict(lambda: {
                        "description": "",
                        "capabilities": []
                    })
                })
            }
        # Aggregate relevance
        persona_map[persona_id]["relevance"] += entry.get("relevance", 0.0)
        # Aggregate job information
        job_id = entry.get("job_id")
        if job_id:
            persona_map[persona_id]["jobs"][job_id]["description"] = entry.get("job_description", "")
            # Aggregate pain information
            pain_id = entry.get("pain_id")
            if pain_id:
                persona_map[persona_id]["jobs"][job_id]["pains"][pain_id]["description"] = entry.get("pain_description", "")
                # Aggregate capability information
                capability_id = entry.get("capability_id")
                if capability_id:
                    persona_map[persona_id]["jobs"][job_id]["pains"][pain_id]["capabilities"].append({
                        "name": entry.get("capability_name", ""),
                        "description": entry.get("capability_description", "")
                    })

    # Filter personas by relevance
    for persona_id, data in persona_map.items():
        if data["relevance"] > threshold:
            final.append(data)
    print("Final aggregated personas:", final)
    return final