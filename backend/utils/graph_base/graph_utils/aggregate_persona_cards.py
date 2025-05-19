from collections import defaultdict

from collections import defaultdict

def aggregate_persona_cards(match_results: list[dict], threshold: float = 0.05):
    grouped = {}
    for entry in match_results:
        # Get the relevance and filter by threshold
        relevance = entry.get("relevance", 0)
        if relevance < threshold:
            continue
        # Extract persona details from the nested persona object
        persona = entry.get("persona", {})
        
        persona_name = persona.get("title", "Unknown Persona")
        dept = persona.get("department", "Unknown Department")
        seniority = persona.get("seniority", "Unknown Seniority")
        job = entry.get("job", "Unknown Job")  # Use "job" directly
        pain = entry.get("pain", "Unknown Pain")  # Use "pain" directly
        capability = entry.get("capability", "Unknown Capability")
        
        # Create a unique key for the persona
        key = f"{persona_name}__{dept}__{seniority}"
        # Initialize the group if it doesn't exist
        if key not in grouped:
            grouped[key] = {
                "persona": {
                    "title": persona_name,
                    "department": dept,
                    "seniority": seniority
                },
                "total_relevance": 0,
                "jobs": defaultdict(set),
                "capabilities": set()
            }

        # Update the group with the current entry
        grouped[key]["total_relevance"] += relevance
        grouped[key]["jobs"][job].add(pain)
        grouped[key]["capabilities"].add(capability)
    # Construct the final output
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
        capabilities = list(obj["capabilities"])

        final.append({
            "persona": persona,
            "relevance": relevance,
            "jobs": jobs,
            "capabilities": capabilities
        })

    return final