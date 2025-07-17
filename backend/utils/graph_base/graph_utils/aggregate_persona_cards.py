from collections import defaultdict
from backend.utils.graph_base.graph import Graph

def aggregate_persona_cards(sub_graph: Graph, match_results: list[dict], threshold: float = 0.4):
    final = []
    for entry in match_results:
        relevance = entry.get("relevance", 0)
        #if relevance < threshold:
        #    continue
        final.append({
            "persona_id": entry.get("persona_id"),
            "persona": entry.get("persona"),
            "relevance": entry.get("relevance"),
            "jobs": entry.get("jobs"),
            "pains": entry.get("pains"),
        })
    print("Final aggregated personas:", final)
    return final