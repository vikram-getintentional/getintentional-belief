import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from backend.utils.graph_base.graph_utils.json_store import load_json
from backend.utils.knowledge_base.canonicalizer import canonicalize_persona
from backend.utils.graph_base.edges.edge_manager import add_edge

# Paths
GRAPH_PATH = Path("backend/utils/graph_base/graph_data")
PERSONA_PATH = GRAPH_PATH / "persona_nodes.json"
JOB_PATH = GRAPH_PATH / "job_nodes.json"
PAIN_PATH = GRAPH_PATH / "pain_nodes.json"
EDGE_PATH = GRAPH_PATH / "graph_edges.json"

# Load canonical maps
CANONICAL_PATH = Path("backend/utils/knowledge_base/canonical_maps")
from backend.utils.knowledge_base.canonical_maps.canonical_loader import load_canonical_map

JOB_CANONICAL_MAP = load_canonical_map(CANONICAL_PATH / "job_to_canonical.json")
PAIN_CANONICAL_MAP = load_canonical_map(CANONICAL_PATH / "pain_to_canonical.json")

model = SentenceTransformer("all-MiniLM-L6-v2")

import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.utils.graph_base.edges.edge_manager import add_edge


def match_capabilities_to_canonical_personas(capabilities: list[dict], base_graph, threshold=0.7):
    results = []
    for cap in capabilities:
        cap_name = cap.get("name", "").strip().lower()
        cap_desc = cap.get("description", "").strip().lower()
        # Try both (name, description) and just name for flexibility
        cap_id = base_graph.node_registry.get(("capability", (cap_name, cap_desc))) \
            or base_graph.node_registry.get(("capability", cap_name))
        if not cap_id:
            continue

        # Find all pain nodes this capability solves
        for edge in base_graph.graph_edges:
            if edge.get("source") == cap_id and edge.get("type") == "solves":
                pain_id = edge.get("target")
                relevance = edge.get("weight", 0.5)
                if relevance < threshold:
                    continue
                pain_value = base_graph.get_node_by_id(pain_id)
                pain_text = pain_value["value"] if pain_value else ""

                 # --- NEW: Find pain triggers for this pain ---
                pain_triggers = []
                for trigger_edge in base_graph.graph_edges:
                    if trigger_edge.get("source") == pain_id and trigger_edge.get("type") == "triggered_by":
                        trigger_id = trigger_edge.get("target")
                        trigger_value = base_graph.get_node_by_id(trigger_id)
                        if trigger_value:
                            pain_triggers.append(trigger_value["value"])
                # If only one trigger, just use the string; else, use the list
                pain_trigger = pain_triggers[0] if len(pain_triggers) == 1 else pain_triggers

                

                # Find all jobs addressed by this pain
                for job_edge in base_graph.graph_edges:
                    if job_edge.get("source") == pain_id and job_edge.get("type") == "addresses":
                        job_id = job_edge.get("target")
                        job_value = base_graph.get_node_by_id(job_id)
                        job_text = job_value["value"] if job_value else ""


                        # Find all personas who perform this job
                        for persona_edge in base_graph.graph_edges:
                            if persona_edge.get("source") == job_id and persona_edge.get("type") == "performed_by":
                                persona_id = persona_edge.get("target")
                                persona_value = base_graph.get_node_by_id(persona_id)
                                if persona_value and isinstance(persona_value["value"], tuple):
                                    title, seniority, department = persona_value["value"]
                                else:
                                    title, seniority, department = "", "", ""
                                results.append({
                                    "persona": {
                                        "title": title,
                                        "department": department,
                                        "seniority": seniority
                                    },
                                    "job": job_text,
                                    "pain": pain_text,
                                    "capability": cap.get("name"),
                                    "relevance": relevance,
                                    "source": edge.get("source", "graph"),
                                    "last_updated": edge.get("last_updated", datetime.utcnow().isoformat()),
                                    "pain_trigger": pain_trigger
                                })
    # Optionally sort and filter as before
    results = [e for e in results if e["relevance"] >= threshold]
    results.sort(key=lambda x: x["relevance"], reverse=True)
    print("Relevance Results generated for:", len(results))
    return results