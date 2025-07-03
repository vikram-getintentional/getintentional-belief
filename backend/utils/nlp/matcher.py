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
    base_graph.node_registry = {
        (key[0].lower(), key[1].lower() if isinstance(key[1], str) else key[1]): value
        for key, value in base_graph.node_registry.items()
    }
    for cap in capabilities:
        cap_name = cap.get("name", "").strip().lower()
        cap_desc = cap.get("description", "").strip().lower()
        # Try both (name, description) and just name for flexibility
        cap_id = base_graph.get_nodes_list("capability", {"name": cap_name, "description": cap_desc}) 
        if not cap_id:
            print(f"Capability '{cap_name}' not found in graph.")
            continue
        else:
            print(f"Found capability '{cap_name}' with ID {cap_id} in graph.")
        

        # Find all pain nodes this capability solves
        pain_ids = base_graph.get_source_nodes_by_target_and_type(cap_id, "solves")
        if not pain_ids:
            print(f"No pains found for capability '{cap_name}'.")
            continue
        for pain_id in pain_ids:
            pain = base_graph.get_node_by_id(pain_id)
            pain_text = pain["text"]
            relevance = pain.get("weight", 0.5)
            if relevance < threshold:
                continue

            # --- NEW: Find pain triggers for this pain ---
            pain_trigger_ids = base_graph.get_source_nodes_by_target_and_type(pain_id, "triggered_by")
            if not pain_trigger_ids:
                print(f"No triggers found for pain '{pain_text}'.")
                continue
            else:
                for pain_trigger_id in pain_trigger_ids:
                    pain_trigger = base_graph.get_node_by_id(pain_trigger_id)
                    if pain_trigger:
                        pain_trigger_text = pain_trigger["text"]
                    
                

            # Find all jobs addressed by this pain
            job_ids = base_graph.get_target_nodes_by_source_and_type(pain["id"], "addresses")
            if not job_ids:
                print(f"No jobs found for pain '{pain_text}'.")
                continue
            for job_id in job_ids:
                job = base_graph.get_node_by_id(job_id)
                job_description = job["description"]
                
                # Find all personas who perform this job
                persona_ids = base_graph.get_source_nodes_by_target_and_type(job['id'], "performed_by")
                for persona_id in persona_ids:
                    persona = base_graph.get_node_by_id(persona_id)
                    if not persona:
                        print(f"No persona found for job '{job_description}'.")
                        continue
                    persona_value = persona["value"]
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
                        "source": cap.get("source", "graph"),
                        "last_updated": cap.get("last_updated", ""),
                        "pain_trigger": pain_trigger
                    })
                    print(f"Matched persona '{title}' with job '{job_description}' and pain '{pain_text}' for capability '{cap_name}'.")
    # Optionally sort and filter as before
    results = [e for e in results if e["relevance"] >= threshold]
    results.sort(key=lambda x: x["relevance"], reverse=True)
    print("Relevance Results generated for:", len(results))
    return results