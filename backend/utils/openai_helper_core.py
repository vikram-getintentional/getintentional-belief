import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict

from backend.utils.inference.capability_to_pain_persona import infer_persona_job_pain_from_capabilities
from backend.utils.graph_base.graph_builder import process_capability_map_to_graph
from backend.utils.knowledge_base.canonical_maps.canonical_loader import load_canonical_map
from backend.utils.nlp.matcher import match_capabilities_to_canonical_personas
from backend.utils.nlp.scorer import persona_relevance_score
from backend.utils.graph_base.graph_builder import convert_rule_matches_to_capability_map, process_capability_map_to_graph
from backend.utils.graph_base.graph_utils.aggregate_persona_cards import aggregate_persona_cards
from backend.utils.graph_base.graph import Graph




# 🔐 Load environment and OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 📚 Canonical maps
# CANONICAL_PATH = Path("backend/utils/knowledge_base/canonical_maps")
# JOB_CANONICAL_MAP = load_canonical_map(CANONICAL_PATH / "job_to_canonical.json")
# PAIN_CANONICAL_MAP = load_canonical_map(CANONICAL_PATH / "pain_to_canonical.json")
# PERSONA_CANONICAL_MAP = load_canonical_map(CANONICAL_PATH / "persona_to_canonical.json")




# 🧠 Rule-based first, fallback to 2-step OpenAI reasoning
def infer_with_rules_then_fallback(product_id, product_subgraph, force_openai=False, retry_depth=0) -> dict:
    print("Starting inference loop: ", product_id)
    from datetime import datetime

    product_node = product_subgraph.get_node_by_id(product_id)
    if not product_node:
        raise ValueError("Product node not found.")
    summary = product_node.get("summary")
    capability_ids = product_subgraph.get_target_nodes_by_source_and_type(product_id, "offered_by")
    capabilities = [
        product_subgraph.get_node_by_id(cid)
        for cid in capability_ids
        if product_subgraph.get_node_by_id(cid)
    ]

    # Add a rule here to check if summary and capabilities are empty. If empty we will prompt the frontend for the user to run value prop again.
    if not summary or not capabilities:
        print("⚠️ Summary or capabilities are empty. Cannot proceed with inference.")
        return {
            "capability_map": [],
            "personas": [],
            "aggregated_personas": [],
            "source": "empty",
            "error": "Summary or capabilities are empty."
        } 

    # Filter valid capabilities
    capabilities = [c for c in capabilities if c.get("name") and c.get("description")]
    print("Filtered capabilities in infer:" , capabilities)

    print("Loaded graph with nodes:", len(product_subgraph.node_registry), "edges:", len(product_subgraph.graph_edges))

    try:
        print("🔁 Running GPT reasoning...")
        capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        print("🧠 [GPT] Generating capability map...")
        gpt_output = infer_persona_job_pain_from_capabilities(summary, capabilities)

        # First pass: Build flattened_capability_map
        flattened_capability_map = []
        for entry in gpt_output:
            cap_id = entry.get("capability_id", "Unknown").strip()
            for pain in entry.get("pains", []):
                pain_desc = pain.get("pain", "Unknown Pain")
                pain_trigger = pain.get("pain_trigger", "")
                relevance_array = pain.get("relevance", [])
                jobs = pain.get("jobs", [])
                for job in jobs:
                    job_desc = job.get("description", "")
                    personas = job.get("personas", [])
                    for persona in personas:
                        persona_title = persona.get("title", "")
                        persona_department = persona.get("department", "")
                        persona_seniority = persona.get("seniority", "")
                        flattened_capability_map.append({
                            "capability_id": cap_id,
                            "pain": pain_desc,
                            "pain_trigger": pain_trigger,
                            "relevance": relevance_array,
                            "job": job_desc,
                            "persona": {
                                "title": persona_title,
                                "department": persona_department,
                                "seniority": persona_seniority,
                            },
                            "source": "openai",
                        })
        print("\n\n\Flattened capability map in Hop0 traversal")

        # Step 3: Normalize and canonicalize the flattened capability map
        print("📊 [Graph] Canonicalizing capability map...")
        if not flattened_capability_map:
            print("⚠️ No valid capability data found in OpenAI output. Cannot proceed.")
            return {
                "capability_map": [],
                "personas": [],
                "aggregated_personas": [],
                "source": "empty",
                "error": "No valid capability data found."
            }
        capability_map = convert_rule_matches_to_capability_map(flattened_capability_map)
        
        flattened_results = process_capability_map_to_graph(capabilities, capability_map)

        print("Calculating capability centralities")
        product_subgraph.update_capability_centralities()

        return {
                "capability_map": capability_map,
                "personas": flattened_results
                
            }

    except Exception as e:
        import traceback
        print("❌ Error in run_two_step_reasoning:", e)
        traceback.print_exc()
        return {
            "capability_map": [],
            "personas": [],
            "error": str(e)
        }


