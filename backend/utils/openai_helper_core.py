import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict

from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data
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
    # Updated query logic
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

        """
        Output is of format:
        {
            "capability_id": "string",
            "capability": "string",
            "pains": [
            {
                "pain": "string",
                "relevance": [0.8, 0.5, 1.0],
                "pain_trigger": {
                        "attribute": "string",
                        "dimension": "string",
                        "direction": "Increase" | "Decrease" | "Change"
                    },
                "jobs": [
                {
                    "description": "Resolve incoming customer tickets in under 24 hours",
                    "impact": 0.9,
                    "personas": [
                    {
                        "job_importance": 0.9,
                        "title": "Customer Support Executive",
                        "department": "Support",
                        "seniority": "Operator"
                    },
                    {
                        "job_importance": 0.8,
                        "title": "Support Team Lead",
                        "department": "Support",
                        "seniority": "Manager"
                    }
                    ]
                }
                ]
            }
        """

        # First pass: Build flattened_capability_map
        flattened_capability_map = []
        for entry in gpt_output:
            cap_id = entry.get("capability_id", "Unknown").strip()
            for pain in entry.get("pains", []):
                pain_desc = pain.get("pain", "Unknown Pain")
                pain_trigger = pain.get("pain_trigger", [])
                pain_trigger_attribute = pain_trigger.get("attribute", "")
                pain_trigger_dimension = pain_trigger.get("dimension", "")
                pain_trigger_direction = pain_trigger.get("direction", "")
                relevance_array = pain.get("relevance", [])
                jobs = pain.get("jobs", [])
                for job in jobs:
                    job_desc = job.get("description", "")
                    job_impact = job.get("impact", 0.0)
                    personas = job.get("personas", [])
                    for persona in personas:
                        persona_title = persona.get("title", "")
                        persona_department = persona.get("department", "")
                        persona_seniority = persona.get("seniority", "")
                        persona_job_importance = persona.get("job_importance", 0.0)
                        flattened_capability_map.append({
                            "capability_id": cap_id,
                            "pain": pain_desc,
                            "pain_trigger": {
                                "attribute": pain_trigger_attribute,
                                "dimension": pain_trigger_dimension,
                                "direction": pain_trigger_direction
                            },
                            "relevance": relevance_array,
                            "job": job_desc,
                            "job_impact": job_impact,
                            "persona": {
                                "title": persona_title,
                                "department": persona_department,
                                "seniority": persona_seniority
                            },
                            "persona_job_importance": persona.get("job_importance", 0.0),
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

        print("Updating cumulative relevance")
        relevance_nodes = product_subgraph.calculate_cumulative_relevance()
        add_or_update_cumulative_relevance_data(product_id, relevance_nodes)
        print("Cumulative relevance json updated successfully.")

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


