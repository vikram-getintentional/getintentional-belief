import os
import json
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data
from backend.utils.graph_base.graph_builder import canonicalize_and_create_zmot_icp_nodes, process_pain_triggers_zmot_icp_map_to_graph
from backend.utils.inference.gpt_prompts.zmot_icp_openai import infer_pain_triggers_zmot_icp




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
def infer_zmot_icp(product_id, product_subgraph, force_openai=False, retry_depth=0) -> dict:
    print("Starting inference loop: ", product_id)
    from datetime import datetime
    # Updated query logic
    product_node = product_subgraph.get_node_by_id(product_id)
    if not product_node:
        raise ValueError("Product node not found.")
    summary = product_node.get("summary")
    pains_list = product_subgraph.get_nodes_list("pain",{})
    if not pains_list:
        print("No pain nodes yet. Do persona inference first...")
        return {
            "capability_map": [],
            "personas": [],
            "aggregated_personas": [],
            "source": "empty",
            "error": "No pain nodes found. Please run persona inference first."
        }
    pain_ids = [node_id for node_id, _ in pains_list]
    pain_jobs_dict = []
    for pain_id in pain_ids:
        pain_node = product_subgraph.get_node_by_id(pain_id)
        if not pain_node:
            continue
        existing_pain_triggers = product_subgraph.get_target_nodes_by_source_and_type(pain_id, "triggered_by")
        if existing_pain_triggers:
            existing_archetypes = product_subgraph.get_target_nodes_by_source_and_type(pain_id, "experienced_in")
            existing_zmot_events = product_subgraph.get_target_nodes_by_source_and_type(pain_id, "zmot_trigger_event")
        if not existing_pain_triggers or not existing_archetypes or not existing_zmot_events:    
            # pain_jobs_dict is only populated if there are no existing pain triggers or if icp or zmot data is missing
            # That way we are not running gpt queries each time for the entire pains repo
            pain_text = pain_node.get("text", "")
            related_job_ids = product_subgraph.get_target_nodes_by_source_and_type(pain_id, "addresses")
            jobs_list = []
            for job_id in related_job_ids:
                job_node = product_subgraph.get_node_by_id(job_id)
                job_description = job_node.get("description", "")
                jobs_list.append(job_description)
            pain_jobs_dict.append({
                "pain_id": pain_id,
                "pain_text": pain_text,
                "jobs_impacted": jobs_list
            })

    try:
        print("🔁 Running GPT reasoning...")
        pains_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        # ---- Batching logic for GPT calls ----
        def batch_list(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        print("🧠 [GPT] Generating ZMOTs map...")
        gpt_output = []
        batch_size = 3  # You can tune this for your token limits
        for batch in batch_list(pain_jobs_dict, batch_size):
            print("🧠 [GPT] Generating ZMOTs map for batch...")
            batch_output = infer_pain_triggers_zmot_icp(summary, batch)
            if isinstance(batch_output, str):
                batch_output = json.loads(batch_output)
            gpt_output.extend(batch_output)
            break  # For testing, we break after the first batch
        # ---- End batching logic ----

        """
        Output is of format:
        - pain_id: string
        - pain_triggers:
            pain_trigger: object of
                - attribute: string
                - dimension: string
                - direction: string
                - icp_archetypes: list of objects
                    - industry: list of strings
                    - revenue: list of strings
                    - employees: list of strings
                    - funding_stage: list of strings
                    - geographies: list of strings
                    - match_score: float
                - zmot_events: list of objects
                    - trigger_event: string
                    - observable_moment: string
                    - trigger_keywords: list of strings
                        - match_score: float
        """

        # First pass: Build flattened_capability_map
        flattened_pains_map = []
        for entry in gpt_output:
            pain_id = entry.get("pain_id", "Unknown").strip()
            pain_triggers = entry.get("pain_triggers", [])
            for pain_trigger in pain_triggers:
                pain_trigger_attribute = pain_trigger.get("attribute", "")
                pain_trigger_dimension = pain_trigger.get("dimension", "")
                pain_trigger_direction = pain_trigger.get("direction", "")
                icp_archetypes = pain_trigger.get("icp_archetypes", [])
                if isinstance(icp_archetypes, dict):
                    icp_archetypes = [icp_archetypes]
                elif not isinstance(icp_archetypes, list):
                    icp_archetypes = []

                zmot_events = pain_trigger.get("zmot_events", [])
                if isinstance(zmot_events, dict):
                    zmot_events = [zmot_events]
                elif not isinstance(zmot_events, list):
                    zmot_events = []

                # If either is empty, still append at least one entry
                if not icp_archetypes:
                    icp_archetypes = [{}]
                if not zmot_events:
                    zmot_events = [{}]

                for icp_archetype in icp_archetypes:
                    industries = icp_archetype.get("industries", [])
                    revenue_ranges = icp_archetype.get("revenues", [])
                    employee_ranges = icp_archetype.get("employees", [])
                    funding_stages = icp_archetype.get("funding_stages", [])
                    geographies = icp_archetype.get("geographies", [])
                    icp_match_score = icp_archetype.get("icp_match_score", 0.0)
                    for zmot_event in zmot_events:
                        trigger_event = zmot_event.get("trigger_event", "")
                        observable_moments = zmot_event.get("observable_moments", [])
                        trigger_keywords = zmot_event.get("trigger_keywords", [])
                        zmot_match_score = zmot_event.get("zmot_match_score", 0.0)

                        flattened_pains_map.append({
                            "pain_id": pain_id,
                            "pain_trigger": {
                                "attribute": pain_trigger_attribute,
                                "dimension": pain_trigger_dimension,
                                "direction": pain_trigger_direction
                            },
                            "icp_archetype": {
                                "industries": industries,
                                "Revenues": revenue_ranges,
                                "Employees": employee_ranges,
                                "Funding Stages": funding_stages,
                                "Geographies": geographies,
                                "icp_match_score": icp_match_score
                            },
                            "zmot_event": {
                                "trigger_event": trigger_event,
                                "observable_moments": observable_moments,
                                "trigger_keywords": trigger_keywords,
                                "zmot_match_score": zmot_match_score
                            },
                            "source": "openai",
                        })
        print("\n\nFlattened pains map in ZMOT discovery")
        print(json.dumps(flattened_pains_map, indent=2))

        # Step 3: Normalize and canonicalize the flattened capability map
        print("📊 [Graph] Canonicalizing capability map...")
        if not flattened_pains_map:
            print("⚠️ No valid data found in OpenAI output. Cannot proceed.")
            return {
                "pains_map": [],
                "source": "empty",
                "error": "No valid data found in OpenAI output."
            }
        pains_map = canonicalize_and_create_zmot_icp_nodes(flattened_pains_map)

        flattened_results = process_pain_triggers_zmot_icp_map_to_graph(pains_map)
        print("ICP & ZMOT Graph processed. Results follow:", flattened_results)

        print("Updating cumulative relevance")
        relevance_nodes = product_subgraph.calculate_cumulative_relevance()
        add_or_update_cumulative_relevance_data(product_id, relevance_nodes)
        print("Cumulative relevance json updated successfully.")

        return {
                "pains_map": pains_map,
                "ZMOT_ICP_Nodes": flattened_results
        }

    except Exception as e:
        import traceback
        print("❌ Error in ZMOT Reasoning:", e)
        traceback.print_exc()
        return {
            "capability_map": [],
            "personas": [],
            "error": str(e)
        }


