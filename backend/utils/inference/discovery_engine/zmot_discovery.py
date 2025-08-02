import os
import json
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

from backend.utils.graph_base.network_graph import get_cumulative_relevance, get_node_by_id, get_nodes_list, get_target_nodes_by_source_and_type
from backend.utils.graph_base.nodes.company_nodes import get_or_create_employees_node, get_or_create_funding_stage_node, get_or_create_geographies_node, get_or_create_industry_node, get_or_create_revenue_node
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data, get_cumulative_relevance_data
from backend.utils.graph_base.graph_builder import canonicalize_and_create_zmot_icp_nodes
from backend.utils.inference.gpt_prompts.zmot_icp_openai import infer_pain_triggers_zmot_icp
import networkx as nx




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
def infer_zmot_icp(product_id, product_subgraph: nx.DiGraph, force_openai=False, retry_depth=0, relevance_threshold=0.5) -> dict:
    print("Starting inference loop: ", product_id)
    from datetime import datetime
    # Updated query logic
    product_node = get_node_by_id(product_subgraph, product_id)
    if not product_node:
        raise ValueError("Product node not found.")
    summary = product_node.get("summary")
    pains_list = get_nodes_list(product_subgraph, "pain", {})
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
        pain_node = get_node_by_id(product_subgraph, pain_id)
        if not pain_node:
            continue
        existing_pain_triggers = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "triggered_by")
        if existing_pain_triggers:
            existing_archetypes = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "experienced_in")
            existing_zmot_events = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "zmot_trigger_event")
        if not existing_pain_triggers or not existing_archetypes or not existing_zmot_events:
            # pain_jobs_dict is only populated if there are no existing pain triggers or if icp or zmot data is missing
            # That way we are not running gpt queries each time for the entire pains repo
            pain_relevance = get_cumulative_relevance_data(product_id, pain_id)
            if pain_relevance > relevance_threshold:
                pain_text = pain_node.get("text", "")
                related_job_ids = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "addresses")
                jobs_list = []
                for job_id in related_job_ids:
                    job_node = get_node_by_id(product_subgraph, job_id)
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
            
        # ---- End batching logic ----

        """
        pain_id: string
        pain_triggers: [
            "attribute": "string",
            "dimension": "string",
            "direction": "string",
            "icp_archetypes": 
            [
                "industries": 
                [
                {"industry": "Healthcare", "match_score": 0.85},
                {"industry": "Finance", "match_score": 0.72},
                ...
                ],  
                "revenues": [
                {"revenue": "1-10M", "match_score": 0.9},
                {"revenue": "10-100M", "match_score": 0.6},
                ...
                ],
                "employees": [
                {"employees": "1-10", "match_score": 0.9},
                {"employees": "11-50", "match_score": 0.6},
                ...
                ],
                "funding_stages": [
                {"funding_stage": "Pre-Seed", "match_score": 0.8},
                {"funding_stage": "Seed", "match_score": 0.6},
                ...
                ],
                "geographies": [
                {"geography": "North America", "match_score": 0.9},
                {"geography": "Europe", "match_score": 0.6},
                ...
                ]
            }}
            ],
            "zmot_events": [
            {{
                "trigger_event": "string",
                "trigger_event_match_score": 0.8,
                "observable_moments": [
                {"observable_moment": "string", "match_score": 0.8}
                ...
                ],
                "trigger_keywords": [
                {"trigger_keyword": "string", "match_score": 0.8},
                ...
                ]
            }},
            {{
                "trigger_event": "string",
                "trigger_event_match_score": 0.8,
                "observable_moments": [
                {"observable_moment": "string", "match_score": 0.8}
                ...
                ],
                "trigger_keywords": [
                {"trigger_keyword": "string", "match_score": 0.8},
                ...
                ]
            }},
            ...
            ]
        }},
        """

        print("\n\nUpdated entries map in ZMOT discovery")
        print(json.dumps(gpt_output, indent=2))

        # Step 3: Normalize and canonicalize the flattened capability map
        print("📊 [Graph] Canonicalizing ZMOT map...")
        if not gpt_output:
            print("⚠️ No valid data found in OpenAI output. Cannot proceed.")
            return {
                "pains_map": [],
                "source": "empty",
                "error": "No valid data found in OpenAI output."
            }
        zmot_map = canonicalize_and_create_zmot_icp_nodes(gpt_output, product_id)

        print("ICP & ZMOT Graph processed. Results follow:", zmot_map)

        #print("Updating cumulative relevance")
        #relevance_nodes = get_cumulative_relevance(product_subgraph)
        #add_or_update_cumulative_relevance_data(product_id, relevance_nodes)
        print("Cumulative relevance json updated successfully.")

        return {
                "pains_map": zmot_map,
                "ZMOT_ICP_Nodes": zmot_map
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
