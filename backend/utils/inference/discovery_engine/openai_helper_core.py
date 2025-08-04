import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict

from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data
from backend.utils.inference.gpt_prompts.capability_to_pain_persona import infer_persona_job_pain_from_capabilities
from backend.utils.knowledge_base.canonical_maps.canonical_loader import load_canonical_map
from backend.utils.nlp.scorer import persona_relevance_score
from backend.utils.graph_base.graph_builder import convert_rule_matches_to_capability_map
from backend.utils.graph_base.network_graph import build_product_graph, get_node_by_id, get_target_nodes_by_source_and_type, update_capability_centralities

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
def infer_with_rules_then_fallback(product_id, product_subgraph: nx.DiGraph, force_openai=False, retry_depth=0) -> dict:
    print("Starting inference loop: ", product_id)
    from datetime import datetime
    # Updated query logic
    product_node = get_node_by_id(product_subgraph, product_id)
    if not product_node:
        raise ValueError("Product node not found.")
    summary = product_node.get("summary")
    domain = product_node.get("domain", "")
    industry = product_node.get("industry", "")
    capability_ids = get_target_nodes_by_source_and_type(product_subgraph, product_id, "offered_by")
    capabilities = [
        get_node_by_id(product_subgraph, cid)
        for cid in capability_ids
        if get_node_by_id(product_subgraph, cid)
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
    filtered_capabilities = []
    for c in capabilities:
        if c.get("name") and c.get("description"):
            filtered_capabilities.append({
                "id": c.get("id"), 
                "name": c.get("name"), 
                "description": c.get("description")
                })
            
    print("Filtered capabilities in infer:" , filtered_capabilities)

    capability_ids_list = [cap.get("id") for cap in filtered_capabilities]

    print("Loaded graph with nodes:", len(product_subgraph.nodes), "edges:", len(product_subgraph.edges))

    try:
        print("🔁 Running GPT reasoning...")
        capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        
        print("🧠 [GPT] Generating capability map...")
        gpt_output = infer_persona_job_pain_from_capabilities(summary, domain, industry, filtered_capabilities)

        """
        Output is of format:
        - capability_id: string
        - capability: string
        - pains: list of
            - pain: string
            - relevance: array of floats
            - jobs: list of
                - description: string
                - impact: float
                - personas: list of
                    - job_importance: float
                    - title: string
                    - department: string
                    - seniority: string
        """

        if not gpt_output:
            print("⚠️ No valid data found in OpenAI output. Cannot proceed.")
            return {
                "capability_map": [],
                "personas": [],
                "aggregated_personas": [],
                "source": "empty",
                "error": "No valid data found in OpenAI output."
            }
        print("GPT output:", gpt_output)
        capability_map = convert_rule_matches_to_capability_map(gpt_output, capability_ids_list, product_id)

        print("\n\nFlattened capability map in Hop0 traversal")
        print(json.dumps(capability_map, indent=2))

        # Step 3: Normalize and canonicalize the flattened capability map
        print("📊 [Graph] Canonicalizing capability map...")
        if not capability_map:
            print("⚠️ No valid capability data found in OpenAI output. Cannot proceed.")
            return {
                "capability_map": [],
                "personas": [],
                "aggregated_personas": [],
                "source": "empty",
                "error": "No valid capability data found."
            }
        
        print("Calculating capability centralities")
        update_capability_centralities(product_subgraph)

        #print("Updating cumulative relevance")
        #relevance_nodes = calculate_cumulative_relevance(product_subgraph)
        #add_or_update_cumulative_relevance_data(product_id, relevance_nodes)
        #print("Cumulative relevance json updated successfully.")

        return {
                "capability_map": capability_map,
                "personas": capability_map
                
            }

    except Exception as e:
        # return {
        #     "capability_map": [],
        #     "personas": [],
        #     "error": str(e)
        # }
        raise e


