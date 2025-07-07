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



# Count total personas, jobs per persona, pains per job
def is_sparse_results(relevance_results, base_graph):
    persona_job_map = {}
    job_pain_map = {}
    cap_pain_map = {}

    for entry in relevance_results:
        raw_persona = entry["persona"]
        job_key = entry["canonical_job"]
        pain = entry["canonical_pain"]
        capability = entry.get("capability", "Unknown Capability")

        # Normalize raw_persona to match the keys in PERSONA_CANONICAL_MAP
        if isinstance(raw_persona, dict):
            # Convert to a tuple key
            persona_key = (
                raw_persona.get("title", "").strip().lower(),
                raw_persona.get("department", "Unknown").strip().lower(),
                raw_persona.get("seniority", "Unknown").strip().lower()
            )
        else:
            # If raw_persona is already a string, normalize it
            persona_key = raw_persona.strip().lower()

        # Canonicalize using the graph's node registry if needed
        persona_id = base_graph.node_registry.get(("persona", persona_key))
        persona_key = persona_id if persona_id else persona_key

        # Count jobs per persona
        if persona_key not in persona_job_map:
            persona_job_map[persona_key] = set()
        persona_job_map[persona_key].add(job_key)

        # Count pains per job
        if job_key not in job_pain_map:
            job_pain_map[job_key] = set()
        job_pain_map[job_key].add(pain)

        # Optional: count pains per capability
        if capability not in cap_pain_map:
            cap_pain_map[capability] = set()
        cap_pain_map[capability].add(pain)
    
    # Only mark as sparse if more than 50% of personas have < 2 jobs, etc.")
    persona_threshold = sum(len(jobs) < 1 for jobs in persona_job_map.values()) > len(persona_job_map) * 0.5
    job_threshold = sum(len(pains) < 1 for pains in job_pain_map.values()) > len(job_pain_map) * 0.5
    capability_threshold = sum(len(pains) < 1 for pains in cap_pain_map.values()) > len(cap_pain_map) * 0.5

    
    print("Sparse results detected:", persona_threshold or job_threshold or capability_threshold)
    return persona_threshold or job_threshold or capability_threshold


# 🧠 Rule-based first, fallback to 2-step OpenAI reasoning
def infer_with_rules_then_fallback(product_id, product_subgraph, force_openai=False, retry_depth=0) -> dict:
    print("Starting inference loop with retry depth:", retry_depth)
    MAX_RETRY_DEPTH = 2

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
        # Step 1: Match capabilities to canonical pains, jobs, and personas
        print("🧠 Trying rule-based relevance scoring...")
        relevance_results = match_capabilities_to_canonical_personas(capabilities, base_graph=product_subgraph)

        sparse = False

        if relevance_results and len(relevance_results) > 0:
            print("Relevance results found. Processing...")
            sparse = is_sparse_results(relevance_results, base_graph=product_subgraph)
            print("Sparse results detected:", sparse)

        if (not relevance_results or len(relevance_results) == 0 or (sparse and not force_openai)):
            print("🔁 Falling back to GPT reasoning...")
            return run_two_step_reasoning(product_subgraph, retry_depth=retry_depth)

        # Step 2: Build capability map and process graph
        print("Not sparse - using historic reelvance map")
        
        flattened_results = relevance_results
        capability_map = convert_rule_matches_to_capability_map(flattened_results)
        aggregated_personas = aggregate_persona_cards(flattened_results, threshold=0.05)
        aggregated_personas = sorted(aggregated_personas, key=lambda x: x["relevance"], reverse=True)
        top_personas = aggregated_personas[:5]
        return {
            "capability_map": capability_map,
                "personas": flattened_results,
                "aggregated_personas": top_personas,
                "source": "rule"
            }

    except Exception as e:
        import traceback
        print("❌ Inference error:", e)
        traceback.print_exc()
        return {
            "capability_map": [],
            "personas": [],
            "aggregated_personas": [],
            "source": "error",
            "error": str(e)
        }

def run_two_step_reasoning(product_subgraph, retry_depth=0, dump_path=None):
    from datetime import datetime
    print("starting run_two_step with capabilitites")
    product_id = product_subgraph.get_node_id("product", {})
    print("Product ID for inference:", product_id)
    product_node = product_subgraph.get_node_by_id(product_id)
    if not product_node:
        raise ValueError("Product node not found.")
    summary = product_node.get("summary")
    capability_nodes = [
        node_data
        for node_id, node_data in product_subgraph.get_nodes_list("capability", {})
    ]
    capabilities_list = [
    {"id":node["id"], "name": node["name"], "description": node["description"]}
    for node in capability_nodes
    if node.get("name") and node.get("description") and node.get("id")
]
    print("Capabilities for inference:", capabilities_list)
    if not summary:
        print("⚠️ Summary is empty. Cannot proceed with inference.")
        return {
            "capability_map": [],
            "personas": [],
            "aggregated_personas": [],
            "source": "empty",
            "error": "Summary is empty."
        }
    
    capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    try:
        # Step 1: Load or generate capability map
        if dump_path:
            print(f"📂 Loading capability map from dump: {dump_path}")
            with open(dump_path, "r") as f:
                capability_map = json.load(f)
        else:
            print("🧠 [GPT] Generating capability map...")
            gpt_output = infer_persona_job_pain_from_capabilities(summary, capabilities_list)

        """
        Manas - here's where I'm stuck. 
           The gpt_output is a list of dictionaries, each containing a capability_id and a list of pains with relevance.
           Each pain has a relevance array with scored values corresponding to every capability in the list.
           The output we want is:
           Every capability node has an edge to every pain node
           The relevance of capability[i]-pain[j] is gpt_output[i]["pains"][j]["relevance"][i]
           If a relevance score already exists for this pain-capability combo - choose the highest value.
        """

        # Build a mapping from capability_id to its index in capabilities_list
        cap_id_to_index = {cap["id"]: idx for idx, cap in enumerate(capabilities_list)}

        # Dict to store max relevance for each (pain, capability) pair
        pain_cap_to_relevance = {}

        # First pass: Build pain_cap_to_relevance
        pain_cap_to_relevance = {}
        for entry in gpt_output:
            cap_id = entry.get("capability_id", "Unknown").strip()
            for pain in entry.get("pains", []):
                pain_desc = pain.get("pain", "Unknown Pain")
                relevance_array = [float(x) for x in pain.get("relevance", [])]
                for idx, rel in enumerate(relevance_array):
                    target_cap_id = capabilities_list[idx]["id"]
                    key = (pain_desc, target_cap_id)
                    prev = pain_cap_to_relevance.get(key, 0.0)
                    pain_cap_to_relevance[key] = max(prev, rel)


        # Second pass: Build flattened_capability_map using the max relevance
        flattened_capability_map = []
        for entry in gpt_output:
            cap_id = entry.get("capability_id", "Unknown").strip()
            for pain in entry.get("pains", []):
                pain_desc = pain.get("pain", "Unknown Pain")
                pain_trigger = pain.get("pain_trigger", "")
                jobs = pain.get("jobs", [])
                # Look up the max relevance for this (pain, capability) pair
                relevance = pain_cap_to_relevance.get((pain_desc, cap_id), 0.0)
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
                            "relevance": relevance,
                            "job": job_desc,
                            "persona": {
                                "title": persona_title,
                                "department": persona_department,
                                "seniority": persona_seniority,
                            },
                            "source": "openai",
                        })
                        print("Flattened map for persona:", cap_id,"+",pain_desc, " with rel: ", relevance)
        print("\n\n\nFull flattened capability map: ", flattened_capability_map)
        
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
        flattened_results = process_capability_map_to_graph(capability_map)

        """
        Source of gpt_output:
        cap 1:{
            pains:{
                pain 1 : {
                    text,
                    relevance_array: [
                    pain1-cap1, pain1-cap2, pain1-cap3,...
                    ]
                },
                pain 2 : {
                    text,
                    relevance_array: [
                    pain2-cap1, pain2-cap2, pain2-cap3,...
                    ]
                }

            }
        }
        cap 2:{
            pains:{
                pain 3 : {
                    text,
                    relevance_array: [
                    pain3-cap1, pain3-cap2, pain3-cap3,...
                    ]
                },
                pain 1 : {
                    text,
                    relevance_array: [
                    pain1-cap1, pain1-cap2, pain1-cap3,...
                    ]
                }

            }
        }

        We want:
        - relevance of cap 1-pain1 = for all pain = pain1 > max(relevance[0])
        - relevance of cap 1-pain2 = for all pain = pain2 > max(relevance[0])
        - relevance of cap 2-pain1 = for all pain = pain1 > max(relevance[1])
        - relevance of cap 2-pain3 = for all pain = pain3 > max(relevance[1])

        """



        print("Updating capability centralities")
        product_subgraph.update_capability_centralities()
        
        # Step 4: Aggregate personas
        aggregated_personas = aggregate_persona_cards(flattened_results, threshold=0.05)
        aggregated_personas = sorted(aggregated_personas, key=lambda x: x["relevance"], reverse=True)
        top_personas = aggregated_personas[:5]
        
        return {
            "capability_map": capability_map,
            "personas": flattened_results,
            "aggregated_personas": top_personas,
            "source": "openai_dump" if dump_path else "openai",
            "retry_depth": retry_depth + 1
        }

    except Exception as e:
        import traceback
        print("❌ Error in run_two_step_reasoning:", e)
        traceback.print_exc()
        return {
            "capability_map": [],
            "personas": [],
            "aggregated_personas": [],
            "source": "error",
            "error": str(e)
        }
