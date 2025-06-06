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
    job_threshold = sum(len(pains) < 2 for pains in job_pain_map.values()) > len(job_pain_map) * 0.5
    capability_threshold = sum(len(pains) < 1 for pains in cap_pain_map.values()) > len(cap_pain_map) * 0.5

    
    print("Sparse results detected:", persona_threshold or job_threshold or capability_threshold)
    return persona_threshold or job_threshold or capability_threshold


# 🧠 Rule-based first, fallback to 2-step OpenAI reasoning
def infer_with_rules_then_fallback(summary, capabilities, force_openai=False, retry_depth=0, base_graph=None) -> dict:
    print("Starting inference loop with retry depth:", retry_depth)
    MAX_RETRY_DEPTH = 2

    # Filter valid capabilities
    capabilities = [c for c in capabilities if c.get("name") and c.get("description")]

    print("Loaded graph with nodes:", len(base_graph.node_registry), "edges:", len(base_graph.graph_edges))

    try:
        # Step 1: Match capabilities to canonical pains, jobs, and personas
        print("🧠 Trying rule-based relevance scoring...")
        relevance_results = match_capabilities_to_canonical_personas(capabilities,base_graph=base_graph)

        if relevance_results and len(relevance_results) > 0:
            print("Relevance results found. Processing...")
            if retry_depth < MAX_RETRY_DEPTH:
                sparse = is_sparse_results(relevance_results, base_graph=base_graph)
                print("Sparse results detected:", sparse)
            else:
                sparse = False  # Don't trigger GPT fallback if we've hit the retry limit

            if sparse and not force_openai:
                print("⚠️ Sparse results. Triggering GPT fallback.")
                return run_two_step_reasoning(summary, capabilities, retry_depth=retry_depth)

            # Step 2: Build capability map and process graph
            print("Not sparse - using historic reelvance map")
            
            # Step 3: Aggregate personas
            flattened_results = relevance_results
            aggregated_personas = aggregate_persona_cards(flattened_results, threshold=0.05)
            aggregated_personas = sorted(aggregated_personas, key=lambda x: x["relevance"], reverse=True)
            top_personas = aggregated_personas[:5]
            return {
                "capability_map": capability_map,
                "personas": flattened_results,
                "aggregated_personas": top_personas,
                "source": "rule"
            }

        # Step 4: Fallback to GPT if no relevance results
        print("🔁 Falling back to GPT reasoning...")
        return run_two_step_reasoning(summary, capabilities, retry_depth=retry_depth)

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

def run_two_step_reasoning(summary, capabilities, retry_depth=0, dump_path=None):
    from datetime import datetime
    print("starting run_two_step with capabilitites")
    capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    try:
        # Step 1: Load or generate capability map
        if dump_path:
            print(f"📂 Loading capability map from dump: {dump_path}")
            with open(dump_path, "r") as f:
                capability_map = json.load(f)
        else:
            print("🧠 [GPT] Generating capability map...")
            gpt_output = infer_persona_job_pain_from_capabilities(summary, capabilities)
            flattened_capability_map = []
            for capability_entry in gpt_output:
                capability = capability_entry.get("capability", "Unknown Capability")
                cap_desc = capability_entry.get("description", "")
                for pain_entry in capability_entry.get("pains", []):
                    pain = pain_entry.get("pain", "Unknown Pain")
                    pain_trigger = pain_entry.get("pain_trigger", "")
                    relevance = float(pain_entry.get("relevance", 0.5))
                    for job_entry in pain_entry.get("jobs", []):
                        job = job_entry.get("description", "Unknown Job")
                        for persona in job_entry.get("personas", []):
                            flattened_capability_map.append({
                                "capability": capability,
                                "capability_description": cap_desc,
                                "pain": pain,
                                "pain_trigger": pain_trigger,
                                "job": job,
                                "persona": persona,
                                "relevance": relevance,
                                "source": "openai"
                            })

        # Step 3: Normalize and canonicalize the capability map
        print("📊 [Graph] Canonicalizing capability map...")
        capability_map = convert_rule_matches_to_capability_map(flattened_capability_map)
        flattened_results = process_capability_map_to_graph(capability_map)
        
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
