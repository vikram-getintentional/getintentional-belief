from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
from backend.utils.graph_base.nodes.pain_trigger_nodes import get_or_create_pain_trigger_node
from backend.utils.graph_base.edges.edge_manager import add_edge
from collections import defaultdict
from datetime import datetime
from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node

# This function creates edges for the node IDs in the flattened list
def process_capability_map_to_graph(capabilities_list, capability_map: dict):
    flattened_results = []
    print("Starting process map")
    for entry in capability_map:
            print("Processing entry:", entry)
            capability_id = entry.get("capability_id")
            pain_id = entry.get("pain_node_id")
            relevance_array = entry.get("relevance", [])
            job_id = entry.get("job_node_id")
            persona_node_id = entry.get("persona_node_id")
            source = entry.get("source", "unknown")
            now = datetime.utcnow().isoformat()
            

            # Add edge: Capability → Pain
            # Looks through cap_list array and sets an edge for each cap to this pain combo with weight = relevance_array[i]
            for i, cap in enumerate(capabilities_list):
                cap_id = cap["id"]
                relevance = relevance_array[i] if i < len(relevance_array) else 0
                
                add_edge(
                    source_id=cap_id,
                    target_id=pain_id,
                    edge_type="solves",
                    weight=relevance,
                    last_updated=now,
                    source=source
                )
                print(f"Added edge from Capability {cap_id} to Pain {pain_id} with weight {relevance}")
            # Add edge: Pain → Job
            add_edge(
                source_id=pain_id,
                target_id=job_id,
                edge_type="addresses",
                weight=1,
                last_updated=now,
                source=source
            )

            # Add edge: Job → Persona
            add_edge(
                source_id=job_id,
                target_id=persona_node_id,
                edge_type="performed_by",
                weight=1,
                last_updated=now,
                source=source
            )

            # (Optional) Add to flattened_results for downstream use
            flattened_results.append(entry)
    print("Finished final flattened results:", flattened_results)
    return flattened_results


# This function gets flattened gpt results and creates canonicalized nodes. Return is the flattened map with node IDs.
def convert_rule_matches_to_capability_map(results):
    from backend.utils.knowledge_base.canonicalizer import (
        canonicalize_pain,
        canonicalize_job,
        canonicalize_persona,
        canonicalize_pain_trigger
    )
    print("Converting rule matches to capability map...")
    pain_cache = set()
    job_cache = set()
    persona_cache = set()
    trigger_cache = set()
    capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for entry in results:
        capability_id = entry.get("capability_id", "Unknown Capability").strip()
        raw_pain = entry.get("pain", entry.get("original_pain", entry.get("canonical_pain", "Unknown Pain"))).strip().lower()
        raw_job = entry.get("job", entry.get("original_job", entry.get("canonical_job", "Unknown Job"))).strip().lower()
        raw_pain_trigger = entry.get("pain_trigger", "").strip().lower()
        relevance_array = entry.get("relevance_array", [])

        persona = entry.get("persona", {})
        if isinstance(persona, dict):
            title = persona.get("title", "").strip().lower()
            dept = persona.get("department", "Unknown").strip().lower()
            seniority = persona.get("seniority", "Unknown").strip().lower()
        else:
            title = "unknown"
            dept = "unknown"
            seniority = "unknown"

        pain_cache.add(raw_pain)
        job_cache.add(raw_job)
        persona_cache.add((title, dept, seniority))
        trigger_cache.add(raw_pain_trigger)

        persona_entry = {
            "persona": {
                "title": title,
                "department": dept,
                "seniority": seniority
            },
            "relevance": entry.get("relevance_array"),
            "source": entry.get("source", "unknown"),
            "pain_trigger": raw_pain_trigger
        }

        persona_list = capability_map[capability_id][raw_pain][raw_job]
        # Only add if not already present
        if all(
            p["persona"]["title"] != persona_entry["persona"]["title"] or
            p["persona"]["department"] != persona_entry["persona"]["department"] or
            p["persona"]["seniority"] != persona_entry["persona"]["seniority"]
            for p in persona_list
        ):
            persona_list.append(persona_entry)

    # Canonicalization (as before)
    canonical_personas = canonicalize_persona(
        [{"title": t, "department": d, "seniority": s} for t, d, s in persona_cache]
    )
    canonical_pains = canonicalize_pain(list(pain_cache))
    canonical_jobs = canonicalize_job(list(job_cache))
    canonical_pain_triggers = canonicalize_pain_trigger(list(trigger_cache))

    # Add nodes
    for pain in canonical_pains.values():
        pain_node = get_or_create_pain_node(pain)
        pain_id = pain_node["id"]
    for pain_trigger in canonical_pain_triggers.values():
        pain_trigger_node = get_or_create_pain_trigger_node(pain_trigger)
        pain_trigger_id = pain_trigger_node["id"]
    for job in canonical_jobs.values():
        job_node = get_or_create_job_node(job)
        job_id = job_node["id"]
    for persona in canonical_personas.values():
        persona_node = get_or_create_persona_node(
            persona["title"], persona["department"], persona["seniority"]
        )
        persona_id = persona_node["id"]

    # update "results" to now include node IDs
    for entry in results:
        entry["pain_node_id"] = pain_id
        entry["job_node_id"] = job_id
        entry["persona_node_id"] = persona_id

    


    print("Updated results blob:" , results)
    return results