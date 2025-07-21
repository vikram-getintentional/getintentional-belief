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
            """
            Canonicalized entry structure:
            {
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
                "pain_node_id": pain_id,
                "job_node_id": job_id,
                "persona_node_id": persona_id,
                "pain_trigger_node_id": pain_trigger_id
            }
            """
            print("Processing entry:", entry)
            capability_id = entry.get("capability_id")
            pain_id = entry.get("pain_node_id")
            relevance_array = entry.get("relevance", [])
            job_id = entry.get("job_node_id")
            persona_node_id = entry.get("persona_node_id")
            pain_trigger_node_id = entry.get("pain_trigger_node_id", None)
            job_impact = entry.get("job_impact", 0.5)
            persona_job_importance = entry.get("persona_job_importance", 0.5)
            source = entry.get("source", "unknown")
            now = datetime.utcnow().isoformat()
            

            
            # Looks through cap_list array and sets an edge for each cap to this pain combo with weight = relevance_array[i]
            for i, cap in enumerate(capabilities_list):
                cap_id = cap["id"]
                relevance = relevance_array[i] if i < len(relevance_array) else 0
                # Add edge: Capability → Pain
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
                weight=job_impact,
                last_updated=now,
                source=source
            )

            # Add edge: Job → Persona
            add_edge(
                source_id=job_id,
                target_id=persona_node_id,
                edge_type="performed_by",
                weight=persona_job_importance,
                last_updated=now,
                source=source
            )

            # Add edge: Pain → Pain Trigger
            add_edge(
                source_id=pain_id,
                target_id=pain_trigger_node_id,
                edge_type="triggered_by",
                weight=1.0,
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

    # Gather unique raw values for canonicalization
    for entry in results:
        raw_pain = entry.get("pain", "").strip()
        raw_job = entry.get("job", "").strip()
        raw_pain_trigger = entry.get("pain_trigger", {})
        raw_persona = entry.get("persona", {})

        pain_cache.add(raw_pain)
        job_cache.add(raw_job)
        if isinstance(raw_persona, dict):
            persona_cache.add((
                raw_persona.get("title", "").strip(),
                raw_persona.get("department", "").strip(),
                raw_persona.get("seniority", "").strip()
            ))
        if isinstance(raw_pain_trigger, dict):
            trigger_cache.add((
                raw_pain_trigger.get("attribute", "").strip(),
                raw_pain_trigger.get("dimension", "").strip(),
                raw_pain_trigger.get("direction", "").strip()
            ))

    # Canonicalize
    canonical_personas = canonicalize_persona([
        {"title": t, "department": d, "seniority": s}
        for (t, d, s) in persona_cache
    ])
    canonical_pains = canonicalize_pain(list(pain_cache))
    canonical_jobs = canonicalize_job(list(job_cache))
    canonical_pain_triggers = canonicalize_pain_trigger([
        {"attribute": a, "dimension": d, "direction": s}
        for (a, d, s) in trigger_cache
    ])

    # Lookup logic for setting right node IDs to entry list
    persona_lookup = {
        (p["title"].strip().lower(), p["department"].strip().lower(), p["seniority"].strip().lower()):
            get_or_create_persona_node(p["title"], p["department"], p["seniority"])["id"]
        for p in canonical_personas.values()
    }
    pain_lookup = {
        p.strip().lower(): get_or_create_pain_node(p)["id"]
        for p in canonical_pains.values()
    }
    job_lookup = {
        j.strip().lower(): get_or_create_job_node(j)["id"]
        for j in canonical_jobs.values()
    }
    pain_trigger_lookup = {
        (pt["attribute"].strip().lower(), pt["dimension"].strip().lower(), pt["direction"].strip().lower()):
            get_or_create_pain_trigger_node(pt["attribute"], pt["dimension"], pt["direction"])["id"]
        for pt in canonical_pain_triggers.values()
    }

    # Assign canonical values and node IDs to each entry
    for entry in results:
        # Canonicalize pain
        raw_pain = entry.get("pain", "").strip()
        pain_canonical = canonical_pains.get(raw_pain, raw_pain)
        entry["pain"] = pain_canonical

        # Canonicalize job
        raw_job = entry.get("job", "").strip()
        job_canonical = canonical_jobs.get(raw_job, raw_job)
        entry["job"] = job_canonical

        # Canonicalize persona
        raw_persona = entry.get("persona", {})
        persona_key = str({
            "title": raw_persona.get("title", "").strip(),
            "department": raw_persona.get("department", "").strip(),
            "seniority": raw_persona.get("seniority", "").strip()
        })
        persona_canonical = canonical_personas.get(persona_key, raw_persona)
        entry["persona"] = persona_canonical

        # Canonicalize pain_trigger
        raw_pain_trigger = entry.get("pain_trigger", {})
        pain_trigger_key = str({
            "attribute": raw_pain_trigger.get("attribute", "").strip(),
            "dimension": raw_pain_trigger.get("dimension", "").strip(),
            "direction": raw_pain_trigger.get("direction", "").strip()
        })
        pain_trigger_canonical = canonical_pain_triggers.get(pain_trigger_key, raw_pain_trigger)
        entry["pain_trigger"] = pain_trigger_canonical

        # Lookup node IDs using canonicalized values
        persona_lookup_key = (
            entry["persona"]["title"].strip().lower(),
            entry["persona"]["department"].strip().lower(),
            entry["persona"]["seniority"].strip().lower()
        )
        if persona_lookup_key not in persona_lookup:
            print("❌ Persona lookup failed for key:", persona_lookup_key)
            print("Available persona keys:", list(persona_lookup.keys()))

        pain_lookup_key = entry["pain"].strip().lower()
        if pain_lookup_key not in pain_lookup:
            print("❌ Pain lookup failed for key:", pain_lookup_key)
            print("Available pain keys:", list(pain_lookup.keys()))

        job_lookup_key = entry["job"].strip().lower()
        if job_lookup_key not in job_lookup:
            print("❌ Job lookup failed for key:", job_lookup_key)
            print("Available job keys:", list(job_lookup.keys()))

        pain_trigger_lookup_key = (
            entry["pain_trigger"].get("attribute", "").strip().lower(),
            entry["pain_trigger"].get("dimension", "").strip().lower(),
            entry["pain_trigger"].get("direction", "").strip().lower()
        )
        if pain_trigger_lookup_key not in pain_trigger_lookup:
            print("❌ Pain trigger lookup failed for key:", pain_trigger_lookup_key)
            print("Available pain trigger keys:", list(pain_trigger_lookup.keys()))

        entry["pain_node_id"] = pain_lookup.get(pain_lookup_key)
        entry["job_node_id"] = job_lookup.get(job_lookup_key)
        entry["persona_node_id"] = persona_lookup.get(persona_lookup_key)
        entry["pain_trigger_node_id"] = pain_trigger_lookup.get(pain_trigger_lookup_key)

    print("Updated results blob:", results)
    return results