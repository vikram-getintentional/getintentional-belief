from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
from backend.utils.graph_base.nodes.pain_trigger_nodes import get_or_create_pain_trigger_node
from backend.utils.graph_base.edges.edge_manager import add_edge
from collections import defaultdict
from datetime import datetime
from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node

def process_capability_map_to_graph(capability_map: dict):
    flattened_results = []
    print("Starting process map")
    for cap, pains in capability_map.items():
        capability_node = get_or_create_capability_node(cap, "")

        for pain_text, jobs in pains.items():
            pain_node = get_or_create_pain_node(pain_text)

            # Add a "solves" edge for each persona entry (relevance is already maxed in upstream logic)
            for job_desc, persona_list in jobs.items():
                for persona_entry in persona_list:
                    relevance = persona_entry.get("relevance", 0.5)
                    if relevance > 0.0:
                        now = datetime.utcnow().isoformat()
                        add_edge(
                            source_id=capability_node["id"],
                            target_id=pain_node["id"],
                            edge_type="solves",
                            weight=relevance,
                            last_updated=now,
                            source=persona_entry.get("source", "openai")
                        )

            for job_desc, persona_list in jobs.items():
                job_node = get_or_create_job_node(job_desc)

                for persona_entry in persona_list:
                    persona = persona_entry["persona"]
                    relevance = persona_entry.get("relevance", 0.5)
                    source = persona_entry.get("source", "unknown")
                    pain_trigger = persona_entry.get("pain_trigger", None)
                    now = datetime.utcnow().isoformat()

                    # Add edge: Pain → Job (weight = 1)
                    add_edge(
                        source_id=pain_node["id"],
                        target_id=job_node["id"],
                        edge_type="addresses",
                        weight=1,
                        last_updated=now,
                        source=source
                    )

                    # Add edge: Job → Persona (weight = 1)
                    persona_node = get_or_create_persona_node(
                        persona["title"],
                        persona["seniority"],
                        persona["department"]
                    )
                    add_edge(
                        source_id=job_node["id"],
                        target_id=persona_node["id"],
                        edge_type="performed_by",
                        weight=1,
                        last_updated=now,
                        source=source
                    )

                    # Add edge: Pain → Pain Trigger (weight = 1)
                    if pain_trigger:
                        pain_trigger_node = get_or_create_pain_trigger_node(pain_trigger)
                        add_edge(
                            source_id=pain_node["id"],
                            target_id=pain_trigger_node["id"],
                            edge_type="triggered_by",
                            weight=1,
                            last_updated=now,
                            source=source
                        )

                    # Add to flattened results
                    flattened_results.append({
                        "capability": cap,
                        "pain": pain_text,
                        "job": job_desc,
                        "persona": persona,
                        "relevance": relevance,
                        "source": source,
                        "pain_trigger": pain_trigger
                    })

    return flattened_results



def convert_rule_matches_to_capability_map(results):
    from backend.utils.knowledge_base.canonicalizer import (
        canonicalize_pain,
        canonicalize_job,
        canonicalize_persona,
        canonicalize_pain_trigger
    )

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
            "relevance": entry.get("relevance", 0.5),
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

    new_capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for cap, pains in capability_map.items():
        for pain_text, jobs in pains.items():
            canonical_pain = canonical_pains.get(pain_text, pain_text)
            canonical_pain_trigger = canonical_pain_triggers.get(pain_text, pain_text)
            for job_desc, persona_list in jobs.items():
                canonical_job = canonical_jobs.get(job_desc, job_desc)
                for persona_entry in persona_list:
                    persona = persona_entry["persona"]
                    key = f"{persona['title']}|{persona['department']}|{persona['seniority']}"
                    canonical_persona = canonical_personas.get(key, persona)
                    new_capability_map[cap][canonical_pain][canonical_job].append({
                        "persona": canonical_persona,
                        "relevance": persona_entry.get("relevance", 0.5),
                        "source": persona_entry.get("source", "unknown"),
                        "pain_trigger": canonical_pain_trigger
                    })
    return new_capability_map