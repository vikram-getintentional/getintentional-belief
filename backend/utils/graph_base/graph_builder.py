from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
from backend.utils.graph_base.nodes.capability_nodes import get_or_create_capability_node
from backend.utils.graph_base.edges.edge_manager import add_edge
from collections import defaultdict

from backend.utils.knowledge_base.canonicalizer import (
    canonicalize_persona,
    canonicalize_job,
    canonicalize_pain
)


def process_capability_map_to_graph(capability_map: list):
    flattened_results = []  # Initialize flattened results
    print("capability map input to process cap: ", capability_map)
    # Add nodes and edges to the graph
    for cap, pains in capability_map.items():
        # Create or retrieve the capability node
        print("capability: ", cap, "cap type: ", type(cap))
        capability_node = get_or_create_capability_node(cap, "")

        for pain_text, jobs in pains.items():
            pain_node = get_or_create_pain_node(pain_text)

            # Add edge: Capability → Pain
            add_edge(
                source_id=capability_node["id"],
                target_id=pain_node["id"],
                edge_type="solves",
                weight=1
            )

            for job_desc, persona_list in jobs.items():
                job_node = get_or_create_job_node(job_desc)

                # Add edge: Pain → Job
                add_edge(
                    source_id=pain_node["id"],
                    target_id=job_node["id"],
                    edge_type="addresses",
                    weight=1
                )

                for persona in persona_list:
                    persona_node = get_or_create_persona_node(
                        persona["title"],
                        persona["seniority"],
                        persona["department"]
                    )

                    # Add edge: Job → Persona
                    add_edge(
                        source_id=job_node["id"],
                        target_id=persona_node["id"],
                        edge_type="performed_by",
                        weight=1
                    )

                    # Add to flattened results
                    flattened_results.append({
                        "capability": cap,
                        "pain": pain_text,
                        "job": job_desc,
                        "persona": persona
                    })

    return flattened_results


def convert_rule_matches_to_capability_map(results):
    from backend.utils.knowledge_base.canonicalizer import (
        canonicalize_pain,
        canonicalize_job,
        canonicalize_persona
    )

    # Initialize caches
    pain_cache = set()
    job_cache = set()
    persona_cache = set()
    capability_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Step 1: Collect pains, jobs, and personas
    for entry in results:
        print("entries in convert rule matches: ", entry)
        capability = entry.get("capability", "Unknown Capability").strip()

        # Update raw_pain extraction
        raw_pain = entry.get("pain", entry.get("original_pain", entry.get("canonical_pain", "Unknown Pain"))).strip().lower()

        # Update raw_job extraction
        raw_job = entry.get("job", entry.get("original_job", entry.get("canonical_job", "Unknown Job"))).strip().lower()

        # Normalize persona
        persona = entry.get("persona", "")
        if isinstance(persona, dict):
            title = persona.get("title", "").strip().lower()
            dept = persona.get("department", "Unknown").strip().lower()
            seniority = persona.get("seniority", "Unknown").strip().lower()
        elif isinstance(persona, str):
            title = persona.strip().lower()
            dept = entry.get("department", "Unknown").strip().lower()
            seniority = entry.get("seniority", "Unknown").strip().lower()
        else:
            title = "unknown"
            dept = "unknown"
            seniority = "unknown"

        # Add to caches
        pain_cache.add(raw_pain)
        job_cache.add(raw_job)
        persona_cache.add((title, dept, seniority))

        # Add raw persona to the capability map
        persona = {"title": title, "department": dept, "seniority": seniority}
        persona_list = capability_map[capability][raw_pain][raw_job]
        if all(
            p["title"] != persona["title"] or
            p["department"] != persona["department"] or
            p["seniority"] != persona["seniority"]
            for p in persona_list
        ):
            persona_list.append(persona)

    # Step 2: Canonicalize pains, jobs, and personas
    print("Graph Builder : Convert_Rule_- canonicalizing personas")
    canonical_personas = canonicalize_persona(
        [{"title": t, "department": d, "seniority": s} for t, d, s in persona_cache]
    )
    print("Graph Builder : Convert_Rule_- canonicalizing pains")
    canonical_pains = canonicalize_pain(list(pain_cache))
    print("Graph Builder : Convert_Rule_- canonicalizing jobs")
    canonical_jobs = canonicalize_job(list(job_cache))

    # Step 3: Replace raw data with canonicalized data
    for cap, pains in capability_map.items():
        for pain_text, jobs in pains.items():
            canonical_pain = canonical_pains.get(pain_text, pain_text)
            for job_desc, persona_list in jobs.items():
                canonical_job = canonical_jobs.get(job_desc, job_desc)
                for i, persona in enumerate(persona_list):
                    canonical_persona = canonical_personas.get(
                        f"{persona['title']}|{persona['department']}|{persona['seniority']}",
                        persona  # Fallback to raw persona if not found
                    )
                    persona_list[i] = canonical_persona

    # Return the constructed capability map
    return capability_map