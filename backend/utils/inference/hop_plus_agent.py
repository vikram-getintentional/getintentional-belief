"""


"""



# Commenting out everything - feel free to reuse or kill
"""
import json
from typing import List, Dict, Any, Set
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.hop_plus_openai import get_upstream_triplets
from backend.utils.graph_base.graph import Graph
from backend.utils.knowledge_base.canonicalizer import canonicalize_job, canonicalize_pain, canonicalize_persona
from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
from backend.utils.graph_base.edges.edge_manager import add_edge

def infer_upstream_with_rules(
    product_subgraph: Graph,
    cap_threshold: float = 0.1,
    relevance_threshold: float = 0.5,
    max_depth: int = 5
):
    """
    Traverse up from product_id to capabilities, pains, jobs, personas.
    For each capability check that capability_centrality > cap_threshold (0.6)
    For each persona, calculate normalized cumulative relevance to product.
    If cumulative relevance > threshold, traverse jobs and upstream pains.
    If no upstream pains, use hop_plus_one_hop (GPT fallback).
    Recurse until cumulative relevance drops below threshold or max_depth reached.
    """
    product_id = product_subgraph.get_node_id("product", {})
    if not product_id:
        raise ValueError("Product ID not found in the provided subgraph.")
    product_subgraph.update_capability_centralities()
    results = []
    visited_personas = set()

    def recurse_from_persona(persona_id, current_depth):
        if current_depth > max_depth or persona_id in visited_personas:
            return
        visited_personas.add(persona_id)
        cumulative_relevance = get_cumulative_relevance_data(product_id, persona_id)
        if cumulative_relevance < relevance_threshold:
            return

        persona_node = product_subgraph.get_node_by_id(persona_id)
        results.append({
            "persona_id": persona_id,
            "persona": persona_node,
            "cumulative_relevance": cumulative_relevance,
        })

        # Step 5: Get jobs "performed_by" this persona
        job_ids = product_subgraph.get_source_nodes_by_target_and_type(persona_id, "performed_by")
        if not job_ids:
            print("No jobs found for persona:", persona_id)
            return
        for job_id in job_ids:
            # Step 6: For each job, get upstream pains "impacted_by" this job
            upstream_pain_ids = product_subgraph.get_target_nodes_by_source_and_type(job_id, "impacts")
            if upstream_pain_ids:
                for up_pain_id in upstream_pain_ids:
                    # Step 7: For each upstream pain, get jobs and personas recursively
                    up_job_ids = product_subgraph.get_target_nodes_by_source_and_type(up_pain_id, "addresses")
                    for up_job_id in up_job_ids:
                        up_persona_ids = product_subgraph.get_target_nodes_by_source_and_type(up_job_id, "performed_by")
                        for up_persona_id in up_persona_ids:
                            recurse_from_persona(up_persona_id, current_depth + 1)
            else:
                # Step 7: If no impacted pains, use GPT fallback
                print(f"No upstream pains for job {job_id}, persona {persona_id}. Using GPT fallback.")
                gpt_results, _ = hop_plus_one_hop(
                    lower_nodes=[{
                        "persona_id": persona_id,
                        "job_id": job_id,
                        "cumulative_criticality": cum_relevance
                    }],
                    graph=product_subgraph,
                    threshold=relevance_threshold,
                    id_to_text={}
                )
                for gpt_node in gpt_results:
                    gpt_persona_id = gpt_node.get("persona_id")
                    if gpt_persona_id:
                        recurse_from_persona(gpt_persona_id, current_depth + 1)

    # Step 2: Get capability nodes with centrality > cap_threshold
    for node in product_subgraph.node_registry.values():
        if node.get("type") == "capability" and node.get("capability_centrality", 0.0) >= cap_threshold:
            cap_id = node["id"]
            # Step 3: Traverse up to pains, jobs, personas
            pain_ids = product_subgraph.get_target_nodes_by_source_and_type(cap_id, "solves")
            for pain_id in pain_ids:
                job_ids = product_subgraph.get_target_nodes_by_source_and_type(pain_id, "addresses")
                for job_id in job_ids:
                    persona_ids = product_subgraph.get_target_nodes_by_source_and_type(job_id, "performed_by")
                    for persona_id in persona_ids:
                        recurse_from_persona(persona_id, current_depth=1)

    return results



# hop_plus_one_hop and add_hop_subgraph remain unchanged

def hop_plus_one_hop(lower_nodes, graph, threshold, id_to_text):
    """
    Processes one hop: collects upstream nodes from the graph, batches GPT requests for missing nodes.
    Returns (all_upstream_nodes, gpt_cache_for_next_hop)
    """
    all_upstream_nodes = []
    gpt_cache = []

    for node in lower_nodes:
        persona_id = node["persona_id"]
        job_id = node["job_id"]
        pain_id = node["pain_id"]
        cumulative_criticality = node["cumulative_criticality"]
        capability = node.get("capability", "")

        node_key = json.dumps({"persona_id": persona_id, "job_id": job_id, "pain_id": pain_id}, sort_keys=True)

        # Only proceed if above threshold
        if cumulative_criticality <= threshold:
            print("Skipping node due to low criticality:", node_key, "with cumulative criticality:", cumulative_criticality)
            continue

        # Check if upstream jobs exist in the graph
        upstream_jobs = graph.get_upstream_jobs_by_id(pain_id)

        if upstream_jobs:
            for upstream in upstream_jobs:
                dep_persona_id = upstream["persona_id"]
                dep_job_id = upstream["job_id"]
                dep_pain_id = upstream["pain_id"]
                impact = upstream.get("impact") or graph.get_edge_weight(pain_id, dep_job_id) or 0.5
                all_upstream_nodes.append({
                    "persona_id": dep_persona_id,
                    "job_id": dep_job_id,
                    "pain_id": dep_pain_id,
                    "impact": float(impact),
                    "from_pain_id": pain_id,
                    "from_job_id": job_id,
                    "from_persona_id": persona_id,
                    "cumulative_criticality": cumulative_criticality * float(impact),
                    "capability": capability
                })
        else:
            # Only add to GPT cache if not in graph and above threshold
            gpt_cache.append({
                "persona": id_to_text.get(persona_id, {}),
                "job": id_to_text.get(job_id, ""),
                "pain": id_to_text.get(pain_id, ""),
                "cumulative_criticality": cumulative_criticality,
                "capability": capability
            })

    # Batch GPT call for all missing upstreams
    if gpt_cache:
        gpt_upstreams = hop_plus_gpt_lookup(gpt_cache)
        # After canonicalization, create/get nodes and get their IDs
        for triplet in gpt_upstreams:
            persona = triplet["persona"]
            job = triplet["job"]
            pain = triplet["pain"]
            persona_node = get_or_create_persona_node(persona.get("title"), persona.get("seniority"), persona.get("department"))
            job_node = get_or_create_job_node(job)
            pain_node = get_or_create_pain_node(pain)
            id_to_text[persona_node["id"]] = persona
            id_to_text[job_node["id"]] = job
            id_to_text[pain_node["id"]] = pain
            all_upstream_nodes.append({
                "persona_id": persona_node["id"],
                "job_id": job_node["id"],
                "pain_id": pain_node["id"],
                "impact": triplet.get("impact", 1.0),
                "from_pain_id": triplet.get("from_pain_id"),
                "from_job_id": triplet.get("from_job_id"),
                "from_persona_id": triplet.get("from_persona_id"),
                "cumulative_criticality": triplet.get("cumulative_criticality", 1.0),
                "capability": triplet.get("capability", "")
            })
        for triplet in gpt_upstreams:
            add_hop_subgraph(
                gpt_upstreams=[triplet],
                id_to_text=id_to_text,
                from_pain_id=triplet.get("from_pain_id"),
                source="hop_plus_gpt"
            )

    print("Total upstream nodes found in this hop:", len(all_upstream_nodes))
    return all_upstream_nodes, gpt_cache

def add_hop_subgraph(
    gpt_upstreams: list,
    id_to_text: dict,
    from_pain_id: str = None,
    from_job_id: str = None,
    from_persona_id: str = None,
    source: str = "hop_plus_gpt"
):
    """
    Adds new upstream nodes and edges to the graph based on GPT results.
    Optionally links them to the original pain/job/persona nodes.
    """
    from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node
    from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
    from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
    from backend.utils.graph_base.edges.edge_manager import add_edge
    from datetime import datetime, timezone

    results = []
    now = datetime.now(timezone.utc).isoformat()

    for triplet in gpt_upstreams:
        persona = triplet["persona"]
        job = triplet["job"]
        pain = triplet["pain"]
        impact = triplet.get("impact", 1.0)
        pain_importance = triplet.get("pain_importance", 1.0)
        capability = triplet.get("capability", "")
        job_criticality = triplet.get("job_criticality", 1.0)

        persona_node = get_or_create_persona_node(persona.get("title"), persona.get("seniority"), persona.get("department"))
        job_node = get_or_create_job_node(job)
        pain_node = get_or_create_pain_node(pain)

        id_to_text[persona_node["id"]] = persona
        id_to_text[job_node["id"]] = job
        id_to_text[pain_node["id"]] = pain

        # Add edges for the upstream subgraph
        add_edge(
            source_id=pain_node["id"],
            target_id=job_node["id"],
            edge_type="addresses",
            weight=pain_importance,
            last_updated=now,
            source=source
        )
        add_edge(
            source_id=job_node["id"],
            target_id=persona_node["id"],
            edge_type="performed_by",
            weight=job_criticality,
            last_updated=now,
            source=source
        )
        # Optionally, link to the original nodes (if provided)
        # Logic if given a "source job" that impacts a "target pain" in upstream traversal
        if triplet.get("from_job_id"):
            add_edge(
                source_id=triplet["from_job_id"],
                target_id=pain_node["id"],
                edge_type="impacts",
                weight=impact,
                last_updated=now,
                source=source
            )
        
        # Logic if given a "source pain" that is impacted_by a "target job" in downstream traversal
        if from_pain_id:
            add_edge(
                source_id=from_pain_id,
                target_id=job_node["id"],
                edge_type="impacts",
                weight=job_criticality,
                last_updated=now,
                source=source
            )

        results.append({
            "persona_id": persona_node["id"],
            "job_id": job_node["id"],
            "pain_id": pain_node["id"],
            "impact": impact,
            "from_pain_id": from_pain_id,
            "from_job_id": from_job_id,
            "from_persona_id": from_persona_id,
            "cumulative_criticality": triplet.get("cumulative_criticality", 1.0),
            "capability": capability
        })
    return results

def hop_plus_gpt_lookup(gpt_cache: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gets higher level persona/job/pain from GPT and canonicalizes results.
    """
    print("Starting hop_plus_gpt_lookup with cache")
    upstream_triplets = []
    persona_cache = set()
    pain_cache = set()
    job_cache = set()

    for entry in gpt_cache:
        print("Processing entry in hop_plus_gpt_lookup:", entry)
        persona = entry["persona"]
        job = entry["job"]
        parent_cumulative_criticality = entry.get("cumulative_criticality", 1.0)
        capability = entry.get("capability", "")
        gpt_results = get_upstream_triplets(persona, job)
        print("GPT results for persona/job:", gpt_results)
        for result in gpt_results:
            original_job = result.get("original_job", "")
            dependent_pains = result.get("dependent_pains", [])
            for pain_obj in dependent_pains:
                pain = pain_obj.get("pain", "")
                pain_impact = float(pain_obj.get("pain_impact", 1.0))
                pain_trigger = pain_obj.get("pain_trigger", "")
                dependent_jobs = pain_obj.get("dependent_jobs", [])
                for job_obj in dependent_jobs:
                    job_desc = job_obj.get("description", "")
                    dependent_personas = job_obj.get("dependent_personas", [])
                    for persona_obj in dependent_personas:
                        canonical_dep_persona = persona_obj
                        canonical_dep_job = job_desc
                        canonical_dep_pain = pain
                        impact = pain_impact

                        # Calculate cumulative_criticality for this triplet
                        new_cumulative_criticality = parent_cumulative_criticality * impact


                        # Add caches
                        persona_cache.add(json.dumps(canonical_dep_persona, sort_keys=True))
                        pain_cache.add(canonical_dep_pain)
                        job_cache.add(canonical_dep_job)
                        upstream_triplets.append({
                            "persona": canonical_dep_persona,
                            "job": canonical_dep_job,
                            "pain": canonical_dep_pain,
                            "impact": impact,
                            "pain_trigger": pain_trigger,
                            "from_persona": persona,
                            "from_job": job,
                            "from_pain": entry.get("pain", ""),
                            "cumulative_criticality": new_cumulative_criticality,
                            "capability": capability
                        })
                        print("Added triplet:", upstream_triplets[-1])

    # Canonicalize caches
    print("Canonicalizing persona, job, and pain caches")
    persona_dict_map = {json.dumps(json.loads(p), sort_keys=True): json.loads(p) for p in persona_cache}
    persona_cache_list = list(persona_dict_map.values())
    pain_cache_list = list(set(pain_cache))
    job_cache_list = list(set(job_cache))
    canonical_dep_persona = canonicalize_persona(persona_cache_list)
    canonical_dep_job = canonicalize_job(job_cache_list)
    canonical_dep_pain = canonicalize_pain(pain_cache_list)

    # Build mapping
    persona_map = {json.dumps(orig, sort_keys=True): canon for orig, canon in zip(persona_cache_list, canonical_dep_persona.values())}
    job_map = {orig: canon for orig, canon in zip(job_cache_list, canonical_dep_job)}
    pain_map = {orig: canon for orig, canon in zip(pain_cache_list, canonical_dep_pain)}

    # Update triplets with canonicalized values (robust to missing keys)
    for triplet in upstream_triplets:
        persona_key = json.dumps(triplet["persona"], sort_keys=True)
        triplet["persona"] = persona_map.get(persona_key, triplet["persona"])
        triplet["job"] = job_map.get(triplet["job"], triplet["job"])
        triplet["pain"] = pain_map.get(triplet["pain"], triplet["pain"])
    return upstream_triplets

"""