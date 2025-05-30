import json
from typing import List, Dict, Any, Set
from backend.utils.inference.hop_plus_openai import get_upstream_triplets
from backend.utils.graph_base.graph import Graph
from backend.utils.knowledge_base.canonicalizer import canonicalize_job, canonicalize_pain, canonicalize_persona
from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
from backend.utils.graph_base.edges.edge_manager import add_edge

def infer_upstream_with_rules(base_nodes: List[Dict[str, Any]], graph: Graph, threshold: float = 0.2, max_depth: int = 3):
    """
    Entry point for Hop+ agentic traversal.
    Enriches base_nodes with cumulative_criticality and calls upstream_node_builder.
    """
    print("Starting Hop+ with Base Nodes:", base_nodes)
    enriched_nodes = []
    for node in base_nodes:
        persona = node["persona"]
        job = node["job"]
        pain = node["pain"]
        capability = node.get("capability", "")

        # Get edge weight (relevance)
        capability_id = graph.get_node_id("capability", capability)
        pain_id = graph.get_node_id("pain", pain)
        relevance = graph.get_edge_weight(capability_id, pain_id)
        if relevance is None:
            relevance = 0.5  # default

        enriched_nodes.append({
            "persona": persona,
            "job": job,
            "pain": pain,
            "cumulative_criticality": float(relevance),
            "capability": capability
        })
    
    print("Enriched Nodes set for upstream traversal:", enriched_nodes)
    return upstream_node_builder(enriched_nodes, graph, threshold, max_depth=max_depth, visited=set())

def upstream_node_builder(lower_nodes: List[Dict[str, Any]], graph: Graph, threshold: float, max_depth: int, visited: Set[str], retry_attempt: int = 0):
    """
    Traverses the graph and gets existing upstream nodes if they exist.
    If not, collects missing data and calls hop_plus_gpt_lookup.
    Returns a set of final upstream nodes.
    """
    if max_depth <= 0:
        return []

    all_upstream_nodes = []
    gpt_cache = []

    for node in lower_nodes:
        persona = node["persona"]
        job = node["job"]
        pain = node["pain"]
        cumulative_criticality = node["cumulative_criticality"]
        capability = node.get("capability", "")

        node_key = json.dumps({"persona": persona, "job": job, "pain": pain}, sort_keys=True)
        if node_key in visited:
            continue
        visited.add(node_key)

        parent_pain_node = get_or_create_pain_node(pain)
        upstream_jobs = graph.get_upstream_jobs(pain)

        if upstream_jobs:
            for upstream in upstream_jobs:
                dep_persona = upstream["persona"]
                dep_job = upstream["job"]
                dep_pain = upstream["pain"]
                dep_job_node = get_or_create_job_node(dep_job)
                impact = upstream.get("impact") or graph.get_edge_weight(parent_pain_node["id"], dep_job_node["id"]) or 0.5
                all_upstream_nodes.append({
                    "persona": dep_persona,
                    "job": dep_job,
                    "pain": dep_pain,
                    "impact": float(impact),
                    "cumulative_criticality": cumulative_criticality * float(impact),
                    "from_pain": pain,
                    "from_job": job,
                    "from_persona": persona,
                    "capability": capability
                })
            print("Existing upstream triplets from graph:", all_upstream_nodes)
        else:
            print("No upstream jobs found for:", node, ". Adding to GPT cache...")
            gpt_cache.append({
                "persona": persona,
                "job": job,
                "pain": pain,
                "cumulative_criticality": cumulative_criticality,
                "capability": capability
            })

    
    if gpt_cache and retry_attempt < 2:
        print("Running GPT lookup for cache. Retry attempt:", retry_attempt)
        gpt_upstreams = hop_plus_gpt_lookup(gpt_cache)
        print("Starting Upstream Node Creation with GPT results:", gpt_upstreams)
        all_upstream_nodes.extend(
            hop_plus_node_creator(gpt_upstreams, graph, threshold, max_depth, visited, retry_attempt + 1)
        )

    return all_upstream_nodes

def hop_plus_gpt_lookup(gpt_cache: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gets higher level persona/job/pain from GPT and canonicalizes results.
    """
    print("Starting hop_plus_gpt_lookup with cache:")
    for ent in gpt_cache:
        print("Cache Entry:", ent, "of type", type(ent))
        for key in ent:
            print(f"Key: {key}, Value: {ent[key]}, Type: {type(ent[key])}")
    print("Starting upstream processing with this data")
    upstream_triplets = []
    persona_cache = set()
    pain_cache = set()
    job_cache = set()

    for entry in gpt_cache:
        print("Processing entry:", entry, "of type", type(entry))
        persona = entry["persona"]
        job = entry["job"]
        gpt_results = get_upstream_triplets(persona, job)
        print("GPT Results", gpt_results, "of type", type(gpt_results))
        for triplet in gpt_results:
            print("Triplet from GPT:", triplet, "of type", type(triplet))
            canonical_dep_persona = triplet["dependent_persona"]
            canonical_dep_job = triplet["dependent_job"]
            canonical_dep_pain = triplet["dependent_pain"]
            impact = float(triplet.get("dependent_pain_impact", 1.0))
            persona_cache.add(json.dumps(canonical_dep_persona, sort_keys=True))
            pain_cache.add(canonical_dep_pain)
            job_cache.add(canonical_dep_job)
            upstream_triplets.append({
                "persona": canonical_dep_persona,
                "job": canonical_dep_job,
                "pain": canonical_dep_pain,
                "impact": impact,
                "from_persona": persona,
                "from_job": job,
                "from_pain": entry["pain"],
                "cumulative_criticality": entry["cumulative_criticality"],
                "capability": entry["capability"]
            })

    # Canonicalize caches
    print("Creating dict maps for canonicalization")
    persona_dict_map = {json.dumps(json.loads(p), sort_keys=True): json.loads(p) for p in persona_cache}
    persona_cache_list = list(persona_dict_map.values())
    pain_cache_list = list(set(pain_cache))
    job_cache_list = list(set(job_cache))
    print("Canonicalizing persona, job, pain with caches")
    canonical_dep_persona = canonicalize_persona(persona_cache_list)
    canonical_dep_job = canonicalize_job(job_cache_list)
    canonical_dep_pain = canonicalize_pain(pain_cache_list)
    print("Canonicalization completed in hop_plus_gpt_lookup. Now doing the mapping thing")

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
        print("Updated Persona:", triplet["persona"], "of type", type(triplet["persona"]))
        print("Updated Job:", triplet["job"], "of type", type(triplet["job"]))
        print("Updated Pain:", triplet["pain"], "of type", type(triplet["pain"]))

    print("Final upstream triplets after GPT lookup:", upstream_triplets)
    for tri in upstream_triplets:
        print("Triplet Persona: ", tri["persona"], "of type", type(tri["persona"]))
        print("Triplet Job: ", tri["job"], "of type", type(tri["job"]))
        print("Triplet Pain: ", tri["pain"], "of type", type(tri["pain"]))

    return upstream_triplets

def hop_plus_node_creator(upstream_triplets: List[Dict[str, Any]], graph: Graph, threshold: float, max_depth: int, visited: Set[str], retry_attempt: int):
    """
    Gets or creates nodes from canonical output, adds edges, and recurses upstream_node_builder.
    """
    all_nodes = []
    for triplet in upstream_triplets:
        dep_persona_node = get_or_create_persona_node(
            triplet["persona"].get("title"),
            triplet["persona"].get("seniority"),
            triplet["persona"].get("department")
        )
        dep_job_node = get_or_create_job_node(triplet["job"])
        dep_pain_node = get_or_create_pain_node(triplet["pain"])
        parent_pain_node = get_or_create_pain_node(triplet["from_pain"])
        add_edge(source_id=dep_job_node["id"], target_id=dep_persona_node["id"], edge_type="performed_by", weight=1)
        add_edge(source_id=dep_pain_node["id"], target_id=dep_job_node["id"], edge_type="addresses", weight=1)
        add_edge(
            source_id=parent_pain_node["id"],
            target_id=dep_job_node["id"],
            edge_type="impacts",
            weight=triplet["impact"]
        )
        # Prepare node for next recursion
        node = {
            "persona": triplet["persona"],
            "job": triplet["job"],
            "pain": triplet["pain"],
            "cumulative_criticality": triplet["cumulative_criticality"] * triplet["impact"],
            "capability": triplet["capability"]
        }
        all_nodes.append(node)
    # Recurse
    if max_depth > 0:
        all_nodes.extend(
            upstream_node_builder(all_nodes, graph, threshold, max_depth - 1, visited, retry_attempt)
        )
    return all_nodes