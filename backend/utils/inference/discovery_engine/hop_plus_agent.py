from collections import defaultdict
from datetime import datetime
import json
from typing import List, Dict, Any, Set
from backend.utils.graph_base.nodes.pain_trigger_nodes import get_or_create_pain_trigger_node
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data 
from backend.utils.inference.gpt_prompts.hop_plus_openai import get_upstream_triplets
from backend.utils.graph_base.graph import Graph
from backend.utils.knowledge_base.canonicalizer import canonicalize_job, canonicalize_pain, canonicalize_persona
from backend.utils.graph_base.nodes.persona_nodes import get_or_create_persona_node
from backend.utils.graph_base.nodes.job_nodes import get_or_create_job_node
from backend.utils.graph_base.nodes.pain_nodes import get_or_create_pain_node
from backend.utils.graph_base.edges.edge_manager import add_edge
from backend.utils.knowledge_base.canonicalizer import (
        canonicalize_pain,
        canonicalize_job,
        canonicalize_persona,
        canonicalize_pain_trigger
    )


def infer_upstream_with_rules(
    product_subgraph: Graph,
    cap_threshold: float = 0.1,
    relevance_threshold: float = 0.5,
    max_depth: int = 3
):

    #1. Get product ID, and initialize arrays/ sets
    product_id = product_subgraph.get_node_id("product", {})
    if not product_id:
        raise ValueError("Product ID not found in the provided subgraph.")
    product_subgraph.update_capability_centralities()
    results = []
    visited_triplets = set()
    triplet_holder = []
    gpt_triplet_cache = []

    #2. Get all capabilities and their IDs
    functional_capabilities_ids, _ = product_subgraph.set_capabilities_relevance(
        capability_threshold=cap_threshold
    )   
    #5. Recurses through next highest relevance triplet from triplet_holder to find upstream triplets. 
    # If no upstream exists, adds current triplet to gpt_Cache for processing. 
    def recurse_from_persona(triplet, current_depth):
        """
        Adds the triplet to visited set to avoid infinite recursion, and removes from triplet_holder.
        Need to ensure that recurse() stops when
        1. max_Depth is reached
        2. there are no more relevant nodes to traverse
        """
        print("Recursing triplet:", triplet)
        persona_id = triplet["persona_id"]
        job_id = triplet["job_id"]
        pain_id = triplet["pain_id"]
        up_nodes_available = False
        triplet_key = (
            persona_id,
            job_id,
            pain_id,
            triplet.get("cumulative_relevance", 1.0)
        )

        if current_depth > max_depth or triplet_key in visited_triplets:
            return
        visited_triplets.add(triplet_key)
        triplet_holder.remove(triplet)

        cumulative_relevance = get_cumulative_relevance_data(product_id, persona_id)
        if cumulative_relevance < relevance_threshold:
            return

        results.append({"triplet": triplet})
        print("Running before step 4")
        # Step 4: For the triplet job, get upstream pains "impacted_by" this job
        upstream_pain_ids = product_subgraph.get_target_nodes_by_source_and_type(job_id, "impacts")
        if upstream_pain_ids:
            for up_pain_id in upstream_pain_ids:
                # Step 5: For each upstream pain, get jobs and personas recursively
                up_job_ids = product_subgraph.get_target_nodes_by_source_and_type(up_pain_id, "addresses")
                if up_job_ids:
                    for up_job_id in up_job_ids:
                        up_persona_ids = product_subgraph.get_target_nodes_by_source_and_type(up_job_id, "performed_by")
                        up_persona_relevance = get_cumulative_relevance_data(product_id, up_persona_ids)
                        if up_persona_ids:
                            for up_persona_id in up_persona_ids:
                                up_triplet = {
                                    "persona_id": up_persona_id,
                                    "job_id": up_job_id,
                                    "pain_id": up_pain_id,
                                    "cumulative_relevance": up_persona_relevance}
                                triplet_holder.append(up_triplet)
                                up_nodes_available = True
                        else:
                            print("No personas found for job:", up_job_id)
                            up_nodes_available = False
                else:
                    print("No jobs found for upstream pain:", up_pain_id)
                    up_nodes_available = False
        else:
            print(f"No upstream pains for job {job_id}, persona {persona_id}. Using GPT fallback.")
            up_nodes_available = False
        if not up_nodes_available:
            gpt_triplet_cache.append(triplet)
            print("GPT Cache length:", len(gpt_triplet_cache))
        print("Process triplet holder length:", len(triplet_holder))
        process_triplet_holder(current_depth)

    # Only runs gpt cache when triplet holder gets empty.
    def process_gpt_cache(current_depth):
        """
        Processes the GPT triplet cache to generate upstream triplets using OpenAI.
        """
        print("Processing GPT cache with length:", len(gpt_triplet_cache))
        if not gpt_triplet_cache:
            print("GPT triplet cache is empty, skipping processing.")
            return
        gpt_results = get_hop_plus(gpt_triplet_cache, sub_graph=product_subgraph, threshold=relevance_threshold, id_to_text={})
        gpt_triplet_cache.clear()  # Clear cache after processing
        for gpt_triplet in gpt_results:
            triplet_holder.append(gpt_triplet)
        current_depth += 1
        process_triplet_holder(current_depth)
    
    #4. Sorts triplets and traverses the most relevant ones first. If triplet_holder is empty - runs gpt on the entire cache.
    def process_triplet_holder(current_depth=1):
        print("processing triplet holder with length:", len(triplet_holder))
        print("Full holder contents \n")
        for item in triplet_holder:
            print(item, "\n")
        print("Current depth:", current_depth)
        if not triplet_holder:
            if gpt_triplet_cache:
                process_gpt_cache(gpt_triplet_cache, current_depth)
            return results
        sorted_triplets = sorted(
            triplet_holder,
            key=lambda x: x.get("cumulative_relevance", 0.5),
            reverse=True
        )
        for triplet in sorted_triplets:
            recurse_from_persona(triplet, current_depth)
    
    # 3. Generate first set of Hop0 triplets and add them to triplet_holder set for processing
    ## triplet_holder works to run recursion on the most relevant triplets first
    for cap_id in functional_capabilities_ids:
        pain_ids = product_subgraph.get_target_nodes_by_source_and_type(cap_id, "solves")
        for pain_id in pain_ids:
            job_ids = product_subgraph.get_target_nodes_by_source_and_type(pain_id, "addresses")
            for job_id in job_ids:
                persona_ids = product_subgraph.get_target_nodes_by_source_and_type(job_id, "performed_by")
                for persona_id in persona_ids:
                    persona_relevance = get_cumulative_relevance_data(product_id, persona_id)
                    triplet = {
                        "persona_id": persona_id,
                        "job_id": job_id,
                        "pain_id": pain_id,
                        "cumulative_relevance": persona_relevance}
                    triplet_holder.append(triplet)
    process_triplet_holder(current_depth=1)
    
    return results


# get_hop_plus processes the cache and gets gpt upstream triplets as output.
# It calls 
def get_hop_plus(triplet_cache, sub_graph=Graph, threshold=0.5, id_to_text={}):
    """
    Processes one hop: collects upstream nodes from the graph, batches GPT requests for missing nodes.
    Returns (all_upstream_nodes, gpt_cache_for_next_hop)
    """
    all_upstream_nodes = []
    gpt_cache = []
    

    for triplet in triplet_cache:
        persona_id = triplet["persona_id"]
        job_id = triplet["job_id"]
        pain_id = triplet["pain_id"]
        relevance = triplet.get("cumulative_relevance", 1.0)
        
        persona_node = sub_graph.get_node_by_id(persona_id)
        job_node = sub_graph.get_node_by_id(job_id)
        pain_node = sub_graph.get_node_by_id(pain_id)
        if not persona_node or not job_node or not pain_node:
            print(f"Missing nodes for input triplet: {triplet}")
            continue
        persona = {
            "title": persona_node.get("title", ""),
            "department": persona_node.get("department", ""),
            "seniority": persona_node.get("seniority", "")
        }
        job = job_node.get("description", "")
        pain = pain_node.get("text", "")
        gpt_cache.append({
            "persona": persona,
            "job": job,
            "pain": pain,
            "relevance": relevance
        })
    gpt_upstreams = []
    if gpt_cache:
        gpt_upstreams = get_upstream_triplets(gpt_cache)
    flattened_results = []
    for entry in gpt_upstreams:
        original_job_id = entry.get("original_job_id")
        upstream_pains = entry.get("upstream_pains", [])
        for pain in upstream_pains:
            pain_desc = pain.get("pain", "")
            impact = pain.get("impact", 1.0)
            pain_trigger = pain.get("pain_trigger", {})
            upstream_jobs = pain.get("upstream_jobs", [])
            for job in upstream_jobs:
                job_desc = job.get("description", "")
                personas = job.get("personas", [])
                for persona in personas:
                    persona_title = persona.get("title", "")
                    persona_department = persona.get("department", "")
                    persona_seniority = persona.get("seniority", "")
                    # Create or get nodes for the triplet
                    flattened_results.append({
                        "original_job_id": original_job_id,
                        "pain": pain_desc,
                        "impact": impact,
                        "pain_trigger": pain_trigger,
                        "job": job_desc,
                        "persona": {
                            "title": persona_title,
                            "department": persona_department,
                            "seniority": persona_seniority
                        },
                        "source": "openAI"
                    })
        print("Flattened results from GPT upstreams:")
        canonicalized_results = canonicalize_and_add_nodes(flattened_results)
        # From canonicalized results create a set of triplets to return
        gpt_triplets = []
        for result in canonicalized_results:
            triplet = {
                "persona_id": result.get("persona_id"),
                "job_id": result.get("job_id"),
                "pain_id": result.get("pain_id"),
                "cumulative_relevance": result.get("cumulative_relevance", 1.0)
            }
            gpt_triplets.append(triplet)
        print("Generated GPT triplets:", gpt_triplets)
    return gpt_triplets

def canonicalize_and_add_nodes(flattened_results):
    """
    Canonicalizes the flattened results and adds nodes to the graph.
    Returns a list of upstream nodes with their IDs.
    """
    print("Converting flattened results to canonicalized nodes...")
    pain_cache = set()
    job_cache = set()
    persona_cache = set()
    trigger_cache = set()
    canonical_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for entry in flattened_results:
        raw_pain = entry.get("pain", entry.get("original_pain", entry.get("canonical_pain", "Unknown Pain"))).strip().lower()
        raw_job = entry.get("job", entry.get("original_job", entry.get("canonical_job", "Unknown Job"))).strip().lower()
        raw_pain_trigger = entry.get("pain_trigger", {})
        if isinstance(raw_pain_trigger, dict):
            trigger_tuple = (
                raw_pain_trigger.get("attribute", "").strip().lower(),
                raw_pain_trigger.get("dimension", "").strip().lower(),
                raw_pain_trigger.get("direction", "").strip().lower()
            )
            trigger_cache.add(trigger_tuple)

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
            }
        }

        persona_list = canonical_map[raw_pain][raw_job]
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
    trigger_dicts = [
        {"attribute": t[0], "dimension": t[1], "direction": t[2]}
        for t in trigger_cache
    ]
    canonical_pain_triggers = canonicalize_pain_trigger(trigger_dicts)

    # 3. Build mapping from raw/canonical to node IDs
    pain_id_map = {}
    for raw, canonical in canonical_pains.items():
        pain_node = get_or_create_pain_node(canonical)
        pain_id_map[raw] = pain_node["id"]

    job_id_map = {}
    for raw, canonical in canonical_jobs.items():
        job_node = get_or_create_job_node(canonical)
        job_id_map[raw] = job_node["id"]

    persona_id_map = {}
    for raw, canonical in canonical_personas.items():
        persona_node = get_or_create_persona_node(
            canonical["title"], canonical["department"], canonical["seniority"]
        )
        persona_id_map[raw] = persona_node["id"]

    pain_trigger_id_map = {}
    for idx, (raw, canonical) in enumerate(canonical_pain_triggers.items()):
        pain_trigger_node = get_or_create_pain_trigger_node(canonical)
        # Use tuple as key for mapping
        trigger_tuple = (
            canonical.get("attribute", "").strip().lower(),
            canonical.get("dimension", "").strip().lower(),
            canonical.get("direction", "").strip().lower()
        )
        pain_trigger_id_map[trigger_tuple] = pain_trigger_node["id"]


    

    #Add edges for original job - upstream pain with edge weight = impact
    for entry in flattened_results:
        raw_pain = entry.get("pain", entry.get("original_pain", entry.get("canonical_pain", "Unknown Pain"))).strip().lower()
        raw_job = entry.get("job", entry.get("original_job", entry.get("canonical_job", "Unknown Job"))).strip().lower()
        persona = entry.get("persona", {})
        title = persona.get("title", "").strip().lower()
        dept = persona.get("department", "Unknown").strip().lower()
        seniority = persona.get("seniority", "Unknown").strip().lower()
        persona_key = (title, dept, seniority)
        raw_pain_trigger = entry.get("pain_trigger", {})
        trigger_tuple = (
            raw_pain_trigger.get("attribute", "").strip().lower(),
            raw_pain_trigger.get("dimension", "").strip().lower(),
            raw_pain_trigger.get("direction", "").strip().lower()
        )

        original_job_id = entry.get("original_job_id")
        pain_id = pain_id_map.get(raw_pain)
        job_id = job_id_map.get(raw_job)
        persona_id = persona_id_map.get(persona_key)
        pain_trigger_id = pain_trigger_id_map.get(trigger_tuple)
        impact = float(entry.get("impact", 1.0))

        # Optionally, add these IDs to the entry for downstream use
        entry["pain_id"] = pain_id
        entry["job_id"] = job_id
        entry["persona_id"] = persona_id
        entry["pain_trigger_id"] = pain_trigger_id

        if not original_job_id or not pain_id or not job_id or not persona_id:
            print(f"Missing IDs in entry: {entry}")
            continue
        now = datetime.utcnow().isoformat()
        add_edge(
            source_id = original_job_id, 
            target_id = pain_id, 
            edge_type = "impacts", 
            weight = impact,
            last_updated = now
        )
        add_edge(
            source_id = pain_id, 
            target_id = job_id, 
            edge_type = "addresses",
            weight = 1,
            last_updated = now
        )
        add_edge(
            source_id = job_id, 
            target_id = persona_id, 
            edge_type = "performed_by",
            weight= 1,
            last_updated = now
        )
        add_edge(
            source_id = pain_id, 
            target_id = pain_trigger_id, 
            edge_type = "triggered_by",
            weight = 1,
            last_updated = now
        )

    print("Updated results blob:" , flattened_results)
    return flattened_results


