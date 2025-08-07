from backend.utils.inference.discovery_engine.agentic_engine.agent_context import AgentContext
from backend.utils.graph_base.network_graph import (
    get_node_by_id, get_node_id, get_nodes_list_ids,
    get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type, update_graph
)
from backend.utils.inference.discovery_engine.agentic_discovery_engine import (
    generate_icps_for_hop0, hop0_inference, pain_source_inference,
    process_hop_plus_gpt_cache, process_upstream_zmot_archetype, process_zmot_gpt_cache
)
import networkx as nx

def run_agentic_loop(product_subgraph: nx.DiGraph, max_depth=3):
    context = AgentContext(max_depth=max_depth)
    while True:
        agentic_inference(product_subgraph, context)
        if (
            not context.hop_plus_cache
            and not context.zmot_cache
            and not context.pain_source_cache
            and not context.zmot_archetype_cache
        ) or context.current_depth >= context.max_depth:
            print("🛑 Agentic loop complete.")
            break
        recursive_agentic_traversal(product_subgraph, context)
        context.current_depth += 1
        

def agentic_inference(product_subgraph: nx.DiGraph, context=None) -> nx.DiGraph:
    product_id = get_node_id(product_subgraph, "product", {})
    if not product_id:
        raise ValueError("❌ Product ID not found in subgraph.")
    if context is None:
        context = AgentContext(max_depth=3)

    # 1. Check for capability nodes
    capability_ids = get_nodes_list_ids(product_subgraph, "capability", {})
    if not capability_ids:
        raise ValueError("❌ No capability nodes found in subgraph.")
    update_graph(product_subgraph)

    # 2. Check for jobs
    job_ids = get_nodes_list_ids(product_subgraph, "job", {})
    if not job_ids:
        print("No jobs found, running Hop0 inference...")
        hop0_jobs = hop0_inference(product_id, product_subgraph, context)
        if not hop0_jobs:
            raise RuntimeError("❌ Hop0 inference failed.")
    update_graph(product_subgraph)

    # 3: Infer pain sources (internal vs external)
    print("🔍 Classifying jobs as internal or external...")
    for job_id in job_ids:
        job_node = get_node_by_id(product_subgraph, job_id)
        if not job_node:
            print(f"⚠️ Job node {job_id} not found in the product subgraph.")
            continue
        pain_source = job_node.get("pain_source", "").strip().lower()
        if not pain_source or pain_source not in ["internal", "external"]:
            print(f"⚠️ Job {job_id} has no valid pain source. Running Pain Source Inference.")
            context.pain_source_cache.append(job_id)
            continue
        else:
            if pain_source == "internal":
                upstream_pains = get_target_nodes_by_source_and_type(product_subgraph, job_id, "solves")
                if upstream_pains:
                    context.mark_visited(job_id)
                if not context.is_visited(job_id):
                    context.hop_plus_cache.append(job_id)
            elif pain_source == "external":
                upstream_zmots = get_target_nodes_by_source_and_type(product_subgraph, job_id, "triggered_by")
                print("🔍 Found upstream ZMOTs for job:", job_id, "->", upstream_zmots)
                if upstream_zmots or len(upstream_zmots) > 0:
                    print("Looping through upstream ZMOTs for job:", job_id)
                    for upstream_zmot_id in upstream_zmots:
                        if upstream_zmot_id not in context.visited_zmot_archetypes and upstream_zmot_id not in context.zmot_archetype_cache:
                            print("Looking for upstream archetypes")
                            upstream_archetypes = get_source_nodes_by_target_and_type(product_subgraph, upstream_zmot_id, "responds_to")
                            if upstream_archetypes:
                                context.mark_visited(job_id)
                                context.visited_zmot_archetypes.add(upstream_zmot_id)
                                continue
                            else:
                                print(f"⚠️ No upstream archetypes found for ZMOT {upstream_zmot_id}. Adding to ZMOT Archetype cache.")
                                if upstream_zmot_id not in context.zmot_archetype_cache:
                                    context.zmot_archetype_cache.append(upstream_zmot_id)
                if not context.is_visited(job_id):
                    context.zmot_cache.append(job_id)
            else:
                print(f"⚠️ Job {job_id} has an unknown pain source: {pain_source}")
                context.pain_source_cache.append(job_id)
    print("🔍 Pain source cache:", context.pain_source_cache)
    print("🔍 Hop+ cache:", context.hop_plus_cache)
    print("🔍 ZMOT cache:", context.zmot_cache)
    print("🔍 ZMOT Archetype cache:", context.zmot_archetype_cache)
    if context.pain_source_cache:
        pain_source_agent(product_subgraph, context)

    print("✅ Agentic inference completed.")
    return product_subgraph

def pain_source_agent(product_subgraph, context):
    if context.pain_source_cache:
        print("🔍 Processing pain source cache...")
        pain_source_gaps = context.pain_source_cache.copy()
        context.pain_source_cache.clear()
        print("Starting pain source inference with Pain Source Gaps:", pain_source_gaps)
        pain_source_results = pain_source_inference(product_subgraph, pain_source_gaps, context)
        print("Pain Source Results:", pain_source_results)

def recursive_agentic_traversal(product_subgraph, context):
    if context.current_depth >= context.max_depth:
        print("🛑 Max depth reached. Stopping.")
        return

    print(f"🔁 Traversal pass at depth {context.current_depth}...")

    # Hop+ loop
    if context.hop_plus_cache:
        hop_plus_jobs = context.hop_plus_cache.copy()
        context.hop_plus_cache.clear()
        results = process_hop_plus_gpt_cache(hop_plus_jobs, product_subgraph, context)
        if results:
            for job_id in hop_plus_jobs:
                if job_id not in context.visited_jobs:
                    context.mark_visited(job_id)
                    

    # ZMOT loop
    if context.zmot_cache:
        zmot_jobs = context.zmot_cache.copy()
        context.zmot_cache.clear()
        results = process_zmot_gpt_cache(zmot_jobs, product_subgraph, context)
        if results:
            for job_id in zmot_jobs:
                if job_id not in context.visited_jobs:
                    context.mark_visited(job_id)
                    
    
    # Deep ZMOT Archetype loop
    if context.zmot_archetype_cache:
        print("🔍 Processing ZMOT Archetype cache...")
        zmot_archetype_events = context.zmot_archetype_cache.copy()
        context.zmot_archetype_cache.clear()
        results = process_upstream_zmot_archetype(product_subgraph, zmot_archetype_events, context)
        if results:
            for zmot_id in zmot_archetype_events:
                context.visited_zmot_archetypes.add(zmot_id)
        
            