from backend.utils.inference.discovery_engine.agentic_engine.agent_context import AgentContext
from backend.utils.graph_base.network_graph import (
    get_node_by_id, get_node_id, get_nodes_list_ids,
    get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type, update_graph
)
from backend.utils.inference.discovery_engine.agentic_discovery_engine import (
    generate_icps_for_hop0, hop0_inference, pain_source_inference,
    process_hop_plus_gpt_cache, process_zmot_gpt_cache
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
        hop0_result = hop0_inference(product_id, product_subgraph, context)
        if not hop0_result or "capability_map" not in hop0_result:
            raise RuntimeError("❌ Hop0 inference failed.")
    update_graph(product_subgraph)

    # 2.5. Get ICP Archetypes with this info
    print("Generating ICP Archetypes if they don't exist...")
    archetype_ids = get_nodes_list_ids(product_subgraph, "archetype", {})
    if not archetype_ids:
        print("ICP archetypes not found. Running inference.")
        job_ids = get_nodes_list_ids(product_subgraph, "job", {})
        pain_ids = get_nodes_list_ids(product_subgraph, "pain", {})
        persona_ids = get_nodes_list_ids(product_subgraph, "persona", {})
        if job_ids and pain_ids and persona_ids:
            generate_icps_for_hop0(product_subgraph, context)

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
                upstream_zmots = get_source_nodes_by_target_and_type(product_subgraph, job_id, "triggered_by")
                if upstream_zmots:
                    context.mark_visited(job_id)
                if not context.is_visited(job_id):
                    context.zmot_cache.append(job_id)
            else:
                print(f"⚠️ Job {job_id} has an unknown pain source: {pain_source}")
                context.pain_source_cache.append(job_id)

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