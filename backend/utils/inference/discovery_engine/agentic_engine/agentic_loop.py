# agentic_loop.py

from typing import Any, Dict, Iterable, List, Optional
from backend.utils.graph_base.schema import EDGES
from backend.utils.inference.discovery_engine.agentic_engine.agent_context import AgentContext
from backend.utils.graph_base.network_graph import (
    get_node_by_id, get_node_id, get_nodes_list_ids,
    get_target_nodes_by_source_and_type, update_graph
)
from backend.utils.inference.discovery_engine.agentic_discovery_engine import (
    expand_frontier_one_layer,
    hop0_inference,
    pain_source_inference,
    process_hop_plus_gpt_cache,
    process_trigger_for_archetype_prevalence,
    process_trigger_for_zmot_boosts     # NEW: build PainTrigger→Archetype
)
from backend.utils.graph_base.icp_catalog import ICP_CATALOG
import networkx as nx


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_agentic_loop(product_subgraph: nx.DiGraph, max_depth: int = 6):
    """
    Main driver: executes incremental agentic inference passes until caches are empty
    or max_depth is reached. Then runs a single archetype discovery pass for all
    remaining Pain Triggers missing Archetype prevalence.
    """
    context = AgentContext(max_depth=max_depth)
    
    while True:
        agentic_inference(product_subgraph, context)

        # Always attempt a traversal pass if depth allows it
        if context.current_depth < context.max_depth:
            recursive_agentic_traversal(product_subgraph, context)
            context.current_depth += 1

        # NOW decide if we’re done
        done = (
            not context.hop_plus_cache
            and not context.pain_source_cache
        )
        depth_stop = context.current_depth >= context.max_depth

        if depth_stop:
            print("🛑 Agentic loop complete.",
                f"(done={done}, depth={context.current_depth}/{context.max_depth})")
            break
    
    # Finalize: run archetype discovery for Pain Triggers
    print("✅ Completed agentic Hop loop passes.")
    #print("🔍 Running final Trigger Event + Archetype discovery for Pain Triggers…")
    #archetype_event_discovery(product_subgraph, context)


    
    
    #print("✅ Completed Archetype discovery.")


# ---------------------------------------------------------------------------
# Core inference pass (pain/trigger–centric)
# ---------------------------------------------------------------------------

def agentic_inference(product_subgraph: nx.DiGraph, context: AgentContext | None = None) -> nx.DiGraph:
    """
    One pass over the graph:
      1) Ensure Product and Capability nodes exist.
      2) Ensure Pain nodes exist (Hop0 bootstrap if empty).
      3) For each Pain (unvisited):
         - If pain_source unknown → queue for pain_source_inference.
         - If internal:
             · If no upstream Jobs (Pain --felt_in--> Job) → queue pain for Hop+.
             · Else mark pain visited.
         - If external:
             · Collect upstream PainTriggers (Pain --triggered_by--> PainTrigger).
             · For each trigger, if it has no ZMOTs (Trigger --accelerated_by--> ZMOT), queue trigger for ZMOT discovery.
             · Mark pain visited ONLY when *all* its triggers have ≥1 ZMOT.
    """
    product_id = get_node_id(product_subgraph, "product", {})
    if not product_id:
        raise ValueError("❌ Product ID not found in subgraph.")
    print("🧠 Starting agentic inference pass for product_id:", product_id)

    if context is None:
        context = AgentContext(max_depth=3)
    product_subgraph = update_graph(product_subgraph)
    print("🔎 Starting inference pass for product_id:", product_id)
    # 1) Capabilities must exist
    capability_ids = get_nodes_list_ids(product_subgraph, "capability", {})
    if not capability_ids:
        raise ValueError("❌ No capability nodes found in subgraph.")
    product_subgraph = update_graph(product_subgraph)

    # 2) Ensure Pain nodes exist (Hop0)
    pain_ids = get_nodes_list_ids(product_subgraph, "pain", {})
    if not pain_ids:
        print("ℹ️ No pains found; running Hop0 inference…")
        hop0_pains = hop0_inference(product_subgraph, context)
        if not hop0_pains:
            raise RuntimeError("❌ Hop0 inference failed to produce pains.")
    # 2.5) Ensure all capabilities have at least one pain
    to_process_cap = []
    for capability_id in capability_ids:
        associated_pains = get_target_nodes_by_source_and_type(product_subgraph, capability_id, "solves")
        if not associated_pains:
            print(f"ℹ️ Capability {capability_id} has no associated pains; adding to Hop0 inference…")
            to_process_cap.append(capability_id)
    if to_process_cap:
        hop0_pains = hop0_inference(product_subgraph, context, cap_ids=to_process_cap)
        if not hop0_pains:
            raise RuntimeError("❌ Hop0 inference failed to produce pains for missing capabilities.")
            
    product_subgraph = update_graph(product_subgraph)
    print("Subgraph updated and saved post Hop0")

    # refresh after potential Hop0
    pain_ids = get_nodes_list_ids(product_subgraph, "pain", {})

    # 3) Classify pains & queue work
    print("🔎 Classifying pains (terminal vs non-terminal) & queuing work…")
    for pain_id in pain_ids:
        print("iterating pain:", pain_id)
        if context.is_pain_visited(pain_id):
            print(f"ℹ️ Pain {pain_id} already visited; skipping.")
            continue
        print("Not visited pain - processing now")

        pain_node = get_node_by_id(product_subgraph, pain_id)
        print("pain node pulled")
        if not pain_node:
            print(f"⚠️ Missing pain node: {pain_id}")
            continue

        pain_source = (pain_node.get("pain_source") or "").strip().lower()
        print("pain source:", pain_source)
        if pain_source not in {"terminal", "non-terminal"}:
            # missing classification → infer later
            print("Missing pain node classification; queuing for inference…")
            context.pain_source_cache.append(pain_id)
            continue

        if pain_source == "non-terminal":
            # Check non-terminal pains for relevance
            print("Processing non-terminal pain:", pain_id)
            upstream_jobs = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "felt_in")
            if upstream_jobs:
                print("Non terminal pain has upstream jobs - marking as visited")
                context.mark_pain_visited(pain_id)
            else:
                if not context.is_pain_visited(pain_id):
                    print("Adding non-terminal pain to Hop+ queue")
                    context.enqueue_hop_plus_pain(pain_id)

        elif pain_source == "terminal":
            print(f"ℹ️ Terminal pain {pain_id} found; Terminating traversal…")
            context.mark_pain_visited(pain_id)
        else:
            # missing/invalid pain_source
            print("Enqueuing missing source for pain node")
            context.enqueue_pain_for_source(pain_id)

    # Resolve missing pain sources immediately (keeps loop tight)
    if context.pain_source_cache:
        print("🧭 Processing pain_source cache…")
        pain_source_agent(product_subgraph, context)

    print(
        "📦 Queues |",
        f"Hop+ pains: {len(context.hop_plus_cache)} |",
        f"Pains missing source: {len(context.pain_source_cache)}"
    )
    print("✅ Inference pass complete.")
    return product_subgraph


# ---------------------------------------------------------------------------
# Cache processors
# ---------------------------------------------------------------------------

def pain_source_agent(product_subgraph: nx.DiGraph, context: AgentContext):
    """
    Resolve unknown pain_source for pains in cache via LLM, then update graph.
    """
    if not context.pain_source_cache:
        return
    print("🧭 Processing pain_source cache…")
    to_infer = context.pain_source_cache.copy()
    context.pain_source_cache.clear()
    _ = pain_source_inference(product_subgraph, to_infer, context)
    product_subgraph = update_graph(product_subgraph)
    print("🧭 Pain source inference complete.")


def recursive_agentic_traversal(product_subgraph: nx.DiGraph, context: AgentContext):
    """
    Executes one traversal step using queued work:
      - Hop+ for internal pains lacking upstream jobs
      - ZMOT discovery for triggers
    """
    if context.current_depth >= context.max_depth:
        print("🛑 Max depth reached. Stopping traversal.")
        return

    print(f"🔁 Traversal pass at depth {context.current_depth}…")

    # 1) Hop+ (grow upstream internal graph for pains)
    if context.hop_plus_cache:
        hop_plus_pains = context.hop_plus_cache.copy()
        context.hop_plus_cache.clear()

        print(f"🪜 Hop+ on {len(hop_plus_pains)} pains…")
        print("Pain IDs:", hop_plus_pains)
        results = process_hop_plus_gpt_cache(hop_plus_pains, product_subgraph, context)
        
        
        product_subgraph = update_graph(product_subgraph)

        if results:
            # Mark these pains visited if they now have upstream jobs
            for pain_id in hop_plus_pains:
                if context.is_pain_visited(pain_id):
                    continue
                upstream_jobs = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "felt_in")
                if upstream_jobs:
                    context.mark_pain_visited(pain_id)
            all_pain_ids = get_nodes_list_ids(product_subgraph, "pain", {})
            for pain_id in all_pain_ids:
                if context.is_pain_visited(pain_id):
                    continue
                upstream_jobs = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "felt_in")
                if upstream_jobs:
                    context.mark_pain_visited(pain_id)
                else:
                    print("Adding newly discovered pain to Hop+ queue")
                    context.enqueue_hop_plus_pain(pain_id)

    print("🔁 Traversal pass complete.")


# ---------------------------------------------------------------------------
# Post-loop: Archetype prevalence for Pain Triggers
# ---------------------------------------------------------------------------

def _pending_triggers_without_edge(G, edge_type: str) -> list[str]:
    trigger_node_ids = get_nodes_list_ids(G, "pain_trigger", {})
    return [
        t for t in trigger_node_ids
        if not get_target_nodes_by_source_and_type(G, t, edge_type)
    ]

def _pending_triggers_without_attr_labels(G) -> list[str]:
    trigger_node_ids = get_nodes_list_ids(G, "pain_trigger", {})
    pending = []
    for t in trigger_node_ids:
        n = get_node_by_id(G, t) or {}
        if not n.get("attribute_scores_labeled"):
            pending.append(t)
    return pending


def archetype_event_discovery(
    product_subgraph: nx.DiGraph,
    context: AgentContext,
    batch_size: int = 10,
    max_passes: int = 20,
) -> None:
    """
    Final discovery loop for pain_triggers:
      1) Per-trigger ATTRIBUTE LABELS (industry / revenue_range / employee_range / funding_stage / geography)
      2) ZMOT events (with observable moments & keywords), independent of archetype nodes
    """
    print("🔍 Running final Trigger Attribute + ZMOT discovery for Pain Triggers…")

    pass_num = 0
    last_pending_zmots = set()
    stagnant_count = 0

    while pass_num < max_passes:
        pass_num += 1
        progress = False
        print(f"— Discovery pass {pass_num}/{max_passes}")

        # 1) Attribute labels for triggers (stored on trigger node)
        pending_attr = _pending_triggers_without_attr_labels(product_subgraph)
        if pending_attr:
            print(f"🏷️ Attribute label discovery for {len(pending_attr)} triggers (batch={batch_size})…")
            for i in range(0, len(pending_attr), batch_size):
                chunk = pending_attr[i:i + batch_size]
                touched = process_trigger_for_archetype_prevalence(chunk, product_subgraph, context) or []
                if touched:
                    progress = True
                product_subgraph = update_graph(product_subgraph)
            print("🏷️ Attribute label discovery pass complete.")

        # 2) ZMOTs (PainTrigger -> accelerated_by -> ZMOT)
        pending_zmots = _pending_triggers_without_edge(product_subgraph, "accelerated_by")
        pending_zmots_set = set(pending_zmots)
        if pending_zmots:
            print(f"⚡ ZMOT discovery for {len(pending_zmots)} triggers (batch={batch_size})…")
            for i in range(0, len(pending_zmots), batch_size):
                chunk = pending_zmots[i:i + batch_size]
                touched = process_trigger_for_zmot_boosts(chunk, product_subgraph, context) or []
                if touched:
                    progress = True
                product_subgraph = update_graph(product_subgraph)
            print("⚡ ZMOT discovery pass complete.")

        # stagnation detection for ZMOTs
        if pending_zmots_set == last_pending_zmots and pending_zmots:
            stagnant_count += 1
        else:
            stagnant_count = 0
        last_pending_zmots = pending_zmots_set

        if stagnant_count >= 2 and pending_zmots:
            print(f"⚠️ Marking {len(pending_zmots)} pain_triggers as attempted (no ZMOTs found after 2 cycles).")
            for t in pending_zmots:
                print(f"No ZMOT data found for {t}. Moving out")
            product_subgraph = update_graph(product_subgraph)
            break

        # Termination checks (all good when attr labels exist and ZMOT edges exist)
        remaining_attr = _pending_triggers_without_attr_labels(product_subgraph)
        remaining_zmots = _pending_triggers_without_edge(product_subgraph, "accelerated_by")

        print(f"Status after pass {pass_num}: "
              f"pending_attr={len(remaining_attr)}, pending_zmots={len(remaining_zmots)}, progress={progress}")

        if not remaining_attr and not remaining_zmots:
            print("✅ All pain_triggers have Attribute Labels and ZMOTs. Done.")
            break
        if not progress:
            print("ℹ️ No progress in this pass. Stopping to avoid infinite loop.")
            break

    if pass_num >= max_passes:
        print("⛔ Reached max_passes without clearing all pending work. You may increase max_passes or inspect inputs.")



#--------------------------------------------
# Slim Frontier Expansion Contract
#--------------------------------------------

def run_frontier_expansion(
    G: nx.DiGraph,
    seed_ids: Iterable[str],
    *,
    only_types: Optional[Iterable[str]] = None,
    waves: int = 1,
    max_items_per_source: int = 5,
    model: str = "gpt-4o-mini",
    ctx: Optional["AgentContext"] = None,
    # fallbacks if ctx is not provided
    product_id: Optional[str] = None,
    product_summary: Optional[str] = None,
    product_industry: Optional[str] = None,
    product_domain: Optional[str] = None,
    client: Optional[Any] = None,
    now_iso_fn: Optional[callable] = None,
    build_product_context_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    seeds: List[str] = list(seed_ids or [])
    waves = max(int(waves or 1), 1)
    agg: Dict[str, Any] = {"expanded_count": 0, "by_type_counts": {}, "new_node_ids": []}
    if ctx is None:
        ctx = AgentContext(max_depth=3)
    for _ in range(waves):
        print("Expanding waves — current seeds:", seeds)
        out = expand_frontier_one_layer(
            G,
            seeds,
            only_types=list(only_types) if only_types else None,
            max_items_per_source=int(max_items_per_source),
            model=model,
            ctx=ctx,
        )
        if isinstance(out, dict):
            agg["expanded_count"] += int(out.get("expanded_count", 0))
            for k, v in (out.get("by_type_counts") or {}).items():
                agg["by_type_counts"][k] = int(agg["by_type_counts"].get(k, 0)) + int(v)
            agg["new_node_ids"].extend(out.get("new_node_ids") or [])

        # If you want “multi-wave” chaining, uncomment:
        # seeds = out.get("new_node_ids") or seeds

    # de-dup
    if agg["new_node_ids"]:
        s = set(); agg["new_node_ids"] = [x for x in agg["new_node_ids"] if not (x in s or s.add(x))]
    print("Frontier expansion complete:", agg)
    update_graph(G)
    return agg

