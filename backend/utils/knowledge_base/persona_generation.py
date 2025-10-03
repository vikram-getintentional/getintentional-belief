<<<<<<< Updated upstream
from typing import Dict, List, Any
from backend.utils.graph_base.graph import Graph
from backend.utils.graph_base.graph_utils.aggregate_persona_cards import aggregate_persona_cards
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data, add_or_update_cumulative_relevance_data
=======
from typing import Dict, List, Any, Optional
import json
from collections import defaultdict
from backend.utils.graph_base.network_graph import calculate_eigenvector_centrality, calculate_pagerank_centrality, calculate_personalized_pagerank_centrality, calculate_soft_or_relevance, get_cumulative_relevance, get_node_by_id, get_node_id, get_nodes_list, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, max_flow, relevance, reverse_belief_weight, update_capability_centralities
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data, get_cumulative_relevance_data
>>>>>>> Stashed changes


from backend.utils.graph_base.graph_data.rcs_utils.save_and_load_rcs import load_rcs_from_json, save_rcs_as_json
from backend.utils.inference.rcs_generators.generate_rcs_fast import generate_rcs

# ---------------- Utils ----------------

<<<<<<< Updated upstream
def get_company_products(base_graph: Graph, company_id: str) -> List[Dict[str, Any]]:
    # This assumes get_nodes_list returns a list of (node_id, node_data) tuples
    return [
        {"id": node_id, "url": node_data.get("url")}
        for node_id, node_data in base_graph.get_nodes_list("product", {"company_id": company_id})
=======
def get_company_products(base_graph: nx.DiGraph, company_id: str) -> List[Dict[str, Any]]:
    return [
        {"id": node_id, "url": node_data.get("url")}
        for node_id, node_data in get_nodes_list(base_graph, "product", {"company_id": company_id})
>>>>>>> Stashed changes
    ]

def _persona_label(sub_graph: nx.DiGraph, persona_id: str) -> Dict[str, str]:
    n = get_node_by_id(sub_graph, persona_id) or {}
    return {
        "title": n.get("title") or n.get("name") or "",
        "department": n.get("department") or "",
        "seniority": n.get("seniority") or "",
    }

def _rcs_cache_key_for_personas(product_id: str) -> str:
    # we use the zmot_id slot for namespacing
    return f"persona_tab::baseline"

# ---------------- NEW: RCS aggregation ----------------

def aggregated_personas_rcs(
    cards: List[Dict[str, Any]],
    *,
    group_on: tuple = ("title", "department", "seniority"),
    min_priority: float = 0.0,
    top_k: int | None = None
) -> List[Dict[str, Any]]:
    """
    Aggregate per-node persona cards into one card per logical persona.
    Grouping key defaults to (title, department, seniority).

    For each group:
      - importance, activation, care, marginal_lift: max
      - priority_score: recomputed = max_importance * max_activation
      - jobs/pains: union by description with max relevance, sorted desc
      - persona_ids: union of ids
    Output shape mirrors your older aggregated format but with RCS metrics added.
    """
    buckets: Dict[tuple, Dict[str, Any]] = {}

    for c in cards or []:
        meta = c.get("persona", {}) or {}
        key = tuple(meta.get(k, "") for k in group_on)

        if key not in buckets:
            buckets[key] = {
                "persona_title": meta.get("title", ""),
                "persona_departments": set([meta.get("department", "")]) if meta.get("department") else set(),
                "persona_seniority": set([meta.get("seniority", "")]) if meta.get("seniority") else set(),
                "persona_ids": set(),
                "importance": 0.0,
                "activation": 0.0,
                "care": 0.0,
                "marginal_lift": 0.0,
                "priority_score": 0.0,
                "jobs": {},   # desc -> max relevance
                "pains": {},  # desc -> max relevance
            }

        agg = buckets[key]
        agg["persona_ids"].add(c.get("persona_id"))
        if meta.get("department"): agg["persona_departments"].add(meta["department"])
        if meta.get("seniority"):  agg["persona_seniority"].add(meta["seniority"])

        # max aggregation for metrics
        imp = float(c.get("importance", 0.0) or 0.0)
        act = float(c.get("activation", 0.0) or 0.0)
        care = float(c.get("care", 0.0) or 0.0)
        ml   = float(c.get("marginal_lift", 0.0) or 0.0)

        agg["importance"]   = max(agg["importance"], imp)
        agg["activation"]   = max(agg["activation"], act)
        agg["care"]         = max(agg["care"], care)
        agg["marginal_lift"] = max(agg["marginal_lift"], ml)

        # union jobs/pains by description, keeping max relevance
        for j in (c.get("jobs") or []):
            desc = (j.get("description") or "").strip()
            if not desc: continue
            r = float(j.get("relevance", 0.0) or 0.0)
            agg["jobs"][desc] = max(agg["jobs"].get(desc, 0.0), r)

        for p in (c.get("pains") or []):
            desc = (p.get("description") or "").strip()
            if not desc: continue
            r = float(p.get("relevance", 0.0) or 0.0)
            agg["pains"][desc] = max(agg["pains"].get(desc, 0.0), r)

    # materialize & finalize
    out: List[Dict[str, Any]] = []
    for agg in buckets.values():
        # recompute priority on max signals
        agg["priority_score"] = float(agg["importance"]) * float(agg["activation"])

        if agg["priority_score"] < min_priority:
            continue

        out.append({
            "persona_title": agg["persona_title"],
            "persona_departments": sorted(list(agg["persona_departments"])),
            "persona_seniority": sorted(list(agg["persona_seniority"])),
            "persona_ids": sorted(list(pid for pid in agg["persona_ids"] if pid)),

            "importance": float(agg["importance"]),
            "activation": float(agg["activation"]),
            "care": float(agg["care"]),
            "marginal_lift": float(agg["marginal_lift"]),
            "priority_score": float(agg["priority_score"]),

            "jobs": sorted(
                [{"description": d, "relevance": r} for d, r in agg["jobs"].items()],
                key=lambda x: x["relevance"], reverse=True
            ),
            "pains": sorted(
                [{"description": d, "relevance": r} for d, r in agg["pains"].items()],
                key=lambda x: x["relevance"], reverse=True
            ),
        })

    out.sort(key=lambda c: c["priority_score"], reverse=True)
    if top_k is not None:
        return out[:top_k]
    return out

# ---------------- RCS Personas (calls aggregation) ----------------

def get_personas_rcs_priority(sub_graph: nx.DiGraph, attribute_dict: Optional[dict] = None, relevance_threshold: float = 0.0, top_k: int = 50) -> list[dict]:
    """
    Ranks personas by Importance × Activation using the RCS engine:
      importance := involvement (from generate_rcs)
      activation := activation (from generate_rcs)
    Returns AGGREGATED persona cards (one per logical persona).
    """
    if sub_graph.number_of_nodes() == 0:
        return []

    product_id = get_product_id_from_subgraph(sub_graph)
    if not product_id:
        return []
    attribute_dict = attribute_dict or {}
    # ---- 1) Load or compute RCS once (baseline, no engaged nodes) ----
    cache_key = _rcs_cache_key_for_personas(product_id)
    cached = load_rcs_from_json(product_id=product_id, attribute_dict=attribute_dict, zmot_id=cache_key)

    if cached:
        causal_graph = cached["causal_graph"]
        rcs_report = cached["rcs_report"]
    else:
        causal_graph, rcs_report = generate_rcs(sub_graph, engaged_nodes=None)
        save_rcs_as_json(product_id=product_id, attribute_dict={},
                         causal_graph=causal_graph, rcs_report=rcs_report, zmot_id=cache_key)

    # ---- 2) Collect per-persona metrics from the report ----
    top_block = (rcs_report or {}).get("top_personas", {}) or {}
    by_involvement = {r["id"]: r for r in top_block.get("by_involvement", [])}
    by_activation  = {r["id"]: r for r in top_block.get("by_activation", [])}
    by_lift        = {r["id"]: r for r in top_block.get("by_marginal_lift", [])}

    metrics: Dict[str, Dict[str, float]] = {}
    for pid in set(list(by_involvement.keys()) + list(by_activation.keys()) + list(by_lift.keys())):
        inv = float(by_involvement.get(pid, {}).get("involvement", 0.0))
        act = float(by_activation.get(pid, {}).get("activation", by_involvement.get(pid, {}).get("activation", 0.0)))
        care = float(by_involvement.get(pid, {}).get("care", by_activation.get(pid, {}).get("care", 0.0)))
        mlift = float(by_lift.get(pid, {}).get("marginal_lift", 0.0))
        metrics[pid] = {
            "involvement": inv,
            "activation": act,
            "care": care,
            "marginal_lift": mlift,
            "priority_score": inv * act
        }

    # ---- 3) Build per-node persona cards (keep for aggregation) ----
    node_cards: List[Dict[str, Any]] = []
    for persona_id, _ in get_nodes_list(sub_graph, "persona", {}):
        m = metrics.get(persona_id, {"involvement": 0.0, "activation": 0.0, "care": 0.0, "marginal_lift": 0.0, "priority_score": 0.0})
        meta = _persona_label(sub_graph, persona_id)

        jobs, pains = [], []
        persona_jobs = get_source_nodes_by_target_and_type(sub_graph, persona_id, "performed_by")
        for job_id in persona_jobs:
            job_node = get_node_by_id(sub_graph, job_id) or {}
            job_text = job_node.get("description") or job_node.get("text") or ""
            job_rel = get_cumulative_relevance_data(product_id, job_id)
            jobs.append({"description": job_text, "relevance": job_rel})

            pain_ids = get_source_nodes_by_target_and_type(sub_graph, job_id, "felt_in")
            for pain_id in pain_ids:
                pain_node = get_node_by_id(sub_graph, pain_id) or {}
                pain_text = pain_node.get("description") or pain_node.get("text") or ""
                pain_rel = get_cumulative_relevance_data(product_id, pain_id)
                pains.append({"description": pain_text, "relevance": pain_rel})

        node_cards.append({
            "persona_id": persona_id,
            "persona": meta,
            "importance": m["involvement"],
            "activation": m["activation"],
            "care": m["care"],
            "marginal_lift": m["marginal_lift"],
            "priority_score": m["priority_score"],
            "jobs": sorted(jobs, key=lambda x: x["relevance"], reverse=True),
            "pains": sorted(pains, key=lambda x: x["relevance"], reverse=True),
        })

    # ---- 4) Aggregate to remove duplicates (title/department/seniority) ----
    aggregated = aggregated_personas_rcs(
        node_cards,
        group_on=("title", "department", "seniority"),
        min_priority=relevance_threshold,
        top_k=top_k
    )
    return aggregated

# ---------------- Legacy relevance-based personas (unchanged) ----------------


def get_product_personas(base_graph: Graph, product_id: str) -> List[Dict[str, Any]]:
    """
    This is a pretty cool traversal function that starts from the product ID, gets capabilities, and bubbles all the way up. 
    Only thing is - its kind of redundant since we're doing this bubble up thingy quite a bit in other places (cum_Rel calculations, product_graph creation, etc.)
    Also get_persona_relevance kind of does the same thing but better... 
    Leaving this here as a breadcrumb for the future but for now, it's just a sad little orphan.
    """
    print("Recalculating capability centralities")
    base_graph.update_capability_centralities()
    print("Starting product persona discovery for product ID:", product_id)
    persona_entries = []
    base_pain_ids = set()

<<<<<<< Updated upstream
    def traverse(job_id, pain_id, capability_id, visited):
        if (job_id, pain_id) in visited:
            return
        visited.add((job_id, pain_id))

        job_personas = base_graph.get_target_nodes_by_source_and_type(job_id, "performed_by")
        for persona_id in job_personas:
            # Compute cumulative_relevance for this persona's pain
            cumulative_relevance = get_cumulative_relevance_data(product_id, persona_id)
            persona_entry = {
                "persona_id": persona_id,
                "relevance": cumulative_relevance,
                "job_id": job_id,
                "pain_id": pain_id,
                "capability_id": capability_id,
                "product_id": product_id
            }
            persona_entries.append(persona_entry)
            
=======
    relevance_nodes = calculate_soft_or_relevance(sub_graph)
    print("Relevance nodes calculated:", relevance_nodes)
    cumulative_relevance = {item["node_id"]: item["relevance"] for item in relevance_nodes}
    for node_id, relevance in cumulative_relevance.items():
        node_data = get_node_by_id(sub_graph, node_id)
        if node_data and node_data.get("node_type") == "persona":
            text = node_data.get("title", "")
        elif node_data and node_data.get("node_type") == "job":
            text = node_data.get("description", "") or node_data.get("text", "")
        elif node_data and node_data.get("node_type") == "pain":
            text = node_data.get("description", "") or node_data.get("text", "")
        elif node_data and node_data.get("node_type") == "capability":
            text = node_data.get("name", "") or node_data.get("description", "")
        elif node_data and node_data.get("node_type") == "product":
            text = node_data.get("url", "")
        else:
            text = node_data.get("name", "") or node_data.get("description", "") or node_data.get("text", "") or "XXXX"
        print(f"Node ID: {node_id}, Node Type: {node_data.get('node_type')}, Text: {text}, Relevance: {relevance}, ")
>>>>>>> Stashed changes

        # Upstream and downstream traversal
        upstream_pains = base_graph.get_source_nodes_by_target_and_type(job_id, "impacts")
        for up_pain in upstream_pains:
            upstream_jobs = base_graph.get_source_nodes_by_target_and_type(up_pain, "solves")
            for up_job in upstream_jobs:
                traverse(up_job, up_pain, capability_id, visited)
        downstream_pains = base_graph.get_target_nodes_by_source_and_type(job_id, "impacts")
        for down_pain in downstream_pains:
            downstream_jobs = base_graph.get_target_nodes_by_source_and_type(down_pain, "solves")
            for down_job in downstream_jobs:
                traverse(down_job, down_pain, capability_id, visited)

<<<<<<< Updated upstream
    capabilities = base_graph.get_target_nodes_by_source_and_type(product_id, "offered_by")
    if not capabilities:
        print(f"No capabilities found for product ID {product_id}.")
    for capability in capabilities:
        pains = base_graph.get_target_nodes_by_source_and_type(capability, "solves")
        base_pain_ids.update(pains)
        for pain in pains:
            jobs = base_graph.get_target_nodes_by_source_and_type(pain, "addresses")
            for job in jobs:
                traverse(job, pain, capability, set())

    # Pass the flat list of persona-job-pain-capability dicts to aggregate_persona_cards
    aggregated_personas = aggregate_persona_cards(base_graph, persona_entries)
    print("Aggregated personas - Product Personas:")
    for data in aggregated_personas:
        print(data,"\n")
    return aggregated_personas


def get_persona_relevance(sub_graph: Graph) -> list[dict]:
    print("Recalculating capability centralities")
    sub_graph.update_capability_centralities()
    print("Starting persona relevance computation")
    personas = []
    product_id = sub_graph.get_node_id("product",{})

    for persona_id, _ in sub_graph.get_nodes_list("persona",{}):
        # Get cumulative_relevance value for this persona from cumulative_relevance.json
        cumulative_relevance = get_cumulative_relevance_data(product_id, persona_id)
        
        print("Cumulative relevance for persona ID:", persona_id, "is", cumulative_relevance)
        # For each job performed by this persona
        persona_jobs = sub_graph.get_source_nodes_by_target_and_type(
            persona_id, "performed_by"
        )
        for job_id in persona_jobs:
            
            # For each pain solved by this job
            pain_ids = sub_graph.get_source_nodes_by_target_and_type(
                job_id, "addresses"
            )
            
            for pain_id in pain_ids:
                
                persona = {
                    "persona_id": persona_id,
                    "relevance": cumulative_relevance,
                    "job_id": job_id,
                    "pain_id": pain_id,
                    "product_id": product_id,
                }
                print("Persona data:", persona)
                personas.append(persona)
                
    aggregated_personas = aggregate_persona_cards(sub_graph, personas)
    print("Aggregated personas - Persona Relevance:")
    for data in aggregated_personas:
        print(data, "\n")
    
    return aggregated_personas
=======
    print("Starting persona traversal")
    print("----------------------------------")
    for persona_id, _ in get_nodes_list(sub_graph, "persona",{}):
        persona_node = get_node_by_id(sub_graph, persona_id)
        print("Processing persona:", persona_id, "with title:", persona_node.get("title", "Unknown"))
        normalized_relevance = get_cumulative_relevance_data(product_id, persona_id)
        persona_node = get_node_by_id(sub_graph, persona_id)
        print("Persona node data:", persona_node)
        jobs = []
        pains = []
        persona_jobs = get_source_nodes_by_target_and_type(sub_graph, persona_id, "performed_by")
        for job_id in persona_jobs:
            job_node = get_node_by_id(sub_graph, job_id)
            if not job_node:
                print(f"Job node not found for ID: {job_id}")
                continue
            job_relevance = get_cumulative_relevance_data(product_id, job_id)
            jobs.append({
                "description": job_node.get("description") or job_node.get("text") or "",
                "relevance": job_relevance
            })
            pain_ids = get_source_nodes_by_target_and_type(sub_graph, job_id, "felt_in")
            for pain_id in pain_ids:
                pain_node = get_node_by_id(sub_graph, pain_id)
                if not pain_node:
                    print(f"Pain node not found for ID: {pain_id}")
                    continue
                pain_relevance = get_cumulative_relevance_data(product_id, pain_id)
                pains.append({
                    "description": pain_node.get("description") or pain_node.get("text") or "",
                    "relevance": pain_relevance
                })

        persona = {
            "persona_id": persona_id,
            "persona": {
                "title": persona_node.get("title"),
                "department": persona_node.get("department"),
                "seniority": persona_node.get("seniority"),
            },
            "relevance": normalized_relevance,
            "jobs": jobs,
            "pains": pains
        }
        print("Persona:", persona)
        print("-----------------------------------")
        personas.append(persona)

    print("Persona relevance computed, total personas found:")
    for persona in personas:
        print("Persona: ", persona["persona"]["title"],
              persona["persona"]["department"],
              persona["persona"]["seniority"],
              "Relevance:", persona["relevance"],
              "Jobs:", len(persona["jobs"]),
              "Pains:", len(persona["pains"])
              )

    aggregated_personas = aggregated_personas_map(sub_graph, personas, threshold=0.2)
    return aggregated_personas


def aggregated_personas_map(sub_graph: nx.DiGraph, match_results: list[dict], threshold: float = 0.0) -> list[dict]:
    final = []
    if not match_results:
        print("No match results found, returning empty list.")
        return final
    persona_map = defaultdict(lambda: {
        "persona_title": "",
        "persona_departments": set(),
        "persona_seniority": set(),
        "persona_ids": set(),
        "max_relevance": 0.0,
        "jobs": set(),
        "pains": set()
    })

    for entry in match_results:
        persona = entry.get("persona", {})
        relevance = entry.get("relevance", 0.0)
        if relevance < threshold:
            continue

        title = persona.get("title", "")
        department = persona.get("department", "")
        seniority = persona.get("seniority", "")
        persona_id = entry.get("persona_id", "")

        persona_map[title]["persona_title"] = title
        persona_map[title]["persona_departments"].add(department)
        persona_map[title]["persona_seniority"].add(seniority)
        persona_map[title]["persona_ids"].add(persona_id)
        persona_map[title]["max_relevance"] = max(persona_map[title]["max_relevance"], relevance)

        jobs = entry.get("jobs", [])
        pains = entry.get("pains", [])
        for job in jobs:
            persona_map[title]["jobs"].add(json.dumps(job))
        for pain in pains:
            persona_map[title]["pains"].add(json.dumps(pain))

    for card in persona_map.values():
        card["persona_departments"] = list(card["persona_departments"])
        card["persona_seniority"] = list(card["persona_seniority"])
        card["persona_ids"] = list(card["persona_ids"])
        card["jobs"] = sorted(list(card["jobs"]), key=lambda x: json.loads(x)["relevance"], reverse=True)
        card["pains"] = sorted(list(card["pains"]), key=lambda x: json.loads(x)["relevance"], reverse=True)
        final.append(card)

    final.sort(key=lambda x: x["max_relevance"], reverse=True)
    print("Aggregate - final output:", final)
    return final
>>>>>>> Stashed changes
