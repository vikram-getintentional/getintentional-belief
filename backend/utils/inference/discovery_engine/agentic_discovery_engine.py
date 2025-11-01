# agentic_discovery_engine.py
# Orchestrates LLM calls and packages HIGHER-CONTEXT inputs for prompts.
from __future__ import annotations
from datetime import datetime
import inspect
from typing import Any, Callable, Dict, List, Optional, DefaultDict, Tuple
from collections import defaultdict
import json
import traceback
import networkx as nx

from backend.utils.graph_base.schema import EDGES, NEXT_HOPS, RAW_FIELDS_BY_TYPE
from backend.utils.inference.discovery_engine.agentic_engine.agent_context import AgentContext, already_linked_fields, minimal_source_fields, product_pack
from backend.utils.inference.gpt_prompts.agentic_prompts import (
    build_archetypes_relevance_matrix,
    build_hop0_prompt,
    build_hop_plus_prompt,
    build_zmot_for_triggers_prompt,
    build_pain_source_prompt,
    build_capability_expansion_prompt,
    build_pain_expansion_prompt,
    build_job_expansion_prompt,
    build_pain_trigger_expansion_prompt,
    build_attribute_value_expansion_prompt,
    build_zmot_event_expansion_prompt
    
)
from backend.utils.graph_base.agent_graph_builder import CREATE_BY_TYPE, RELATION_BY_PAIR, CanonManager
from backend.utils.graph_base.network_graph import (
    get_node_by_id,
    get_node_id,
    get_nodes_list_ids,
    get_product_id_from_subgraph,
    get_source_nodes_by_target_and_type,
    get_target_nodes_by_source_and_type,
    update_graph,
)
from backend.utils.graph_base.icp_catalog import ICP_CATALOG

# ---------------- LLM util ----------------

def _extract_json_block(text: str) -> str:
    if not text:
        return "[]"
    start = text.find("[")
    obj_start = text.find("{")
    if start == -1 and obj_start == -1:
        return "[]"
    if start == -1 or (obj_start != -1 and obj_start < start):
        start = obj_start
    stack, end = [], start
    for i, ch in enumerate(text[start:], start=start):
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                break
            top = stack.pop()
            if (top == "[" and ch != "]") or (top == "{" and ch != "}"):
                break
            if not stack:
                end = i + 1
                break
    return text[start:end] if end > start else "[]"


def _llm_json(client, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> Any:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    content = resp.choices[0].message.content
    raw_json = _extract_json_block(content)
    try:
        return json.loads(raw_json)
    except Exception:
        try:
            return json.loads(raw_json.replace(",]", "]").replace(",}", "}"))
        except Exception:
            print("❌ LLM JSON parse failed.\nRAW:\n", content)
            traceback.print_exc()
            return []

# ---------------- context packers (read from G) ----------------

def _node_text(n: Dict[str, Any]) -> str:
    # Try common fields; fall back to id
    return n.get("description") or n.get("name") or n.get("title") or n.get("event") or n.get("attribute") or n.get("text") or n.get("id")

def _gather_pain_contexts(G, pain_ids: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for pid in pain_ids:
        p = get_node_by_id(G, pid)
        if not p:
            print(f"⚠️ Warning: Pain ID {pid} not found in graph. Breaking Hop+ context gathering.")
            continue

        jobs = []
        all_personas = []  # accumulate across jobs

        for jid in get_source_nodes_by_target_and_type(G, pid, "solves") or []:
            jn = get_node_by_id(G, jid)
            if not jn:
                continue
            edge_importance = None
            try:
                edge_importance = G[pid][jid].get("weight")
            except Exception:
                pass
            jobs.append({
                "job_to_be_done": jn.get("description") or _node_text(jn),
                "importance": edge_importance if edge_importance is not None else 0.6
            })

            # personas (via job -> performed_by)
            for per_id in get_target_nodes_by_source_and_type(G, jid, "performed_by") or []:
                per = get_node_by_id(G, per_id)
                if not per:
                    continue
                all_personas.append({
                    "title": per.get("title",""),
                    "department": per.get("department",""),
                    "seniority": per.get("seniority","")
                })

        metrics = []
        for mid in get_target_nodes_by_source_and_type(G, pid, "expressed_as") or []:
            mn = get_node_by_id(G, mid)
            if not mn:
                continue
            metrics.append({"metric": mn.get("metric") or _node_text(mn)})

        triggers = []
        for tid in get_target_nodes_by_source_and_type(G, pid, "triggered_by") or []:
            tn = get_node_by_id(G, tid)
            if not tn:
                continue
            triggers.append({"attribute": tn.get("attribute","")})

        out.append({
            "pain_id": pid,
            "pain_text": _node_text(p),
            "felt_in_jobs": jobs[:3],
            "personas": all_personas[:3],  # FIX: accumulated personas
            "perceived_metrics": metrics[:2],
            "pain_triggers": triggers[:2],
        })
    return out



def _gather_trigger_contexts(G, trigger_ids: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tid in trigger_ids:
        t = get_node_by_id(G, tid)
        if not t: 
            continue
        # linked pains (reverse of pain ->triggered_by)
        pains = get_source_nodes_by_target_and_type(G, tid, "triggered_by") or []
        linked_pains = []
        solved_by_jobs = set()
        felt_in_jobs = set()
        personas = []

        for pid in pains:
            pn = get_node_by_id(G, pid)
            if pn:
                linked_pains.append(_node_text(pn))
            # jobs that solve for these pains
            for jid in get_source_nodes_by_target_and_type(G, pid, "solves") or []:
                jn = get_node_by_id(G, jid)
                if jn:
                    solved_by_jobs.add(_node_text(jn))
                # personas for those jobs
                for per_id in get_target_nodes_by_source_and_type(G, jid, "performed_by") or []:
                    per = get_node_by_id(G, per_id)
                    if per:
                        personas.append({
                            "title": per.get("title",""),
                            "department": per.get("department",""),
                            "seniority": per.get("seniority","")
                        })
            # jobs where these pains are "Felt_in"
            for jid in get_target_nodes_by_source_and_type(G, pid, "felt_in") or []:
                jn = get_node_by_id(G, jid)
                if jn:
                    felt_in_jobs.add(_node_text(jn))
                # personas for those jobs
                for per_id in get_target_nodes_by_source_and_type(G, jid, "performed_by") or []:
                    per = get_node_by_id(G, per_id)
                    if per:
                        personas.append({
                            "title": per.get("title",""),
                            "department": per.get("department",""),
                            "seniority": per.get("seniority","")
                        })


        out.append({
            "pain_trigger_id": tid,
            "attribute": t.get("attribute",""),
            "linked_pains": linked_pains[:4],
            "solved_by_jobs": list(solved_by_jobs)[:4],
            "felt_in_jobs": list(felt_in_jobs)[:4], 
            "personas": personas[:4],
        })
    return out


def _gather_capability_contexts(G, capability_ids: List[str]) -> List[Dict[str, Any]]:
    out = []
    for cid in capability_ids:
        c = get_node_by_id(G, cid)
        if not c:
            continue
        out.append({
            "capability_id": cid,
            "name": c.get("name",""),
            "description": c.get("description","")
        })
    return out

def _gather_archetype_contexts(G, archetype_ids: List[str]) -> List[Dict[str, Any]]:
    def _safe_text(n: dict) -> str:
        return n.get("description") or n.get("attribute") or n.get("text") or n.get("name") or ""

    out: List[Dict[str, Any]] = []

    for aid in archetype_ids:
        a = get_node_by_id(G, aid)
        if not a:
            continue

        # pain_trigger --prevalent_in--> archetype
        trigger_ids = get_source_nodes_by_target_and_type(G, aid, "prevalent_in") or []

        related_triggers: List[Dict[str, Any]] = []
        triggered_pains: Dict[str, str] = {}   # id -> text
        solved_jobs:   Dict[str, str] = {}     # id -> text

        persona_set = set()
        personas: List[Dict[str, str]] = []

        for tid in trigger_ids:
            tn = get_node_by_id(G, tid)
            if not tn:
                continue

            # edge weight from trigger to archetype (baseline prevalence)
            try:
                edge_data = G[tid][aid]
                if isinstance(edge_data, dict) and "weight" in edge_data:
                    prevalence = float(edge_data.get("weight") or 0.0)
                else:
                    # networkx can nest attrs under first key sometimes; normalize
                    first_key = next(iter(edge_data.keys()))
                    prevalence = float(edge_data[first_key].get("weight", 0.0))
            except Exception:
                prevalence = 0.0

            related_triggers.append({
                "pain_trigger_id": tid,
                "attribute": tn.get("attribute") or _safe_text(tn),
                "avg_prevalence": prevalence
            })

            # pains --triggered_by--> pain_trigger
            for pid in get_source_nodes_by_target_and_type(G, tid, "triggered_by") or []:
                pn = get_node_by_id(G, pid)
                if pn:
                    triggered_pains[pid] = pn.get("description") or _safe_text(pn)

                # job --solves--> pain (upstream jobs that solve this pain)
                for jid in get_source_nodes_by_target_and_type(G, pid, "solves") or []:
                    jn = get_node_by_id(G, jid)
                    if jn:
                        solved_jobs[jid] = jn.get("description") or _safe_text(jn)

                    # job --performed_by--> persona
                    for per_id in get_target_nodes_by_source_and_type(G, jid, "performed_by") or []:
                        per = get_node_by_id(G, per_id)
                        if per:
                            key = (per.get("title",""), per.get("department",""), per.get("seniority",""))
                            if key not in persona_set:
                                persona_set.add(key)
                                personas.append({
                                    "title": per.get("title",""),
                                    "department": per.get("department",""),
                                    "seniority": per.get("seniority","")
                                })

        # Optional human-friendly label for grounding
        label = " | ".join(filter(None, [
            a.get("industry",""),
            a.get("funding_stage",""),
            a.get("geography","")
        ])) or aid

        out.append({
            "archetype_id": aid,  # canonical graph id (use this in ZMOT output)
            "label": label,
            "industry": a.get("industry",""),
            "revenue_range": a.get("revenue_range",""),
            "employee_range": a.get("employee_range",""),
            "funding_stage": a.get("funding_stage",""),
            "geography": a.get("geography",""),

            # keep tight; sorted by prevalence desc then cap
            "related_pain_triggers": sorted(
                related_triggers, key=lambda r: r.get("avg_prevalence", 0.0), reverse=True
            )[:6],

            # small but diverse anchors (id + text)
            "anchor_examples": {
                "pains":  [{"pain_id": k, "text": v} for k, v in list(triggered_pains.items())[:4]],
                "jobs":   [{"job_id": k,  "text": v} for k, v in list(solved_jobs.items())[:4]],
                "personas": personas[:4]
            }
        })

    return out




# ---------------- helpers to pull context ----------------

def _extract_product_context(G, context):
    product_id = get_node_id(G, "product", {})
    if not product_id:
        raise ValueError("❌ Product ID not found in graph.")

    p = get_node_by_id(G, product_id) or {}
    summary  = p.get("summary")  or getattr(context, "product_summary",  "")
    domain   = p.get("domain")   or getattr(context, "domain",           "")
    industry = p.get("industry") or getattr(context, "industry",         "")

    capability_ids = get_nodes_list_ids(G, "capability", {}) or []
    client  = getattr(context, "client",  None)
    builder = getattr(context, "builder", None)

    if client is None:
        raise RuntimeError("❌ LLM client missing on context.")
    if builder is None:
        raise RuntimeError("❌ CanonManager builder missing on context.")
    
    # 🔒 ensure the builder is graph-bound
    if getattr(builder, "G", None) is None:
        if hasattr(builder, "bind_graph") and callable(builder.bind_graph):
            builder.bind_graph(G)
        else:
            # fallback if someone passed a plain object with a .G slot
            setattr(builder, "G", G)

    # Optionally set product_id / data_source if the instance didn’t get them
    if not getattr(builder, "product_id", None):
        try:
            setattr(builder, "product_id", product_id)
        except Exception:
            pass

    return product_id, summary, domain, industry, capability_ids, client, builder

def _collect_attribute_enums_from_arch_ctx(arch_ctx: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    enums = {"industry": [], "revenue_range": [], "employee_range": [], "funding_stage": [], "geography": []}
    if not arch_ctx:
        return enums
    def add(key, val):
        if val and val not in enums[key]:
            enums[key].append(val)
    for a in arch_ctx:
        add("industry", a.get("industry",""))
        add("revenue_range", a.get("revenue_range",""))
        add("employee_range", a.get("employee_range",""))
        add("funding_stage", a.get("funding_stage",""))
        add("geography", a.get("geography",""))
    # Remove empties
    for k in list(enums.keys()):
        enums[k] = [v for v in enums[k] if v]
    return enums

# ---------------- public API ----------------

def hop0_inference(G, context, cap_ids=None, enrich_capabilities: bool = True) -> list[str]:
    """
    Hop0: capability -> pains -> jobs/personas.
    Returns list of job node ids touched/created.
    """
    print("Processing Hop0 inference...")
    product_id, summary, domain, industry, capability_ids, client, builder = _extract_product_context(G, context)
    if not capability_ids:
        raise ValueError("❌ No capability nodes found in graph for Hop0.")
    if cap_ids is not None:
        capability_ids = [cid for cid in capability_ids if cid in cap_ids]
    cap_ctx = _gather_capability_contexts(G, capability_ids) if enrich_capabilities else None
    if enrich_capabilities and (not cap_ctx or len(cap_ctx) == 0):
        print("⚠️ Warning: No valid capability contexts gathered; skipping Hop0 inference.")
        return []
    print("Capability context gathered. Starting LLM call...")
    user_prompt = build_hop0_prompt(summary, domain, industry, capability_ids, capability_context=cap_ctx)
    print("User prompt for Hop0:", user_prompt)

    data = _llm_json(
        client,
        "You produce Hop0 (capability→pains→felt_in jobs/personas) JSON only.",
        user_prompt,
        temperature=0.12,
    )
    builder.ingest_hop0(hop0_json=data, capability_ids=capability_ids, product_id=product_id)
    touched = builder.flush(G)
    product_subgraph = update_graph(G)
    print("Hop0 inference completed. Touched nodes:", touched)
    print("------------------------------------------------")
    return touched or []


def process_hop_plus_gpt_cache(pain_ids: list[str], G, context) -> list[str]:
    """
    Expand internal pains upstream → solving_jobs/personas and their downstream pains.
    """
    print("Processing Hop+ inference...")
    if not pain_ids:
        return []
    _, summary, domain, industry, _, client, builder = _extract_product_context(G, context)
    
    pain_ctx = _gather_pain_contexts(G, pain_ids)
    if not pain_ctx or len(pain_ctx) == 0:
        print("⚠️ Warning: No valid pain contexts gathered; skipping Hop+ inference.")
        return []
    print("Pain context gathered. Starting LLM call with pain context: \n", pain_ctx)
    print("----------------- LLM call for Hop+ ----------------")
    user_prompt = build_hop_plus_prompt(summary, domain, industry, pain_ctx)
    print("User prompt for Hop+:", user_prompt)

    data = _llm_json(
        client,
        "You expand internal pains upstream into jobs/personas and their downstream pains.",
        user_prompt,
        temperature=0.15,
    )
    
    builder.ingest_hop_plus(gpt_outputs=data, product_id=get_node_id(G, "product", {}))
    touched = builder.flush(G)
    product_subgraph = update_graph(G)
    print("Hop+ inference completed. Touched nodes:", touched)
    print("------------------------------------------------")
    return touched or []




def process_trigger_for_zmot_boosts(trigger_ids: list[str], G, context, enrich_triggers: bool = True) -> list[str]:
    """
    Map PainTrigger → ZMOT events (with per-attribute boosts + labels).
    """
    print("Processing ZMOT inference...")
    if not trigger_ids:
        return []

    _, summary, domain, industry, _, client, builder = _extract_product_context(G, context)

    trig_ctx = _gather_trigger_contexts(G, trigger_ids) if enrich_triggers else [{"pain_trigger_id": tid} for tid in trigger_ids]
    if not trig_ctx:
        print("⚠️ No valid trigger contexts; skipping.")
        return []

    user_prompt = build_zmot_for_triggers_prompt(
        summary, domain, industry,
        trigger_contexts=trig_ctx,
        attributes_dict=context.attributes_dict or ICP_CATALOG
    )
    print("User prompt for ZMOT:", user_prompt)

    data = _llm_json(
        client,
        "You map pain triggers to ZMOTs with observable moments & keywords; per-attribute boosts use LABELS only.",
        user_prompt,
        temperature=0.2,
    )
    print("LLM response for ZMOT:", json.dumps(data, indent=2))
    builder.ingest_zmot(results=data, product_id=get_node_id(G, "product", {}))
    touched = builder.flush(G)
    update_graph(G)
    print("ZMOT inference completed. Touched nodes:", touched)
    return touched or []



def process_trigger_for_archetype_prevalence(trigger_ids: list[str], G, context, enrich_triggers: bool = True) -> list[str]:
    """
    Map PainTrigger → AttributeValue prevalence (labels only).
    """
    print("Processing Trigger Attribute Prevalence inference...")
    if not trigger_ids:
        return []
    _, summary, domain, industry, _, client, builder = _extract_product_context(G, context)

    trig_ctx = _gather_trigger_contexts(G, trigger_ids) if enrich_triggers else [{"pain_trigger_id": tid} for tid in trigger_ids]
    if not trig_ctx:
        print("⚠️ No valid trigger contexts; skipping.")
        return []

    user_prompt = build_archetypes_relevance_matrix(
        summary, domain, industry, trig_ctx, attributes_dict=context.attributes_dict or ICP_CATALOG
    )
    print("User prompt for Trigger Attribute Prevalence:", user_prompt)

    data = _llm_json(
        client,
        "You map pain triggers to attribute values with relevance/likelihood LABELS only.",
        user_prompt,
        temperature=0.15,
    )
    print("LLM response for Trigger Attribute Prevalence:", json.dumps(data, indent=2))
    builder.ingest_trigger_attribute_matrix(results=data, product_id=get_node_id(G, "product", {}))
    touched = builder.flush(G)
    update_graph(G)
    print("Trigger Attribute Prevalence inference completed. Touched nodes:", touched)
    return touched or []



def pain_source_inference(G, pain_ids: list[str], context, enrich_pains: bool = True) -> dict[str, str]:
    """
    Classify pains as internal|external and write result back to the graph.
    Returns mapping pain_id -> source for convenience.
    """
    print("Processing Pain Source inference...")
    if not pain_ids:
        return {}
    _, summary, domain, industry, _, client, _ = _extract_product_context(G, context)

    pain_ctx = _gather_pain_contexts(G, pain_ids) if enrich_pains else [{"pain_id": pid} for pid in pain_ids]
    if not pain_ctx or len(pain_ctx) == 0:
        print("⚠️ Warning: No valid pain contexts gathered; skipping Pain Source inference.")
        return {}
    print("Pain context gathered. Starting LLM call...")
    user_prompt = build_pain_source_prompt(summary, domain, industry, pain_ctx)
    print("User prompt for Pain Source:", user_prompt)

    data = _llm_json(
        client,
        "You classify pains as internal or external, JSON only.",
        user_prompt,
        temperature=0.0,
    )

    result: dict[str, str] = {}
    rows = data if isinstance(data, list) else []
    for row in rows:
        print("Processing row:", row)
        pid = row.get("pain_id")
        src = (row.get("pain_source") or "").strip().lower()
        if pid and src in {"terminal", "non-terminal"}:
            n = get_node_by_id(G, pid)
            if n is not None:
                n["pain_source"] = src
            result[pid] = src

    product_subgraph = update_graph(G)
    print("Pain Source inference completed. Result:", result)
    print("------------------------------------------------")
    return result


#---- Single Frontier Node Addition Logic ----
def _get_product_pack(product_id: str) -> Dict[str, Any]:
    """
    Minimal, stable product pack used by prompts.
    Return keys you already include today in your agentic prompts
    (summary, domain, industry, value_prop, capabilities, etc.)
    """
    # TODO: replace with your real loader
    return {
        "product_id": product_id,
        "summary": "",   # fill from your cache/db
        "domain": "",
        "industry": "",
    }

def _seed_context(G: nx.DiGraph, node_id: str) -> Dict[str, Any]:
    """
    Minimal seed-node context for the LLM. Keep it compact & deterministic.
    """
    d = G.nodes.get(node_id, {})
    return {
        "id": node_id,
        "type": d.get("type") or d.get("node_type"),
        "title": d.get("title"),
        "label": d.get("label"),
        # raw-ish fields by type (keep short)
        "raw": {
            "description": d.get("description"),
            "pain_source": d.get("pain_source"),
            "metric": d.get("metric"),
            "attribute": d.get("attribute"),
            "dimension": d.get("dimension"),
            "name": d.get("name"),
            "event": d.get("event"),
            "text": d.get("text"),
            "title_persona": d.get("title"),
            "department": d.get("department"),
            "seniority": d.get("seniority"),
        }
    }

def _already_linked(G: nx.DiGraph, src: str, target_type: str) -> List[Dict[str, Any]]:
    """
    Provide a compact list of currently-linked targets (by type) so the LLM avoids duplicates.
    """
    out = []
    for _, tgt, data in G.out_edges(src, data=True):
        td = G.nodes.get(tgt, {})
        ttype = td.get("type") or td.get("node_type")
        if ttype == target_type:
            out.append({
                "id": tgt,
                "label": td.get("label") or td.get("title"),
                "description": td.get("description"),
                "metric": td.get("metric"),
                "attribute": td.get("attribute"),
                "dimension": td.get("dimension"),
                "name": td.get("name"),
                "event": td.get("event"),
                "text": td.get("text"),
                "title_persona": td.get("title"),
                "department": td.get("department"),
                "seniority": td.get("seniority"),
            })
    return out

# Target schemas (mirror RAW_FIELDS_BY_TYPE)
RAW_FIELDS_BY_TYPE: Dict[str, List[str]] = {
    "pain": ["description", "pain_source"],
    "job": ["description"],
    "perceived_metric": ["metric"],
    "pain_trigger": ["attribute"],
    "attribute_value": ["dimension", "name"],
    "zmot_event": ["event"],
    "observable_moment": ["text"],
    "keyword": ["text"],
    "persona": ["title", "department", "seniority", "linkedin_profiles"],
}

# Legal expansions per *source* type
SOURCE_TO_TARGETS: Dict[str, List[str]] = {
    "capability": ["pain"],
    "pain": ["job", "pain_trigger", "perceived_metric"],
    "job": ["persona", "pain"],  # solves path
    "pain_trigger": ["attribute_value", "zmot_event"],
    "attribute_value": ["zmot_event"],
    "zmot_event": ["observable_moment", "keyword"],
    "persona": ["job"],  # optional
}

# Relation defaults keyed by (source_type, target_type)
RELATION_BY_PAIR: Dict[Tuple[str, str], str] = {
    ("capability", "pain"): "solves",
    ("pain", "job"): "felt_in",
    ("pain", "perceived_metric"): "expressed_as",
    ("pain", "pain_trigger"): "triggered_by",
    ("pain_trigger", "attribute_value"): "prevalent_in",
    ("pain_trigger", "zmot_event"): "associated_zmot",
    ("attribute_value", "zmot_event"): "associated_zmot",
    ("zmot_event", "observable_moment"): "observed_in",
    ("zmot_event", "keyword"): "keyword",
    ("job", "persona"): "performed_by",
    ("job", "pain"): "solves",
}

def _ntype(G: nx.DiGraph, node_id: str) -> str:
    nd = G.nodes.get(node_id, {}) or {}
    return nd.get("type") or nd.get("node_type") or ""

def _minimal_source_fields(G: nx.DiGraph, node_id: str) -> Dict[str, Any]:
    """Small summary per source used in prompts—safe across types."""
    nd = G.nodes.get(node_id, {}) or {}
    t = _ntype(G, node_id)
    # pick the canonical fields you store on nodes
    if t == "capability":
        return {"id": node_id, "type": t, "name": nd.get("name"), "description": nd.get("description")}
    if t == "pain":
        return {"id": node_id, "type": t, "description": nd.get("description"), "pain_source": nd.get("pain_source")}
    if t == "job":
        return {"id": node_id, "type": t, "description": nd.get("description")}
    if t == "persona":
        return {"id": node_id, "type": t, "title": nd.get("title"), "department": nd.get("department"), "seniority": nd.get("seniority")}
    if t == "perceived_metric":
        return {"id": node_id, "type": t, "metric": nd.get("metric")}
    if t == "pain_trigger":
        return {"id": node_id, "type": t, "attribute": nd.get("attribute")}
    if t == "attribute_value":
        return {"id": node_id, "type": t, "dimension": nd.get("dimension"), "name": nd.get("name")}
    if t == "zmot_event":
        return {"id": node_id, "type": t, "event": nd.get("event")}
    if t == "observable_moment":
        return {"id": node_id, "type": t, "text": nd.get("text")}
    if t == "keyword":
        return {"id": node_id, "type": t, "text": nd.get("text")}
    return {"id": node_id, "type": t}

def _already_linked_targets(G: nx.DiGraph, source_id: str, target_type: str) -> List[Dict[str, Any]]:
    out = []
    for _, tgt, edata in G.out_edges(source_id, data=True):
        if _ntype(G, tgt) == target_type:
            out.append({"id": tgt, "label": G.nodes[tgt].get("label"), "title": G.nodes[tgt].get("title")})
    return out

def _bucket_by_target_type(G: nx.DiGraph, seed_ids: List[str], only_types: List[str] | None) -> Dict[str, List[str]]:
    """
    Decide which target types to expand for each seed, then bucket the seeds.
    If `only_types` is provided, use it. Otherwise infer from source type.
    """
    buckets: Dict[str, List[str]] = {}
    for sid in seed_ids:
        st = _ntype(G, sid)
        if only_types:
            target_types = only_types
        else:
            # default expansion map: what a source typically expands to in one hop
            if st == "capability":
                target_types = ["pain"]
            elif st == "pain":
                target_types = ["job", "pain_trigger", "perceived_metric"]
            elif st == "job":
                target_types = ["persona", "pain"]  # pain here means "solves pain"
            elif st == "pain_trigger":
                target_types = ["attribute_value", "zmot_event"]
            elif st == "attribute_value":
                target_types = ["zmot_event"]  # optional; depends on your graph
            elif st == "zmot_event":
                target_types = ["observable_moment", "keyword"]
            else:
                target_types = []

        for t in target_types:
            buckets.setdefault(t, []).append(sid)

    # dedupe seeds per bucket
    for t in list(buckets.keys()):
        buckets[t] = list(dict.fromkeys(buckets[t]))
    return buckets


def _ingest_with_pid(fn, items, product_id):
            try:
                return fn(items, product_id=product_id)
            except TypeError:
                return fn(items)

def _normalize_type(t: str) -> str:
    t = (t or "").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "jobs": "job",
        "pains": "pain",
        "personas": "persona",
        "metrics": "perceived_metric",
        "metric": "perceived_metric",
        "pain_triggers": "pain_trigger",
        "attributes": "attribute_value",
        "attribute_values": "attribute_value",
        "zmots": "zmot_event",
        "zmot": "zmot_event",
        "moments": "observable_moment",
        "keywords": "keyword",
    }.get(t, t)

def _make_expansion_contexts_for_frontier(G, source_ids: List[str], target_type: str) -> List[Dict[str, Any]]:
    need_fields = RAW_FIELDS_BY_TYPE.get(target_type, [])
    out = []
    for sid in source_ids:
        out.append({
            "source_id": sid,
            "source": _minimal_source_fields(G, sid),
            "already_linked": _already_linked_targets(G, sid, target_type),
            "need_fields": need_fields
        })
    return out


EDGE_REL_KEYS = ("relevance_label", "likelihood_label")
EDGE_BOOST_PAIR = ("pain_trigger", "zmot_event")  # only this pair expects boost_label from LLM

def _normalize_edge_scored_proposals(
    proposals: List[Dict[str, Any]],
    source_type: str,
    target_type: str,
) -> List[Dict[str, Any]]:
    """
    Clone each proposal and attach an 'edge_scores' object containing:
      - relevance_label (if present)
      - likelihood_label (if present)
      - boost_label (ONLY for pain_trigger -> zmot_event, if present)
    We keep labels also on the proposal root for backward compatibility,
    but ingest_* functions can standardize on 'edge_scores'.
    """
    out: List[Dict[str, Any]] = []
    for p in proposals or []:
        # Shallow clone is fine; we do not mutate the original list
        q = dict(p)

        # Collect labels if present
        rel = p.get("relevance_label")
        lik = p.get("likelihood_label")
        boost = p.get("boost_label") if (source_type, target_type) == EDGE_BOOST_PAIR else None

        edge_scores: Dict[str, Any] = {}
        if rel is not None:
            edge_scores["relevance_label"] = rel
        if lik is not None:
            edge_scores["likelihood_label"] = lik
        if boost is not None:
            edge_scores["boost_label"] = boost

        if edge_scores:
            q["edge_scores"] = edge_scores

        out.append(q)
    return out


def expand_frontier_one_layer(
    G: nx.DiGraph,
    seed_ids: List[str],
    *,
    only_types: Optional[List[str]] = None,
    max_items_per_source: int = 5,
    model: str = "gpt-4o-mini",
    ctx: Optional["AgentContext"] = None,
) -> Dict[str, int]:
    """
    Expand exactly one hop from the provided seeds, grouped by target type.
    Returns counts per target type. Mutates G in place (adds nodes/edges).
    """

    # --- 1) Extract shared context exactly like hop0_inference ---
    product_id, summary, domain, industry, capability_ids, client, builder = _extract_product_context(G, context=ctx)
    product_ctx = {"summary": summary, "domain": domain, "industry": industry}
    now_iso = datetime.now().isoformat()

    if builder is None:
        raise RuntimeError("expand_frontier_one_layer: Canon builder is None (check _extract_product_context)")
    if client is None:
        raise RuntimeError("expand_frontier_one_layer: LLM client is None (check _extract_product_context)")

    # Bucket seeds by source type
    by_source: Dict[str, List[str]] = defaultdict(list)
    for sid in seed_ids:
        st = _ntype(G, sid)
        if only_types and st not in (only_types or []):
            continue
        by_source[st].append(sid)

    counts: Dict[str, int] = {}

    for source_type, sources in by_source.items():
        # Build contexts (need_fields is per target, but our prompts already ask for all)
        expansion_node_ids = sources
        expansion_node_contexts = []
        for sid in sources:
            expansion_node_contexts.append({
                "source_id": sid,
                "source": _minimal_source_fields(G, sid),
                "already_linked": [],  # optional: can prefill per-target later
                "need_fields": []      # prompts already encode what to emit
            })

        # --- 2) Select prompt builder for this source_type ---
        if source_type == "job":
            user_prompt = build_job_expansion_prompt(
                product_summary=summary, domain=domain, industry=industry,
                expansion_node_ids=expansion_node_ids,
                expansion_node_contexts=expansion_node_contexts,
                max_items_per_source=max_items_per_source,
            )
        elif source_type == "pain":
            user_prompt = build_pain_expansion_prompt(
                product_summary=summary, domain=domain, industry=industry,
                expansion_node_ids=expansion_node_ids,
                expansion_node_contexts=expansion_node_contexts,
                max_items_per_source=max_items_per_source,
            )
        elif source_type == "capability":
            user_prompt = build_capability_expansion_prompt(
                product_summary=summary, domain=domain, industry=industry,
                expansion_node_ids=expansion_node_ids,
                expansion_node_contexts=expansion_node_contexts,
                max_items_per_source=max_items_per_source,
            )
        elif source_type == "pain_trigger":
            user_prompt = build_pain_trigger_expansion_prompt(
                product_summary=summary, domain=domain, industry=industry,
                expansion_node_ids=expansion_node_ids,
                expansion_node_contexts=expansion_node_contexts,
                max_items_per_source=max_items_per_source,
            )
        elif source_type == "attribute_value":
            user_prompt = build_attribute_value_expansion_prompt(
                product_summary=summary, domain=domain, industry=industry,
                expansion_node_ids=expansion_node_ids,
                expansion_node_contexts=expansion_node_contexts,
                max_items_per_source=max_items_per_source,
            )
        elif source_type == "zmot_event":
            user_prompt = build_zmot_event_expansion_prompt(
                product_summary=summary, domain=domain, industry=industry,
                expansion_node_ids=expansion_node_ids,
                expansion_node_contexts=expansion_node_contexts,
                max_items_per_source=max_items_per_source,
            )
        else:
            raise RuntimeError(f"Unsupported source_type for frontier: {source_type}")

        print(f"[frontier] Calling LLM for source_type={source_type} with {len(sources)} sources...")
        # Optional but VERY handy for debugging malformed label returns
        # print("User prompt for frontier expansion:", user_prompt)

        resp = _llm_json(
            client=client,
            system_prompt="You return JSON ONLY. Never include prose.",
            user_prompt=user_prompt,
            temperature=0.12,
        )

        # --- 3) Normalize response into batches per target type ---
        batches: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        allowed_targets = SOURCE_TO_TARGETS.get(source_type, [])

        key_map = {
            "pain": "pain",
            "pains": "pain",
            "job": "job",
            "jobs": "job",
            "persona": "persona",
            "personas": "persona",
            "perceived_metric": "perceived_metric",
            "metric": "perceived_metric",
            "metrics": "perceived_metric",
            "pain_trigger": "pain_trigger",
            "pain_triggers": "pain_trigger",
            "attribute_value": "attribute_value",
            "attributes": "attribute_value",
            "attribute_values": "attribute_value",
            "zmot_event": "zmot_event",
            "zmot": "zmot_event",
            "observable_moment": "observable_moment",
            "observable_moments": "observable_moment",
            "keyword": "keyword",
            "keywords": "keyword",
        }

        rows = []
        if isinstance(resp, dict):
            rows = resp.get("items") or resp.get("results") or []
        elif isinstance(resp, list):
            rows = resp

        for row in rows or []:
            src = row.get("source_id")
            tblocks = row.get("targets") or {}
            if not src or not isinstance(tblocks, dict):
                continue

            for raw_k, block in tblocks.items():
                tcanon = key_map.get(raw_k)
                if not tcanon:
                    continue
                if tcanon not in allowed_targets:
                    continue
                if not isinstance(block, dict):
                    continue

                proposals = block.get("proposals") or []
                if not proposals:
                    continue

                # --- NEW: attach edge_scores to each proposal based on labels present ---
                scored_proposals = _normalize_edge_scored_proposals(
                    proposals=proposals,
                    source_type=source_type,
                    target_type=tcanon,
                )

                relation = RELATION_BY_PAIR.get((source_type, tcanon))
                batches[tcanon].append({
                    "source_id": src,
                    "relation": relation,
                    "proposals": scored_proposals,            # carries edge_scores inside each proposal
                    "evidence": block.get("evidence") or [],  # block-level why
                })

        # Debug: how much we’re about to ingest for this source_type
        dbg_counts = {k: sum(len(it.get("proposals") or []) for it in v) for k, v in batches.items()}
        print(f"LLM Response normalized for {source_type}:", json.dumps(dbg_counts))

        # --- 4) Ingest per target type (builder.* will now see edge_scores) ---
        created_total = 0
        def _ct(items: List[Dict[str, Any]]) -> int:
            return sum(len(it.get("proposals") or []) for it in items)

        if batches.get("pain"):
            if source_type == "job":
                _ingest_with_pid(builder.ingest_frontier_solves_pain, batches["pain"], product_id)
            else:
                _ingest_with_pid(builder.ingest_frontier_pain, batches["pain"], product_id)
            created_total += _ct(batches["pain"])

        if batches.get("job"):
            _ingest_with_pid(builder.ingest_frontier_job, batches["job"], product_id)
            created_total += _ct(batches["job"])

        if batches.get("persona"):
            _ingest_with_pid(builder.ingest_frontier_persona, batches["persona"], product_id)
            created_total += _ct(batches["persona"])

        if batches.get("perceived_metric"):
            _ingest_with_pid(builder.ingest_frontier_perceived_metric, batches["perceived_metric"], product_id)
            created_total += _ct(batches["perceived_metric"])

        if batches.get("pain_trigger"):
            _ingest_with_pid(builder.ingest_frontier_pain_trigger, batches["pain_trigger"], product_id)
            created_total += _ct(batches["pain_trigger"])

        if batches.get("attribute_value"):
            _ingest_with_pid(builder.ingest_frontier_attribute_value, batches["attribute_value"], product_id)
            created_total += _ct(batches["attribute_value"])

        if batches.get("zmot_event"):
            _ingest_with_pid(builder.ingest_frontier_zmot_event, batches["zmot_event"], product_id)
            created_total += _ct(batches["zmot_event"])

        if batches.get("observable_moment"):
            _ingest_with_pid(builder.ingest_frontier_observable_moment, batches["observable_moment"], product_id)
            created_total += _ct(batches["observable_moment"])

        if batches.get("keyword"):
            _ingest_with_pid(builder.ingest_frontier_keyword, batches["keyword"], product_id)
            created_total += _ct(batches["keyword"])

        # Flush *inside* the loop so each source_type commit lands
        touched = builder.flush(G)
        update_graph(G)

        counts[source_type] = created_total
        print(f"Ingested total for {source_type}: {created_total}")

    # mark timestamp and return per-source_type counts
    try:
        G.graph["last_frontier_expand"] = now_iso
    except Exception:
        pass
    return counts