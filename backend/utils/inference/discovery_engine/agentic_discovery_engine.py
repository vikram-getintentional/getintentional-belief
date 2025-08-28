# agentic_discovery_engine.py
# Orchestrates LLM calls and packages HIGHER-CONTEXT inputs for prompts.
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import traceback

from backend.utils.graph_base.schema import EDGES
from backend.utils.inference.gpt_prompts.agentic_prompts import (
    build_archetypes_relevance_matrix,
    build_hop0_prompt,
    build_hop_plus_prompt,
    build_zmot_for_triggers_prompt,
    build_pain_source_prompt,
)
from backend.utils.graph_base.agent_graph_builder import CanonManager
from backend.utils.graph_base.network_graph import (
    get_node_by_id,
    get_node_id,
    get_nodes_list_ids,
    get_source_nodes_by_target_and_type,
    get_target_nodes_by_source_and_type,
    update_graph,
)

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
        # solving jobs
        jobs = []
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
            personas = []
            for per_id in get_target_nodes_by_source_and_type(G, jid, "performed_by") or []:
                per = get_node_by_id(G, per_id)
                if not per: 
                    continue
                personas.append({
                    "title": per.get("title",""),
                    "department": per.get("department",""),
                    "seniority": per.get("seniority","")
                })

        # perceived metrics
        metrics = []
        for mid in get_target_nodes_by_source_and_type(G, pid, "expressed_as") or []:
            mn = get_node_by_id(G, mid)
            if not mn: 
                continue
            metrics.append({
                "metric": mn.get("metric") or _node_text(mn)
            })

        # triggers
        triggers = []
        for tid in get_target_nodes_by_source_and_type(G, pid, "triggered_by") or []:
            tn = get_node_by_id(G, tid)
            if not tn: 
                continue
            triggers.append({
                "attribute": tn.get("attribute","")
            })

        out.append({
            "pain_id": pid,
            "pain_text": _node_text(p),
            "felt_in_jobs": jobs[:3],
            "personas": personas[:3],
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

    return product_id, summary, domain, industry, capability_ids, client, builder


# ---------------- public API ----------------

def hop0_inference(G, context, enrich_capabilities: bool = True) -> list[str]:
    """
    Hop0: capability -> pains -> jobs/personas.
    Returns list of job node ids touched/created.
    """
    print("Processing Hop0 inference...")
    product_id, summary, domain, industry, capability_ids, client, builder = _extract_product_context(G, context)
    if not capability_ids:
        raise ValueError("❌ No capability nodes found in graph for Hop0.")

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
    Map PainTrigger × Archetype → ZMOT events (with observable moments & keywords).
    - Tries to scope archetypes to only those already linked to the triggers via 'prevalent_in'.
    - If none are linked yet, falls back to all archetypes in the product subgraph.
    """
    print("Processing ZMOT inference...")
    if not trigger_ids:
        print("ℹ️ No trigger_ids provided; skipping ZMOT inference.")
        return []

    # ---- pull base product context
    product_id, summary, domain, industry, _, client, builder = _extract_product_context(G, context)

    # ---- gather trigger contexts
    trig_ctx = (
        _gather_trigger_contexts(G, trigger_ids)
        if enrich_triggers
        else [{"pain_trigger_id": tid} for tid in trigger_ids]
    )

    if not trig_ctx or len(trig_ctx) == 0:
        print("⚠️ Warning: No valid trigger contexts gathered; skipping ZMOT inference.")
        return []

    # ---- gather archetypes scoped to these triggers (prevalent_in), else all archetypes
    prevalent_edge = EDGES.get("PREVALENT_IN", "prevalent_in")
    linked_arch_ids: set[str] = set()
    for tid in trigger_ids:
        # triggers →(prevalent_in)→ archetypes
        for aid in get_target_nodes_by_source_and_type(G, tid, prevalent_edge) or []:
            linked_arch_ids.add(aid)

    if not linked_arch_ids:
        # fallback: all archetypes in the product subgraph
        linked_arch_ids = set(get_nodes_list_ids(G, "archetype", {}) or [])
        if linked_arch_ids:
            print(f"ℹ️ No archetypes linked to triggers; using all {len(linked_arch_ids)} archetypes in graph.")
        else:
            print("ℹ️ No archetype nodes exist in graph; proceeding with trigger-only context.")

    arch_ctx = _gather_archetype_contexts(G, list(linked_arch_ids)) if linked_arch_ids else []


    print("Trigger context gathered. Starting LLM call...")
    user_prompt = build_zmot_for_triggers_prompt(
            summary,
            domain,
            industry,
            archetype_contexts=arch_ctx,      # <-- updated builder signature
            trigger_contexts=trig_ctx,
        )
    print("User prompt for ZMOT:", user_prompt)

    data = _llm_json(
        client,
        "You map pain triggers to ZMOTs with observable moments & keywords.",
        user_prompt,
        temperature=0.2,
    )
    print("LLM response for ZMOT:", json.dumps(data, indent=2))
    builder.ingest_zmot(results=data, product_id=get_node_id(G, "product", {}))
    touched = builder.flush(G)
    product_subgraph = update_graph(G)
    print("ZMOT inference completed. Touched nodes:", touched)
    print("------------------------------------------------")
    return touched or []


def process_trigger_for_archetype_prevalence(trigger_ids: list[str], G, context, icp_catalog: dict | None = None, enrich_triggers: bool = True) -> list[str]:
    """
    Map PainTrigger → ICP archetype prevalence.
    """
    print("Processing Trigger Archetype Prevalence inference...")
    if not trigger_ids:
        return []
    _, summary, domain, industry, _, client, builder = _extract_product_context(G, context)

    trig_ctx = _gather_trigger_contexts(G, trigger_ids) if enrich_triggers else [{"pain_trigger_id": tid} for tid in trigger_ids]
    if not trig_ctx or len(trig_ctx) == 0:
        print("⚠️ Warning: No valid trigger contexts gathered; skipping Trigger Archetype Prevalence inference.")
        return []
    print("Trigger context gathered. Starting LLM call...")
    user_prompt = build_archetypes_relevance_matrix(summary, domain, industry, trig_ctx)
    print("User prompt for Trigger Archetype Prevalence:", user_prompt)

    data = _llm_json(
        client,
        "You map pain triggers to ICP archetypes and prevalence scores.",
        user_prompt,
        temperature=0.15,
    )
    print("LLM response for Trigger Archetype Prevalence:", json.dumps(data, indent=2))
    builder.ingest_archetypes(results=data, product_id=get_node_id(G, "product", {}))
    touched = builder.flush(G)
    product_subgraph = update_graph(G)
    print("Trigger Archetype Relevance Matrix inference completed. Touched nodes:", touched)
    print("------------------------------------------------")
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