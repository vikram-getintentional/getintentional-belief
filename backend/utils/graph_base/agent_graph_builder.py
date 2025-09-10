# agent_graph_builder.py
# Stateful, batch-first graph builder. Collect → canonicalize-in-batch → write nodes/edges.

from __future__ import annotations
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib

import networkx as nx
from networkx.exception import NetworkXError

# batch-first canonicalizer
from backend.utils.knowledge_base import canonicalizer as CAN

from backend.utils.knowledge_base.canonicalizer import (
    canonicalize_pain,
    canonicalize_job,
    canonicalize_persona,
    canonicalize_perceived_metric,
    canonicalize_pain_trigger,
    canonicalize_trigger_events,
    canonicalize_observable_moments,
    canonicalize_keywords,
    canonicalize_archetypes,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _slug(s: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in (s or "")).split())

def _stable_key(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]

def _upsert_node(G: nx.DiGraph, ntype: str, key_parts: Iterable[str], attrs: Optional[Dict[str, Any]] = None) -> str:
    node_key = _stable_key([ntype] + list(key_parts))
    node_id = f"{ntype}:{node_key}"
    if node_id not in G:
        # Set both type and node_type
        G.add_node(node_id, **{"type": ntype, "node_type": ntype, "id": node_id})
    if attrs:
        for k, v in attrs.items():
            if v is None:
                continue
            # avoid blowing away existing attrs; fill only if missing
            if k not in G.nodes[node_id] or G.nodes[node_id][k] in (None, "", []):
                G.nodes[node_id][k] = v
    # Ensure node_type is always present and correct
    G.nodes[node_id]["node_type"] = ntype
    G.nodes[node_id]["type"] = ntype
    return node_id

def _would_create_cycle(G: nx.DiGraph, src: str, tgt: str) -> bool:
    """Adding src->tgt would create a directed cycle if tgt can already reach src (or src==tgt)."""
    if not src or not tgt:
        return False
    if src == tgt:
        return True
    try:
        return nx.has_path(G, tgt, src)
    except NetworkXError:
        # If either node missing, no path exists yet.
        return False

def _upsert_edge(
        G: nx.DiGraph, 
        src: str, 
        type: str, 
        tgt: str, 
        weight: Optional[float] = 1.0, 
        attrs: Optional[Dict[str, Any]] = None, 
        prevent_cycles: bool = True, 
        on_cycle: str = "skip", # "skip" | "raise" | "tag"
        ) -> None:
    # Guard against immediate cycles
    if prevent_cycles and _would_create_cycle(G, src, tgt):
        msg = f"[cycle-blocked] {src} - [{type}] -> {tgt}"
        if on_cycle == "raise":
            raise ValueError(msg)
        if on_cycle == "tag":
            if not G.has_edge(src,tgt):
                G.add_edge(src, tgt, type=f"{type}_blocked_cycle", blocked=True)
            edata = G[src][tgt]
            if isinstance(edata, dict) and "type" not in edata:
                # networkx may wrap attributes under an arbitrary key for multigraphs; normalize
                first_key = next(iter(edata.keys()))
                edata = edata[first_key]
            edata["blocked_reason"]="cycle"
            edata["requested_type"] = type
            return
        # Default: skip but print
        print(msg)
        return
        

    if not G.has_edge(src, tgt):
        G.add_edge(src, tgt, type=type)
    edata = G[src][tgt]
    if isinstance(edata, dict) and "type" not in edata:
        # networkx may wrap attributes under an arbitrary key for multigraphs; normalize
        first_key = next(iter(edata.keys()))
        edata = edata[first_key]
    edata["type"] = type
    edata["weight"] = float(weight)
    if attrs:
        for k, v in attrs.items():
            if v is None:
                continue
            edata[k] = v

# Relevance Labels to Floats:
RELEVANCE_LABEL_MAP = {
    "Critical": 1.0,
    "Core": 0.8,
    "Supportive": 0.6,
    "Ancillary": 0.4,
    "Out-of-scope": 0.0,
}

LIKELIHOOD_LABEL_MAP = {
    "Essential": 0.95,
    "Expected": 0.75,
    "Common": 0.6,
    "Rare": 0.4,
    "Unlikely": 0.2,
}

BOOST_LABEL_MAP = {
    "Very High": 0.95,
    "High": 0.75,
    "Medium": 0.5,
    "Low": 0.3,
    "Negligible": 0.1,
}

def label_to_float(label: str, label_type: str = "relevance") -> float:
    """
    Maps a relevance or likelihood label to a float value.
    label_type: "relevance" or "likelihood"
    """
    if label_type == "relevance":
        return RELEVANCE_LABEL_MAP.get(label, 0.0)
    elif label_type == "likelihood":
        return LIKELIHOOD_LABEL_MAP.get(label, 0.0)
    elif label_type == "boost":
        return BOOST_LABEL_MAP.get(label, 0.0)
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

# canonical node constructors (use canonical strings/fields)

def _product(G: nx.DiGraph, product_id: str, attrs: Optional[Dict[str, Any]] = None) -> str:
    return _upsert_node(G, "product", [product_id], attrs or {"id": product_id})

def _capability(
    G,
    cap_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    attrs: Optional[Dict[str, Any]] = None
) -> str:
    key_parts = [cap_id] if cap_id else [name or "", description or ""]
    # Don't set "id" to cap_id (which may be None); let _upsert_node generate the node_id
    base_attrs = {"name": name, "description": description}
    if attrs:
        base_attrs.update(attrs)
    node_id = _upsert_node(G, "capability", key_parts, base_attrs)
    G.nodes[node_id]["id"] = node_id  # Always set the id attribute to the canonical node ID
    return node_id

def _pain(G, canonical_label: str, pain_source: Optional[str]) -> str:
    return _upsert_node(G, "pain", [_slug(canonical_label)], {"description": canonical_label, "pain_source": pain_source})

def _job(G, canonical_label: str) -> str:
    return _upsert_node(G, "job", [_slug(canonical_label)], {"description": canonical_label})

def _persona(G, title: str, department: str, seniority: str) -> str:
    return _upsert_node(G, "persona", [_slug(title), _slug(department), (seniority or "").lower()],
                        {"title": title, "department": department, "seniority": seniority})

def _metric(G, metric: str) -> str:
    return _upsert_node(G, "perceived_metric", [_slug(metric)], {"metric": metric})

def _trigger(G, attribute: str) -> str:
    return _upsert_node(G, "pain_trigger", [_slug(attribute)], {"attribute": attribute})

def _zmot(G, event: str) -> str:
    return _upsert_node(G, "zmot_event", [_slug(event)], {"event": event})

def _observable(G, text: str) -> str:
    return _upsert_node(G, "observable_moment", [_slug(text)], {"text": text})

def _keyword(G, text: str) -> str:
    return _upsert_node(G, "keyword", [_slug(text)], {"text": text})

def _archetype(G, a: Dict[str, Any]) -> str:
    return _upsert_node(G, "archetype",
                        [a.get("industry","").lower(), a.get("revenue_range","").lower(),
                         a.get("employee_range","").lower(), a.get("funding_stage","").lower(),
                         a.get("geography","").lower()],
                        {"industry": a.get("industry",""),
                         "revenue_range": a.get("revenue_range",""),
                         "employee_range": a.get("employee_range",""),
                         "funding_stage": a.get("funding_stage",""),
                         "geography": a.get("geography","")})

# -----------------------------------------------------------------------------
# CanonManager — batch-first collector + writer
# -----------------------------------------------------------------------------

class CanonManager:
    """
    Collect RAW items → batch-canonicalize with canonicalizer.BatchCanonicalizer → write nodes/edges.

    Usage per stage:
      builder.ingest_hop0(...)
      builder.flush(G)     # canonicalize + write
    """

    def __init__(self):
        self.buf: Dict[str, set] = {
            "pain": set(),
            "job": set(),
            "persona": [],                # (title, dept, seniority)
            "perceived_metric": set(),       # (metric)
            "pain_trigger": set(),           # (attribute)
            "zmot_event": set(),
            "observable_moment": set(),
            "keyword": set(),
            "archetype": [],              # (industry, revenue_range, employee_range, funding_stage, geography)
        }
        self.canon: Dict[str, Dict[Any, Dict[str, Any]]] = {k: {} for k in self.buf.keys()}

        # write plan contains raw payloads grouped by op
        self.plan: List[Tuple[str, Dict[str, Any]]] = []
        self._current_product_id: Optional[str] = None

        # internal batch canonicalizer instance
        self._bc = CAN.BatchCanonicalizer()

    # --------- ingest (collect raw) ---------

    def ingest_hop0(self, hop0_json: List[Dict[str, Any]], capability_ids: List[str], product_id: str):
        print("Hop0 Ingestion started...")
        print("Hop0 Output:\n", json.dumps(hop0_json, indent=2))
        self._current_product_id = product_id
        for cap in hop0_json or []:
            for p in cap.get("pains", []) or []:
                self.buf["pain"].add(p.get("pain","").strip())
                for m in p.get("perceived_metrics", []) or []:
                    text = m.get("text","").strip() if isinstance(m, dict) else (m or "").strip()
                    self.buf["perceived_metric"].add(text)
                
                for j in p.get("felt_in_jobs", []) or []:
                    self.buf["job"].add(j.get("job_to_be_done","").strip())
                    for pr in j.get("personas", []) or []:
                        self.buf["persona"].append({
                            "title": pr.get("title",""),
                            "department": pr.get("department",""),
                            "seniority": pr.get("seniority","")
                        })
                    for sp in j.get("solving_pains", []) or []:
                        self.buf["pain"].add(sp.get("pain","").strip())
                        for sm in sp.get("perceived_metrics", []) or []:
                            text = sm.get("text","").strip() if isinstance(sm, dict) else (sm or "").strip()
                            self.buf["perceived_metric"].add(text)
                        for st in sp.get("pain_triggers", []) or []:
                            text = st.get("text","").strip() if isinstance(st, dict) else (st or "").strip()
                            self.buf["pain_trigger"].add(text)
        self.plan.append(("hop0", {"hop0_json": hop0_json, "capability_ids": capability_ids, "product_id": product_id}))

    def ingest_hop_plus(self, gpt_outputs: List[Dict[str, Any]], product_id: str):
        print("Hop+ Ingestion started...")
        print("Hop+ GPT output:\n", json.dumps(gpt_outputs, indent=2))
        self._current_product_id = product_id
        for item in gpt_outputs or []:
            for sj in item.get("felt_in_jobs", []) or []:
                self.buf["job"].add(sj.get("job_to_be_done","").strip())
                for pr in sj.get("personas", []) or []:
                    self.buf["persona"].append({
                            "title": pr.get("title",""),
                            "department": pr.get("department",""),
                            "seniority": pr.get("seniority","")
                        })
                for dp in sj.get("solving_pains", []) or []:
                    self.buf["pain"].add(dp.get("pain","").strip())
                    for m in dp.get("perceived_metrics", []) or []:
                        text = m.get("text","").strip() if isinstance(m, dict) else (m or "").strip()
                        if text:
                            self.buf["perceived_metric"].add(text)
                    for t in dp.get("pain_triggers", []) or []:
                        text = t.get("text","").strip() if isinstance(t, dict) else (t or "").strip()
                        if text:
                            self.buf["pain_trigger"].add(text)
        self.plan.append(("hop_plus", {"gpt_outputs": gpt_outputs, "product_id": product_id}))

    def ingest_zmot(self, results: Dict[str, Any], product_id: str):
        """
        Ingest ZMOT results in matrix format:
        results:
            "archetype_id": "string",
            "original_pain_trigger_id": "string",
            "zmot_triggers": [
                "trigger_event": "string",
                "match_score": float,
                "boost_score": float,
                "observable_moments": [{"observable_moment": "string", "match_score": float}],
                "trigger_keywords": [{"keyword": "string", "match_score": float}]
            ]
        ]
        """
        print("ZMOT Ingestion started...")
        self._current_product_id = product_id

        # Normalize to rows
        rows = results.get("zmot_matrix") if isinstance(results, dict) else results
        rows = rows or []

        for item in rows:
            for z in item.get("zmot_triggers", []) or []:
                ev = (z.get("trigger_event") or "").strip()
                if ev: self.buf["zmot_event"].add(ev)
                for om in z.get("observable_moments", []) or []:
                    omt = om.get("observable_moment")
                    if isinstance(om, dict):
                        text = omt.get("text", "").strip() if isinstance(omt, dict) else (omt or "").strip()
                        self.buf["observable_moment"].add(text)
                for kw in z.get("trigger_keywords", []) or []:
                    k = (kw.get("keyword"))
                    if isinstance(kw, dict):
                        text = kw.get("text", "").strip()
                        self.buf["keyword"].add(text)

        # Enqueue ONE op that the emitter understands
        self.plan.append(("zmot", {"results": results, "product_id": product_id}))
        print("ZMOT Ingestion complete.")




    def ingest_archetypes(self, results: Dict[str, Any], product_id: str):
        print("Archetype Ingestion started...")
        self._current_product_id = product_id
        if not (isinstance(results, dict) and 
            "archetypes" in results and 
            "relevance_matrix" in results):
            print("⚠️ ingest_archetypes expected dict with 'archetypes' and 'relevance_matrix'. Skipping.")
            return
        if isinstance(results, dict) and "archetypes" in results and "relevance_matrix" in results:
            for a in results.get("archetypes", []) or []:
                self.buf["archetype"].append({
                    "industry": a.get("industry",""),
                    "revenue_range": a.get("revenue_range",""),
                    "employee_range": a.get("employee_range",""),
                    "funding_stage": a.get("funding_stage",""),
                    "geography": a.get("geography","")
                })
        self.plan.append(("archetype", {"results": results, "product_id": product_id}))
        print("Archetype Ingestion complete.")
        return


    # --------- flush (canonicalize batch + write) ---------

    def flush(self, G: nx.DiGraph) -> List[str]:
        """
        1) Batch-canonicalize everything buffered.
        2) Emit graph nodes/edges with canonical IDs.
        3) Clear buffers for the next stage.
        Returns a list of "touched" node ids (useful for tests).
        """
        touched: List[str] = []
        if self._current_product_id:
            _product(G, self._current_product_id)

        # 1) canonicalize batches
        self.canon = {}
        if "pain" in self.buf:
            self.canon["pain"] = canonicalize_pain(self.buf["pain"])
        if "job" in self.buf:
            self.canon["job"] = canonicalize_job(self.buf["job"])
        if "persona" in self.buf:
            self.canon["persona"] = canonicalize_persona(self.buf["persona"])
        if "perceived_metric" in self.buf:
            self.canon["perceived_metric"] = canonicalize_perceived_metric(self.buf["perceived_metric"])
        if "pain_trigger" in self.buf:
            self.canon["pain_trigger"] = canonicalize_pain_trigger(self.buf["pain_trigger"])
        if "zmot_event" in self.buf:
            self.canon["zmot_event"] = canonicalize_trigger_events(self.buf["zmot_event"])
        if "observable_moment" in self.buf:
            self.canon["observable_moment"] = canonicalize_observable_moments(self.buf["observable_moment"])
        if "keyword" in self.buf:
            self.canon["keyword"] = canonicalize_keywords(self.buf["keyword"])
        if "archetype" in self.buf:
            self.canon["archetype"] = canonicalize_archetypes(self.buf["archetype"])

        # 2) resolve plan and write
        for op, payload in self.plan:
            if op == "hop0":
                touched += self._emit_hop0(G, payload)
            elif op == "hop_plus":
                touched += self._emit_hop_plus(G, payload)
            elif op == "zmot":
                touched += self._emit_zmot(G, payload)
            elif op == "archetype":
                touched += self._emit_archetype(G, payload)

        # 3) clear buffers and plan
        for k in self.buf:
            self.buf[k].clear()
        self.plan.clear()

        return list(dict.fromkeys(touched))  # unique

    # --------- internal emitters (canonicalized writes) ---------

    def _emit_hop0(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("Hop0 Emission started...")
        hop0_json = payload["hop0_json"]
        print("Hop0 Emission Payload:\n", json.dumps(hop0_json, indent=2))
        capability_ids = payload["capability_ids"]
        product_id = payload["product_id"]

        touched: List[str] = []
        _product(G, product_id)
        print("Starting capability traversal")
        caps_list = []
        for cap_obj_ordered in hop0_json or []:
            cap_id = cap_obj_ordered.get("capability_id")
            if not cap_id:
                print("Capability with missing ID. Using blank instead.")
                cap_id = ""
            caps_list.append(cap_id)

        for cap_obj in hop0_json or []:
            cap_id = cap_obj.get("capability_id")
            if not cap_id:
                continue
            for p in cap_obj.get("pains", []) or []:
                print("Processing pains...")
                print("pain content:", p, "of type:", type(p))
                pain_raw = p.get("pain","").strip()
                pain_can = self.canon["pain"].get(pain_raw, {"canonical_label": pain_raw})
                if isinstance(pain_can, dict):
                    pain_id = _pain(G, pain_can.get("canonical_label", pain_raw), p.get("pain_source"))
                else:
                    pain_id = _pain(G, pain_can, p.get("pain_source"))
                touched.append(pain_id)
                relevance_label = p.get("relevance_label", "")
                likelihood_label = p.get("likelihood_label", "")
                relevance = label_to_float(relevance_label, "relevance")
                likelihood = label_to_float(likelihood_label, "likelihood")
                
                _upsert_edge(G, cap_id, "solves", pain_id, weight = relevance, attrs={"relevance": relevance, "likelihood": likelihood})       
                print("Capability-Pain Nodes and Edges Added")
                    

                for metric in p.get("perceived_metrics", []) or []:
                    print("Processing perceived metrics...")
                    text = metric.get("text", "").strip() if isinstance(metric, dict) else (metric or "").strip()
                    can_metric = self.canon["perceived_metric"].get(text, text)
                    m_id = _metric(G, can_metric)
                    relevance = label_to_float(metric.get("relevance_label", ""), "relevance") if isinstance(metric, dict) else 0.0
                    likelihood = label_to_float(metric.get("likelihood_label", ""), "likelihood") if isinstance(metric, dict) else 0.0
                    _upsert_edge(G, pain_id, "expressed_as", m_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                    touched.append(m_id)
                print("Pain-Metric Nodes and Edges Added")

                

                for j in p.get("felt_in_jobs", []) or []:
                    print("Processing felt_in_jobs...")
                    job_raw = j.get("job_to_be_done","").strip()
                    job_can = self.canon["job"].get(job_raw, {"canonical_label": job_raw})
                    if isinstance(job_can, dict):
                        job_id = _job(G, job_can.get("canonical_label"))
                    else:
                        job_id = _job(G, job_can)
                    relevance = label_to_float(j.get("relevance_label", ""), "relevance")
                    likelihood = label_to_float(j.get("likelihood_label", ""), "likelihood")
                    _upsert_edge(G, pain_id, "felt_in", job_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                    touched.append(job_id)
                    print("Pain-Job Nodes and Edges Added")
                    for pr in j.get("personas", []) or []:
                        key = {
                            "title": pr.get("title",""),
                            "department": pr.get("department",""),
                            "seniority": pr.get("seniority","")
                        }
                        pr_can = self.canon["persona"].get(str(key), key)
                        persona_id = _persona(G, pr_can["title"], pr_can["department"], pr_can["seniority"])
                        relevance = label_to_float(pr.get("relevance_label", ""), "relevance") if isinstance(pr, dict) else 0.0
                        likelihood = label_to_float(pr.get("likelihood_label", ""), "likelihood") if isinstance(pr, dict) else 0.0
                        _upsert_edge(G, job_id, "performed_by", persona_id, weight = relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                        touched.append(persona_id)
                    print("Persona nodes added")
                    
                    for sp in j.get("solving_pains", []) or []:
                        print("Solving Pain nodes started")
                        pain_raw = sp.get("pain","").strip()
                        pain_can = self.canon["pain"].get(pain_raw, {"canonical_label": pain_raw})
                        if isinstance(pain_can, dict):
                            sp_id = _pain(G, pain_can.get("canonical_label"), sp.get("pain_source"))
                        else:
                            sp_id = _pain(G, pain_can, sp.get("pain_source"))
                        if sp_id == pain_id:
                            print("Solving pain is the same as original pain. Skipping.")
                            continue
                        relevance = label_to_float(sp.get("relevance_label", ""), "relevance")
                        likelihood = label_to_float(sp.get("likelihood_label", ""), "likelihood")
                        _upsert_edge(G, job_id, "solves", sp_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                        touched.append(sp_id)

                        for metric in sp.get("perceived_metrics", []) or []:
                            text = metric.get("text", "").strip() if isinstance(metric, dict) else (metric or "").strip()
                            can_metric = self.canon["perceived_metric"].get(text, text)
                            m_id = _metric(G, can_metric)
                            relevance = label_to_float(metric.get("relevance_label", ""), "relevance") if isinstance(metric, dict) else 0.0
                            likelihood = label_to_float(metric.get("likelihood_label", ""), "likelihood") if isinstance(metric, dict) else 0.0
                            _upsert_edge(G, sp_id, "expressed_as", m_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                            touched.append(m_id) 
                        print("Pain-Metric Nodes and Edges Added")

                        for attr in sp.get("pain_triggers", []) or []:
                            print("Processing pain triggers...")
                            text = attr.get("text", "").strip() if isinstance(attr, dict) else (attr or "").strip()
                            can_attr = self.canon["pain_trigger"].get(text, text)
                            t_id = _trigger(G, can_attr)
                            relevance = label_to_float(attr.get("relevance_label", ""), "relevance") if isinstance(attr, dict) else 0.0
                            likelihood = label_to_float(attr.get("likelihood_label", ""), "likelihood") if isinstance(attr, dict) else 0.0
                            _upsert_edge(G, sp_id, "triggered_by", t_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                            touched.append(t_id)
                        print("Pain-Trigger Nodes and Edges Added")
                        print("Solving Pain - Trigger Nodes added")
                    print("Solving pain nodes completed")
                print("Felt in jobs nodes completed")
            print("Emission for Hop0 completed")

        return touched

    def _emit_hop_plus(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("Hop Plus Emission started...")
        gpt_outputs = payload["gpt_outputs"]
        print("Hop+ Emission Payload:\n", json.dumps(gpt_outputs, indent=2))
        product_id = payload["product_id"]
        _product(G, product_id)
        touched: List[str] = []

        for item in gpt_outputs or []:
            orig_pain_id = item.get("original_pain_id")
            if not orig_pain_id:
                continue
            if orig_pain_id not in G:
                # create placeholder if missing
                print("Original pain ID not found. Skipping entry but something is wrong.")
                continue

            for sj in item.get("felt_in_jobs", []) or []:
                print("Processing felt_in_jobs...")
                job_raw = sj.get("job_to_be_done","").strip()
                job_can = self.canon["job"].get(job_raw, {"canonical_label": job_raw})
                if isinstance(job_can, dict):
                    job_id = _job(G, job_can.get("canonical_label"))
                else:
                    job_id = _job(G, job_can)
                relevance = label_to_float(sj.get("relevance_label", ""), "relevance")
                likelihood = label_to_float(sj.get("likelihood_label", ""), "likelihood")
                touched.append(job_id)
                _upsert_edge(G, orig_pain_id, "felt_in", job_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                print("Pain-Job Nodes and Edges Added")
                

                for pr in sj.get("personas", []) or []:
                    print("Processing personas...")
                    key = {
                        "title": pr.get("title",""),
                        "department": pr.get("department",""),
                        "seniority": pr.get("seniority","")
                    }
                    pr_can = self.canon["persona"].get(str(key), key)
                    persona_id = _persona(G, pr_can["title"], pr_can["department"], pr_can["seniority"])
                    relevance = label_to_float(pr.get("relevance_label", ""), "relevance") if isinstance(pr, dict) else 0.0
                    likelihood = label_to_float(pr.get("likelihood_label", ""), "likelihood") if isinstance(pr, dict) else 0.0
                    touched.append(persona_id)
                    _upsert_edge(G, job_id, "performed_by", persona_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                print("Persona nodes added")

                for dp in sj.get("solving_pains", []) or []:
                    print("Processing solving pains...")
                    pain_raw = dp.get("pain","").strip()
                    pain_can = self.canon["pain"].get(pain_raw, {"canonical_label": pain_raw})
                    pain_source = dp.get("pain_source")
                    if isinstance(pain_can, dict):
                        dp_id = _pain(G, pain_can.get("canonical_label"), pain_source)
                    else:
                        dp_id = _pain(G, pain_can, pain_source)
                    touched.append(dp_id)
                    if dp_id == orig_pain_id:
                        print("Solving pain is the same as original pain. Skipping.")
                        continue
                    relevance = label_to_float(dp.get("relevance_label", ""), "relevance")
                    likelihood = label_to_float(dp.get("likelihood_label", ""), "likelihood")
                    _upsert_edge(G, job_id, "solves", dp_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                    

                    for metric in dp.get("perceived_metrics", []) or []:
                        text = metric.get("text", "").strip() if isinstance(metric, dict) else (metric or "").strip()
                        can_metric = self.canon["perceived_metric"].get(text, text)
                        m_id = _metric(G, can_metric)
                        relevance = label_to_float(metric.get("relevance_label", ""), "relevance") if isinstance(metric, dict) else 0.0
                        likelihood = label_to_float(metric.get("likelihood_label", ""), "likelihood") if isinstance(metric, dict) else 0.0
                        _upsert_edge(G, dp_id, "expressed_as", m_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                    print("Pain-Metric Nodes and Edges Added")

                    for attr in dp.get("pain_triggers", []) or []:
                        print("Processing pain triggers...")
                        text = attr.get("text", "").strip() if isinstance(attr, dict) else (attr or "").strip()
                        can_attr = self.canon["pain_trigger"].get(text, text)
                        t_id = _trigger(G, can_attr)
                        relevance = label_to_float(attr.get("relevance_label", ""), "relevance") if isinstance(attr, dict) else 0.0
                        likelihood = label_to_float(attr.get("likelihood_label", ""), "likelihood") if isinstance(attr, dict) else 0.0
                        _upsert_edge(G, dp_id, "triggered_by", t_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                        touched.append(t_id)
                    print("Pain-Trigger Nodes and Edges Added")
        print("Hop+ Emission completed")        

        return touched

    def _emit_zmot(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("ZMOT Emission started...")
        results = payload["results"]
        product_id = payload["product_id"]
        _product(G, product_id)
        touched: List[str] = []

        # Support both {"zmot_matrix":[...]} and just a list
        rows = results.get("zmot_matrix") if isinstance(results, dict) else results
        rows = rows or []

        canon_z = self.canon.get("zmot_event", {}) or {}
        canon_om = self.canon.get("observable_moment", {}) or {}
        canon_kw = self.canon.get("keyword", {}) or {}

        for item in rows:
            trig_id = (item.get("original_pain_trigger_id") or "").strip()
            arch_id = (item.get("archetype_id") or "").strip()

            if not trig_id:
                print("⚠️ Missing original_pain_trigger_id; skipping row.")
                continue
            if trig_id not in G:
                print(f"⚠️ Trigger ID {trig_id} not found in graph; skipping row.")
                continue

            touched.append(trig_id)

            for z in item.get("zmot_triggers", []) or []:
                print("Processing ZMOT trigger…")
                ev_raw = (z.get("trigger_event") or "").strip()
                if not ev_raw:
                    print("⚠️ Missing trigger_event; skipping this zmot_trigger.")
                    continue

                ev_can = canon_z.get(ev_raw, {"canonical_label": ev_raw})
                ev_label = ev_can.get("canonical_label", ev_raw) if isinstance(ev_can, dict) else ev_can
                zmot_id = _zmot(G, ev_label)
                touched.append(zmot_id)

                # PainTrigger → accelerated_by → ZMOT
                likelihood = label_to_float(z.get("event_to_archetype_likelihood_label", ""), "likelihood")
                boost = label_to_float(z.get("boost_label", ""), "boost")
                

                
                if not arch_id or arch_id not in G:
                    print(f"⚠️ Missing or unknown archetype_id '{arch_id}'; skipping archetype↔event edge.")
                    continue
                _upsert_edge(
                    G, arch_id, "relevant_event", zmot_id,
                    weight=likelihood if likelihood is not None else 0.0,
                    attrs={"boost": boost, "likelihood": likelihood}
                )
                _upsert_edge(
                    G, trig_id, "accelerated_by", zmot_id,
                    weight=likelihood if likelihood is not None else 0.0,
                    attrs={"archetype_id": arch_id, "likelihood": likelihood, "boost": boost}
                )
                touched.append(arch_id)

                # ZMOT → observed_in → ObservableMoment(s)
                for om in z.get("observable_moments", []) or []:
                    om_raw = om.get("observable_moment")
                    if not om_raw:
                        continue
                    om_text = om_raw.get("text", "").strip() if isinstance(om_raw, dict) else (om_raw or "").strip()
                    if not om_text:
                        print("⚠️ Missing observable_moment text; skipping this observable moment.")
                        continue

                    om_can = canon_om.get(om_text, {"canonical_label": om_text})
                    om_label = om_can.get("canonical_label", om_text) if isinstance(om_can, dict) else om_can
                    om_id = _observable(G, om_label)
                    relevance = label_to_float(om_raw.get("relevance_label", ""), "relevance") if isinstance(om_raw, dict) else 0.0
                    likelihood = label_to_float(om_raw.get("likelihood_label", ""), "likelihood") if isinstance(om_raw, dict) else 0.0
                    _upsert_edge(G, zmot_id, "observed_in", om_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})

                # ZMOT → associated_with → Keyword(s)
                for kw in z.get("trigger_keywords", []) or []:
                    kw_raw = kw.get("keyword")
                    if not kw_raw:
                        continue
                    kw_text = kw_raw.get("text", "").strip() if isinstance(kw_raw, dict) else (kw_raw or "").strip()
                    kw_can = canon_kw.get(kw_text, {"canonical_label": kw_text})
                    kw_label = kw_can.get("canonical_label", kw_text) if isinstance(kw_can, dict) else kw_can
                    kw_id = _keyword(G, kw_label)
                    relevance = label_to_float(kw_raw.get("relevance_label", ""), "relevance") if isinstance(kw_raw, dict) else 0.0
                    likelihood = label_to_float(kw_raw.get("likelihood_label", ""), "likelihood") if isinstance(kw_raw, dict) else 0.0
                    touched.append(kw_id)
                    _upsert_edge(G, zmot_id, "associated_with", kw_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})

        print("ZMOT Emission completed")
        return list(dict.fromkeys(touched))  # unique


    def _emit_archetype(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("Archetype Emission started...")
        data = payload["results"]
        product_id = payload["product_id"]
        _product(G, product_id)
        touched: List[str] = []
        gpt_to_archid: Dict[str, str] = {}

        for a in data.get("archetypes", []) or []:
            print("Processing archetypes...")
            gpt_id = a.get("archetype_id")
            if not gpt_id:
                print("⚠️ Missing archetype_id on item; keeping node but not mapping.")
            # Use a canonical key (stringified dict)
            key_dict = {
                "industry": a.get("industry",""),
                "revenue_range": a.get("revenue_range",""),
                "employee_range": a.get("employee_range",""),
                "funding_stage": a.get("funding_stage",""),
                "geography": a.get("geography","")
            }
            key_str = json.dumps(key_dict, sort_keys=True)
            a_can = self.canon.get("archetype", {}).get(key_str, key_dict)
            arch_id = _archetype(G, a_can)

            # store provenance & JSON fields for debugging/analytics
            if gpt_id:
                G.nodes[arch_id]["gpt_archetype_id"] = gpt_id
            G.nodes[arch_id]["industry"] = a_can["industry"]
            G.nodes[arch_id]["revenue_range"] = a_can["revenue_range"]
            G.nodes[arch_id]["employee_range"] = a_can["employee_range"]
            G.nodes[arch_id]["funding_stage"] = a_can["funding_stage"]
            G.nodes[arch_id]["geography"] = a_can["geography"]

            gpt_to_archid[gpt_id] = arch_id


        for row in data.get("relevance_matrix", []) or []:
            
            print("Processing relevance matrix...")
            arch_ref = row.get("archetype_ref")
            if not arch_ref:
                print("Relevance matrix entry without archetype_ref. Skipping.")
                continue
            arch_node = gpt_to_archid.get(arch_ref)
            if not arch_node:
                print(f"Archetype reference {arch_ref} not found in gpt_to_archid mapping. Skipping.")
                continue
            for ts in row.get("trigger_scores", []) or []:
                pain_trigger_id = ts.get("pain_trigger_id")
                if not pain_trigger_id:
                    print("Trigger score entry without pain_trigger_id. Skipping.")
                    continue
                if pain_trigger_id not in G:
                    print(f"Pain trigger ID {pain_trigger_id} not found in graph. Skipping.")
                    continue
                relevance = label_to_float(ts.get("relevance_label", ""), "relevance") if isinstance(ts, dict) else 0.0
                likelihood = label_to_float(ts.get("likelihood_label", ""), "likelihood") if isinstance(ts, dict) else 0.0
                _upsert_edge(G, pain_trigger_id, "prevalent_in", arch_node, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                touched.append(pain_trigger_id)
                

        print("Archetype Emission completed")
        return touched
