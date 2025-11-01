from datetime import datetime
import re
import networkx as nx
from typing import Any, Dict, List, Set, Optional

from backend.utils.graph_base.agent_graph_builder import (
    _attribute_value, 
    _product, 
    _capability, 
    _pain, 
    _job, 
    _persona,
    _slug,
    _stable_key, 
    _trigger,
    _metric,
    _upsert_edge,
    _upsert_node,
    _zmot, 
    _keyword, 
    _observable, 
    current_timestamp
    )
from backend.utils.graph_base.network_graph import (
    _set_node_label,
    build_product_graph, 
    get_edge_attribute, 
    get_node_by_id, 
    get_product_id_from_subgraph, 
    get_target_nodes_by_source_and_type, 
    update_graph
    )


def _normalize_node(G: nx.DiGraph, n_id: Any) -> Dict:
    raw = get_node_by_id(G, n_id) or {}
    node_type = raw.get("type") or raw.get("node_type") or ""
    label = _set_node_label(G, n_id) if callable(_set_node_label) else (raw.get("label") or raw.get("title") or str(n_id))
    title = raw.get("title") or raw.get("label") or label
    content = raw.get("content") or raw.get("description") or raw.get("text") or ""
    source_nodes = G.predecessors(n_id) if n_id in G else []
    sources = {}
    for src_id in source_nodes:
        relevance = get_edge_attribute(G, src_id, n_id, "relevance") or 0.0
        likelihood = get_edge_attribute(G, src_id, n_id, "likelihood") or 0.0
        boost = get_edge_attribute(G, src_id, n_id, "boost") or 0.0
        if not relevance or not likelihood:
            print("Rel or likelihood missing for:", src_id, "->", n_id)
        sources[str(src_id)] = {
            "relevance": relevance,
            "likelihood": likelihood,
            "boost": boost,
        }
            
    return {
        "id": str(n_id),
        "label": label,
        "type": node_type,
        "title": title,
        "content": content,
        "sources": sources,
        "properties": raw.get("properties", {}),
        "raw": {k: v for k, v in raw.items()},
    }


def serialize_graph_for_frontend(G: nx.DiGraph) -> Dict:
    """
    Produce a JSON-friendly dict:
      {
        "nodes": [{ id, type, title, content, properties, raw, sources? }],
        "edges": [{ source, target, relation, weight, relevance, likelihood, boost }]
      }
    Frontend will use nodes[] plus edges[] (and build outgoing-edge index).
    """
    nodes: List[Dict] = []
    edges: List[Dict] = []

    for n_id, n_attrs in G.nodes(data=True):
        # use your _normalize_node to keep consistent fields (it already includes sources)
        try:
            nn = _normalize_node(G, n_id)
        except Exception:
            raw = get_node_by_id(G, n_id) or {}
            nn = {
                "id": str(n_id),
                "type": raw.get("type") or raw.get("node_type") or "",
                "title": raw.get("label") or str(n_id),
                "content": raw.get("content") or raw.get("description") or "",
                "properties": raw.get("properties", {}),
                "raw": dict(raw),
                "sources": {},  # fallback
            }
        nodes.append(nn)

    for src, tgt, ed in G.out_edges(data=True):
        # normalize edge attributes
        if isinstance(ed, dict):
            relation = ed.get("relation") or ed.get("type") or ed.get("edge_type") or None
            weight = ed.get("weight")
            relevance = ed.get("relevance")
            likelihood = ed.get("likelihood")
            boost = ed.get("boost")
        else:
            relation = ed if isinstance(ed, str) else None
            weight = relevance = likelihood = boost = None

        edges.append({
            "source": str(src),
            "target": str(tgt),
            "relation": relation,
            "weight": weight,
            "relevance": relevance,
            "likelihood": likelihood,
            "boost": boost,
            "raw": ed if isinstance(ed, dict) else {},
        })

    return {"nodes": nodes, "edges": edges}

#==================================================
# GRAPH MERGE / USER EDIT MANAGEMENT
#==================================================

#---- Helpers and code for merge/ update user graph edits

def _now() -> str:
    return datetime.now().isoformat()

# Editable, canonical (top-level) fields by type
EDITABLE_FIELDS: Dict[str, List[str]] = {
    "capability": ["name", "description"],
    "pain": ["description", "pain_source"],
    "job": ["description"],
    "persona": ["title", "department", "seniority", "linkedin_profiles"],
    "perceived_metric": ["metric"],
    "metric": ["metric"],
    "pain_trigger": ["attribute"],
    "attribute_value": ["dimension", "name"],
    "zmot_event": ["event"],
    "observable_moment": ["text"],
    "keyword": ["text"],
}

# -------- field extraction (prefer top-level; fallback to nested raw for compatibility) ----------
def _extract_edit_fields(ntype: str, incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Pull user-editable canonical fields from top-level (preferred)."""
    t = (ntype or "").lower()
    allowed = set(EDITABLE_FIELDS.get(t, []))
    out: Dict[str, Any] = {}
    for k in allowed:
        if k in incoming and incoming[k] is not None:
            out[k] = incoming[k]
    # Back-compat: if some UIs still send under src["raw"], we can (optionally) fallback:
    if not out and isinstance(incoming.get("raw"), dict):
       nested = incoming["raw"]
       for k in allowed:
           if k in nested and nested[k] is not None:
                out[k] = nested[k]
    return out

def _extract_node_evidence(incoming: Dict[str, Any]) -> Optional[str]:
    # Node evidence, if present; prefer top-level "evidence"
    if "evidence" in incoming and incoming["evidence"] is not None:
        return incoming["evidence"]
    nested = incoming.get("raw") if isinstance(incoming.get("raw"), dict) else {}
    if "evidence" in nested and nested["evidence"] is not None:
        return nested["evidence"]
    return None

# -----------------------------
# 2) Canonicalize → canonical_label
# -----------------------------
_WS = re.compile(r"\s+")

def _canon_text(s: Optional[str]) -> str:
    """Basic canonicalization: trim and collapse whitespace. Extend if needed."""
    if not s:
        return ""
    return _WS.sub(" ", str(s)).strip()

def _canonical_label_for(ntype: str, f: Dict[str, Any]) -> str:
    """
    Build the canonical_label used by agent_graph_builder for ID stability.
    Keep this exactly in sync with how your builder expects to form the identity.
    """
    t = (ntype or "").lower()

    if t == "capability":
        # Prefer name; optionally include description to disambiguate
        name = _canon_text(f.get("name"))
        desc = _canon_text(f.get("description"))
        return name or desc

    if t == "pain":
        return _canon_text(f.get("description"))

    if t == "job":
        return _canon_text(f.get("description"))

    if t == "persona":
        parts = [
            _canon_text(f.get("title")),
            _canon_text(f.get("seniority")),
            _canon_text(f.get("department")),
        ]
        # Only include non-empty parts, order matters for stable key
        return " • ".join([p for p in parts if p])

    if t in ("perceived_metric", "metric"):
        return _canon_text(f.get("metric"))

    if t in ("pain_trigger", "trigger"):
        return _canon_text(f.get("attribute"))

    if t == "attribute_value":
        dim = _canon_text(f.get("dimension")).lower()
        name = _canon_text(f.get("name"))
        return f"{dim}:{name}" if dim or name else ""

    if t == "zmot_event":
        return _canon_text(f.get("event"))

    if t == "observable_moment":
        return _canon_text(f.get("text"))

    if t == "keyword":
        return _canon_text(f.get("text"))

    # Unknown types: try a generic canonical label from any present strings
    return _canon_text(" ".join([str(v) for v in f.values() if isinstance(v, str)]))

# ------------------------------------
# 3) CREATE via canonical_label only
# ------------------------------------
def _create_node_canonical(G: nx.DiGraph, ntype: str, fields: Dict[str, Any]) -> str:
    """
    Create nodes using the agent_graph_builder canonical_label API.
    Only pass canonical_label (and any required extra kwargs like pain_source).
    """
    t = (ntype or "").lower()
    canonical_label = _canonical_label_for(t, fields)

    if t == "capability":
        # If your capability builder expects canonical_label, use that.
        # If it expects (name/description), you can keep the two-arg version:
        # return _capability(G, name=fields.get("name"), description=fields.get("description"))
        return _capability(G, canonical_label=canonical_label)

    if t == "pain":
        # pain_source is an extra qualifier often used in your code
        return _pain(G, canonical_label=canonical_label, pain_source=fields.get("pain_source"))

    if t == "job":
        return _job(G, canonical_label=canonical_label)

    if t == "persona":
        # If builder supports extra fields, pass them in; identity is canonical_label
        return _persona(
            G,
            title=fields.get("title"),
            department=fields.get("department"),
            seniority=fields.get("seniority"),
            linkedin_profiles=fields.get("linkedin_profiles"),
        )

    if t in ("perceived_metric", "metric"):
        return _metric(G, metric=canonical_label)

    if t in ("pain_trigger", "trigger"):
        return _trigger(G, attribute=canonical_label)

    if t == "attribute_value":
        return _attribute_value(
            G, 
            dimension=fields.get("dimension"),
            name=canonical_label)

    if t == "zmot_event":
        return _zmot(G, event=canonical_label)

    if t == "observable_moment":
        return _observable(G, text=canonical_label)

    if t == "keyword":
        return _keyword(G, text=canonical_label)

    # Fallback for unknown type
    return _upsert_node(G, t or "node", [], attrs={"canonical_label": canonical_label}, data_source="user_input")

# ------------------------------------
# (Optional) use canonical_label to predict ID without creating
# ------------------------------------
def _canonical_id_for(ntype: str, fields: Dict[str, Any]) -> Optional[str]:
    """
    Predict the node ID (no mutation) using _stable_key and the same canonical_label
    that the builder would use.
    """
    t = (ntype or "").lower()
    cl = _canonical_label_for(t, fields)
    if not cl:
        return None
    return f"{t}:{_stable_key([t, cl])}"
# -------- merge canonical fields into existing node (no raw/label/title/content) ----------
def _merge_node_in_place(G: nx.DiGraph, node_id: str, ntype: str, incoming: Dict[str, Any]) -> bool:
    nd = G.nodes[node_id]
    fields = _extract_edit_fields(ntype, incoming)
    evidence = _extract_node_evidence(incoming)

    print("Data for update:")
    print("Existing node:", node_id)
    print("Incoming fields:", fields)
    print("Incoming evidence:", evidence)

    changed = False
    for k, v in fields.items():
        print(f"Checking field '{k}': existing value: {nd.get(k)}, incoming value: {v}")
        if nd.get(k) != v:
            nd[k] = v
            changed = True
            print(f" - updated field '{k}' to: {v}")

    # write evidence if provided (top-level)
    if evidence is not None and nd.get("evidence") != evidence:
        nd["evidence"] = evidence
        changed = True

    # remove any lingering raw from legacy nodes to keep schema flat
    if "raw" in nd:
        nd.pop("raw", None)

    # provenance only
    nd["data_source"] = "user_input"
    nd["last_updated"] = _now()
    return changed

# -------- main upsert ----------
def add_or_update_graph(G: nx.DiGraph, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
    id_map: Dict[str, str] = {}
    summary = {"nodes_created": 0, "nodes_updated": 0, "edges_upserted": 0}

    # ----- NODES -----
    for n in nodes or []:
        provided_id = n.get("id")
        ntype = (n.get("type") or n.get("node_type") or "").lower()
        fields = _extract_edit_fields(ntype, n)  # top-level canonical fields
        # A) update by provided id
        if isinstance(provided_id, str) and provided_id in G.nodes:
            if _merge_node_in_place(G, provided_id, ntype, n):
                summary["nodes_updated"] += 1
                print(f"Node {provided_id} updated in place. Data: {G.nodes[provided_id]}")
            id_map[provided_id] = provided_id
            continue

        # B) update by canonical id (if present)
        canon_id = _canonical_id_for(ntype, fields) if fields else None
        if canon_id and canon_id in G.nodes:
            if _merge_node_in_place(G, canon_id, ntype, n):
                summary["nodes_updated"] += 1
            if provided_id:
                id_map[provided_id] = canon_id
            continue

        # C) create new canonically
        created_id = _create_node_canonical(G, ntype, fields)
        # stamp provenance + evidence; strip raw if builder added any
        node_ref = G.nodes[created_id]
        node_ref["data_source"] = "user_input"
        node_ref["last_updated"] = _now()
        if "raw" in node_ref:
            node_ref.pop("raw", None)
        ev = _extract_node_evidence(n)
        if ev is not None:
            node_ref["evidence"] = ev
        if isinstance(provided_id, str):
            id_map[provided_id] = created_id
        summary["nodes_created"] += 1

    # ----- EDGES -----
    for e in edges or []:
        src_in, tgt_in = e.get("source"), e.get("target")
        src = id_map.get(src_in, src_in)
        tgt = id_map.get(tgt_in, tgt_in)
        if not src or not tgt or src not in G.nodes or tgt not in G.nodes:
            continue

        # accept evidence at top-level or under raw.evidence (compat)
        edge_evidence = e.get("evidence")
        if edge_evidence is None and isinstance(e.get("raw"), dict):
            edge_evidence = e["raw"].get("evidence")

        attrs = {
            "relation": e.get("relation"),
            "relevance": e.get("relevance"),
            "likelihood": e.get("likelihood"),
            "weight": e.get("weight"),
            "boost": e.get("boost"),
            "evidence": edge_evidence,   # <-- top-level on edge
        }

        _upsert_edge(
            G,
            src=src,
            type=str(e.get("relation") or "related_to"),
            tgt=tgt,
            weight=e.get("weight", 1.0),
            attrs=attrs,
            prevent_cycles=True,
            on_cycle="skip",
            data_source="user_input",
        )

        if G.has_edge(src, tgt):
            G[src][tgt]["data_source"] = "user_input"
            G[src][tgt]["last_updated"] = _now()

        summary["edges_upserted"] += 1

    G = update_graph(G)
    print("Graph updated info:")
    for n in nodes:
        nid = n.get("id")
        node = G.nodes[nid] if nid in G.nodes else None
        if node:
            print(f"Node {nid}: {node}")
        else:
            print(f"Node {nid} not found in graph.")
    product_lookup_id = G.graph.get("product_lookup_id")
    redo_graph = build_product_graph(product_lookup_id)
    print("Redone graph updated info:")
    for n in nodes:
        nid = n.get("id")
        node = redo_graph.nodes[nid] if nid in redo_graph.nodes else None
        if node:
            print(f"Node {nid}: {node}")
        else:
            print(f"Node {nid} not found in redone graph.")

    return G, {"summary": summary, "id_map": id_map}