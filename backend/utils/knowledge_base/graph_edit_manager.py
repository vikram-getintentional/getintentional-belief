from datetime import datetime
import networkx as nx
from typing import Any, Dict, List, Set, Optional

from backend.utils.graph_base.agent_graph_builder import (
    _attribute_value, 
    _product, 
    _capability, 
    _pain, 
    _job, 
    _persona, 
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
        weight = get_edge_attribute(G, src_id, n_id, "weight") or 0.0
        if not relevance or not likelihood:
            print("Rel or likelihood missing for:", src_id, "->", n_id)
        sources[str(src_id)] = {
            "relevance": relevance,
            "likelihood": likelihood,
            "weight": weight,
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

NODE_CREATORS = {
    "capability": _capability,
    "pain": _pain,
    "job": _job,
    "persona": _persona,
    "perceived_metric": _metric,
    "pain_trigger": _trigger,
    "zmot_event": _zmot,
    "observable_moment": _observable,
    "keyword": _keyword,
    "attribute_value": _attribute_value,
}


def add_or_update_graph(G: nx.DiGraph, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Add or update nodes and edges in graph G using the canonical ID system.
    Nodes: List of dicts with at least 'id' or 'type' and identifying content.
    Edges: List of dicts with at least 'source' and 'target'.
    """
    for n in nodes:
        ntype = n.get("type")
        raw = n.get("raw", {})
        data_source = "user_input"

        # Ensure we always use canonical key logic
        creator_fn = NODE_CREATORS.get(ntype)
        if not creator_fn:
            print(f"⚠️ Unknown node type '{ntype}' — skipping node.")
            continue

        # Construct using canonical logic per node type
        if ntype == "capability":
            node_id = creator_fn(G, name=raw.get("name"), description=raw.get("description"), data_source=data_source)
        elif ntype == "pain":
            node_id = creator_fn(G, canonical_label=raw.get("description"), pain_source=raw.get("pain_source"), data_source=data_source)
        elif ntype == "job":
            node_id = creator_fn(G, canonical_label=raw.get("description"), data_source=data_source)
        elif ntype == "persona":
            node_id = creator_fn(
                G,
                title=raw.get("title"),
                department=raw.get("department"),
                seniority=raw.get("seniority"),
                data_source=data_source
            )
        elif ntype == "perceived_metric":
            node_id = creator_fn(G, metric=raw.get("metric"), data_source=data_source)
        elif ntype == "pain_trigger":
            node_id = creator_fn(G, attribute=raw.get("attribute"), data_source=data_source)
        elif ntype == "attribute_value":
            node_id = creator_fn(G, dimension=raw.get("dimension"), name=raw.get("name"), data_source=data_source)
        elif ntype == "zmot_event":
            node_id = creator_fn(G, event=raw.get("event"), data_source=data_source)
        elif ntype == "observable_moment":
            node_id = creator_fn(G, text=raw.get("text"), data_source=data_source)
        elif ntype == "keyword":
            node_id = creator_fn(G, text=raw.get("text"), data_source=data_source)
        else:
            # Fallback for any missing type
            node_id = _upsert_node(G, ntype or "unknown", [n.get("label", ""), n.get("content", "")], n, data_source)

        # Merge updates
        node_data = G.nodes[node_id]
        for k, v in raw.items():
            if v is not None:
                node_data[k] = v
        node_data["data_source"] = data_source
        node_data["last_updated"] = current_timestamp()

    # ---- Edges ----
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        if not src or not tgt:
            continue

        attrs = {
            "relation": e.get("relation"),
            "relevance": e.get("relevance"),
            "likelihood": e.get("likelihood"),
            "weight": e.get("weight"),
            "boost": e.get("boost"),
            "evidence": e.get("evidence"),
        }

        _upsert_edge(G, src, e.get("relation", "related_to"), tgt, weight=e.get("weight", 1.0), attrs=attrs, data_source="user_input")
    print("Graph updated with user edits.")
    update_graph(G)

    return G