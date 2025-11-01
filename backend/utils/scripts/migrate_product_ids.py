# backend/utils/scripts/migrate_product_ids.py
import json, os, networkx as nx
from typing import Any, Dict, List
from networkx.readwrite import json_graph

LOOKUP_PATH = "backend/utils/graph_base/graph_data/company_product_lookup.json"
GRAPH_DIR   = "backend/utils/graph_base/graph_data/network_json"

def _coerce_to_node_link(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts various legacy graph JSON shapes and normalizes to node-link:
      {nodes: [...], links: [...]} with id-preserving nodes.
    """
    # 1) Already node-link
    if isinstance(data.get("nodes"), list) and isinstance(data.get("links"), list):
        return data

    # 2) nodes list + edges list  -> rename edges -> links
    if isinstance(data.get("nodes"), list) and isinstance(data.get("edges"), list):
        links = []
        for e in data["edges"]:
            # allow {'u':..,'v':..} or {'source':..,'target':..}
            src = e.get("source") or e.get("u") or e.get("from")
            tgt = e.get("target") or e.get("v") or e.get("to")
            attrs = {k: v for k, v in e.items() if k not in ("source","target","u","v","from","to")}
            links.append({"source": src, "target": tgt, **attrs})
        return {"nodes": data["nodes"], "links": links}

    # 3) nodes dict keyed by id, + edges/links variants
    if isinstance(data.get("nodes"), dict):
        nodes_list: List[Dict[str, Any]] = []
        for nid, attrs in data["nodes"].items():
            if isinstance(attrs, dict):
                node = {"id": nid, **attrs}
            else:
                node = {"id": nid}
            nodes_list.append(node)

        if "links" in data and isinstance(data["links"], list):
            return {"nodes": nodes_list, "links": data["links"]}

        if "edges" in data and isinstance(data["edges"], list):
            links = []
            for e in data["edges"]:
                src = e.get("source") or e.get("u") or e.get("from")
                tgt = e.get("target") or e.get("v") or e.get("to")
                attrs = {k: v for k, v in e.items() if k not in ("source","target","u","v","from","to")}
                links.append({"source": src, "target": tgt, **attrs})
            return {"nodes": nodes_list, "links": links}

        # no edges/links — assume empty
        return {"nodes": nodes_list, "links": []}

    # 4) Unknown/empty → empty node-link
    return {"nodes": [], "links": []}

def migrate_pid():
    # load lookup
    if not os.path.exists(LOOKUP_PATH):
        print("No company_product_lookup.json found; nothing to migrate.")
        return

    with open(LOOKUP_PATH, "r") as f:
        lookup = json.load(f)

    os.makedirs(GRAPH_DIR, exist_ok=True)

    for company_id, canonical_pid in lookup.items():
        path = os.path.join(GRAPH_DIR, f"{canonical_pid}_graph.json")
        # load legacy (or new) json safely
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = json.load(f)
            node_link = _coerce_to_node_link(raw)
            G = json_graph.node_link_graph(node_link, directed=True, multigraph=False)
        else:
            G = nx.DiGraph()

        # ensure product node id == canonical_pid
        product_node = None
        for nid, d in list(G.nodes(data=True)):
            if d.get("node_type") == "product":
                product_node = nid
                break

        if product_node is None:
            G.add_node(canonical_pid, node_type="product", id=canonical_pid, company_id=company_id)
        elif product_node != canonical_pid:
            nx.relabel_nodes(G, {product_node: canonical_pid}, copy=False)
            G.nodes[canonical_pid]["id"] = canonical_pid

        # stamp canonical id on graph
        G.graph["product_lookup_id"] = canonical_pid

        # write back as node-link
        with open(path, "w") as f:
            json.dump(json_graph.node_link_data(G), f, indent=2)
        print("Migrated:", path)
