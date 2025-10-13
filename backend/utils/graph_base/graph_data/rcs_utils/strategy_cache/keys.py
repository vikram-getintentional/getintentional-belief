# backend/utils/strategy_cache/keys.py
import hashlib, json
from typing import Any, Dict, Iterable, List, Mapping, Tuple

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def canon_node_set(node_ids: Iterable[str]) -> List[str]:
    # IDs only; sorted & deduped
    return sorted(set(node_ids))

def inputs_hash_from_initial_nodes(initial_nodes: Iterable[str], attributes: Mapping[str, Any] | None = None, zmot: Any | None = None) -> str:
    payload = {
        "initial_nodes": canon_node_set(initial_nodes),
        # optional: include stable attribute + ZMOT representation if they factor into strategy
        "attributes": attributes or {},
        "zmot": zmot or None,
    }
    return sha256_hex(_stable_json(payload))

def engaged_hash_from_nodes(engaged_nodes: Iterable[str]) -> str:
    return sha256_hex(_stable_json({"engaged_nodes": canon_node_set(engaged_nodes)}))

def algo_hash_from_params(version: str, params: Dict[str, Any]) -> str:
    # version could be a git SHA or semantic version of the algo
    payload = {"version": version, "params": params or {}}
    return sha256_hex(_stable_json(payload))

# Graph hashing
def graph_hash_from_graph(product_graph) -> str:
    """
    Normalize nodes/edges to a stable string and hash it.
    Assumes each node has: id, type (optional), and each edge has: u, v, weight, label (optional).
    """
    # Build stable lists
    nodes = []
    for n, data in product_graph.nodes(data=True):
        nodes.append({"id": str(n), "type": data.get("type"), "attrs": _minify(data)})
    nodes.sort(key=lambda x: x["id"])

    edges = []
    for u, v, data in product_graph.edges(data=True):
        edges.append({
            "u": str(u),
            "v": str(v),
            "w": round(float(data.get("weight", 1.0)), 12),
            "label": data.get("label"),
            "attrs": _minify(data),
        })
    edges.sort(key=lambda x: (x["u"], x["v"], x["label"] or ""))

    normalized = {"nodes": nodes, "edges": edges}
    return sha256_hex(_stable_json(normalized))

def _minify(d: Dict[str, Any]) -> Dict[str, Any]:
    # keep only deterministic primitives; drop volatile runtime fields
    keep = {}
    for k, v in d.items():
        if k in ("weight", "type", "label"):
            continue  # already handled
        if isinstance(v, (str, int, float, bool)) or v is None:
            keep[k] = v
    return keep
