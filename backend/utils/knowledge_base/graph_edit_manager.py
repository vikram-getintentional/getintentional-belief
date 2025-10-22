import networkx as nx
from typing import Dict, List, Any, Set
from backend.utils.graph_base.network_graph import _set_node_label, get_node_by_id, get_nodes_list_ids, get_product_id_from_subgraph, get_target_nodes_by_source_and_type

def _normalize_node(G: nx.DiGraph, n: Any) -> Dict:
    data = G.nodes[n]
    node_type = data.get("type") or data.get("node_type") or ""
    label = _set_node_label(G, n)
    title = data.get("title") or data.get("label") or label
    content = data.get("content") or data.get("description") or data.get("text") or ""
    return {
        "id": str(n),
        "label": label,
        "type": node_type,
        "title": title,
        "content": content,
        "relevance": data.get("relevance"),
        "likelihood": data.get("likelihood"),
        "properties": data.get("properties", {}),
        "raw": {k: v for k, v in data.items()},
    }

def _get_next_nodes(G: nx.DiGraph, node_id: Any) -> List[Dict]:
    out = []
    if node_id not in G:
        return out
    node = get_node_by_id(G, node_id)
    node_type = node.get("node_type") or ""
    if node_type == "product":
        target_edge = "offers"
    elif node_type == "capability":
        target_edge = "solves"
    elif node_type == "pain":
        target_edge = "felt_in"
    elif node_type == "job":
        target_edge = "solves"
    if not target_edge: return out

    targets = get_target_nodes_by_source_and_type(G, node_id, target_edge)
    for tgt in targets:
        ndata = get_node_by_id(G,tgt)
        ntype = ndata.get("node_type") or ""
        out.append(_normalize_node(G, tgt))
    return out

def build_hops_from_graph(G: nx.DiGraph) -> List[Dict]:
    """
    Build an ordered list of hop objects:
      Hop 0: { capabilities: [...], felt_pains: [...], jobs: [...], personas: [...], solving_pains: [...] }
      Hop 1+: seeds are jobs from previous hop; each hop returns the same structure but filled from the seeds.
    """
    hops: List[Dict] = []

    seed_nodes = []
    product_id = get_product_id_from_subgraph(G)
    seed_nodes.append(product_id)
    if not seed_nodes:
        return hops
    current_hop = 0
    unseen_nodes: Set[Any] = set()

    while(len(seed_nodes) > 0 and current_hop < 5):
        hop_output = {}
        hop_output["hop"] = current_hop
        current_hop += 1
        loop_seeds = seed_nodes.copy()
        seed_nodes = []

        for seed in loop_seeds:
            next_jc_nodes = _get_next_nodes(G, seed)
            jc_out = []
            pn_out = []
            jtbd_out = []
            persona_out = []
            if not next_jc_nodes or len(next_jc_nodes) <= 0:
                break
            for jc in next_jc_nodes:
                if jc["id"] not in hop_output:
                    jc_out.append(jc)
                next_pn_nodes = _get_next_nodes(G, jc)
                if not next_pn_nodes or len(next_pn_nodes) <= 0:
                    continue
                for pn in next_pn_nodes:
                    if pn["id"] not in hop_output:
                        pn_out.append(pn)
                next_jtbd_nodes = _get_next_nodes(G, pn)
                if not next_jtbd_nodes or len(next_jtbd_nodes) <= 0:
                    continue
                for jtbd in next_jtbd_nodes:
                    if jtbd["id"] not in hop_output:
                        jtbd_out.append(jtbd)
                        seed_nodes.append([jtbd["id"]])
                    next_pers_nodes = get_target_nodes_by_source_and_type(G, jtbd["id"], "performed_by")
                    if not next_pers_nodes or len(next_pers_nodes) <= 0:
                        continue
                    for pers in next_pers_nodes:
                        if pers not in hop_output:
                            persona_out.append(pers)
            hop_output = {
                "hop_depth": current_hop,
                "jc": jc_out,
                "pn": pn_out,
                "jtbd": jtbd_out,
                "personas": persona_out
            }
            print("Hop Output: ", hop_output)
            hops.append(hop_output)
    return hops