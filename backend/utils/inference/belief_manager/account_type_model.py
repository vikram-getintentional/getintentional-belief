from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Set, Dict, Any
import networkx as nx

@dataclass
class SubgraphSelector:
    # exactly one of these can be provided; both can be combined via union if you like
    attribute_nodes: Optional[Iterable[str]] = None   # current RCS seed behavior
    seed_nodes: Optional[Iterable[str]] = None        # personas/jobs/pains/caps from account
    depth: int = 3                                    # expansion radius
    allow_types: Optional[Set[str]] = None            # restrict traversal by node type
    allow_edges: Optional[Set[str]] = None            # restrict traversal by edge "relation" label

def build_account_subgraph(G: nx.DiGraph, sel: SubgraphSelector) -> nx.DiGraph:
    """
    Returns a trimmed subgraph that unions attribute-driven and account-driven seeds,
    then expands k steps along allowed edge types. Keeps product node if present.
    """
    # 1) collect start seeds
    seeds: Set[str] = set()
    if sel.attribute_nodes:
        seeds |= set(sel.attribute_nodes)
    if sel.seed_nodes:
        seeds |= set(sel.seed_nodes)

    if not seeds:
        # fallback: current behavior implies attributes; return whole graph if needed
        return G

    # 2) bounded, typed expansion (BFS)
    allow_types = sel.allow_types or {"persona","job","pain","capability","product","attribute"}
    allow_edges = sel.allow_edges or {"performed_by","felt_in","solves","enables","relates_to","belongs_to","to_product"}

    frontier = list(seeds)
    seen = set(seeds)
    for _ in range(max(0, sel.depth)):
        nxt = []
        for u in frontier:
            # expand both directions but filter by edge semantics and node types
            for v in G.successors(u):
                rel = str(G.edges[u, v].get("relation", G.edges[u, v].get("type","")))
                if rel in allow_edges and str(G.nodes[v].get("type","")) in allow_types and v not in seen:
                    seen.add(v); nxt.append(v)
            for v in G.predecessors(u):
                rel = str(G.edges[v, u].get("relation", G.edges[v, u].get("type","")))
                if rel in allow_edges and str(G.nodes[v].get("type","")) in allow_types and v not in seen:
                    seen.add(v); nxt.append(v)
        frontier = nxt

    # keep product node, even if not reached (prevents accidental pruning)
    for n, d in G.nodes(data=True):
        if str(d.get("type","")).lower() == "product":
            seen.add(n)

    return G.subgraph(seen).copy()
