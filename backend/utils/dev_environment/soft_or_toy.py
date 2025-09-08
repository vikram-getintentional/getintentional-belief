"""
Toy demonstration of the soft‑OR relevance propagation used to score archetypes.

Replicates the example from spec.md and prints per‑iteration relevance values.

Graph:
  Product(P) --0.6--> pain_trigger(T1) --0.5--> archetype(A)
  Product(P) --0.2------------------------------> archetype(A)

Run:
  python utils/dev_environment/soft_or_toy.py
"""
from __future__ import annotations

import networkx as nx


def soft_or_iterative(G: nx.DiGraph, product_id: str, damping: float = 0.85, rounds: int = 3) -> dict[str, float]:
    # Initialize
    rel = {n: 0.0 for n in G.nodes()}
    rel[product_id] = 1.0
    print(f"Seed: rel({product_id})=1.0; others=0.0; damping={damping}")

    for t in range(1, rounds + 1):
        new_rel = rel.copy()
        for v in G.nodes():
            if v == product_id:
                new_rel[v] = 1.0
                continue
            # incoming contributions
            belief_inputs = []
            for u, _, data in G.in_edges(v, data=True):
                w = float(data.get("weight", 0.0))
                belief_inputs.append(rel[u] * w)
            updated = 1.0
            for b in belief_inputs:
                updated *= (1.0 - b)
            updated = 1.0 - updated
            new_rel[v] = damping * updated + (1.0 - damping) * rel[v]

        rel = new_rel
        pretty = ", ".join(f"{n}={rel[n]:.4f}" for n in G.nodes())
        print(f"Iteration {t}: {pretty}")
    return rel


def build_toy_graph() -> tuple[nx.DiGraph, str]:
    G = nx.DiGraph()
    # Nodes
    G.add_node("P", node_type="product")
    G.add_node("T1", node_type="pain_trigger")
    G.add_node("A", node_type="archetype")
    # Edges with weights
    G.add_edge("P", "T1", weight=0.6)
    G.add_edge("P", "A", weight=0.2)
    G.add_edge("T1", "A", weight=0.5)
    return G, "P"


if __name__ == "__main__":
    G, product_id = build_toy_graph()
    soft_or_iterative(G, product_id, damping=0.85, rounds=3)

