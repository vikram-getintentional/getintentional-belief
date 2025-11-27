# backend/utils/inference/rcs_generators/persona_map/persona_paths.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import math
import networkx as nx

def _softmax(xs: List[float]) -> List[float]:
    if not xs: return []
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex) or 1.0
    return [x / Z for x in ex]

def expected_next_personas(PG: nx.DiGraph, engaged_personas: List[str], k: int = 5) -> List[Tuple[str, float]]:
    """
    Aggregate outgoing persona->persona probabilities from currently engaged personas.
    """
    acc: Dict[str, float] = {}
    engaged = set(engaged_personas or [])
    for p in engaged:
        for _, v, d in PG.out_edges(p, data=True):
            # only persona targets
            if (PG.nodes[v] or {}).get("node_type") != "persona":
                continue
            acc[v] = acc.get(v, 0.0) + float(d.get("prob", d.get("weight", 0.0)) or 0.0)
    if not acc:
        return []
    items = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:k]
    # optional softmax polish
    probs = _softmax([v for _, v in items])
    return [(items[i][0], probs[i]) for i in range(len(items))]

def beam_search_persona_paths_on_PG(
    PG: nx.DiGraph,
    start_personas: List[str],
    beam_width: int = 3,
    max_depth: int = 6,
) -> List[Dict[str, Any]]:
    """
    Persona-only beam search on the reduced graph.
    Score step with edge prob (or weight if prob missing).
    """
    def out_edges_prob(u: str) -> List[Tuple[str, float]]:
        outs = []
        for _, v, d in PG.out_edges(u, data=True):
            if (PG.nodes[v] or {}).get("node_type") != "persona":
                continue
            p = float(d.get("prob", d.get("weight", 0.0)) or 0.0)
            outs.append((v, p))
        return outs

    paths: List[Tuple[List[str], float]] = []
    # init beams from start priors
    beams = [([s], 1.0) for s in (start_personas or [])]

    for _ in range(max_depth):
        new_beams: List[Tuple[List[str], float]] = []
        for path, score in beams:
            u = path[-1]
            outs = out_edges_prob(u)
            if not outs:
                paths.append((path, score))
                continue
            outs = sorted(outs, key=lambda kv: kv[1], reverse=True)[:beam_width]
            for v, p in outs:
                if p <= 0.0:
                    continue
                new_beams.append((path + [v], score * p))
        if not new_beams:
            break
        new_beams = sorted(new_beams, key=lambda ps: ps[1], reverse=True)[:beam_width]
        beams = new_beams

    paths.extend(beams)
    # format
    ranked = sorted(paths, key=lambda ps: ps[1], reverse=True)
    out = [{"personas": p, "score": s, "probability": s, "path": p} for p, s in ranked]
    return out[:8]
