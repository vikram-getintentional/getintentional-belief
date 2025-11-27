# backend/utils/inference/rcs_generators/persona_map/persona_belief_engine.py

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math
import networkx as nx

# ---- import your helpers (names/relations as per your snippet) ----
from backend.utils.graph_base.network_graph import (
    get_nodes_list_ids,
    get_source_nodes_by_target_and_type,
    get_target_nodes_by_source_and_type,
)
from backend.utils.inference.rcs_generators.graph_algorithms import _normalize_01, _ppr, get_involvement_activation_report
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import get_graphwin



# -------------------- low-level helpers --------------------
def _nt(G: nx.DiGraph, n: str) -> str:
    return (G.nodes.get(n) or {}).get("node_type", "")


def _ew(G: nx.DiGraph, u: str, v: str, key: str) -> float:
    data = G.get_edge_data(u, v) or {}
    try:
        return float(data.get(key, 0.0))
    except Exception:
        return 0.0

def _ensure_node(dst: nx.DiGraph, src: nx.DiGraph, nid: str) -> None:
    if nid not in dst:
        dst.add_node(nid, **(src.nodes[nid] if nid in src else {}))

def _softmax(xs: List[float]) -> List[float]:
    if not xs: return []
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex) or 1.0
    return [x / Z for x in ex]

def _default_seed(PG: PersonaGraph, k=1):
    persons = PG._personas()
    if not persons:
        return []
    cand = []
    for p in persons:
        out_prob = sum(PG.G.edges[p, v].get("prob", 0.0) for _, v in PG.G.out_edges(p))
        cand.append((out_prob, p))
    cand.sort(reverse=True)
    return [pid for _, pid in cand[:k]] or persons[:1]


STOP = "persona:__STOP__"

def _attach_stop_edges(PG: nx.DiGraph, prior_offpath_rate: float):
    if STOP not in PG:
        PG.add_node(STOP, node_type="persona", label="STOP/UNKNOWN")
    for p, d in PG.nodes(data=True):
        if d.get("node_type") != "persona" or p == STOP:
            continue
        if not PG.has_edge(p, STOP):
            PG.add_edge(p, STOP, weight=prior_offpath_rate,
                        likelihood=prior_offpath_rate, n_obs=0)


def _renorm_with_dirichlet(PG: nx.DiGraph, kappa: float | Dict[str, float]) -> None:
    """
    Set edge probs to posterior mean of Dirichlet(n_obs + kappa * likelihood).
    `kappa` can be a single float or a per-persona dict.
    """
    def _k(p: str) -> float:
        return float(kappa.get(p, 0.0)) if isinstance(kappa, dict) else float(kappa)

    for p, nd in PG.nodes(data=True):
        if nd.get("node_type") != "persona":
            continue
        outs = list(PG.out_edges(p, data=True))
        if not outs:
            continue

        k = _k(p)
        numer = []
        denom = 0.0
        for _, v, ed in outs:
            prior = float(ed.get("likelihood", ed.get("weight", 0.0)) or 0.0)
            nobs  = float(ed.get("n_obs", 0.0) or 0.0)
            post  = nobs + k * prior
            numer.append((ed, post))
            denom += post

        if denom > 0.0:
            for ed, post in numer:
                ed["prob"] = post / denom
        else:
            for ed, _ in numer:
                ed["prob"] = 0.0


# -------------------- main PersonaGraph wrapper --------------------
@dataclass
class PersonaGraph:
    """A reduced graph containing only persona and product nodes."""
    G: nx.DiGraph
    product_graph: nx.DiGraph         # <-- add
    original_graph: nx.DiGraph        # <-- add

    # ---------- construction ----------
    @classmethod
    def from_product_graph(
        cls,
        product_graph: nx.DiGraph,
        *,
        original_graph: Optional[nx.DiGraph] = None,
        combine: str = "sum",          # "sum" | "noisy_or"
        normalize_outgoing: bool = True,
        prior_offpath_rate: float | None = None,   # <-- NEW
        dirichlet_kappa: float | Dict[str, float] | None = None,  # <-- NEW
    ) -> "PersonaGraph":
        OG = original_graph or product_graph
        PG = nx.DiGraph()
        details: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        G = product_graph
        print("Building PersonaGraph from product graph with Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

        personas = get_nodes_list_ids(G, "persona", {})
        
        for p1 in personas:
            _ensure_node(PG, G, p1)
        

            # (p1) --performed_by--> (job j1)  ; edge key: "relevance"
            jobs_of_persona = get_source_nodes_by_target_and_type(G, p1, "performed_by")
        
            if not jobs_of_persona:
                continue
            for j1 in jobs_of_persona:
        
                if _nt(G, j1) != "job":
                    continue
                r1 = _ew(G, j1, p1, "relevance") if G.has_edge(j1, p1) else _ew(G, p1, j1, "relevance")
                if r1 <= 0.0:
                    continue

                # (j1) --felt_in--> (pain)
                pains_of_job = get_source_nodes_by_target_and_type(G, j1, "felt_in")
        
                for pain in pains_of_job:
        
                    if _nt(G, pain) != "pain":
                        continue
                    l1 = _ew(G, pain, j1, "likelihood") if G.has_edge(pain, j1) else _ew(G, j1, pain, "likelihood")
                    if l1 <= 0.0:
                        continue

                    # (pain) --solves--> (job | capability)
                    nodes_of_pain = get_source_nodes_by_target_and_type(G, pain, "solves")
        
                    for mid in nodes_of_pain:
        
                        l2 = _ew(G, mid, pain, "likelihood") if G.has_edge(mid, pain) else _ew(G, pain, mid, "likelihood")
                        if l2 <= 0.0:
                            continue

                        t = _nt(G, mid)
                        if t == "job":
                            # (job mid) --performed_by--> (persona p2) ; relevance
                            upstream_personas = get_target_nodes_by_source_and_type(G, mid, "performed_by")
                            for p2 in upstream_personas:
                                if _nt(G, p2) != "persona" or p2 == p1:
                                    continue
                                r2 = _ew(G, mid, p2, "relevance")
                                if r2 <= 0.0:
                                    continue
                                w = r1 * l1 * l2 * r2
                                _ensure_node(PG, G, p2)
                                _accumulate_edge(PG, p1, p2, w, details)

                        elif t == "capability":
                            # (capability mid) --offered_by--> (product)
                            product_nodes = get_source_nodes_by_target_and_type(G, mid, "offers")
                            for prod in product_nodes:
                                if _nt(G, prod) != "product":
                                    continue
                                l3 = _ew(G, prod, mid, "likelihood") if G.has_edge(prod, mid) else _ew(G, mid, prod, "likelihood")
                                if l3 <= 0.0:
                                    continue
                                w = r1 * l1 * l2 * l3
                                _ensure_node(PG, G, prod)
                                _accumulate_edge(PG, p1, prod, w, details)

        if combine == "noisy_or":
            for (u, v) in list(PG.edges()):
                contribs = [d["contrib"] for d in details[(u, v)] if d.get("contrib", 0.0) > 0.0]
                if contribs:
                    p = 1.0
                    for c in contribs:
                        p *= (1.0 - c)
                    val = 1.0 - p
                    PG.edges[u, v]["likelihood"] = val
                    PG.edges[u, v]["weight"] = val

        # 1) Structural uncertainty: add STOP/UNKNOWN edges from every persona
        if prior_offpath_rate is not None:
            _attach_stop_edges(PG, prior_offpath_rate)

        # 2) Normalize: plain renorm or Bayesian posterior renorm
        if normalize_outgoing:
            if dirichlet_kappa is None:
                _renormalize_outgoing(PG)
            else:
                _renorm_with_dirichlet(PG, dirichlet_kappa)

        # stamp type to be explicit (useful if PG created standalone)
        for n in PG.nodes():
            if n == STOP:
                PG.nodes[n]["node_type"] = "persona"
                continue
            src = (product_graph.nodes.get(n) or original_graph.nodes.get(n) or {})
            if "node_type" in src:
                PG.nodes[n]["node_type"] = src["node_type"]
            else:
                # best-effort fallback: persona if it has any outgoing to persona/product
                PG.nodes[n]["node_type"] = "persona"

        return cls(G=PG, product_graph=product_graph, original_graph=OG)
    
    def _mk_engaged_nodes(self, engaged_personas: List[str]) -> List[Dict[str, float]]:
        print("Making engaged nodes from personas:", engaged_personas)
        return [{"id": pid, "occurrence": 1.0} for pid in engaged_personas]
    
    # persona_belief_engine.py (add)
    def _hydrate_from_rcs_scores(
        self,
        engaged_nodes: Optional[List[Dict[str, float]]] = None,
        *,
        alpha: float = 0.85,
        attr_prior: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Recompute persona/job/pain scores conditioned on engaged_nodes,
        and stamp persona-level scores back onto self.G nodes.
        """
        engaged_nodes = engaged_nodes or []
        print("Hydrating RCS scores for engaged nodes:", engaged_nodes)


        report = get_involvement_activation_report(
            G=self.product_graph,
            Original_G=self.original_graph,
            engaged_nodes=engaged_nodes,
            alpha=alpha,
            attr_prior=attr_prior,
        )
        print("Computed involvement/activation report in PersonaGraph")
        p_scores = report.get("persona_scores", {}) or {}
        print("Persona scores computed")


        # Stamp persona-level scores back onto the reduced graph
        for pid, ps in p_scores.items():
            if pid in self.G and self.G.nodes[pid].get("node_type") == "persona":
                nd = self.G.nodes[pid]
                nd["perceptibility"] = float(ps.get("perceptibility", 0.0))
                nd["proximity"]      = float(ps.get("proximity", 0.0))
                nd["involvement"]    = float(ps.get("involvement", 0.0))
                nd["activation"]     = float(ps.get("activation", 0.0))
                nd["strength"]       = float(ps.get("strength", 0.0))

    # ---------- query API ----------
    def expected_next(self, engaged_personas: List[str], k: int = 5, t: int | None = None) -> List[Tuple[str, float]]:
        acc_prob: Dict[str, float] = {}
        for p in engaged_personas or []:
            for _, v, d in self.G.out_edges(p, data=True):
                if _nt(self.G, v) != "persona" or v == STOP:
                    continue
                w = d.get("prob_persona", None)
                if w is None:
                    w = float(d.get("prob", d.get("weight", 0.0)) or 0.0)
                acc_prob[v] = acc_prob.get(v, 0.0) + float(w)

        # Fallback if no persona outs:
        if not acc_prob:
            per = self.perceptibility()
            pool = [pid for pid in per.keys() if pid not in set(engaged_personas) and pid != STOP]
            pool = sorted(pool, key=lambda pid: per[pid], reverse=True)[:k]
            probs = _softmax([per[pid] for pid in pool])
            return list(zip(pool, probs))

        # --- hybrid re-rank ---
        per = self.perceptibility()
        prox = self.proximity()
        act = {p: float(self.G.nodes[p].get("activation", 0.0)) for p in acc_prob.keys()}

        # stage-aware weights: earlier -> favor perceptibility
        step = 0 if t is None else max(0, int(t))
        w_per, w_prox, w_act, w_edge = (0.50, 0.20, 0.10, 0.20) if step <= 1 else (0.30, 0.25, 0.15, 0.30)

        scored = []
        for pid, edge_p in acc_prob.items():
            s = (w_edge * edge_p) + (w_per * per.get(pid, 0.0)) + (w_prox * prox.get(pid, 0.0)) + (w_act * act.get(pid, 0.0))
            scored.append((pid, s))

        scored.sort(key=lambda kv: kv[1], reverse=True)
        top = scored[:k]
        probs = _softmax([v for _, v in top])
        return [(top[i][0], probs[i]) for i in range(len(top))]


    def beam_paths(
        self, start_personas: List[str], *, beam_width: int = 3, max_depth: int = 6, keep: int = 8
    ) -> List[Dict[str, Any]]:
        def outs(u: str):
            for _, v, d in self.G.out_edges(u, data=True):
                if _nt(self.G, v) == "persona" and v != STOP:
                    yield v, float(d.get("prob", d.get("weight", 0.0)) or 0.0)

        beams: List[Tuple[List[str], float]] = [([s], 1.0) for s in (start_personas or [])]
        paths: List[Tuple[List[str], float]] = []
        for _ in range(max_depth):
            new_beams: List[Tuple[List[str], float]] = []
            for path, score in beams:
                u = path[-1]
                if u == STOP:
                    paths.append((path, score))
                    continue
                cand = sorted(list(outs(u)), key=lambda kv: kv[1], reverse=True)[:beam_width]
                if not cand:
                    paths.append((path, score))
                    continue
                for v, p in cand:
                    if p <= 0: 
                        continue
                    new_beams.append((path + [v], score * p))
            if not new_beams:
                break
            new_beams = sorted(new_beams, key=lambda ps: ps[1], reverse=True)[:beam_width]
            beams = new_beams
        paths.extend(beams)
        ranked = sorted(paths, key=lambda ps: ps[1], reverse=True)[:keep]
        seen = set()
        uniq = []
        for p, s in ranked:
            key = tuple(p)
            if key in seen: 
                continue
            seen.add(key)
            uniq.append((p, s))
        return [{"personas": p, "probability": s, "score": s, "path": p} for p, s in uniq[:keep]]

    # ---------- metrics ----------
    def graphwin(self, *, alpha: float = 0.85, engaged_personas: Optional[List[str]] = None) -> Dict[str, float]:
        Gw = nx.DiGraph()
        for u, v, d in self.G.edges(data=True):
            w = float(d.get("prob", d.get("likelihood", 0.0)) or 0.0)
            if w > 0:
                Gw.add_edge(u, v, weight=w)

        if engaged_personas:
            seeds = {p: 1.0 for p in engaged_personas if p in Gw and p != STOP}
        else:
            seeds = {n: 1.0 for n, dd in self.G.nodes(data=True)
                    if dd.get("node_type") == "persona" and n != STOP}

        if not seeds:
            return {}

        pr = _ppr(Gw, seeds, alpha=alpha, weight_key="weight")
        out = {n: float(pr.get(n, 0.0)) for n, dd in self.G.nodes(data=True)
            if dd.get("node_type") == "persona" and n != STOP}
        out["_product_mass"] = float(sum(pr.get(n, 0.0) for n, dd in self.G.nodes(data=True)
                                        if dd.get("node_type") == "product"))
        return _normalize_01(out)


    def perceptibility(self, *, alpha: float = 0.85) -> Dict[str, float]:
        vals = {p: float(d.get("perceptibility", 0.0))
                for p, d in self.G.nodes(data=True) if d.get("node_type") == "persona"}
        # normalize to guard against different scales
        return _normalize_01(vals)

    def proximity(self, *, alpha: float = 0.85) -> Dict[str, float]:
        vals = {p: float(d.get("proximity", 0.0))
                for p, d in self.G.nodes(data=True) if d.get("node_type") == "persona"}
        return _normalize_01(vals)


    # ---------- learning ----------
    def learn_from_sequence(
        self,
        observed_personas_in_order: List[str],
        observed_products: Optional[List[str]] = None,
        kappa: float = 0.2,
    ) -> None:
        """Update start priors and edges with simple online blending; renormalize."""
        if not observed_personas_in_order:
            return
        first = observed_personas_in_order[0]
        self.G.nodes[first]["start_count"] = self.G.nodes[first].get("start_count", 0) + 1
        for u, v in zip(observed_personas_in_order, observed_personas_in_order[1:]):
            if u == STOP or v == STOP:  # don’t learn STOP
                continue
            if not self.G.has_edge(u, v):
                self.G.add_edge(u, v, weight=0.0, n_obs=0, count=0)
            d = self.G[u][v]
            d["n_obs"] = d.get("n_obs", 0) + 1
            d["count"] = d.get("count", 0) + 1
            d["weight"] = (1 - kappa) * d.get("weight", 0.0) + kappa * 1.0
        if observed_products:
            for p in observed_personas_in_order:
                for prod in observed_products:
                    if self.G.has_edge(p, prod):
                        e = self.G.edges[p, prod]
                        e["n_obs"] = e.get("n_obs", 0) + 1
                        e["weight"] = (1 - kappa) * float(e.get("weight", 0.0)) + kappa * 1.0
        _renormalize_outgoing(self.G)
        self._hydrate_from_rcs_scores(
            engaged_nodes=self._mk_engaged_nodes(observed_personas_in_order),
        )

    # ---------- utils ----------
    def _personas(self) -> List[str]:
        return [n for n, d in self.G.nodes(data=True) if d.get("node_type") == "persona"]

def _accumulate_edge(
    PG: nx.DiGraph,
    src: str,
    dst: str,
    w: float,
    details: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> None:
    if w <= 0.0:
        return
    if not PG.has_edge(src, dst):
        PG.add_edge(src, dst, likelihood=0.0, weight=0.0, count=0, n_obs=0)
    e = PG.edges[src, dst]
    e["likelihood"] += w
    e["weight"] = e["likelihood"]
    e["count"] += 1
    details[(src, dst)].append({"contrib": w})

def _renormalize_outgoing(PG: nx.DiGraph) -> None:
    bad = []
    for p, nd in PG.nodes(data=True):
        if nd.get("node_type") != "persona":
            continue

        outs = list(PG.out_edges(p, data=True))
        # 1) Full renorm (including STOP/product) → 'prob'
        s_all = sum(float(d.get("weight", 0.0)) for _, _, d in outs)
        if s_all > 0.0:
            for _, _, d in outs:
                d["prob"] = float(d.get("weight", 0.0)) / s_all
        else:
            for _, _, d in outs:
                d["prob"] = 0.0

        # 2) Persona-only renorm (exclude STOP) → 'prob_persona'
        persona_outs = [(u, v, d) for u, v, d in outs if _nt(PG, v) == "persona" and v != STOP]
        s_p = sum(float(d.get("weight", 0.0)) for _, _, d in persona_outs)
        for _, _, d in persona_outs:
            d["prob_persona"] = float(d.get("weight", 0.0)) / s_p if s_p > 0 else 0.0

        # sanity check for full prob (not used for gating UI, just log)
        s_check = sum(float(d.get("prob", 0.0)) for _, _, d in outs)
        if abs(s_check - 1.0) > 1e-6 and outs:
            bad.append((p, s_check))

    if bad:
        print(f"[PG] renorm_check: bad={len(bad)} of personas; sample={bad[:3]}")





# --- Prediction-only helpers (no learning) ---

def predict_snapshot(
    PG: PersonaGraph,
    *,
    engaged_personas: List[str],
    k_next: int = 5,
    beam_width: int = 3,
    max_depth: int = 6,
    attr_prior: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Stateless snapshot: given currently engaged personas, predict:
      - expected_next (top-K next personas)
      - walk_paths (beam on persona-only graph, seeded by last engaged persona if any)
      - metrics: graphwin, perceptibility, proximity
    """
    seed = engaged_personas if engaged_personas else _default_seed(PG)
    print("Engaged personas for snapshot prediction:", seed)
    engaged_nodes = PG._mk_engaged_nodes(seed)
    print("starting first hydrate")

    # >>> hydrate contextually before any metric reads
    PG._hydrate_from_rcs_scores(engaged_nodes=engaged_nodes, attr_prior=attr_prior)
    print("completed first hydrate with graph Nodes:", PG.G.number_of_nodes(), "Edges:", PG.G.number_of_edges())
    print("Sample PersonaGraph nodes after hydrate:")
    for n, d in list(PG.G.nodes(data=True))[:5]:
        print(f"  Node: {n} Data: {d}")
    print("Sample PersonaGraph edges after hydrate:")
    for u, v, d in list(PG.G.edges(data=True))[:5]:
        print(f"  Edge: {u} -> {v} Data: {d}")
        
    expected = PG.expected_next(seed, k=k_next)
    print("Computed expected next personas:", expected)
    paths = PG.beam_paths(start_personas=seed, beam_width=beam_width, max_depth=max_depth, keep=8)
    metrics = {
        "graphwin": PG.graphwin(engaged_personas=seed),
        "perceptibility": PG.perceptibility(),
        "proximity": PG.proximity(),
    }
    involvement_raw = {
        pid: float(PG.G.nodes[pid].get("involvement", 0.0))
        for pid, data in PG.G.nodes(data=True)
        if data.get("node_type") == "persona" and pid != STOP
    }
    if involvement_raw:
        metrics["involvement"] = _normalize_01(involvement_raw)
    else:
        metrics["involvement"] = {}

    return {
        "expected_next": [
            {"persona": str(pid), "prob": float(prob)} for (pid, prob) in list(expected)[:k_next]
        ],
        "walk_paths": paths,
        "metrics": metrics,
    }


def predict_sequence(
    PG: PersonaGraph,
    *,
    observed_personas_in_order: List[str],
    k_next: int = 5,
    beam_width: int = 3,
    max_depth: int = 6,
    attr_prior: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Stateless sequence predictor: for each time step t,
    predicts BEFORE consuming observation at t (no learning).
    Returns a list of steps: [{"t": t, "predicted_topK": [...], "walk_paths": [...], "metrics": {...}}]
    """
    engaged: List[str] = []
    steps: List[Dict[str, Any]] = []
    for t, pid in enumerate(observed_personas_in_order):
        snap = predict_snapshot(
            PG,
            engaged_personas=engaged,
            k_next=k_next,
            beam_width=beam_width,
            max_depth=max_depth,
            attr_prior=attr_prior,
        )
        steps.append({
            "t": t,
            "state_personas": list(engaged),
            "predicted_topK": snap["expected_next"],
            "walk_paths": snap["walk_paths"],
            "metrics": snap["metrics"],
        })
        engaged.append(pid)
    return steps
