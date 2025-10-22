# ============================
# File: backend/utils/inference/rcs_generators/generate_rcs_fast.py
# ============================
from __future__ import annotations
import json
import re
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
from itertools import combinations
import networkx as nx
import openai

from backend.utils.graph_base.network_graph import (
    _set_node_label as _set_node_label,
    get_nodes_list_ids,
    get_product_id_from_subgraph,  # if used
)

from backend.utils.inference.rcs_generators.graph_algorithms import (
    build_concern_backlog_from_activation_breakdown,
    get_involvement_activation_report,  # {"core_scores": {...}, "persona_scores": {...}}
    _stage_rank,                          # stage ordering helper
)
from backend.utils.inference.rcs_generators.rcs_computations.graphwin_runtime import get_graphwin

# -----------------------------------------------------------------------------
# Light helpers (UI packaging, no math)
# -----------------------------------------------------------------------------
_STAGE_ORDER = {"problem": 0, "pain": 1, "execution": 2, "resolution": 3}
_STAGE_WEIGHT = {"problem": 0.85, "pain": 0.90, "execution": 0.80, "resolution": 0.75}

_STOP = {
    "the","and","or","to","of","in","for","on","a","an","with",
    "by","from","at","as","into","is","are","be","being","been",
    "your","their","our","this","that","these","those","via",
    "data","team","teams","process","processes","system","systems"
}

def _extract_keywords(label: str) -> list:
    # crude keywordizer: split on non-letters, drop stopwords/short tokens
    toks = re.split(r"[^a-zA-Z0-9]+", (label or "").lower())
    out = [t for t in toks if len(t) >= 3 and t not in _STOP]
    return out[:8]

def _nice_join(words, limit=5):
    words = [w.capitalize() for w in words[:limit]]
    if not words: 
        return ""
    if len(words) == 1:
        return words[0]
    if len(words) == 2:
        return f"{words[0]} & {words[1]}"
    return f"{', '.join(words[:-1])} & {words[-1]}"

def _trajectory_phrase(steps):
    seen = []
    for s in steps:
        st = _stage_norm(s.get("stage"))
        if st and st not in seen:
            seen.append(st)
    ladder = " → ".join(w.capitalize() for w in seen)
    return ladder or "Problem → Resolution"

def _best_label_for_stage(steps, stage_name):
    # pick the highest-lift concern at a specific stage, return its label and id
    sname = _stage_norm(stage_name)
    rows = [s for s in steps if _stage_norm(s.get("stage")) == sname]
    if not rows:
        return None, None
    best = max(rows, key=lambda r: float(r.get("lift_proxy", 0.0)))
    return (best.get("concern_label") or best.get("cid")), best.get("cid")





def _nt(G: nx.DiGraph, n: str) -> str:
    d = G.nodes.get(n, {})
    return d.get("node_type") or d.get("type") or "unknown"

def rcs_prepare(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
) -> Tuple[nx.DiGraph, Optional[str], List[Dict]]:
    engaged_nodes = engaged_nodes or []

    keep_types = {
        "product", "capability", "job", "pain", "pain_trigger",
        "persona", "zmot_event", "attribute_value"
    }

    # Start with a type-pruned copy (keeps attribute_value)
    Gp = G.copy()
    for n in list(Gp.nodes):
        if _nt(Gp, n) not in keep_types:
            Gp.remove_node(n)

    product_id = get_product_id_from_subgraph(Gp)

    # *** KEY FIX: only use engaged attribute ids for path pruning ***
    engaged_attr_ids: List[str] = []
    for e in engaged_nodes:
        nid = e.get("id")
        if not nid or nid not in Gp:
            continue
        if _nt(Gp, nid) == "attribute_value":
            engaged_attr_ids.append(nid)

    # If we have engaged attributes, prune to their paths to product.
    # Otherwise, keep the type-pruned graph as-is (don’t over-prune).
    if engaged_attr_ids and product_id:
        Gp = _prune_to_attr_product_paths(Gp, product_id, engaged_attr_ids)

    return Gp, product_id, engaged_nodes

def _prune_to_attr_product_paths(G: nx.DiGraph, product_id: Optional[str], engaged_attr_ids: List[str]) -> nx.DiGraph:
    if not product_id or not engaged_attr_ids:
        return G.copy()

    try:
        descendants_of_product = set(nx.descendants(G, product_id))
    except nx.NetworkXError:
        descendants_of_product = set()
    descendants_of_product.add(product_id)

    keep_nodes: Set[str] = set()
    for attr_id in engaged_attr_ids:
        if attr_id not in G:
            continue
        try:
            ancestors_of_attr = set(nx.ancestors(G, attr_id))
        except nx.NetworkXError:
            ancestors_of_attr = set()
        ancestors_of_attr.add(attr_id)
        keep_nodes |= (ancestors_of_attr & descendants_of_product)

    # *** ensure seeds + product survive even if graph is sparse ***
    keep_nodes.update([n for n in engaged_attr_ids if n in G])
    if product_id in G:
        keep_nodes.add(product_id)

    if not keep_nodes:
        return G.copy()
    return G.subgraph(keep_nodes).copy()

#---- Synth human-ready campaign names----
def synthesize_sequence_theme_humanized(G, steps, persona_label=None, use_llm=True):
    base = synthesize_sequence_theme(G, steps, persona_label)
    
    if not use_llm:
        return base
    
    prompt = (
        f"Create a human-friendly campaign title and subtitle summarizing this sequence:\n"
        f"Persona: {persona_label}\n"
        f"Steps: {[s.get('concern_label') for s in steps]}\n"
        f"Base title: {base['title']}\n"
        f"Base subtitle: {base['subtitle']}\n\n"
        f"Output as JSON {{'title': str, 'subtitle': str}}."
    )

    try:
        llm_output = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        refined = json.loads(llm_output["choices"][0]["message"]["content"])
        base.update({
            "title": refined.get("title", base["title"]),
            "subtitle": refined.get("subtitle", base["subtitle"])
        })
    except Exception:
        pass

    return base


def synthesize_sequence_theme(G, steps, persona_label=None):
    """
    Create a human-readable theme from a sequence's concerns + stage trajectory.
    """
    if not steps:
        return {"title": persona_label or "Program", "subtitle": ""}

    # 1) collect weighted keywords by stage
    kw = Counter()
    pain_kw = Counter()
    prob_kw = Counter()
    exec_kw = Counter()
    res_kw = Counter()

    for s in steps:
        stage = _stage_norm(s.get("stage"))
        w = _STAGE_WEIGHT.get(stage, 0.8)
        label = s.get("concern_label") or s.get("cid") or ""
        for t in _extract_keywords(label):
            kw[t] += w
            if stage == "pain":
                pain_kw[t] += w
            elif stage == "problem":
                prob_kw[t] += w
            elif stage == "execution":
                exec_kw[t] += w
            elif stage == "resolution":
                res_kw[t] += w

    # 2) primary topic: prefer pain/problem keywords; fallback to global
    topic_terms = []
    for pool in (pain_kw, prob_kw, kw):
        topic_terms = [t for t,_ in pool.most_common(3)]
        # drop overly generic overlaps
        topic_terms = [t for t in topic_terms if t not in {"issue","issues","risk","risks","improve","improvement"}]
        if topic_terms:
            break

    # 3) resolution/capability phrase (if any)
    res_label, res_id = _best_label_for_stage(steps, "resolution")
    res_terms = _extract_keywords(res_label or "")
    res_phrase = _nice_join(res_terms, limit=3)

    # 4) build title
    base_topic = _nice_join(topic_terms, limit=3)
    if base_topic and res_phrase:
        title = f"{base_topic}: Toward {res_phrase}"
    elif base_topic:
        title = base_topic
    elif res_phrase:
        title = f"Toward {res_phrase}"
    else:
        # last resort: use first concern label
        title = (steps[0].get("concern_label") or steps[0].get("cid") or "Program").title()

    # small cleanups
    title = re.sub(r"\s+", " ", title).strip()
    title = title[:90]

    # 5) subtitle = trajectory + (optional persona)
    traj = _trajectory_phrase(steps)
    if persona_label and persona_label != "coalition":
        subtitle = f"{traj} · {persona_label}"
    else:
        subtitle = traj

    return {"title": title, "subtitle": subtitle} 

# -----------------------------------------------------------------------------
# Concern coalitions / sequences (presentation logic)
# -----------------------------------------------------------------------------
def _jobs_of_concern(G, cid):
    jobs = set()
    for u, v, _ in G.in_edges(cid, data=True):
        if _nt(G, u) == "job":
            jobs.add(u)
    for u, v, _ in G.out_edges(cid, data=True):
        if _nt(G, v) == "job":
            jobs.add(v)
    return jobs

def _pains_of_concern(G, cid):
    pains = set()
    for u, v, _ in G.in_edges(cid, data=True):
        if _nt(G, u) == "pain":
            pains.add(u)
    for u, v, _ in G.out_edges(cid, data=True):
        if _nt(G, v) == "pain":
            pains.add(v)
    return pains

def _concern_similarity(G, a, b):
    aj, bj = _jobs_of_concern(G, a["cid"]), _jobs_of_concern(G, b["cid"])
    ap, bp = _pains_of_concern(G, a["cid"]), _pains_of_concern(G, b["cid"])
    def jacc(s1, s2):
        if not s1 and not s2:
            return 0.0
        return len(s1 & s2) / max(1, len(s1 | s2))
    s = 0.6 * jacc(aj, bj) + 0.35 * jacc(ap, bp)
    if (a.get("stage") or "") == (b.get("stage") or ""):
        s += 0.05
    return s

def build_concern_coalitions(
    G,
    concern_backlog,
    *,
    sim_threshold: float = 0.35,
    max_coalitions: int = 6,
    max_items_per: int = 8,
):
    items = [
        {"pid": r.get("pid"), "persona_label": r.get("persona_label"), "cid": r.get("cid"), "stage": r.get("stage"),
         "lift_proxy": float(r.get("lift_proxy", 0.0)),
         "concern_label": r.get("concern_label") or r.get("cid")}
        for r in concern_backlog
        if r.get("cid") in G
    ][:60]
    used = set()
    coalitions = []
    for x in items:
        kx = (x["pid"], x["cid"])
        if kx in used:
            continue
        group = [x]
        used.add(kx)
        for y in items:
            ky = (y["pid"], y["cid"])
            if ky in used:
                continue
            if _concern_similarity(G, x, y) >= sim_threshold:
                group.append(y)
                used.add(ky)
        rep = max(group, key=lambda z: z["lift_proxy"])
        coalitions.append({
            "coalition_id": f"coco:{rep['cid']}",
            "label": rep.get("concern_label", rep["cid"]),
            "members": sorted(
                [{"persona": g["pid"], "persona_label": g["persona_label"], "concern_id": g["cid"], "stage": g["stage"],
                  "lift_proxy": float(g["lift_proxy"]),
                  "label": g.get("concern_label", g["cid"])} for g in group],
                key=lambda a: a["lift_proxy"], reverse=True
            )[:max_items_per],
        })
        if len(coalitions) >= max_coalitions:
            break
    return coalitions



def _stage_rank(stage): return _STAGE_ORDER.get((stage or "").lower(), 1)



from collections import defaultdict

def build_concern_sequences(
    G,
    concern_backlog,
    concern_coalitions,
    *,
    max_sequences: int = 6,
    max_len: int = 4,
    per_persona_limit: int = 1,
    global_cid_cap: int = 3,
):
    """
    Build stage-increasing concern sequences with:
      - per_persona_limit: cap sequences per persona
      - global_cid_cap: cap total usage of the same (coalition-normalized) concern across all sequences
      - concern_coalitions: cluster similar concerns so we diversify
    """

    # ---------- Coalition indexing ----------
    # cid -> coalition key (prefer coalition_id; fallback to cid)
    cid_to_coal = {}
    coal_label = {}
    for coco in (concern_coalitions or []):
        coco_id = coco.get("coalition_id")
        label = coco.get("label") or coco_id
        for m in coco.get("members", []):
            cid = m.get("concern_id")
            if cid:
                cid_to_coal[cid] = coco_id
        if coco_id:
            coal_label[coco_id] = label

    def coalition_key(cid: str) -> str:
        return cid_to_coal.get(cid, cid)  # normalize to coalition if present

    # ---------- Collect per-persona, per-stage ranked rows ----------
    by_persona_all = defaultdict(list)
    by_persona_stage = defaultdict(lambda: defaultdict(list))
    for r in (concern_backlog or []):
        cid = r.get("cid")
        pid = r.get("pid")
        if not cid or cid not in G or not pid:
            continue
        row = {
            "pid": pid,
            "persona_label": r.get("persona_label") or (G.nodes.get(pid, {}).get("label") or pid),
            "cid": cid,
            "stage": _stage_norm(r.get("stage")),
            "lift_proxy": float(r.get("lift_proxy", 0.0)),
            "concern_label": r.get("concern_label") or (G.nodes.get(cid, {}).get("label") or cid),
        }
        by_persona_all[pid].append(row)
        by_persona_stage[pid][row["stage"]].append(row)

    # sort each persona’s pool: stage ascending, then lift desc
    for pid, rows in by_persona_all.items():
        rows.sort(key=lambda z: (_STAGE_ORDER.get(_stage_norm(z["stage"]), 99), -z["lift_proxy"]))
    for pid, stage_map in by_persona_stage.items():
        for stg, rows in stage_map.items():
            rows.sort(key=lambda z: (-z["lift_proxy"]))

    # ---------- Caps & counters ----------
    persona_seq_count = defaultdict(int)     # sequences emitted per persona
    global_coal_usage = defaultdict(int)     # counts by coalition_key

    sequences = []

    # Iterate personas in priority order (by the best lift row they have)
    persona_ids_sorted = sorted(
        by_persona_all.keys(),
        key=lambda pid: (max((r["lift_proxy"] for r in by_persona_all[pid]), default=0.0)),
        reverse=True
    )

    for pid in persona_ids_sorted:
        if len(sequences) >= max_sequences:
            break
        if persona_seq_count[pid] >= per_persona_limit:
            continue

        rows = by_persona_all[pid]
        i = 0
        while i < len(rows) and len(sequences) < max_sequences and persona_seq_count[pid] < per_persona_limit:
            start = rows[i]
            seq = []
            used_cids = set()
            last_stage_rank = -1

            # helper: try to append a step (with cap checks & alternatives)
            def try_append_step(target_row):
                cid = target_row["cid"]
                stg = target_row["stage"]
                k = coalition_key(cid)
                # check global cap and local duplicates
                if global_coal_usage[k] >= global_cid_cap or cid in used_cids:
                    # try alternate of same stage for this persona
                    for alt in by_persona_stage[pid].get(stg, []):
                        alt_cid = alt["cid"]
                        alt_k = coalition_key(alt_cid)
                        if alt_cid in used_cids:
                            continue
                        if global_coal_usage[alt_k] >= global_cid_cap:
                            continue
                        # respect strictly increasing stage
                        if _STAGE_ORDER.get(stg, -1) > last_stage_rank:
                            seq.append(alt)
                            used_cids.add(alt_cid)
                            return True
                    return False
                else:
                    if _STAGE_ORDER.get(stg, -1) > last_stage_rank:
                        seq.append(target_row)
                        used_cids.add(cid)
                        return True
                return False

            # always try to start with the chosen row
            if not try_append_step(start):
                i += 1
                continue

            last_stage_rank = _STAGE_ORDER.get(seq[-1]["stage"], -1)

            # extend with strictly increasing stages, honoring caps
            for j in range(i + 1, len(rows)):
                if len(seq) >= max_len:
                    break
                candidate = rows[j]
                sj = _STAGE_ORDER.get(candidate["stage"], -1)
                if sj <= last_stage_rank:
                    continue
                if try_append_step(candidate):
                    last_stage_rank = _STAGE_ORDER.get(seq[-1]["stage"], -1)

            if seq:
                # theme using the (possibly coalition-normalized) concerns
                persona_label = seq[0]["persona_label"]
                theme = synthesize_sequence_theme_humanized(G, seq, persona_label=persona_label, use_llm=False)

                # finalize sequence
                sequences.append({
                    "sequence_id": f"seq:{pid}:{'|'.join([s['cid'] for s in seq])}",
                    "theme": theme,
                    "sequence": [
                        {
                            "persona": s["pid"],
                            "persona_label": s["persona_label"],
                            "cid": s["cid"],
                            "stage": s["stage"],
                            "lift_proxy": s["lift_proxy"],
                            "concern_label": s["concern_label"],
                        }
                        for s in seq
                    ],
                    "final_win": float(sum(s["lift_proxy"] for s in seq) / max(1, len(seq))),
                })

                # update caps: count each step against global coalition cap
                for s in seq:
                    global_coal_usage[coalition_key(s["cid"])] += 1

                persona_seq_count[pid] += 1

            i += 1
            # hard stop if persona hit limit
            if persona_seq_count[pid] >= per_persona_limit:
                break

    # final rank & truncate
    sequences.sort(key=lambda s: s["final_win"], reverse=True)
    return sequences[:max_sequences]


from itertools import combinations
from collections import defaultdict



def _label(G, nid, fallback=None):
    d = G.nodes.get(nid, {})
    return d.get("label") or d.get("name") or d.get("title") or fallback or nid

def _stage_norm(stage: str) -> str:
    return (stage or "").lower()

def build_persona_coalitions(
    G,
    top_personas_list,
    *,
    # NEW: pass the backlog we already computed
    concern_backlog=None,
    # tuning
    topK_per_persona: int = 6,
    max_pairs: int = 8,
    min_overlap: float = 0.20,
):
    """
    Build persona coalitions by measuring overlap of their top concerns.
    If concern_backlog is provided, we derive each persona's 'concern set' from it.
    Otherwise, we fall back to the legacy edge-based Jaccard on jobs/pains.
    """
    # --------------------------------------------
    # Prefer concern_backlog driven coalitions
    # --------------------------------------------
    if concern_backlog:
        # 1) Build per-persona shortlist of concerns (topK by lift_proxy)
        per_persona = defaultdict(list)
        for r in concern_backlog:
            pid = r.get("pid")
            if not pid:
                continue
            per_persona[pid].append({
                "cid": r.get("cid"),
                "stage": _stage_norm(r.get("stage")),
                "lift": float(r.get("lift_proxy", 0.0)),
                "label": r.get("concern_label") or r.get("cid"),
            })

        # keep only personas we care about (top list intersection)
        top_ids = [p["id"] for p in top_personas_list if p.get("id")]
        per_persona = {
            pid: sorted(rows, key=lambda x: x["lift"], reverse=True)[:topK_per_persona]
            for pid, rows in per_persona.items() if pid in top_ids
        }
        # If < 2 personas have concerns, nothing to pair
        if sum(1 for v in per_persona.values() if v) < 2:
            return []

        # 2) Build pairwise overlaps
        def concern_key(row):
            # identical concern id defines the atomic item; you can widen later (e.g., same job family)
            return row["cid"]

        out_pairs = []
        for a, b in combinations(per_persona.keys(), 2):
            A = per_persona.get(a, [])
            B = per_persona.get(b, [])
            if not A or not B:
                continue

            setA = {concern_key(x) for x in A}
            setB = {concern_key(x) for x in B}
            inter = setA & setB
            union = setA | setB

            if not union:
                continue

            # Plain Jaccard overlap
            jacc = len(inter) / float(len(union))

            if jacc < min_overlap:
                continue

            # Lift quality on the shared concerns (stage-weighted sum of min(lift_a, lift_b))
            # Build quick lookups
            idxA = {x["cid"]: x for x in A}
            idxB = {x["cid"]: x for x in B}
            shared = []
            weighted_sum = 0.0
            for cid in inter:
                ra, rb = idxA[cid], idxB[cid]
                stage = ra["stage"] if ra["stage"] == rb["stage"] else ra["stage"] or rb["stage"]
                w = _STAGE_WEIGHT.get(stage, 0.85)
                # be conservative: take the weaker lift among the two personas, then stage-weight it
                min_lift = min(ra["lift"], rb["lift"])
                weighted = w * min_lift
                weighted_sum += weighted
                shared.append({
                    "concern_id": cid,
                    "stage": stage,
                    "label": _label(G, cid, fallback=ra.get("label") or rb.get("label")),
                    "lift_a": ra["lift"],
                    "lift_b": rb["lift"],
                    "weighted_pair_lift": weighted,
                })

            # Compatibility: blend of breadth (jaccard) and quality (normalized weighted_sum)
            # Simple normalization: divide by sum of topK weights to keep score in [0, ~1]
            denom = (
                sum(x["lift"] for x in A) + sum(x["lift"] for x in B)
            ) or 1.0
            quality = min(1.0, weighted_sum / denom)
            compatibility = 0.6 * jacc + 0.4 * quality

            out_pairs.append({
                "coalition_id": f"pcoal:{a}|{b}",
                "pair": [a, b],
                "labels": [_set_node_label(G, a), _set_node_label(G, b)],
                "compatibility": round(float(compatibility), 6),
                "shared_concerns": sorted(shared, key=lambda s: s["weighted_pair_lift"], reverse=True),
                "overlap_jaccard": round(float(jacc), 6),
                "overlap_size": len(inter),
            })

        # Sort by compatibility and cap
        out_pairs.sort(key=lambda x: x["compatibility"], reverse=True)
        return out_pairs[:max_pairs]

def get_best_org_theme(G: nx.DiGraph, core_scores: Dict[str, Dict]) -> Optional[Dict]:
    pain_trigger_nodes = get_nodes_list_ids(G, "pain_trigger", {})
    pain_triggers_scored = []
    for p in pain_trigger_nodes:
        if p in core_scores:
            score = core_scores.get(p, {})
            pain_triggers_scored.append({
                "id": p,
                "label": _set_node_label(G, p),
                "involvement": float(score.get("involvement", 0.0)),
                "activation": float(score.get("activation", 0.0)),
                "strength": float(score.get("strength", 0.0)),
            })
        else:
            continue
    if pain_triggers_scored:
        return max(pain_triggers_scored, key=lambda x: (x["involvement"], x["strength"]))

    return None


# -----------------------------------------------------------------------------
# MAIN: generate_rcs (deterministic; math delegated to graph_algorithms)
# -----------------------------------------------------------------------------
def generate_rcs(
    G: nx.DiGraph,
    *,
    engaged_nodes: Optional[List[Dict]] = None,
    boost_factor: float = 2.0,
    top_concerns_per_persona: int = 5,
    top_personas: int = 50,
    plays_per_concern: int = 3,
) -> Tuple[nx.DiGraph, Dict]:
    engaged_nodes = engaged_nodes or []
    # 0) Prepare/prune
    Gp, product_id, engaged_nodes = rcs_prepare(G, engaged_nodes=engaged_nodes)
    # 1) Baseline GraphWin (PPR-based)
    baseline_block = get_graphwin(Gp, engaged_nodes=engaged_nodes)
    Gp.graph["win_likelihood"] = float(baseline_block.get("win_likelihood", 0.0))

    # 2) Core + Persona scores (math in graph_algorithms)
    scores = get_involvement_activation_report(Gp, G, engaged_nodes=engaged_nodes)
    core_scores    = scores.get("core_scores", {}) or {}
    persona_scores = scores.get("persona_scores", {}) or {}
    activation_breakdown = scores.get("activation_breakdown", {}) or {}
    


    pa_rows = [{"id": pid, **vals} for pid, vals in persona_scores.items()]
    top_by_inv = sorted(pa_rows, key=lambda r: r.get("involvement", 0.0), reverse=True)
    top_by_act = sorted(pa_rows, key=lambda r: r.get("activation", 0.0),  reverse=True)
    top_by_str = sorted(pa_rows, key=lambda r: r.get("strength",   0.0),  reverse=True)
    
    # 3) Concerns backlog (uses core_scores)
    concern_backlog, concerns_by_persona = build_concern_backlog_from_activation_breakdown(G, activation_breakdown, stage_weights=None, top_k_per_persona=5)

    
    
    # 4) Coalitions (concern- and persona-level)
    concern_coalitions  = build_concern_coalitions(Gp, concern_backlog, sim_threshold=0.35, max_coalitions=6, max_items_per=8)
    
    persona_coalitions = build_persona_coalitions(
        G,
        top_by_inv[:10],
        concern_backlog=concern_backlog,     # << use the new path
        topK_per_persona=6,
        max_pairs=8,
        min_overlap=0.15,                    # slightly looser to surface pairs
    )


    # 5) Concern sequences
    concern_sequences = build_concern_sequences(
        Gp,
        concern_backlog,
        concern_coalitions=concern_coalitions,
        max_sequences=6,
        max_len=4,
        per_persona_limit=1,
        global_cid_cap=3,
    )

    # 6) Optional reverse-case-study stages & zmot theme (leave empty for now)
    stages: List[Dict] = []

    org_theme = get_best_org_theme(Gp, core_scores)


    # 7) Final report (stable keys)
    report = {
        "baseline": baseline_block,
        "top_personas": {
            "by_involvement": top_by_inv,
            "by_activation":  top_by_act,
            "by_strength":    top_by_str,
        },
        "concerns_by_persona": concerns_by_persona,
        "concerns_flat":       concern_backlog,   # keep as exact alias
        "concern_backlog":     concern_backlog,
        "concern_coalitions":  concern_coalitions,
        "concern_sequences":   concern_sequences,
        "coalitions":          persona_coalitions,
        "stages":              stages,
        "zmot_theme":          org_theme,
        # pass-throughs:
        "core_scores":         core_scores,
        "persona_scores":      persona_scores,
    }

    return Gp, report

