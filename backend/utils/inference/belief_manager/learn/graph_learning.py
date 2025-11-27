# backend/utils/inference/belief_manager/learn/graph_learning.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Callable, Any, Optional
from collections import defaultdict
import math
import uuid
from datetime import datetime

# --------------------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------------------

NodeKey = Tuple[str, str]       # e.g., ("persona","persona:abc"), ("job","job:xyz"), ("pain","pain:123")
EdgeKey = Tuple[NodeKey, NodeKey]

@dataclass
class EdgeBayes:
    alpha: float = 1.0
    beta: float = 1.0
    # optional metadata
    last_updated_at: Optional[str] = None
    rationale: Optional[str] = None

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def n(self) -> float:
        return self.alpha + self.beta

@dataclass
class PersonaMeta:
    perceptibility_prior: float = 0.0  # [-1, +1] small nudges
    label: Optional[str] = None
    # (optional) uncertainty on perceptibility
    perceptibility_n: float = 1.0

@dataclass
class GraphStore:
    """Minimal facade over your real graph/tables. Replace dicts with your stores."""
    edges: Dict[EdgeKey, EdgeBayes] = field(default_factory=lambda: defaultdict(EdgeBayes))
    outgoing: Dict[NodeKey, List[NodeKey]] = field(default_factory=lambda: defaultdict(list))
    incoming: Dict[NodeKey, List[NodeKey]] = field(default_factory=lambda: defaultdict(list))
    personas: Dict[str, PersonaMeta] = field(default_factory=dict)  # key: persona_id (e.g., "persona:abcd")

    def ensure_edge(self, src: NodeKey, dst: NodeKey) -> EdgeBayes:
        if dst not in self.outgoing[src]:
            self.outgoing[src].append(dst)
        if src not in self.incoming[dst]:
            self.incoming[dst].append(src)
        return self.edges[(src, dst)]

# --------------------------------------------------------------------------------------
# Injection points (plug your existing services here)
# --------------------------------------------------------------------------------------

PickJobFn              = Callable[[str], str]  # persona_id -> job_id
DominantJobsFn         = Callable[[str], List[str]]
CandidatePainsIntoJob  = Callable[[str], List[Tuple[str, float]]]  # job_id -> [(pain_id, score)]
InferPainBetweenFn     = Callable[[str, str, str], List[Tuple[str, float]]]  # (job_from, p_from, p_to) -> pains
PersonaMatchScoreFn    = Callable[[str], float]  # persona_id -> match score in this run [0..1]
ExpectedStartFn        = Callable[[], List[str]] # returns expected topK personas at start for this ICP

# --------------------------------------------------------------------------------------
# Core utilities
# --------------------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _beta_incr_edge(
    G: GraphStore, src: NodeKey, dst: NodeKey, 
    w: float, penalize_siblings: bool = True, rho: float = 0.3, rationale: Optional[str] = None
) -> None:
    e = G.ensure_edge(src, dst)
    e.alpha += max(0.0, w)
    e.last_updated_at = _now_iso()
    if rationale:
        e.rationale = rationale

    if penalize_siblings:
        for sib in G.outgoing[src]:
            if sib == dst: 
                continue
            se = G.edges[(src, sib)]
            se.beta += max(0.0, w * rho)
            se.last_updated_at = e.last_updated_at

def _safe_mean(G: GraphStore, src: NodeKey, dst: NodeKey, default: float = 0.5) -> float:
    e = G.edges.get((src, dst))
    return e.mean if e else default

def _conf_score(alpha_beta_sum: float) -> float:
    """Simple monotone map for 'how sure' we are (0..1)."""
    return 1.0 - math.exp(-alpha_beta_sum / 10.0)

# --------------------------------------------------------------------------------------
# 1) Back-score latent edges in the full graph
# --------------------------------------------------------------------------------------

def backscore_latent_edges(
    G: GraphStore,
    observed_path: List[str],  # list of persona_ids in observed order
    persona_match_score: PersonaMatchScoreFn,
    pick_job_for_persona: PickJobFn,
    dominant_jobs: DominantJobsFn,
    infer_pain_between: InferPainBetweenFn,
    candidate_pains_into_job: CandidatePainsIntoJob,
    corridor_conf: float = 1.0,
    split_across_topk: int = 2,
    perceptibility_bump: float = 0.10,
    early_k: int = 3,
) -> Dict[str, Any]:

    """
    Performs 4-credit decomposition for every observed hop and bumps latent incoming edges
    for early gatekeepers. Returns a dict of atomic deltas applied (for audit/debug).
    """
    deltas = {
        "persona_corridor_updates": [],
        "four_credit_updates": [],
        "latent_incoming_updates": [],
        "perceptibility_updates": []
    }

    # credit each observed persona→persona hop
    for p_from, p_to in zip(observed_path, observed_path[1:]):
        ms_from = max(0.0, min(1.0, persona_match_score(p_from)))
        ms_to   = max(0.0, min(1.0, persona_match_score(p_to)))
        w = ms_from * ms_to * corridor_conf

        nk_from = ("persona", p_from)
        nk_to   = ("persona", p_to)

        _beta_incr_edge(G, nk_from, nk_to, w, rationale="observed corridor hop")
        deltas["persona_corridor_updates"].append({"from": p_from, "to": p_to, "w": w})

        # 4-credit decomposition
        job_from = pick_job_for_persona(p_from)
        job_to   = pick_job_for_persona(p_to)
        pains = infer_pain_between(job_from, p_from, p_to)
        if not pains:
            # fallback to candidate pains into the next job
            candidates = candidate_pains_into_job(job_to)
            pains = candidates[:split_across_topk] if candidates else []

        # split w across top-k pains
        per = (w * 1.0) / 4.0
        if pains:
            per_pain_share = per / max(1, min(split_across_topk, len(pains)))
        else:
            per_pain_share = 0.0

        _beta_incr_edge(G, ("persona", p_from), ("job", job_from), per, rationale="obs: persona→job")
        deltas["four_credit_updates"].append({"src": ("persona", p_from), "dst": ("job", job_from), "w": per})

        if pains:
            for pain_id, score in pains[:split_across_topk]:
                _beta_incr_edge(G, ("job", job_from), ("pain", pain_id), per_pain_share, rationale="obs: job→pain")
                _beta_incr_edge(G, ("pain", pain_id), ("job", job_to), per_pain_share, rationale="obs: pain→job")
                deltas["four_credit_updates"].extend([
                    {"src": ("job", job_from), "dst": ("pain", pain_id), "w": per_pain_share},
                    {"src": ("pain", pain_id), "dst": ("job", job_to), "w": per_pain_share},
                ])

        _beta_incr_edge(G, ("job", job_to), ("persona", p_to), per, rationale="obs: job→persona")
        deltas["four_credit_updates"].append({"src": ("job", job_to), "dst": ("persona", p_to), "w": per})

    # (we are *not* hardcoding gatekeepers; early-ness is learned over many episodes)
    for idx, pid in enumerate(observed_path[:early_k]):
        G.personas.setdefault(pid, PersonaMeta())
        G.personas[pid].perceptibility_prior += perceptibility_bump
        G.personas[pid].perceptibility_n += 1.0
        deltas["perceptibility_updates"].append(
            {"persona": pid, "delta": perceptibility_bump}
        )

        for job in dominant_jobs(pid):
            for pain_id, score in candidate_pains_into_job(job)[:split_across_topk]:
                bump = perceptibility_bump * max(0.2, min(1.0, score))
                _beta_incr_edge(
                    G,
                    ("pain", pain_id),
                    ("job", job),
                    bump,
                    penalize_siblings=False,
                    rationale="latent incoming pain→job bump (early persona)",
                )
                deltas["latent_incoming_updates"].append(
                    {
                        "src": ("pain", pain_id),
                        "dst": ("job", job),
                        "w": bump,
                    }
                )


    return deltas

# --------------------------------------------------------------------------------------
# 2) Full inference of recommended changes (ranked)
# --------------------------------------------------------------------------------------

@dataclass
class EdgeReco:
    src: NodeKey
    dst: NodeKey
    delta_prob: float
    confidence: float
    why: str

@dataclass
class PersonaReco:
    persona: str
    delta_perceptibility: float
    confidence: float
    why: str

@dataclass
class LatentNodeProposal:
    node_type: str  # "pain"
    label_suggestions: List[str]
    attach_edges: List[EdgeReco]
    confidence: float
    why: str

@dataclass
class InferenceResult:
    edge_updates: List[EdgeReco]
    perceptibility_updates: List[PersonaReco]
    latent_node_proposals: List[LatentNodeProposal]
    deemphasis_updates: List[EdgeReco]
    debug: Dict[str, Any] = field(default_factory=dict)

def generate_full_inference(
    G: GraphStore,
    observed_path: List[str],
    expected_topk_start: List[str],
    corridor_floor_delta: float = 0.05,
    de_emphasize_delta: float = -0.06,
    min_reco_abs_delta: float = 0.03
) -> InferenceResult:
    """
    Inspects posterior means vs pre-hop baselines and emits a convergent set of recommendations.
    """
    edge_recos: List[EdgeReco] = []
    deemp_recos: List[EdgeReco] = []
    persona_recos: List[PersonaReco] = []
    latent_proposals: List[LatentNodeProposal] = []

    # A) corridor edges along the observed path: propose up-weights proportional to certainty
    for a, b in zip(observed_path, observed_path[1:]):
        src, dst = ("persona", a), ("persona", b)
        e = G.edges.get((src, dst))
        if not e:
            continue
        conf = _conf_score(e.n)
        # delta suggested relative to a floor + certainty
        delta = max(corridor_floor_delta, 0.15 * conf)
        if delta >= min_reco_abs_delta:
            edge_recos.append(EdgeReco(src, dst, delta, conf, "observed corridor; posterior strengthened"))

    # B) perceptibility: personas that appear early on observed but not expected
    observed_set = set(observed_path[:3])  # early 3
    expected_set = set(expected_topk_start[:5])
    unexpected_early = [p for p in observed_set if p not in expected_set]
    for pid in unexpected_early:
        meta = G.personas.get(pid, PersonaMeta())
        conf = min(0.9, 0.5 + 0.1 * meta.perceptibility_n)
        persona_recos.append(PersonaReco(pid, +0.10, conf, "unexpected early appearance"))

    # C) de-emphasize early edges to personas that never appeared
    missing = [p for p in expected_topk_start[:5] if p not in set(observed_path)]
    for miss in missing:
        # de-emphasize any persona→miss early edges found in the graph
        for src in list(G.incoming.get(("persona", miss), [])):
            e = G.edges.get((src, ("persona", miss)))
            if not e:
                continue
            conf = _conf_score(e.n)
            delta = de_emphasize_delta * (0.5 + 0.5 * conf)  # stronger deemphasis if we're confident they didn't occur
            if abs(delta) >= min_reco_abs_delta:
                deemp_recos.append(EdgeReco(src, ("persona", miss), delta, conf, "did not appear; baseline over-weighted"))

    # D) latent proposals: scan for gatekeepers (SE/Legal/etc.) appearing early; suggest upstream pains if weak
    for pid in observed_set:
        # heuristic: if many incoming edges into this persona's jobs are weak, propose latent pains
        # we don't know labels—emit suggestions; UI can rename/confirm
        # collect a couple of attach edges with small means
        weak_attach: List[EdgeReco] = []
        for src, dst in list(G.edges.keys()):
            if dst == ("persona", pid):
                # look for job→persona edges with low mean
                e = G.edges[(src, dst)]
                if src[0] == "job" and e.mean < 0.45 and e.n < 20:
                    weak_attach.append(EdgeReco(src, dst, +0.10, _conf_score(e.n), "early persona needs stronger incoming"))
        if weak_attach:
            latent_proposals.append(LatentNodeProposal(
                node_type="pain",
                label_suggestions=["Security/Compliance Risk", "Integration/Data Access Risk", "Contract Cycle Drag"],
                attach_edges=weak_attach[:3],
                confidence=0.55,
                why="early gatekeeper with weak upstream attachments"
            ))

    return InferenceResult(
        edge_updates=edge_recos,
        perceptibility_updates=persona_recos,
        latent_node_proposals=latent_proposals,
        deemphasis_updates=deemp_recos,
        debug={"observed_path": observed_path, "expected_topk_start": expected_topk_start}
    )

# --------------------------------------------------------------------------------------
# 3) Human-readable summary for frontend
# --------------------------------------------------------------------------------------

def build_frontend_summary(
    account_id: str,
    expected_topk_start: List[str],
    observed_path: List[str],
    inference: InferenceResult
) -> Dict[str, Any]:
    hit_at_1 = bool(observed_path and expected_topk_start and observed_path[0] == expected_topk_start[0])
    hit_at_3 = any(p in expected_topk_start[:3] for p in observed_path[:3])

    persona_path_theses = []
    if inference.edge_updates:
        corridor_pairs = [[e.src[1], e.dst[1]] for e in inference.edge_updates]
        corridor_flat  = list({x for pair in corridor_pairs for x in pair})
        persona_path_theses.append({
            "thesis": "Observed corridors are hotter than assumed",
            "edges": corridor_pairs,
            "personas": corridor_flat,
            "delta_edge_prob": f"+{min(0.15, max(e.delta_prob for e in inference.edge_updates)):.02f}",
            "confidence": round(max(e.confidence for e in inference.edge_updates), 2),
            "rationale": "Corridor fired in observed path; posterior strengthened."
        })

    latent_node_theses = []
    for lp in inference.latent_node_proposals:
        latent_node_theses.append({
            "thesis": "An upstream pain trigger is under-weighted",
            "candidate_labels": lp.label_suggestions,
            "attach_to": [[e.src[1], e.dst[1]] for e in lp.attach_edges],
            "proposed_delta": "+0.10 on pain→job; +0.10 on job→persona",
            "confidence": lp.confidence,
            "rationale": lp.why
        })

    deemph = []
    for d in inference.deemphasis_updates:
        deemph.append({
            "from": d.src[1],
            "to": d.dst[1],
            "delta_prob": f"{d.delta_prob:+.02f}",
            "confidence": round(d.confidence, 2),
            "why": d.why
        })

    percept = []
    for pr in inference.perceptibility_updates:
        percept.append({
            "persona": pr.persona,
            "delta": f"{pr.delta_perceptibility:+.02f}",
            "confidence": round(pr.confidence, 2),
            "why": pr.why
        })

    return {
        "account_id": account_id,
        "summary_version": 2,
        "persona_path_diff": {
            "expected_topK_start": expected_topk_start,
            "observed_path": observed_path,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3
        },
        "learnings": {
            "A_persona_path_theses": persona_path_theses,
            "B_latent_node_theses": latent_node_theses,
            "C_deemphasis": deemph,
            "D_perceptibility": percept
        },
        "recommendations": {
            "edge_updates": [
                {
                    "from": e.src[1], "to": e.dst[1],
                    "delta_prob": f"{e.delta_prob:+.02f}",
                    "confidence": round(e.confidence, 2),
                    "why": e.why
                } for e in inference.edge_updates
            ],
            "perceptibility_updates": percept,
            "latent_node_proposals": [
                {
                    "node_type": lp.node_type,
                    "label_suggestions": lp.label_suggestions,
                    "attach_edges": [[e.src[1], e.dst[1], f"{e.delta_prob:+.02f}"] for e in lp.attach_edges],
                    "confidence": round(lp.confidence, 2),
                    "why": lp.why
                } for lp in inference.latent_node_proposals
            ],
            "deemphasis_updates": deemph
        }
    }

# --------------------------------------------------------------------------------------
# Convenience: end-to-end one-shot
# --------------------------------------------------------------------------------------

def learn_and_summarize(
    G: GraphStore,
    account_id: str,
    observed_path: List[str],
    expected_topk_start: List[str],
    persona_match_score: PersonaMatchScoreFn,
    pick_job_for_persona: PickJobFn,
    dominant_jobs: DominantJobsFn,
    infer_pain_between: InferPainBetweenFn,
    candidate_pains_into_job: CandidatePainsIntoJob,
) -> Dict[str, Any]:

    """Runs back-scoring, builds recommendations, returns frontend summary JSON."""
    # 1) back-score (mutates G)
    _ = backscore_latent_edges(
        G,
        observed_path,
        persona_match_score,
        pick_job_for_persona,
        dominant_jobs,
        infer_pain_between,
        candidate_pains_into_job,
    )

    # 2) infer convergent recommendations (does NOT mutate G)
    inference = generate_full_inference(G, observed_path, expected_topk_start)
    # 3) human-readable summary
    return build_frontend_summary(account_id, expected_topk_start, observed_path, inference)


#--------------------------------------------------------------------------------
# Save learnings as epsidoes - TBD
#--------------------------------------------------------------------------------
def record_learning_episode(
    account_id: str,
    product_id: str,
    diffs: Dict[str, Any],
    neighborhoods: Dict[str, Any],
    recommendations: Dict[str, Any],
    when_iso: Optional[str] = None,
    episode_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Persist one learning episode (diffs -> neighborhoods -> recos).

    This remains a lightweight hook:
      - Always returns the payload dict
      - Best-effort: if SHM models / DB are available AND episode_id is provided,
        it will also create ShmLearningUpdate rows for the edge/node recos.
    """
    print("Recording learning episode...")
    ts = when_iso or _now_iso()
    payload = {
        "account_id": account_id,
        "product_id": product_id,
        "timestamp": ts,
        "diffs": diffs,
        "neighborhoods": neighborhoods,
        "recommendations": recommendations,
    }

    # --- Best-effort SHM persistence for learning updates ---
    if episode_id:
        try:
            from sqlalchemy.orm import Session
            from backend.database import get_db
            from backend.super_models.shm.episode import (
                ShmLearningUpdate,
                ShmUpdateType,
            )

            db: Session = next(get_db())
            try:
                recs = recommendations or {}
                edge_recs = (
                    recs.get("edge_updates_ranked")
                    or recs.get("edge_updates")
                    or []
                )
                node_recs = (
                    recs.get("node_updates_ranked")
                    or recs.get("perceptibility_updates")
                    or []
                )

                # Edge-level updates (persona/job/pain/etc. edges)
                for r in edge_recs:
                    u = r.get("u") or r.get("from")
                    v = r.get("v") or r.get("to")

                    # delta may be named "delta" or "delta_prob" and may be a string
                    raw_delta = (
                        r.get("delta")
                        if "delta" in r
                        else r.get("delta_prob")
                    )
                    delta: float
                    try:
                        delta = float(raw_delta)
                    except Exception:
                        try:
                            delta = float(
                                str(raw_delta or "0")
                                .replace("%", "")
                                .replace("+", "")
                            )
                        except Exception:
                            delta = 0.0

                    conf_val = r.get("confidence")
                    try:
                        conf = float(conf_val) if conf_val is not None else None
                    except Exception:
                        conf = None

                    upd = ShmLearningUpdate(
                        episode_id=episode_id,
                        product_id=product_id,
                        account_id=account_id,
                        update_type=ShmUpdateType.bayes_param_update,
                        band=r.get("band"),
                        u_node_id=u,
                        v_node_id=v,
                        node_id=None,
                        field=None,
                        delta=delta,
                        confidence=conf,
                        payload=r,
                    )
                    db.add(upd)
                    print("[graph_learning - edge] Updating ShmLearningUpdate with :", upd)

                # Node-level updates (e.g. perceptibility bumps)
                for r in node_recs:
                    node_id = r.get("node_id") or r.get("persona")
                    raw_delta = r.get("delta")

                    try:
                        delta = float(raw_delta)
                    except Exception:
                        try:
                            delta = float(
                                str(raw_delta or "0")
                                .replace("%", "")
                                .replace("+", "")
                            )
                        except Exception:
                            delta = 0.0

                    conf_val = r.get("confidence")
                    try:
                        conf = float(conf_val) if conf_val is not None else None
                    except Exception:
                        conf = None

                    field = r.get("field") or "perceptibility"

                    upd = ShmLearningUpdate(
                        episode_id=episode_id,
                        product_id=product_id,
                        account_id=account_id,
                        update_type=ShmUpdateType.bayes_param_update,
                        band=None,
                        u_node_id=None,
                        v_node_id=None,
                        node_id=node_id,
                        field=field,
                        delta=delta,
                        confidence=conf,
                        payload=r,
                    )
                    db.add(upd)
                    print("[graph_learning - node] Updating ShmLearningUpdate with :", upd)

                db.commit()
            except Exception as e:
                db.rollback()
                print(
                    "[graph_learning] Failed to persist ShmLearningUpdate rows:",
                    repr(e),
                )
            finally:
                db.close()
        except Exception as e:
            # Swallow import/DB errors; keep the main learning flow working
            print("[graph_learning] SHM models/DB not available:", repr(e))

    print("Recorded learning episode payload.")
    return payload


