# backend/utils/crm_management/persona_normalization.py
from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import datetime as dt
import numpy as np

from backend.database import get_db
from backend.utils.crm_management.person_models import AccountPerson, AccountPersonJob
from backend.utils.crm_management.target_account_manager import get_account_by_id
from backend.utils.graph_base.network_graph import (
    build_product_graph,
    get_edge_attribute,
    get_node_by_id,
    get_nodes_list_ids,
    get_source_nodes_by_target_and_type,  # persona is target; jobs are sources
    _set_node_label,
)
from backend.utils.knowledge_base.canonicalizer import (
    canonicalize_persona,   # expects List[{"title","department","seniority"}] -> Dict[str, dict]
    canonicalize_job,       # returns mapping original_text -> canonical_label
)
from backend.utils.knowledge_base.canonical_maps.canonical_loader import load_embeddings
from backend.utils.embedding.embed_utils import get_embedding


# ---- Persona embedding table (legacy fallback) ----
# Expected key format: "title|department|seniority" (all lowercased/trimmed)
EMB: Dict[str, List[float]] = load_embeddings("persona") or {}

# Simple cache to avoid recompute across a request burst
_PERSON_CACHE: Dict[str, Dict[str, Any]] = {}


# ------------------------ small helpers ------------------------

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _infer_seniority_from_title(title: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in ["chief", "cxo", "cfo", "ceo", "coo", "cto", "cmo", "vp", "vice president", "head"]):
        return "Executive"
    if "director" in t:
        return "Executive"
    if any(k in t for k in ["manager", "lead", "owner"]):
        return "Manager"
    if any(k in t for k in ["senior", "sr."]):
        return "Senior"
    return "Operator"


SENIORITY_ORDER = ["junior", "operator", "manager", "senior", "executive"]


def _normalize_seniority(s: str) -> str:
    t = (s or "").strip().lower()
    if not t:
        return ""
    # normalize common variants
    if t in {"cxo", "chief", "cfo", "ceo", "coo", "cto", "cmo", "president", "vp", "vice president", "head", "director"}:
        return "executive"
    if "vice" in t or "vp" in t or "chief" in t or "cxo" in t:
        return "executive"
    if "director" in t or "head" in t:
        return "executive"
    if "manager" in t or "owner" in t or "lead" in t:
        return "manager"
    if t.startswith("sr") or "senior" in t:
        return "senior"
    if t == "exec":
        return "executive"
    if t in {"ic", "indiv contributor", "individual contributor"}:
        return "operator"
    # if already one of ours, keep it
    if t in SENIORITY_ORDER:
        return t
    return t  # fallthrough


def _persona_key(title: str, department: str, seniority: str) -> str:
    return f"{(title or '').strip().lower()}|{(department or '').strip().lower()}|{(seniority or '').strip().lower()}"


def _normalize_actor_to_canonical_triplet(actor: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Canonicalize title/department; clamp seniority to the fixed ladder.
    Returns: (title_c, dept_c, seniority_c)
    """
    title = (actor.get("title") or actor.get("role") or "").strip()
    dept  = (actor.get("department") or "").strip()
    snr   = (actor.get("seniority") or "").strip() or _infer_seniority_from_title(title)

    c_map = canonicalize_persona([{"title": title, "department": dept, "seniority": snr}]) or {}
    canon = list(c_map.values())[0] if c_map else {"title": title, "department": dept, "seniority": snr}
    c_title = (canon.get("title") or title).strip().lower()
    c_dept  = (canon.get("department") or dept).strip().lower()
    c_snr   = _normalize_seniority((canon.get("seniority") or snr).strip())
    return c_title, c_dept, c_snr


def _persona_node_parts(nid: str, data: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Return (title, department, seniority) lowercased for a persona node.
    Prefer parsing id 'title|department|seniority'; else read node attrs.
    """
    node_type = (data.get("node_type") or data.get("type") or "").strip().lower()

    t = (data.get("title") or data.get("label") or "").strip().lower()
    d = (data.get("department") or "").strip().lower()
    s_raw = (data.get("seniority") or "").strip()

    if node_type == "canonical_persona":
        t = (data.get("label") or t).strip().lower()
        departments = data.get("typical_departments") or []
        if departments and not d:
            d = (departments[0] or "").strip().lower()
        dist = data.get("typical_seniority_distribution") or {}
        if dist:
            s_raw = max(dist.items(), key=lambda kv: kv[1])[0]
    elif node_type == "persona_variant":
        t = (data.get("title") or data.get("label") or t).strip().lower()
        d = (data.get("department") or d).strip().lower()
        s_raw = (data.get("seniority") or s_raw).strip()

    # Fallback to parsing from id if attrs are incomplete
    if (not t or not d or not s_raw) and "|" in nid:
        parts = nid.split("|")
        if len(parts) >= 3:
            t = t or parts[0].strip().lower()
            d = d or parts[1].strip().lower()
            s_raw = s_raw or parts[2].strip()

    s = _normalize_seniority(s_raw)
    return t, d, s


def _fuzzy(a: str, b: str) -> float:
    """
    Lightweight token overlap score in [0,1].
    Works well for titles like 'finance controller' vs 'controller, finance'.
    """
    A = {tok for tok in a.replace("/", " ").replace(",", " ").split() if tok}
    B = {tok for tok in b.replace("/", " ").replace(",", " ").split() if tok}
    if not A and not B:
        return 0.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union


# -------------------------------------------------------------------
# Persona Resolver – incremental, thesis-aware matcher
# -------------------------------------------------------------------

def _actor_identity_key(actor: Dict[str, Any], canonical_meta: Dict[str, Any]) -> str:
    """
    Build a stable key for this actor so the caller can store resolver_state
    per (product_id, account_id, actor_key).

    Prefer email; fall back to canonical title|dept|seniority.
    """
    email = (actor.get("email") or "").strip().lower()
    if email:
        return f"email:{email}"
    return f"{canonical_meta.get('title','')}|{canonical_meta.get('department','')}|{canonical_meta.get('seniority','')}"


def _init_resolution_state(
    actor_key: str,
    canonical_meta: Dict[str, Any],
    prev_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Initialize or reuse a resolution state for an actor.
    Caller is responsible for persisting this across engagements.
    """
    if prev_state:
        # shallow copy to avoid mutating caller's dict in-place
        state = dict(prev_state)
        state.setdefault("mass", dict(prev_state.get("mass", {})))
        state.setdefault("posterior", dict(prev_state.get("posterior", {})))
        state.setdefault("thesis_history", list(prev_state.get("thesis_history", [])))
        state.setdefault("current_thesis", prev_state.get("current_thesis", {
            "persona_id": None,
            "status": "unresolved",
            "confidence": 0.0,
        }))
        state.setdefault("actor_key", actor_key)
        state.setdefault("canonical_meta", canonical_meta)
        state.setdefault("episodes", int(prev_state.get("episodes", 0)))
        return state

    return {
        "actor_key": actor_key,
        "canonical_meta": canonical_meta,
        "episodes": 0,
        "mass": {},           # persona_id -> accumulated evidence (positive only)
        "posterior": {},      # persona_id -> probability
        "current_thesis": {   # our current "best story" about this actor
            "persona_id": None,
            "status": "unresolved",  # unresolved|tentative|stable|revised|new_node_candidate
            "confidence": 0.0,
        },
        "thesis_history": [], # list of thesis transitions over time
        "last_resolved": None,
    }


def _graph_prior_for_persona(
    G,
    persona_id: str,
    account_persona_ids: Optional[List[str]] = None,
    alpha: float = 0.2,
) -> float:
    """
    Simple graph-signature prior:
    - Count connections between this persona and other personas already
      seen in the account (edges in either direction).
    - prior = 1 + alpha * degree.

    This nudges us toward personas that "fit" the current account committee.
    """
    if not account_persona_ids:
        return 1.0

    deg = 0
    for other in account_persona_ids:
        if other == persona_id:
            continue
        if G.has_edge(persona_id, other) or G.has_edge(other, persona_id):
            deg += 1
    return 1.0 + alpha * float(deg)


def _update_resolution_state_from_match(
    state: Dict[str, Any],
    match: Dict[str, Any],
    G,
    account_persona_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    One more engagement episode for this actor:
      - Add evidence for best + alternates.
      - Fold in a graph prior wrt other personas in this account.
      - Recompute a normalized posterior over persona candidates.

    IMPORTANT: This only ever uses POSITIVE evidence (what we actually saw).
               Absence-of-evidence is not treated as negative evidence here.
    """
    state["episodes"] = int(state.get("episodes", 0)) + 1
    mass: Dict[str, float] = state.setdefault("mass", {})

    best_id: Optional[str] = match.get("best")
    best_score = float(match.get("score") or 0.0)
    alternates: List[str] = match.get("alternates", []) or []

    eps = 1e-3  # small floor so new candidates can enter the pool

    # 1) Add evidence from current episode
    if best_id:
        mass[best_id] = mass.get(best_id, eps) + best_score

    # Light evidence to alternates – plausible but weaker
    alt_boost = 0.3 * best_score
    for alt in alternates:
        if alt == best_id:
            continue
        mass[alt] = mass.get(alt, eps) + alt_boost

    # 2) Apply graph-signature prior (connectivity with other personas in this account)
    if account_persona_ids:
        for pid in list(mass.keys()):
            prior = _graph_prior_for_persona(G, pid, account_persona_ids)
            mass[pid] *= prior

    # 3) Normalize → posterior
    Z = sum(mass.values()) or 1.0
    posterior = {pid: float(val) / Z for pid, val in mass.items()}
    state["mass"] = mass
    state["posterior"] = posterior
    return state


def _stage_thesis_transition(
    state: Dict[str, Any],
    new_persona_id: Optional[str],
    new_status: str,
    new_conf: float,
    reason: str,
) -> None:
    """
    Append a thesis stage transition IF it is meaningfully different from the last one.
    """
    history: List[Dict[str, Any]] = state.setdefault("thesis_history", [])
    current = state.get("current_thesis") or {
        "persona_id": None,
        "status": "unresolved",
        "confidence": 0.0,
    }

    changed = (
        current.get("persona_id") != new_persona_id
        or current.get("status") != new_status
        or abs(float(current.get("confidence", 0.0)) - float(new_conf)) > 1e-6
    )

    if changed:
        history.append({
            "t": dt.datetime.utcnow().isoformat() + "Z",
            "from_persona_id": current.get("persona_id"),
            "to_persona_id": new_persona_id,
            "from_status": current.get("status"),
            "to_status": new_status,
            "confidence": float(new_conf),
            "episodes": int(state.get("episodes", 0)),
            "reason": reason,
        })
        state["thesis_history"] = history

    state["current_thesis"] = {
        "persona_id": new_persona_id,
        "status": new_status,
        "confidence": float(new_conf),
    }


def _decide_and_stage_thesis(
    state: Dict[str, Any],
    min_episodes_for_stable: int = 2,
    revise_margin: float = 0.20,
) -> Tuple[Optional[str], str]:
    """
    Decide how far we are in the "thesis" about this actor AND record
    thesis transitions over time.

    Status values:
      - unresolved         : no good explanation yet
      - tentative          : one persona is ahead but not rock-solid
      - stable             : strong, consistent winner
      - revised            : we had a previous thesis and this new one
                             clearly explains observations better
      - new_node_candidate : even after multiple episodes, no persona
                             explains this actor well → likely new node

    We explicitly avoid flapping: we only revise/flip if the new thesis
    is clearly better than the old one (revise_margin).
    """
    posterior: Dict[str, float] = state.get("posterior", {})
    episodes = int(state.get("episodes", 0))

    prev_thesis = state.get("current_thesis") or {
        "persona_id": None,
        "status": "unresolved",
        "confidence": 0.0,
    }
    prev_id = prev_thesis.get("persona_id")
    prev_status = prev_thesis.get("status")

    if not posterior:
        _stage_thesis_transition(
            state,
            new_persona_id=None,
            new_status="unresolved",
            new_conf=0.0,
            reason="No posterior yet; not enough evidence to form a thesis.",
        )
        return None, "unresolved"

    sorted_p = sorted(posterior.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best_p = sorted_p[0]
    second_p = sorted_p[1][1] if len(sorted_p) > 1 else 0.0

    status = "unresolved"
    resolved_id: Optional[str] = None
    reason = "Posterior still too flat; no persona clearly explains observations."

    # Candidate statuses based on shape of posterior
    if episodes >= 3 and best_p < 0.45:
        # Even with some data, no persona is doing a good job.
        status = "new_node_candidate"
        resolved_id = None
        reason = (
            "Multiple episodes but no persona has high posterior; "
            "this actor may represent a missing persona node."
        )
    elif (
        episodes >= min_episodes_for_stable
        and best_p >= 0.75
        and (best_p - second_p) >= 0.30
    ):
        status = "stable"
        resolved_id = best_id
        reason = "Strong, consistent winner persona that clearly dominates alternatives."
    elif best_p >= 0.5 and (best_p - second_p) >= 0.15:
        status = "tentative"
        resolved_id = best_id
        reason = "One persona is ahead, but alternatives still have non-trivial mass."
    else:
        status = "unresolved"
        resolved_id = None
        reason = (
            "Evidence accumulated, but posterior remains ambiguous; "
            "we should keep questioning our current thesis."
        )

    # Temporal stability and revision logic:
    # If we already have a thesis, we only overturn it if the new candidate
    # explains observations clearly better.
    if prev_id and prev_id in posterior:
        prev_p = posterior[prev_id]
        # Case 1: new candidate is different from previous
        if resolved_id and resolved_id != prev_id:
            if (best_p - prev_p) >= revise_margin:
                # New thesis clearly better → mark as revised
                status = "revised"
                reason = (
                    f"Persona {resolved_id} now has materially higher posterior "
                    f"({best_p:.2f}) than previous thesis {prev_id} ({prev_p:.2f}); "
                    "revising thesis."
                )
            else:
                # New evidence is not strong enough to overturn old thesis
                resolved_id = prev_id
                status = prev_status or "stable"
                reason = (
                    "New candidate persona is only marginally better; keeping prior thesis "
                    "to avoid overfitting to noise."
                )
        # Case 2: best_id == prev_id but confidence changed
        elif resolved_id == prev_id:
            if status == "stable" and prev_status != "stable":
                reason = "Existing thesis reinforced; moved from tentative to stable."
        # Case 3: new_node_candidate vs previous stable/tentative thesis
        if status == "new_node_candidate" and prev_status in ("stable", "tentative"):
            # We do NOT immediately throw away a previous thesis just because
            # posterior flattened a bit; we prefer the previous story unless
            # a specific alternative is clearly better.
            resolved_id = prev_id
            status = prev_status
            reason = (
                "Posterior flattened but no alternative persona clearly superior; "
                "keeping previous thesis instead of inventing a new node prematurely."
            )

    # Record transition + update current_thesis
    _stage_thesis_transition(
        state,
        new_persona_id=resolved_id,
        new_status=status,
        new_conf=posterior.get(resolved_id, 0.0) if resolved_id else 0.0,
        reason=reason,
    )

    # Keep a shortcut for "stable" persona if any
    if status in ("stable", "revised") and resolved_id:
        state["last_resolved"] = resolved_id

    return resolved_id, status


# ------------------------ legacy, embedding-based persona matcher ------------------------

def match_persona_for_actor(actor: Dict[str, Any]) -> Dict[str, Any]:
    """
    Embedding/canonical fallback. Kept for backward compatibility.
    """
    name = (actor.get("name") or "").strip()
    title = (actor.get("title") or "").strip()
    dept = (actor.get("department") or "").strip()
    cache_key = f"{name}|{title}|{dept}"

    if cache_key in _PERSON_CACHE:
        return _PERSON_CACHE[cache_key]

    c_title, c_dept, c_snr = _normalize_actor_to_canonical_triplet(actor)
    q_key = _persona_key(c_title, c_dept, c_snr)

    # 1) Exact-key fast path (already embedded)
    if q_key in EMB:
        result = {
            "best": q_key,
            "score": 1.0,
            "alternates": [],
            "canonical_meta": {
                "title": c_title,
                "department": c_dept,
                "seniority": c_snr,
            },
        }
        _PERSON_CACHE[cache_key] = result
        return result

    # 2) Cosine fallback over embedding table
    q_vec_list = get_embedding(q_key) or []
    if not q_vec_list:
        result = {
            "best": q_key,
            "score": 0.0,
            "alternates": [],
            "canonical_meta": {
                "title": c_title,
                "department": c_dept,
                "seniority": c_snr,
            },
        }
        _PERSON_CACHE[cache_key] = result
        return result

    q_vec = np.array(q_vec_list, dtype=np.float32)
    scored: List[Tuple[str, float]] = []
    for k, vec in EMB.items():
        v = np.array(vec, dtype=np.float32)
        scored.append((k, _cos(q_vec, v)))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:3] if scored else [(q_key, 0.0)]
    result = {
        "best": top[0][0],
        "score": float(top[0][1]),
        "alternates": [x[0] for x in top[1:]],
        "canonical_meta": {"title": c_title, "department": c_dept, "seniority": c_snr},
    }
    _PERSON_CACHE[cache_key] = result
    return result


# ------------------------ NEW: graph-aware persona matcher + resolver ------------------------

def match_persona_for_actor_in_graph(
    product_id: str,
    actor: Dict[str, Any],
    resolver_state: Optional[Dict[str, Any]] = None,
    account_persona_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Preferred matcher: actor → persona NODE in the product graph.

    Now with a *thesis-aware resolver* on top:

      - We still compute a per-engagement match (exact + fuzzy) like before.
      - We then UPDATE an incremental resolution_state that the caller
        should persist per (product_id, account_id, actor_key).
      - We DO NOT blindly trust a single engagement; we accumulate evidence
        and only change our thesis when the new one explains observations
        clearly better than the old one.

    Returns (backward-compatible + rich resolution info):
      {
        best: "<persona_node_id>",            # for this episode (may be the resolved thesis persona)
        best_label: str,
        score: float,                         # raw per-engagement match score
        alternates: [ids...],
        canonical_meta: {...},
        resolution: {
          status: "unresolved"|"tentative"|"stable"|"revised"|"new_node_candidate",
          resolved_persona_id: Optional[str],
          confidence: float,                  # posterior mass on resolved_persona_id (if any)
          posterior: {persona_id: prob},      # current posterior over candidates
          thesis_history: [ ... ],            # staged thesis transitions
          state: { ... }                      # updated resolver_state → persist this
        }
      }
    """
    try:
        G = build_product_graph(product_id)
    except Exception:
        # Graph build failed → fall back entirely to legacy embedding-only matcher
        legacy = match_persona_for_actor(actor)
        legacy["note"] = "graph_build_failed_fallback"
        return legacy

    # --- Canonical actor triple ---
    c_title, c_dept, c_snr = _normalize_actor_to_canonical_triplet(actor)
    query_id = _persona_key(c_title, c_dept, c_snr)

    # --- 1) Exact node-id match (strong, cheap path) ---
    if query_id in G and G.nodes[query_id].get("type") == "persona":
        base_result = {
            "best": query_id,
            "best_label": _set_node_label(G, query_id),
            "score": 1.0,
            "alternates": [],
            "canonical_meta": {"title": c_title, "department": c_dept, "seniority": c_snr},
        }
    else:
        # --- 2) Fuzzy matching over persona nodes ---
        runners: List[Tuple[str, str, float]] = []
        persona_nodes = get_nodes_list_ids(G, "persona", {})
        canonical_nodes = get_nodes_list_ids(G, "canonical_persona", {})
        all_personas = persona_nodes + canonical_nodes
        for p in all_personas:
            p_node = get_node_by_id(G, p)
            if not p_node:
                continue
            t, d, s = _persona_node_parts(p, p_node)
            score = (
                0.5 * _fuzzy(c_title, t)
                + 0.3 * _fuzzy(c_dept, d)
                + 0.2 * (1.0 if _normalize_seniority(s) == c_snr else 0.0)
            )
            if score > 0.0:
                runners.append((p, _set_node_label(G, p), score))

        if not runners:
            # No persona nodes or nothing similar → legacy fallback
            legacy = match_persona_for_actor(actor)
            legacy["note"] = "no_persona_nodes_fallback"
            return legacy

        runners.sort(key=lambda x: x[2], reverse=True)
        best_id, best_label, best_score = runners[0]
        alternates = [nid for (nid, _, _) in runners[1:4]]

        base_result = {
            "best": best_id,
            "best_label": best_label,
            "score": float(best_score),
            "alternates": alternates,
            "canonical_meta": {"title": c_title, "department": c_dept, "seniority": c_snr},
        }

    # ---------------------------
    # 3) Persona Resolver overlay
    # ---------------------------
    try:
        canonical_meta = base_result["canonical_meta"]
        actor_key = _actor_identity_key(actor, canonical_meta)

        state = _init_resolution_state(actor_key, canonical_meta, resolver_state)

        # Update state with this new engagement episode (positive evidence only)
        state = _update_resolution_state_from_match(
            state,
            base_result,
            G,
            account_persona_ids=account_persona_ids,
        )

        resolved_id, status = _decide_and_stage_thesis(state)

        # For backward compat:
        # - If we have a resolved thesis persona, we override "best"
        #   so downstream consumers see the thesis persona.
        # - If not, we keep the per-episode best as-is.
        if resolved_id:
            base_result["best"] = resolved_id
            base_result["best_label"] = _set_node_label(G, resolved_id)

        current_thesis = state.get("current_thesis", {})
        base_result["resolution"] = {
            "status": status,
            "resolved_persona_id": current_thesis.get("persona_id"),
            "confidence": float(current_thesis.get("confidence", 0.0)),
            "posterior": state.get("posterior", {}),
            "thesis_history": state.get("thesis_history", []),
            # caller SHOULD persist this keyed by (product_id, account_id, actor_key)
            "state": state,
        }

    except Exception as e:
        # Fail-safe: keep old behavior, but surface error for debugging
        base_result["resolution_error"] = str(e)

    return base_result


# ------------------------ upsert person ------------------------

def upsert_person_from_engagement(product_id: str, account_id: str, actor: Dict[str, Any]) -> str:
    """
    Ensure a person exists within (product_id, account_id) and normalize their meta.
    Returns person_id or "" if account not found.
    """
    print("Upserting actor from engagement for account:", account_id)
    db: Session = next(get_db())
    try:
        if not get_account_by_id(product_id, account_id):
            return ""

        name = (actor.get("name") or "").strip() or "Unknown"
        title = (actor.get("title") or "").strip()
        dept  = (actor.get("department") or "").strip()
        snr   = (actor.get("seniority") or "").strip()

        person = (
            db.query(AccountPerson)
              .filter(AccountPerson.product_id == product_id,
                      AccountPerson.account_id == account_id,
                      AccountPerson.name == name)
              .first()
        )
        if not person:
            person = AccountPerson(
                product_id=product_id,
                account_id=account_id,
                name=name,
                title=title or None,
                department=dept or None,
                seniority=snr or None,
                engagement_count=0,
                last_seen_at=datetime.now(timezone.utc),
            )
            db.add(person)
            db.flush()

        # Prefer richer incoming strings
        if title and (not person.title or len(title) > len(person.title)):
            person.title = title
        if dept and (not person.department or len(dept) > len(person.department)):
            person.department = dept
        if snr and (not person.seniority or len(snr) > len(person.seniority)):
            person.seniority = snr

        actor_payload = {
            "name": name,
            "title": person.title or "",
            "department": person.department or "",
            "seniority": person.seniority or "",
        }
        resolved_persona_id: Optional[str] = None
        canonical_meta: Dict[str, Any] = {}
        try:
            match = match_persona_for_actor_in_graph(product_id, actor_payload)
        except Exception:
            match = None
        if match:
            resolution = match.get("resolution") or {}
            resolved_persona_id = resolution.get("resolved_persona_id") or match.get("best")
            canonical_meta = match.get("canonical_meta") or {}

        if resolved_persona_id:
            person.canonical_persona_id = resolved_persona_id

        meta_department = (canonical_meta.get("department") or "").strip().lower()
        meta_seniority = (canonical_meta.get("seniority") or "").strip().lower()
        if meta_department:
            person.canonical_department = meta_department
        if meta_seniority:
            person.canonical_seniority = meta_seniority

        if not resolved_persona_id:
            # Fallback to text-only canonicalization for legacy coverage
            persona_map = canonicalize_persona([{
                "title": person.title or "",
                "department": person.department or "",
                "seniority": person.seniority or _infer_seniority_from_title(person.title or ""),
            }]) or {}
            canon = list(persona_map.values())[0] if persona_map else None
            if canon and not person.canonical_persona_id:
                c_key = _persona_key(
                    canon.get("title", ""),
                    canon.get("department", ""),
                    canon.get("seniority", ""),
                )
                person.canonical_persona_id = c_key or None
            if canon:
                if not person.canonical_department and canon.get("department"):
                    person.canonical_department = canon.get("department").strip().lower()
                if not person.canonical_seniority and canon.get("seniority"):
                    person.canonical_seniority = canon.get("seniority").strip().lower()

        # Stats
        person.engagement_count = (person.engagement_count or 0) + 1
        person.last_seen_at = datetime.now(timezone.utc)

        db.commit()
        return person.id
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ------------------------ job suggestions ------------------------

def suggest_jobs_for_person(product_id: str, person: AccountPerson, k: int = 6) -> List[Dict[str, Any]]:
    """
    Hop-0 suggestions: Job(source) -[performed_by]-> Persona(target)
    """
    print("Suggesting jobs for person:", person.id)
    try:
        G = build_product_graph(product_id)
    except Exception:
        print("Failed to build product graph in job suggestion:", product_id)
        return []

    actor = {
        "title": person.title or "",
        "department": person.department or "",
        "seniority": person.seniority or "",
    }
    try:
        match = match_persona_for_actor_in_graph(product_id, actor)
        persona_node_id = match.get("best")
    except Exception:
        persona_node_id = None

    if not persona_node_id or persona_node_id not in G:
        return []
    print("Running job search for persona node id:", persona_node_id)

    job_node_ids = get_source_nodes_by_target_and_type(G, persona_node_id, "performed_by") or []
    print("Found job nodes for persona", persona_node_id, ":", job_node_ids)
    if not job_node_ids:
        return []

    out: List[Dict[str, Any]] = []
    for job_id in job_node_ids:
        job_label = _set_node_label(G, job_id)
        relevance = float(get_edge_attribute(G, job_id, persona_node_id, "relevance") or 0.0)
        likelihood = float(get_edge_attribute(G, job_id, persona_node_id, "likelihood") or 0.0)
        score = 0.6 * relevance + 0.4 * likelihood

        job_map = canonicalize_job([job_label]) or {}
        canonical_label = job_map.get(job_label, job_label)

        out.append({
            "job_text": job_label,
            "canonical_job_id": canonical_label,
            "relevance": relevance,
            "likelihood": likelihood,
            "score": score,
            "evidence": {"edge": {"source": job_id, "target": persona_node_id, "type": "performed_by"}},
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:max(1, int(k))]


def upsert_job_suggestions(product_id: str, account_id: str, person_id: str, suggestions: List[Dict[str, Any]]) -> None:
    db: Session = next(get_db())
    try:
        for s in suggestions:
            canon_id = (s.get("canonical_job_id") or "").strip()
            exists = (
                db.query(AccountPersonJob)
                  .filter(AccountPersonJob.product_id == product_id,
                          AccountPersonJob.account_id == account_id,
                          AccountPersonJob.person_id == person_id,
                          AccountPersonJob.canonical_job_id == canon_id)
                  .first()
            )
            if exists:
                continue
            row = AccountPersonJob(
                product_id=product_id,
                account_id=account_id,
                person_id=person_id,
                source="suggested_graph",
                status="suggested",
                job_text=s.get("job_text"),
                canonical_job_id=canon_id or None,
                meta={"score": s.get("score"), "relevance": s.get("relevance"),
                      "likelihood": s.get("likelihood"), "evidence": s.get("evidence")},
            )
            db.add(row)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def set_person_jobs_status(product_id: str, account_id: str, person_id: str, job_ids: List[str], status: str) -> None:
    """
    status ∈ {"confirmed","rejected"}
    """
    db: Session = next(get_db())
    try:
        q = (
            db.query(AccountPersonJob)
              .filter(AccountPersonJob.product_id == product_id,
                      AccountPersonJob.account_id == account_id,
                      AccountPersonJob.person_id == person_id,
                      AccountPersonJob.id.in_(job_ids))
        )
        for row in q.all():
            row.status = status
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def list_people_for_account(product_id: str, account_id: str) -> List[Dict[str, Any]]:
    db: Session = next(get_db())
    try:
        people = (
            db.query(AccountPerson)
              .filter(AccountPerson.product_id == product_id,
                      AccountPerson.account_id == account_id)
              .order_by(AccountPerson.name.asc())
              .all()
        )
        out: List[Dict[str, Any]] = []
        for p in people:
            out.append({
                "id": p.id,
                "name": p.name,
                "title": p.title,
                "department": p.department,
                "seniority": p.seniority,
                "canonical_persona_id": p.canonical_persona_id,
                "engagement_count": p.engagement_count,
                "last_seen_at": p.last_seen_at.isoformat() if p.last_seen_at else None,
            })
        return out
    finally:
        db.close()


def list_person_jobs(product_id: str, account_id: str, person_id: str) -> List[Dict[str, Any]]:
    db: Session = next(get_db())
    try:
        rows = (
            db.query(AccountPersonJob)
              .filter(AccountPersonJob.product_id == product_id,
                      AccountPersonJob.account_id == account_id,
                      AccountPersonJob.person_id == person_id)
              .all()
        )
        return [{
            "id": r.id,
            "job_text": r.job_text,
            "canonical_job_id": r.canonical_job_id,
            "source": r.source,
            "status": r.status,
            "meta": r.meta,
        } for r in rows]
    finally:
        db.close()
