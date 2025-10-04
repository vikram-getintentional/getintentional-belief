# canonicalizer.py — Option A (project-path aware, batch canonicalization)

from __future__ import annotations
import re
import uuid
from pathlib import Path
import os, json
from typing import Dict, List, Any
from collections import defaultdict

from backend.utils.knowledge_base.canonical_maps.canonical_loader import (
    save_canonical_map,
    load_embeddings,  # optional but available in your tree
)
from backend.utils.knowledge_base.canonical_maps.canonical_utils import (
    assign_canonical_labels,
    cluster_items,
)
from backend.utils.embedding.embed_utils import generate_and_save_embeddings

# ---- DEFAULT GRAPH DIRECTORY -----------------------------------

def _default_graph_dir() -> str:
    # mirror graph_loader's default path: backend/utils/graph_base/graph_data
    here = os.path.abspath(os.path.dirname(__file__))  # .../backend/utils/knowledge_base
    base_dir = os.path.dirname(os.path.dirname(here))  # .../backend/utils
    graph_dir = os.path.join(base_dir, "graph_base", "graph_data")
    os.makedirs(graph_dir, exist_ok=True)
    return graph_dir

# ---- BATCH CANONICALIZER ----------------------------------------

class BatchCanonicalizer:
    """
    Minimal, file-backed batch canonicalizer compatible with CanonManager expectations.
    Writes:
      - <out_dir>/<node_type>_nodes.json          (list of nodes)
      - <out_dir>/graph_edges.json                (dict: {product_id: [edges]})
    """
    def __init__(self, out_dir: str | None = None):
        # allow CanonManager() to call without args
        self.out_dir = out_dir or _default_graph_dir()
        self._nodes: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)   # node_type -> id -> node
        self._edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)        # product_id -> edges

    # ---- nodes ----
    def upsert_node(self, node_type: str, node: Dict[str, Any]) -> None:
        nid = node.get("id")
        if not nid:
            return
        self._nodes[node_type][nid] = node

    def upsert_nodes(self, node_type: str, nodes: List[Dict[str, Any]]) -> None:
        for n in nodes:
            self.upsert_node(node_type, n)

    # legacy aliases
    put_node = upsert_node
    put_nodes = upsert_nodes

    # ---- edges ----
    def add_edge(self, product_id: str, edge: Dict[str, Any]) -> None:
        if edge.get("source") in (None, "null") or edge.get("target") in (None, "null"):
            return
        self._edges[product_id].append(edge)

    def add_edges(self, product_id: str, edges: List[Dict[str, Any]]) -> None:
        for e in edges:
            self.add_edge(product_id, e)

    # legacy aliases
    put_edge = add_edge
    put_edges = add_edges

    # ---- flush/write ----
    def flush(self, *args, **kwargs) -> None:
        """Write buffered nodes/edges to disk. Accepts & ignores extra args for API-compat."""
        os.makedirs(self.out_dir, exist_ok=True)

        # write per-type node files
        for node_type, by_id in self._nodes.items():
            path = os.path.join(self.out_dir, f"{node_type}_nodes.json")
            with open(path, "w") as f:
                json.dump(list(by_id.values()), f, indent=2)

        # merge & write edges dict keyed by product_id
        edges_path = os.path.join(self.out_dir, "graph_edges.json")
        existing = {}
        if os.path.exists(edges_path):
            try:
                with open(edges_path, "r") as f:
                    existing = json.load(f) or {}
            except Exception:
                existing = {}

        for pid, edges in self._edges.items():
            existing.setdefault(pid, [])
            existing[pid].extend(edges)

        with open(edges_path, "w") as f:
            json.dump(existing, f, indent=2)

        # reset buffers
        self._nodes.clear()
        self._edges.clear()

    # keep aliases tolerant, too
    write = flush
    save = flush


# ---- PATH RESOLUTION--------------------------------------------
try:
    from backend.utils.knowledge_base.canonical_maps import canonical_paths as _cp

    CANONICAL_MAP_PATH = _cp.CANONICAL_MAP_PATH
    EMBEDDING_OUTPUT_PATH = _cp.EMBEDDING_OUTPUT_PATH

    PATHS = {
        "job": _cp.JOB_CANONICAL_MAP_PATH,
        "pain": _cp.PAIN_CANONICAL_MAP_PATH,
        "persona": _cp.PERSONA_CANONICAL_MAP_PATH,
        "pain_trigger": _cp.PAIN_TRIGGER_CANONICAL_MAP_PATH,
        "zmot_event": _cp.TRIGGER_EVENT_CANONICAL_MAP_PATH,
        "observable_moment": _cp.OBSERVABLE_MOMENT_CANONICAL_MAP_PATH,
        "perceived_metric": _cp.METRIC_CANONICAL_MAP_PATH,
        "keyword": _cp.KEYWORD_CANONICAL_MAP_PATH,
        "archetype": getattr(_cp, "ARCHETYPE_CANONICAL_MAP_PATH",
                             CANONICAL_MAP_PATH / "archetype_to_canonical.json"),
    }
except Exception:
    # Very unlikely in your project, but safe fallback
    CANONICAL_MAP_PATH = Path("backend/utils/knowledge_base/canonical_maps/")
    EMBEDDING_OUTPUT_PATH = CANONICAL_MAP_PATH / "canonical_embeddings/"
    PATHS = {
        "job": CANONICAL_MAP_PATH / "job_to_canonical.json",
        "pain": CANONICAL_MAP_PATH / "pain_to_canonical.json",
        "persona": CANONICAL_MAP_PATH / "persona_to_canonical.json",
        "pain_trigger": CANONICAL_MAP_PATH / "pain_trigger_to_canonical.json",
        "zmot_event": CANONICAL_MAP_PATH / "trigger_event_to_canonical.json",
        "observable_moment": CANONICAL_MAP_PATH / "observable_moment_to_canonical.json",
        "perceived_metric": CANONICAL_MAP_PATH / "metric_to_canonical.json",
        "keyword": CANONICAL_MAP_PATH / "keyword_to_canonical.json",
        "archetype": CANONICAL_MAP_PATH / "archetype_to_canonical.json",
    }

# Ensure dirs exist
CANONICAL_MAP_PATH.mkdir(parents=True, exist_ok=True)
EMBEDDING_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# ---- OPTIONAL: preload embedding tables (safe if absent) -------------------
try:
    JOB_EMBEDDINGS = load_embeddings("job")
    PAIN_EMBEDDINGS = load_embeddings("pain")
    PERSONA_EMBEDDINGS = load_embeddings("persona")
    ATTRIBUTE_EMBEDDINGS = load_embeddings("attribute")
    DEPARTMENT_EMBEDDINGS = load_embeddings("department")
    PAIN_TRIGGER_EMBEDDINGS = load_embeddings("pain_trigger")
    OBSERVABLE_MOMENT_EMBEDDINGS = load_embeddings("observable_moment")
    TITLE_EMBEDDINGS = load_embeddings("title")
    TRIGGER_EVENT_EMBEDDINGS = load_embeddings("trigger_event")
    PERCEIVED_METRIC_EMBEDDINGS = load_embeddings("perceived_metric")
    KEYWORD_EMBEDDINGS = load_embeddings("keyword")
except Exception as e:
    print(f"⚠️ Embedding preload failed: {e}")
    JOB_EMBEDDINGS = {}
    PAIN_EMBEDDINGS = {}
    PERSONA_EMBEDDINGS = {}
    KEYWORD_EMBEDDINGS = {}
    ATTRIBUTE_EMBEDDINGS = {}
    DEPARTMENT_EMBEDDINGS = {}
    PAIN_TRIGGER_EMBEDDINGS = {}
    OBSERVABLE_MOMENT_EMBEDDINGS = {}
    TITLE_EMBEDDINGS = {}
    TRIGGER_EVENT_EMBEDDINGS = {}
    PERCEIVED_METRIC_EMBEDDINGS = {}

# ---- UTILS -----------------------------------------------------------------
def generate_new_id(prefix="pain"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

# ---- CANONICALIZERS (batch) ------------------------------------------------
def _cluster_and_label(items: List[str], embed_key: str) -> Dict[str, str]:
    """
    Shared helper: generate embeddings for `items`, cluster, then assign labels.
    Returns a map original -> canonical_label.
    """
    uniq = list(dict.fromkeys([s.strip() for s in items if s and s.strip()]))
    if not uniq:
        return {}
    embs = generate_and_save_embeddings(uniq, embed_key)
    if len(embs) <= 1:
        # trivial case: identity map
        return {uniq[0]: uniq[0]} if uniq else {}
    clusters = cluster_items(embs)
    label_map = assign_canonical_labels(clusters)
    return {s: label_map[s] for s in uniq}

def canonicalize_job(jobs: List[str]) -> Dict[str, str]:
    jobs = list(set(jobs))
    if not jobs:
        print("⚠️ No jobs provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(jobs, "job")
    save_canonical_map(canonical_map, PATHS["job"])
    return canonical_map

def canonicalize_pain(pains: List[str]) -> Dict[str, str]:
    pains = list(set(pains))
    if not pains:
        print("⚠️ No pains provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(pains, "pain")
    save_canonical_map(canonical_map, PATHS["pain"])
    return canonical_map

def canonicalize_perceived_metric(perceived_metrics: List[str]) -> Dict[str, str]:
    perceived_metrics = list(set(perceived_metrics))
    if not perceived_metrics:
        print("⚠️ No perceived metrics provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(perceived_metrics, "perceived_metric")
    save_canonical_map(canonical_map, PATHS["perceived_metric"])
    return canonical_map

# ---- Attributes (helper for triggers/personas) -----------------------------
def canonicalize_attributes(attributes: List[str]) -> Dict[str, str]:
    attributes = list(set(attributes))
    if not attributes:
        print("⚠️ No attributes provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(attributes, "attribute")
    # attributes are helper-only; no dedicated file persisted here
    return canonical_map

# ---- Pain Triggers ---------------------------------------------------------
def canonicalize_pain_trigger(pain_triggers: List[str]) -> Dict[str, str]:
    """
    Input triggers: ["attribute", ...]
    Returns map: attribute -> canonical_attribute
    """
    if not pain_triggers:
        print("⚠️ No pain triggers provided for canonicalization.")
        return {}
    canonical_map = canonicalize_attributes(list(set(pain_triggers)))
    save_canonical_map(canonical_map, PATHS["pain_trigger"])
    return canonical_map

# ---- Trigger Events (ZMOT) -------------------------------------------------
def canonicalize_trigger_events(trigger_events: List[str]) -> Dict[str, str]:
    trigger_events = list(set(trigger_events))
    if not trigger_events:
        print("⚠️ No trigger events provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(trigger_events, "trigger_event")
    save_canonical_map(canonical_map, PATHS["zmot_event"])
    return canonical_map

# ---- Observable Moments ----------------------------------------------------
def canonicalize_observable_moments(observable_moments: List[str]) -> Dict[str, str]:
    observable_moments = list(set(observable_moments))
    if not observable_moments:
        print("⚠️ No observable moments provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(observable_moments, "observable_moment")
    save_canonical_map(canonical_map, PATHS["observable_moment"])
    return canonical_map

# ---- Keywords --------------------------------------------------------------
def canonicalize_keywords(keywords: List[str]) -> Dict[str, str]:
    keywords = list(set(keywords))
    if not keywords:
        print("⚠️ No keywords provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(keywords, "keyword")
    save_canonical_map(canonical_map, PATHS["keyword"])
    return canonical_map

# ---- Personas --------------------------------------------------------------
def canonicalize_titles(titles: List[str]) -> Dict[str, str]:
    titles = list(set(titles))
    if not titles:
        print("⚠️ No titles provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(titles, "title")
    return canonical_map

def canonicalize_departments(departments: List[str]) -> Dict[str, str]:
    departments = list(set(departments))
    if not departments:
        print("⚠️ No departments provided for canonicalization.")
        return {}
    canonical_map = _cluster_and_label(departments, "department")
    return canonical_map

def canonicalize_persona(personas: List[dict]) -> Dict[str, dict]:
    """
    Personas: [{title, department, seniority}, ...]
    We canonicalize title/department by clustering; seniority is normalized.
    Returns: str(persona_raw_dict) -> {title, department, seniority}
    """
    if not personas:
        print("⚠️ No personas provided for canonicalization.")
        return {}

    title_cache = { (p.get("title") or "").strip().lower() for p in personas if p.get("title") }
    dept_cache = { (p.get("department") or "").strip().lower() for p in personas if p.get("department") }

    canonical_titles = canonicalize_titles(list(title_cache))
    canonical_depts = canonicalize_departments(list(dept_cache))

    SENIORITY_NORMALIZATION = {
        "intern": "Junior",
        "junior": "Junior",
        "associate": "Operator",
        "mid level": "Operator",
        "mid-level": "Operator",
        "mid-senior level": "Manager",
        "senior": "Senior",
        "senior level": "Senior",
        "lead": "Senior",
        "director": "Executive",
        "vp": "Executive",
        "c-level": "Executive",
        "c level": "Executive",
        "executive": "Executive",
        "manager": "Manager",
    }

    def normalize_seniority(s: str) -> str:
        if not s:
            return "Operator"
        return SENIORITY_NORMALIZATION.get(s.strip().lower(), "Operator")

    out: Dict[str, dict] = {}
    for orig in personas:
        title_raw = (orig.get("title") or "").strip()
        dept_raw = (orig.get("department") or "").strip()
        seniority_raw = (orig.get("seniority") or "").strip()

        canon = {
            "title": canonical_titles.get(title_raw.lower(), title_raw),
            "department": canonical_depts.get(dept_raw.lower(), dept_raw),
            "seniority": normalize_seniority(seniority_raw),
        }
        out[str(orig)] = canon

    save_canonical_map(out, PATHS["persona"])
    return out

# ---- Archetypes ------------------------------------------------------------
def canonicalize_archetypes(archetypes: List[dict]) -> Dict[str, dict]:
    """
    Archetype objects:
      {
        "industry": str,
        "revenue_range": str,
        "employee_range": str,
        "funding_stage": str,
        "geography": str
      }
    Canonicalization here is light-touch (lower/strip + consistent buckets).
    Returns: str(orig_dict) -> canonical_dict
    """
    if not archetypes:
        print("⚠️ No archetypes provided for canonicalization.")
        return {}

    # Light normalization tables (extend as needed)
    REVENUE_BUCKETS = {
        "0-1m": "0-1M", "1-10m": "1-10M", "10-100m": "10-100M", "100-500m": "100-500M",
        "500m-1b": "500M-1B", "1b+": "1B+",
    }
    EMP_BUCKETS = {
        "1-10": "1-10", "11-50": "11-50", "51-200": "51-200", "201-500": "201-500",
        "501-1000": "501-1000", "1000-5000": "1000-5000", "5000-10000": "5000-10000", "10000+": "10000+",
    }
    FUNDING = {
        "pre-seed": "Pre-Seed", "seed": "Seed", "series a": "Series A",
        "series b": "Series B", "series c+": "Series C+", "public": "Public"
    }
    GEO = {
        "north america": "North America", "europe": "Europe", "asia": "Asia",
        "south america": "South America", "africa": "Africa", "australia": "Australia",
    }

    def norm(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("$", "")
        s = s.replace("\u2013", "-")  # en dash to hyphen
        s = s.replace("–", "-")       # just in case
        s = s.lower()
        s = re.sub(r"\s+", "", s)     # remove all whitespace
        return s

    out: Dict[str, dict] = {}
    for a in archetypes:
        ind = norm(a.get("industry", ""))
        rev = REVENUE_BUCKETS.get(norm(a.get("revenue_range", "")).lower(), norm(a.get("revenue_range", "")))
        emp = EMP_BUCKETS.get(norm(a.get("employee_range", "")).lower(), norm(a.get("employee_range", "")))
        fund = FUNDING.get(norm(a.get("funding_stage", "")).lower(), norm(a.get("funding_stage", "")))
        geo = GEO.get(norm(a.get("geography", "")).lower(), norm(a.get("geography", "")))

        canon = {
            "industry": ind,
            "revenue_range": rev,
            "employee_range": emp,
            "funding_stage": fund,
            "geography": geo,
        }
        # Use a canonical string key for lookup
        key = json.dumps({
            "industry": a.get("industry", ""),
            "revenue_range": a.get("revenue_range", ""),
            "employee_range": a.get("employee_range", ""),
            "funding_stage": a.get("funding_stage", ""),
            "geography": a.get("geography", "")
        }, sort_keys=True)
        out[key] = canon

    save_canonical_map(out, PATHS["archetype"])
    return out


# ---- Local quick test ------------------------------------------------------
if __name__ == "__main__":
    pains = [
        "difficulty in managing subscriptions and ensuring no missed payments.",
        "manual management of subscriptions leading to missed payments.",
        "inflexible pricing structures that don't reflect usage.",
        "pricing structures that don't align with customer usage.",
    ]
    jobs = [
        "subscription management and billing",
        "pricing strategy and management",
        "customer retention and growth",
    ]
    personas = [
        {"title": "Billing Manager", "department": "Finance", "seniority": "Mid-Senior level"},
        {"title": "Pricing Analyst", "department": "Marketing", "seniority": "Mid level"},
        {"title": "Customer Success Manager", "department": "Customer Success", "seniority": "Senior level"},
    ]

    print("→ canonicalize_pain:", canonicalize_pain(pains))
    print("→ canonicalize_job:", canonicalize_job(jobs))
    print("→ canonicalize_persona:", canonicalize_persona(personas))