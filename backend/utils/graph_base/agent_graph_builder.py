# backend/utils/graph_base/agent_graph_builder.py
# agent_graph_builder.py
# Stateful, batch-first graph builder. Collect → canonicalize-in-batch → write nodes/edges.

from __future__ import annotations
import json
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import time
from datetime import datetime

import networkx as nx
from networkx.exception import NetworkXError

# batch-first canonicalizer
from backend.utils.knowledge_base import canonicalizer as CAN

from backend.utils.knowledge_base.canonicalizer import (
    canonicalize_pain,
    canonicalize_job,
    canonicalize_persona,
    canonicalize_perceived_metric,
    canonicalize_pain_trigger,
    canonicalize_trigger_events,
    canonicalize_observable_moments,
    canonicalize_keywords,
    canonicalize_archetypes,
)
from backend.utils.graph_base.icp_catalog import ICP_CATALOG
from backend.utils.graph_base.persona_schema import (
    CanonicalPersonaAttributes,
    PersonaVariantAttributes,
    DEFAULT_CANONICAL_INFLUENCE_SCALARS,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def current_timestamp() -> str:
    """Return ISO8601 timestamp string."""
    return datetime.now().isoformat()

def _slug(s: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in (s or "")).split())

def _stable_key(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]

def _upsert_node(G: nx.DiGraph, ntype: str, key_parts: Iterable[str], attrs: Optional[Dict[str, Any]] = None, data_source="llm") -> str:
    node_key = _stable_key([ntype] + list(key_parts))
    node_id = f"{ntype}:{node_key}"
    if node_id not in G:
        # Set both type and node_type
        G.add_node(node_id, **{"type": ntype, "node_type": ntype, "id": node_id})
    if attrs:
        for k, v in attrs.items():
            if v is None:
                continue
            # avoid blowing away existing attrs; fill only if missing
            if k not in G.nodes[node_id] or G.nodes[node_id][k] in (None, "", []):
                G.nodes[node_id][k] = v
    # Ensure node_type is always present and correct
    G.nodes[node_id]["node_type"] = ntype
    G.nodes[node_id]["type"] = ntype
    G.nodes[node_id]["data_source"] = data_source
    G.nodes[node_id]["last_updated"] = current_timestamp()
    return node_id

def _would_create_cycle(G: nx.DiGraph, src: str, tgt: str) -> bool:
    """Adding src->tgt would create a directed cycle if tgt can already reach src (or src==tgt)."""
    if not src or not tgt:
        return False
    if src == tgt:
        return True
    try:
        return nx.has_path(G, tgt, src)
    except NetworkXError:
        # If either node missing, no path exists yet.
        return False

def _upsert_edge(
        G: nx.DiGraph, 
        src: str, 
        type: str, 
        tgt: str, 
        weight: Optional[float] = 1.0, 
        attrs: Optional[Dict[str, Any]] = None, 
        prevent_cycles: bool = True, 
        on_cycle: str = "skip", # "skip" | "raise" | "tag",
        data_source: str = "llm"
        ) -> None:
    # Guard against immediate cycles
    if prevent_cycles and _would_create_cycle(G, src, tgt):
        msg = f"[cycle-blocked] {src} - [{type}] -> {tgt}"
        if on_cycle == "raise":
            raise ValueError(msg)
        if on_cycle == "tag":
            if not G.has_edge(src,tgt):
                G.add_edge(src, tgt, type=f"{type}_blocked_cycle", blocked=True)
            edata = G[src][tgt]
            if isinstance(edata, dict) and "type" not in edata:
                # networkx may wrap attributes under an arbitrary key for multigraphs; normalize
                first_key = next(iter(edata.keys()))
                edata = edata[first_key]
            edata["blocked_reason"]="cycle"
            edata["requested_type"] = type
            return
        # Default: skip but print
        print(msg)
        return
        

    if not G.has_edge(src, tgt):
        G.add_edge(src, tgt, type=type)
    edata = G[src][tgt]
    if isinstance(edata, dict) and "type" not in edata:
        # networkx may wrap attributes under an arbitrary key for multigraphs; normalize
        first_key = next(iter(edata.keys()))
        edata = edata[first_key]
    edata["type"] = type
    edata["weight"] = float(weight)
    if attrs:
        for k, v in attrs.items():
            if v is None:
                continue
            edata[k] = v
    edata["data_source"] = data_source
    edata["last_updated"] = current_timestamp()

# Relevance Labels to Floats:
RELEVANCE_LABEL_MAP = {
    "Critical": 1.0,
    "Core": 0.8,
    "Supportive": 0.6,
    "Ancillary": 0.4,
    "Out-of-scope": 0.0,
}

LIKELIHOOD_LABEL_MAP = {
    "Essential": 0.95,
    "Expected": 0.75,
    "Common": 0.6,
    "Rare": 0.4,
    "Unlikely": 0.2,
}

BOOST_LABEL_MAP = {
    "Very High": 0.95,
    "High": 0.75,
    "Medium": 0.5,
    "Low": 0.3,
    "Negligible": 0.1,
} 

def label_to_float(label: str, label_type: str = "relevance") -> float:
    """
    Maps a relevance or likelihood label to a float value.
    label_type: "relevance" or "likelihood"
    """
    if label_type == "relevance":
        return RELEVANCE_LABEL_MAP.get(label, 0.0)
    elif label_type == "likelihood":
        return LIKELIHOOD_LABEL_MAP.get(label, 0.0)
    elif label_type == "boost":
        return BOOST_LABEL_MAP.get(label, 0.0)
    else:
        raise ValueError(f"Unknown label_type: {label_type}")

# canonical node constructors (use canonical strings/fields)


def _product(G, node_id: str, attrs: dict | None = None) -> str:
    """Upsert a product node using the *provided* node_id (canonical id)."""
    attrs = attrs or {}
    if node_id in G:
        G.nodes[node_id].update({"node_type": "product", "id": node_id, **attrs})
    else:
        G.add_node(node_id, node_type="product", id=node_id, **attrs)
    return node_id


def _capability(
    G,
    cap_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    attrs: Optional[Dict[str, Any]] = None
) -> str:
    key_parts = [cap_id] if cap_id else [name or "", description or ""]
    # Don't set "id" to cap_id (which may be None); let _upsert_node generate the node_id
    base_attrs = {"name": name, "description": description}
    if attrs:
        base_attrs.update(attrs)
    node_id = _upsert_node(G, "capability", key_parts, base_attrs)
    G.nodes[node_id]["id"] = node_id  # Always set the id attribute to the canonical node ID
    return node_id

def _pain(G, canonical_label: str, pain_source: Optional[str], data_source = "llm") -> str:
    return _upsert_node(G, "pain", [_slug(canonical_label)], {"description": canonical_label, "pain_source": pain_source, "data_source": data_source})

def _job(G, canonical_label: str, data_source = "llm") -> str:
    return _upsert_node(G, "job", [_slug(canonical_label)], {"description": canonical_label, "data_source": data_source})

def _persona(G, title: str, department: str, seniority: str, sample_profiles: List[str] = None, data_source = "llm") -> str:
    return _upsert_node(G, "persona", [_slug(title), _slug(department), (seniority or "").lower()],
                        {"title": title, "department": department, "seniority": seniority, "sample_profiles": sample_profiles, "data_source": data_source})

def _canonical_persona(
    G,
    payload: Dict[str, Any],
    data_source: str = "llm",
) -> str:
    attrs = CanonicalPersonaAttributes(
        canonical_persona_id=payload.get("canonical_persona_id") or payload.get("id") or "",
        label=payload.get("label") or payload.get("title") or "",
        description=payload.get("description") or payload.get("persona_description") or "",
        core_jobs=payload.get("core_jobs") or [],
        supporting_jobs=payload.get("supporting_jobs") or [],
        core_pains=payload.get("core_pains") or payload.get("pains") or [],
        example_titles=payload.get("example_titles") or payload.get("titles") or [],
        typical_departments=payload.get("typical_departments") or payload.get("departments") or [],
        typical_seniority_distribution=payload.get("typical_seniority_distribution") or {},
        default_influence_scalars=payload.get("default_influence_scalars") or payload.get("influence_scalars") or {},
        meta=payload.get("meta") or {},
    )
    node_attrs = attrs.to_node_attrs()
    if not node_attrs.get("title"):
        node_attrs["title"] = node_attrs.get("label")
    if not node_attrs.get("department"):
        departments = node_attrs.get("typical_departments") or []
        if departments:
            node_attrs["department"] = departments[0]
    if not node_attrs.get("seniority"):
        dist = node_attrs.get("typical_seniority_distribution") or {}
        if dist:
            node_attrs["seniority"] = max(dist.items(), key=lambda kv: kv[1])[0]
    node_attrs["meta"]["source"] = node_attrs["meta"].get("source") or data_source
    key_parts = [
        attrs.canonical_persona_id or _slug(node_attrs["label"]),
        node_attrs["label"].lower(),
    ]
    return _upsert_node(G, "canonical_persona", key_parts, node_attrs, data_source=data_source)

def _persona_variant(
    G,
    payload: Dict[str, Any],
    data_source: str = "llm",
) -> str:
    attrs = PersonaVariantAttributes(
        persona_variant_id=payload.get("persona_variant_id") or payload.get("id") or "",
        canonical_persona_id=payload.get("canonical_persona_id") or "",
        title=payload.get("title") or payload.get("label") or "",
        department=payload.get("department") or "",
        seniority=(payload.get("seniority") or "").lower(),
        team_context=payload.get("team_context") or payload.get("team") or "",
        influence_scalars=payload.get("influence_scalars"),
        crm_person_ids=payload.get("crm_person_ids") or payload.get("people") or [],
        meta=payload.get("meta") or {},
    )
    canonical_scalars = (
        payload.get("canonical_influence_scalars")
        or payload.get("default_influence_scalars")
        or DEFAULT_CANONICAL_INFLUENCE_SCALARS
    )
    node_attrs = attrs.to_node_attrs(canonical_influence=canonical_scalars)
    node_attrs["meta"]["source"] = node_attrs["meta"].get("source") or data_source
    key_parts = [
        node_attrs["persona_variant_id"],
        node_attrs["canonical_persona_id"],
        node_attrs.get("title") or "",
        node_attrs.get("seniority") or "",
    ]
    return _upsert_node(G, "persona_variant", key_parts, node_attrs, data_source=data_source)

def _metric(G, metric: str, data_source = "llm") -> str:
    return _upsert_node(G, "perceived_metric", [_slug(metric)], {"metric": metric, "data_source": data_source})

def _trigger(G, attribute: str, data_source = "llm") -> str:
    return _upsert_node(G, "pain_trigger", [_slug(attribute)], {"attribute": attribute, "data_source": data_source})

def _zmot(G, event: str, data_source = "llm") -> str:
    return _upsert_node(G, "zmot_event", [_slug(event)], {"event": event, "data_source": data_source})

def _observable(G, text: str, data_source = "llm") -> str:
    return _upsert_node(G, "observable_moment", [_slug(text)], {"text": text, "data_source": data_source})

def _keyword(G, text: str, data_source = "llm") -> str:
    return _upsert_node(G, "keyword", [_slug(text)], {"text": text, "data_source": data_source})

def _archetype(G, a: Dict[str, Any], data_source = "llm") -> str:
    return _upsert_node(G, "archetype",
                        [a.get("industry","").lower(), a.get("revenue_range","").lower(),
                         a.get("employee_range","").lower(), a.get("funding_stage","").lower(),
                         a.get("geography","").lower()],
                        {"industry": a.get("industry",""),
                         "revenue_range": a.get("revenue_range",""),
                         "employee_range": a.get("employee_range",""),
                         "funding_stage": a.get("funding_stage",""),
                         "geography": a.get("geography",""),
                         "data_source": data_source})

def _attribute_value(G, dimension: str, name: str, data_source = "llm") -> str:
    return _upsert_node(
        G, "attribute_value",
        [dimension.lower(), _slug(name)],
        {"dimension": dimension, "name": name, "data_source": data_source}
    )

def _stype(G: nx.DiGraph, node_id: str) -> str:
    nd = G.nodes.get(node_id, {}) or {}
    return nd.get("type") or nd.get("node_type") or ""

 #=== Frontier creation & relations (add-only) ===

# Create targets by type (ID stability remains via your builders)
CREATE_BY_TYPE = {
    "capability":       lambda G, r: _capability(G, canonical_label=(r.get("name") or r.get("description") or "")),
    "pain":             lambda G, r: _pain(G, canonical_label=(r.get("description") or ""), pain_source=r.get("pain_source")),
    "job":              lambda G, r: _job(G, canonical_label=(r.get("description") or "")),
    "persona":          lambda G, r: _persona(
        G,
        canonical_label=" • ".join([x for x in [r.get("title"), r.get("seniority"), r.get("department")] if x]),
        title=r.get("title"), department=r.get("department"), seniority=r.get("seniority"),
        linkedin_profiles=r.get("linkedin_profiles"),
    ),
    "canonical_persona": lambda G, r: _canonical_persona(G, r),
    "persona_variant":  lambda G, r: _persona_variant(G, r),
    "perceived_metric": lambda G, r: _metric(G, canonical_label=(r.get("metric") or "")),
    "pain_trigger":     lambda G, r: _trigger(G, canonical_label=(r.get("attribute") or "")),
    "attribute_value":  lambda G, r: _attribute_value(G, canonical_label="{}:{}".format((r.get("dimension") or "").lower(), r.get("name") or "")),
    "zmot_event":       lambda G, r: _zmot(G, canonical_label=(r.get("event") or "")),
    "observable_moment":lambda G, r: _observable(G, canonical_label=(r.get("text") or "")),
    "keyword":          lambda G, r: _keyword(G, canonical_label=(r.get("text") or "")),
}

# Edge relation map (source_type -> target_type)
RELATION_BY_PAIR = {
    ("capability","pain"): "solves",
    ("pain","job"): "felt_in",
    ("pain","perceived_metric"): "expressed_as",
    ("pain","pain_trigger"): "triggered_by",
    ("pain_trigger","attribute_value"): "prevalent_in",
    ("attribute_value","zmot_event"): "associated_zmot",
    ("pain_trigger","zmot_event"): "leads_to_zmot",
    ("zmot_event","observable_moment"): "observed_in",
    ("zmot_event","keyword"): "keyword",
    ("job","persona"): "performed_by",
    ("job","canonical_persona"): "performed_by",
    ("job","pain"): "solves",
    ("canonical_persona","persona_variant"): "has_variant",
    ("persona_variant","canonical_persona"): "variant_of",
}


# -----------------------------------------------------------------------------
# CanonManager — batch-first collector + writer
# -----------------------------------------------------------------------------

class CanonManager:
    """
    Collect RAW items → batch-canonicalize with canonicalizer.BatchCanonicalizer → write nodes/edges.

    Usage per stage:
      builder.ingest_hop0(...)
      builder.flush(G)     # canonicalize + write
    """

    def __init__(
        self,
        product_id: Optional[str] = None,
        data_source: str = "llm_frontier",
        G: Optional[nx.DiGraph] = None,
        **_ignored,   # tolerate future kwargs
    ):
        self.data_source = data_source
        self.product_id = product_id
        self.G = G

        self.buf: Dict[str, Any] = {
            "pain": set(),
            "job": set(),
            "persona": [],                # list of dicts
            "perceived_metric": set(),
            "pain_trigger": set(),
            "zmot_event": set(),
            "observable_moment": set(),
            "keyword": set(),
            "archetype": [],              # list of dicts
            "attribute_value": set(),
        }
        self.canon: Dict[str, Dict[Any, Dict[str, Any]]] = {k: {} for k in self.buf.keys()}
        self.plan: List[Tuple[str, Dict[str, Any]]] = []
        self._current_product_id: Optional[str] = None
        self._bc = CAN.BatchCanonicalizer()

    # keep your existing bind_graph, and add a set_graph for older callers
    def set_graph(self, G: nx.DiGraph):
        self.G = G
        return self
    @staticmethod
    def _init_canon_manager(G, product_id: Optional[str]) -> Any:
        """
        Create and bind a CanonManager regardless of signature differences across versions.
        Tries several ctor shapes and graph-binding styles.
        """
        # 1) construct
        cm = None
        ctor_errors = []

        for kwargs in (
            {"product_id": product_id, "data_source": "llm_frontier"},
            {"data_source": "llm_frontier"},
            {},  # bare
        ):
            try:
                cm = CanonManager(**kwargs)  # type: ignore
                break
            except TypeError as e:
                ctor_errors.append((kwargs, str(e)))
                cm = None

        if cm is None:
            # Last resort: raise the most informative error
            raise TypeError(f"CanonManager ctor mismatch. Tried: {ctor_errors}")

        # 2) bind graph
        if hasattr(cm, "bind_graph") and callable(getattr(cm, "bind_graph")):
            bound = cm.bind_graph(G)
            # some implementations return a new instance, others return None/self
            cm = bound or cm
        elif hasattr(cm, "set_graph") and callable(getattr(cm, "set_graph")):
            cm.set_graph(G)
        else:
            # best-effort field injection
            setattr(cm, "G", G)

        # 3) stash product_id if the instance uses it later (and ctor didn't take it)
        if product_id and not getattr(cm, "product_id", None):
            try:
                setattr(cm, "product_id", product_id)
            except Exception:
                pass

        return cm
    
    # ---------- bind a live NetworkX graph before ingesting ----------
    def bind_graph(self, G: nx.DiGraph):
        self.G = G
        return self
    
    # ---------- tiny helper for edge attrs ----------
    def _edge_attrs(self, relation: str | None, evidence: str | None) -> dict:
        attrs: dict = {}
        if relation:
            attrs["relation"] = relation
        if evidence:
            attrs["evidence"] = evidence
        # NOTE: if you want to add relevance/likelihood/weight/boost later from LLM,
        #       this is the one place to merge them.
        return attrs
    
        # ---------- edge scoring helpers (frontier) ----------
    def _edge_scores_from_raw(self, raw: Dict[str, Any], source_type: str, target_type: str) -> Dict[str, float]:
        """
        Pull labels from raw['edge_scores'] (preferred) or raw['*_label'] fallbacks,
        then map → floats via label_to_float(). Returns dict with possible keys:
        {'relevance','likelihood','boost'} (missing keys omitted).
        """
        es = (raw or {}).get("edge_scores") or {}

        # Prefer edge_scores; fallback to top-level labels for backward compatibility
        rel_label = es.get("relevance_label") or raw.get("relevance_label")
        lik_label = es.get("likelihood_label") or raw.get("likelihood_label")
        # Only meaningful for pain_trigger -> zmot_event
        boost_label = es.get("boost_label") or raw.get("boost_label")

        out: Dict[str, float] = {}
        if rel_label:
            out["relevance"] = label_to_float(rel_label, "relevance")
        if lik_label:
            out["likelihood"] = label_to_float(lik_label, "likelihood")

        if (source_type, target_type) == ("pain_trigger", "zmot_event") and boost_label:
            out["boost"] = label_to_float(boost_label, "boost")

        return out

    def _score_and_upsert_edge(
        self,
        G: nx.DiGraph,
        *,
        src: str,
        relation: str,
        tgt: str,
        source_type: str,
        target_type: str,
        raw: Dict[str, Any],
        evidence: Optional[str] = None,
    ) -> None:
        """
        Decide edge weight and attrs based on labels present on proposal.
        Default weight = relevance, EXCEPT for pain_trigger -> zmot_event, where
        weight = boost (if present) else relevance.
        """
        scores = self._edge_scores_from_raw(raw, source_type, target_type)
        # default weight
        weight = scores.get("relevance", 1.0)
        # special-case: pain_trigger -> zmot_event prefers boost
        if (source_type, target_type) == ("pain_trigger", "zmot_event"):
            weight = scores.get("boost", scores.get("relevance", 1.0))

        attrs = {"relation": relation}
        if evidence:
            attrs["evidence"] = evidence
        # attach numeric scores if present
        attrs.update({k: v for k, v in scores.items()})

        _upsert_edge(G, src, relation, tgt, weight=weight, attrs=attrs)

    
    # ---- capability -> pain (usually produced by frontier expansion from capability) ----
    def ingest_frontier_pain(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None, "CanonManager.bind_graph(G) must be called before ingest."
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "pain"), "solve")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                desc = (raw.get("description") or "").strip()
                pain_source = raw.get("pain_source")
                tgt = _pain(self.G, canonical_label=desc, pain_source=pain_source, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="pain", raw=raw, evidence=ev
                )

    # ---- pain -> job ----
    def ingest_frontier_job(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "job"), "felt_in")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                desc = (raw.get("description") or "").strip()
                tgt = _job(self.G, canonical_label=desc, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="job", raw=raw, evidence=ev
                )

    # ---- job -> persona ----
    def ingest_frontier_persona(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "persona"), "performed_by")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                title = (raw.get("title") or "").strip()
                department = raw.get("department")
                seniority = raw.get("seniority")
                tgt = _persona(self.G, title=title, department=department, seniority=seniority, data_source=self.data_source)

                # Optional: persist persona.linkedin_profiles to node (if present)
                profiles = raw.get("linkedin_profiles") or []
                if profiles:
                    self.G.nodes[tgt]["linkedin_profiles"] = profiles  # [{"url": "...", "bio": "..."}, ...]

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="persona", raw=raw, evidence=ev
                )

    # ---- job -> pain (solves) ----
    def ingest_frontier_solves_pain(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "pain"), "solves")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                desc = (raw.get("description") or "").strip()
                pain_source = raw.get("pain_source")
                tgt = _pain(self.G, canonical_label=desc, pain_source=pain_source, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="pain", raw=raw, evidence=ev
                )

    # ---- pain -> perceived_metric ----
    def ingest_frontier_perceived_metric(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "perceived_metric"), "expressed_as")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                metric = (raw.get("metric") or "").strip()
                tgt = _metric(self.G, metric=metric, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="perceived_metric", raw=raw, evidence=ev
                )

    # ---- pain -> pain_trigger ----
    def ingest_frontier_pain_trigger(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "pain_trigger"), "triggered_by")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                attribute = (raw.get("attribute") or "").strip()
                tgt = _trigger(self.G, attribute=attribute, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="pain_trigger", raw=raw, evidence=ev
                )

    # ---- pain_trigger -> attribute_value ----
    def ingest_frontier_attribute_value(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "attribute_value"), "prevalent_in")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                dim = (raw.get("dimension") or "").strip()
                name = (raw.get("name") or "").strip()
                tgt = _attribute_value(self.G, dimension=dim, name=name, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="attribute_value", raw=raw, evidence=ev
                )

    # ---- pain_trigger -> zmot_event ----
    def ingest_frontier_zmot_event(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "zmot_event"), "associated_zmot")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                event = (raw.get("event") or "").strip()
                tgt = _zmot(self.G, event=event, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="zmot_event", raw=raw, evidence=ev
                )

    # ---- zmot_event -> observable_moment ----
    def ingest_frontier_observable_moment(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "observable_moment"), "observed_in")  # <-- typo fixed below
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                text = (raw.get("text") or "").strip()
                tgt = _observable(self.G, text=text, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="observable_moment", raw=raw, evidence=ev
                )

    # ---- zmot_event -> keyword ----
    def ingest_frontier_keyword(self, batch: list[dict], *, product_id: str = "") -> None:
        assert self.G is not None
        for rec in batch:
            src = rec.get("source_id")
            if not src or src not in self.G:
                continue
            st = _stype(self.G, src)
            relation = rec.get("relation") or RELATION_BY_PAIR.get((st, "keyword"), "keyword")
            props = rec.get("proposals") or []
            evids = rec.get("evidence") or []

            for i, raw in enumerate(props):
                text = (raw.get("text") or "").strip()
                tgt = _keyword(self.G, text=text, data_source=self.data_source)

                ev = evids[i] if i < len(evids) else None
                self._score_and_upsert_edge(
                    self.G, src=src, relation=relation, tgt=tgt,
                    source_type=st, target_type="keyword", raw=raw, evidence=ev
                )

    # ----------------- Batch Non-frontier Methods below -----------------

    """def __init__(self):
        self.buf: Dict[str, set] = {
            "pain": set(),
            "job": set(),
            "persona": [],                # (title, dept, seniority)
            "perceived_metric": set(),       # (metric)
            "pain_trigger": set(),           # (attribute)
            "zmot_event": set(),
            "observable_moment": set(),
            "keyword": set(),
            "archetype": [],              # (industry, revenue_range, employee_range, funding_stage, geography)
            "attribute_value": set(),
        }
        self.canon: Dict[str, Dict[Any, Dict[str, Any]]] = {k: {} for k in self.buf.keys()}

        # write plan contains raw payloads grouped by op
        self.plan: List[Tuple[str, Dict[str, Any]]] = []
        self._current_product_id: Optional[str] = None

        # internal batch canonicalizer instance
        self._bc = CAN.BatchCanonicalizer()"""
        

    # --------- ingest (collect raw) ---------

    def ingest_hop0(self, hop0_json: List[Dict[str, Any]], capability_ids: List[str], product_id: str):
        print("Hop0 Ingestion started...")
        print("Hop0 Output:\n", json.dumps(hop0_json, indent=2))
        self._current_product_id = product_id
        for cap in hop0_json or []:
            for p in cap.get("pains", []) or []:
                self.buf["pain"].add(p.get("pain","").strip())
                for m in p.get("perceived_metrics", []) or []:
                    text = m.get("text","").strip() if isinstance(m, dict) else (m or "").strip()
                    self.buf["perceived_metric"].add(text)
                
                for j in p.get("felt_in_jobs", []) or []:
                    job_text = j.get("job_to_be_done","").strip()
                    self.buf["job"].add(job_text)
                    job_related_pains: List[str] = []
                    primary_pain = (p.get("pain") or "").strip()
                    if primary_pain:
                        job_related_pains.append(primary_pain)
                    for sp in j.get("solving_pains", []) or []:
                        sp_pain = (sp.get("pain") or "").strip()
                        if sp_pain:
                            job_related_pains.append(sp_pain)
                    job_weight = label_to_float(j.get("relevance_label", ""), "relevance") or 1.0
                    for pr in j.get("personas", []) or []:
                        self.buf["persona"].append({
                            "title": pr.get("title",""),
                            "department": pr.get("department",""),
                            "seniority": pr.get("seniority",""),
                            "job_to_be_done": job_text,
                            "job_weight": job_weight,
                            "pains": list(job_related_pains),
                            "team_context": pr.get("team") or pr.get("team_context"),
                            "source": "hop0",
                        })
                    for sp in j.get("solving_pains", []) or []:
                        self.buf["pain"].add(sp.get("pain","").strip())
                        for sm in sp.get("perceived_metrics", []) or []:
                            text = sm.get("text","").strip() if isinstance(sm, dict) else (sm or "").strip()
                            self.buf["perceived_metric"].add(text)
                        for st in sp.get("pain_triggers", []) or []:
                            text = st.get("text","").strip() if isinstance(st, dict) else (st or "").strip()
                            self.buf["pain_trigger"].add(text)
        self.plan.append(("hop0", {"hop0_json": hop0_json, "capability_ids": capability_ids, "product_id": product_id}))

    def ingest_hop_plus(self, gpt_outputs: List[Dict[str, Any]], product_id: str):
        print("Hop+ Ingestion started...")
        print("Hop+ GPT output:\n", json.dumps(gpt_outputs, indent=2))
        self._current_product_id = product_id
        for item in gpt_outputs or []:
            for sj in item.get("felt_in_jobs", []) or []:
                self.buf["job"].add(sj.get("job_to_be_done","").strip())
                job_related_pains: List[str] = []
                observed_pain = (item.get("pain") or "").strip()
                if observed_pain:
                    job_related_pains.append(observed_pain)
                for sp in sj.get("solving_pains", []) or []:
                    sp_pain = (sp.get("pain") or "").strip()
                    if sp_pain:
                        job_related_pains.append(sp_pain)
                job_weight = label_to_float(sj.get("relevance_label", ""), "relevance") or 1.0
                for pr in sj.get("personas", []) or []:
                    self.buf["persona"].append({
                            "title": pr.get("title",""),
                            "department": pr.get("department",""),
                            "seniority": pr.get("seniority",""),
                            "job_to_be_done": sj.get("job_to_be_done","").strip(),
                            "job_weight": job_weight,
                            "pains": list(job_related_pains),
                            "team_context": pr.get("team") or pr.get("team_context"),
                            "source": "hop_plus",
                        })
                for dp in sj.get("solving_pains", []) or []:
                    self.buf["pain"].add(dp.get("pain","").strip())
                    for m in dp.get("perceived_metrics", []) or []:
                        text = m.get("text","").strip() if isinstance(m, dict) else (m or "").strip()
                        if text:
                            self.buf["perceived_metric"].add(text)
                    for t in dp.get("pain_triggers", []) or []:
                        text = t.get("text","").strip() if isinstance(t, dict) else (t or "").strip()
                        if text:
                            self.buf["pain_trigger"].add(text)
        self.plan.append(("hop_plus", {"gpt_outputs": gpt_outputs, "product_id": product_id}))

    def ingest_trigger_attribute_matrix(self, results: Dict[str, Any], product_id: str):
        """
        results: {
        "pain_triggers": [
            {"pain_trigger_id":"...", "industry":[...], "revenue_range":[...], "employee_range":[...], "funding_stage":[...], "geography":[...] }
        ]
        }
        """
        print("Trigger×Attribute Ingestion started...")
        self._current_product_id = product_id
        self.plan.append(("trigger_attr", {"results": results, "product_id": product_id}))
        print("Trigger×Attribute Ingestion complete.")

    def ingest_zmot(self, results: Dict[str, Any], product_id: str):
        """
        Ingest ZMOT results (labels-only schema):
        {
        "zmot":[
            {
            "pain_trigger_id": "...",
            "events":[
                {
                "event_id": "string-slug",
                "trigger_event": "string",
                "observable_moments":[{"observable_moment":{"text":"...","relevance_label":"...","likelihood_label":"..."}}],
                "trigger_keywords":[{"keyword":{"text":"...","relevance_label":"...","likelihood_label":"..."}}],
                "boosts": {
                    "industry":[...],
                    "revenue_range":[...],
                    "employee_range":[...],
                    "funding_stage":[...],
                    "geography":[...]
                }
                }
            ]
            }
        ]
        }
        """
        print("ZMOT Ingestion started...")
        self._current_product_id = product_id

        rows = results.get("zmot") if isinstance(results, dict) else results
        rows = rows or []

        # Pre-buffer for canonicalization (used by _emit_zmot)
        for item in rows:
            for z in item.get("events", []) or []:
                ev = (z.get("trigger_event") or "").strip()
                if ev:
                    self.buf["zmot_event"].add(ev)

                for om in z.get("observable_moments", []) or []:
                    omt = om.get("observable_moment")
                    if isinstance(omt, dict):
                        text = (omt.get("text") or "").strip()
                        if text:
                            self.buf["observable_moment"].add(text)

                for kw in z.get("trigger_keywords", []) or []:
                    k = kw.get("keyword")
                    if isinstance(k, dict):
                        text = (k.get("text") or "").strip()
                        if text:
                            self.buf["keyword"].add(text)

        # Enqueue emit
        self.plan.append(("zmot", {"results": results, "product_id": product_id}))
        print("ZMOT Ingestion complete.")






    def ingest_archetypes(self, results: Dict[str, Any], product_id: str):
        print("Archetype Ingestion started...")
        self._current_product_id = product_id
        if not (isinstance(results, dict) and 
            "archetypes" in results and 
            "relevance_matrix" in results):
            print("⚠️ ingest_archetypes expected dict with 'archetypes' and 'relevance_matrix'. Skipping.")
            return
        if isinstance(results, dict) and "archetypes" in results and "relevance_matrix" in results:
            for a in results.get("archetypes", []) or []:
                self.buf["archetype"].append({
                    "industry": a.get("industry",""),
                    "revenue_range": a.get("revenue_range",""),
                    "employee_range": a.get("employee_range",""),
                    "funding_stage": a.get("funding_stage",""),
                    "geography": a.get("geography","")
                })
        self.plan.append(("archetype", {"results": results, "product_id": product_id}))
        print("Archetype Ingestion complete.")
        return


    # --------- flush (canonicalize batch + write) ---------

    def flush(self, G: nx.DiGraph) -> List[str]:
        """
        1) Batch-canonicalize everything buffered.
        2) Emit graph nodes/edges with canonical IDs.
        3) Clear buffers for the next stage.
        Returns a list of "touched" node ids (useful for tests).
        """
        touched: List[str] = []
        product_node_id = None
        if self._current_product_id:
            product_node_id = f"product:{_stable_key(['product', self._current_product_id])}"
 

        # 1) canonicalize batches
        self.canon = {}
        if "pain" in self.buf:
            self.canon["pain"] = canonicalize_pain(self.buf["pain"])
        if "job" in self.buf:
            self.canon["job"] = canonicalize_job(self.buf["job"])
        if "persona" in self.buf:
            self.canon["persona"] = canonicalize_persona(
                self.buf["persona"],
                job_map=self.canon.get("job"),
                pain_map=self.canon.get("pain"),
            )
        if "perceived_metric" in self.buf:
            self.canon["perceived_metric"] = canonicalize_perceived_metric(self.buf["perceived_metric"])
        if "pain_trigger" in self.buf:
            self.canon["pain_trigger"] = canonicalize_pain_trigger(self.buf["pain_trigger"])
        if "zmot_event" in self.buf:
            self.canon["zmot_event"] = canonicalize_trigger_events(self.buf["zmot_event"])
        if "observable_moment" in self.buf:
            self.canon["observable_moment"] = canonicalize_observable_moments(self.buf["observable_moment"])
        if "keyword" in self.buf:
            self.canon["keyword"] = canonicalize_keywords(self.buf["keyword"])
        if "archetype" in self.buf:
            self.canon["archetype"] = canonicalize_archetypes(self.buf["archetype"])

        self._materialize_canonical_personas(G)

        # 2) resolve plan and write
        for op, payload in self.plan:
            if op == "hop0":
                touched += self._emit_hop0(G, payload)
            elif op == "hop_plus":
                touched += self._emit_hop_plus(G, payload)
            elif op == "zmot":
                touched += self._emit_zmot(G, payload)
            elif op == "archetype":
                touched += self._emit_archetype(G, payload)
            elif op == "trigger_attr":                 # ← add
                touched += self._emit_trigger_attr(G, payload)
            elif op == "trigger_attr_scores":          # ← add
                touched += self._emit_trigger_attr_scores(G, payload)



        # 3) clear buffers and plan
        for k in self.buf:
            self.buf[k].clear()
        self.plan.clear()

        # Remove orphan product node if it was created but has no edges (no inbound/outbound)
        if product_node_id and product_node_id in G:
            # degree == 0 means truly dangling
            try:
                if G.degree(product_node_id) == 0:
                    print(f"[gc] removing orphan product node {product_node_id}")
                    G.remove_node(product_node_id)
            except Exception:
                # be defensive: ignore cleanup errors
                pass

        return list(dict.fromkeys(touched))  # unique

    # --------- internal emitters (canonicalized writes) ---------

    def _materialize_canonical_personas(self, G: nx.DiGraph) -> None:
        persona_result = self.canon.get("persona")
        if not getattr(persona_result, "canonical_personas", None):
            return

        canonical_nodes = self.canon.setdefault("canonical_persona_nodes", {})
        variant_nodes = self.canon.setdefault("persona_variant_nodes", {})

        for node in persona_result.canonical_personas:
            canonical_id = node.get("canonical_persona_id")
            if not canonical_id:
                continue
            node_id = _canonical_persona(G, node, data_source=self.data_source)
            canonical_nodes[canonical_id] = node_id

        for variant in persona_result.persona_variants:
            canonical_id = variant.get("canonical_persona_id")
            if not canonical_id:
                continue
            canonical_node_id = canonical_nodes.get(canonical_id)
            node_id = _persona_variant(
                G,
                variant,
                data_source=self.data_source,
            )
            variant_nodes[variant.get("persona_variant_id")] = node_id
            if canonical_node_id:
                _upsert_edge(G, canonical_node_id, "has_variant", node_id, weight=1.0)
                _upsert_edge(G, node_id, "variant_of", canonical_node_id, weight=1.0)

    def _resolve_persona_nodes(self, persona_key: Dict[str, Any]) -> tuple[Optional[str], Optional[str], Dict[str, Any]]:
        persona_result = self.canon.get("persona") or {}
        canonical_entry = persona_result.get(str(persona_key), {}) if isinstance(persona_result, dict) else {}

        canonical_persona_id = canonical_entry.get("canonical_persona_id")
        variant_id = canonical_entry.get("persona_variant_id")

        canonical_node_id = None
        variant_node_id = None
        if canonical_persona_id:
            canonical_node_id = (
                self.canon.get("canonical_persona_nodes", {}) or {}
            ).get(canonical_persona_id)
        if variant_id:
            variant_node_id = (
                self.canon.get("persona_variant_nodes", {}) or {}
            ).get(variant_id)
        return canonical_node_id, variant_node_id, canonical_entry

    def _emit_hop0(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("Hop0 Emission started...")
        hop0_json = payload["hop0_json"]
        print("Hop0 Emission Payload:\n", json.dumps(hop0_json, indent=2))
        capability_ids = payload["capability_ids"]
        product_id = payload["product_id"]

        touched: List[str] = []
        _product(G, product_id)
        print("Starting capability traversal")
        caps_list = []
        for cap_obj_ordered in hop0_json or []:
            cap_id = cap_obj_ordered.get("capability_id")
            if not cap_id:
                print("Capability with missing ID. Using blank instead.")
                cap_id = ""
            caps_list.append(cap_id)

        for cap_obj in hop0_json or []:
            cap_id = cap_obj.get("capability_id")
            if not cap_id:
                continue
            for p in cap_obj.get("pains", []) or []:
                print("Processing pains...")
                print("pain content:", p, "of type:", type(p))
                pain_raw = p.get("pain","").strip()
                pain_can = self.canon["pain"].get(pain_raw, {"canonical_label": pain_raw})
                if isinstance(pain_can, dict):
                    pain_id = _pain(G, pain_can.get("canonical_label", pain_raw), p.get("pain_source"))
                else:
                    pain_id = _pain(G, pain_can, p.get("pain_source"))
                touched.append(pain_id)
                relevance_label = p.get("relevance_label", "")
                likelihood_label = p.get("likelihood_label", "")
                relevance = label_to_float(relevance_label, "relevance")
                likelihood = label_to_float(likelihood_label, "likelihood")
                
                _upsert_edge(G, cap_id, "solves", pain_id, weight = relevance, attrs={"relevance": relevance, "likelihood": likelihood})       
                print("Capability-Pain Nodes and Edges Added")
                    

                for metric in p.get("perceived_metrics", []) or []:
                    print("Processing perceived metrics...")
                    text = metric.get("text", "").strip() if isinstance(metric, dict) else (metric or "").strip()
                    can_metric = self.canon["perceived_metric"].get(text, text)
                    m_id = _metric(G, can_metric)
                    relevance = label_to_float(metric.get("relevance_label", ""), "relevance") if isinstance(metric, dict) else 0.0
                    likelihood = label_to_float(metric.get("likelihood_label", ""), "likelihood") if isinstance(metric, dict) else 0.0
                    _upsert_edge(G, pain_id, "expressed_as", m_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                    touched.append(m_id)
                print("Pain-Metric Nodes and Edges Added")

                

                for j in p.get("felt_in_jobs", []) or []:
                    print("Processing felt_in_jobs...")
                    job_raw = j.get("job_to_be_done","").strip()
                    job_can = self.canon["job"].get(job_raw, {"canonical_label": job_raw})
                    if isinstance(job_can, dict):
                        job_id = _job(G, job_can.get("canonical_label"))
                    else:
                        job_id = _job(G, job_can)
                    relevance = label_to_float(j.get("relevance_label", ""), "relevance")
                    likelihood = label_to_float(j.get("likelihood_label", ""), "likelihood")
                    _upsert_edge(G, pain_id, "felt_in", job_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                    touched.append(job_id)
                    print("Pain-Job Nodes and Edges Added")
                    for pr in j.get("personas", []) or []:
                        key = {
                            "title": pr.get("title",""),
                            "department": pr.get("department",""),
                            "seniority": pr.get("seniority","")
                        }
                        canonical_node_id, variant_node_id, canonical_entry = self._resolve_persona_nodes(key)
                        target_node_id = canonical_node_id or variant_node_id
                        if not target_node_id:
                            pr_can = canonical_entry or key
                            target_node_id = _persona(G, pr_can.get("title",""), pr_can.get("department",""), pr_can.get("seniority",""))
                        relevance = label_to_float(pr.get("relevance_label", ""), "relevance") if isinstance(pr, dict) else 0.0
                        likelihood = label_to_float(pr.get("likelihood_label", ""), "likelihood") if isinstance(pr, dict) else 0.0
                        _upsert_edge(G, job_id, "performed_by", target_node_id, weight = relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                        touched.append(target_node_id)
                    print("Persona nodes added")
                    
                    for sp in j.get("solving_pains", []) or []:
                        print("Solving Pain nodes started")
                        pain_raw = sp.get("pain","").strip()
                        pain_can = self.canon["pain"].get(pain_raw, {"canonical_label": pain_raw})
                        if isinstance(pain_can, dict):
                            sp_id = _pain(G, pain_can.get("canonical_label"), sp.get("pain_source"))
                        else:
                            sp_id = _pain(G, pain_can, sp.get("pain_source"))
                        if sp_id == pain_id:
                            print("Solving pain is the same as original pain. Skipping.")
                            continue
                        relevance = label_to_float(sp.get("relevance_label", ""), "relevance")
                        likelihood = label_to_float(sp.get("likelihood_label", ""), "likelihood")
                        _upsert_edge(G, job_id, "solves", sp_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                        touched.append(sp_id)

                        for metric in sp.get("perceived_metrics", []) or []:
                            text = metric.get("text", "").strip() if isinstance(metric, dict) else (metric or "").strip()
                            can_metric = self.canon["perceived_metric"].get(text, text)
                            m_id = _metric(G, can_metric)
                            relevance = label_to_float(metric.get("relevance_label", ""), "relevance") if isinstance(metric, dict) else 0.0
                            likelihood = label_to_float(metric.get("likelihood_label", ""), "likelihood") if isinstance(metric, dict) else 0.0
                            _upsert_edge(G, sp_id, "expressed_as", m_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                            touched.append(m_id) 
                        print("Pain-Metric Nodes and Edges Added")

                        for attr in sp.get("pain_triggers", []) or []:
                            print("Processing pain triggers...")
                            text = attr.get("text", "").strip() if isinstance(attr, dict) else (attr or "").strip()
                            can_attr = self.canon["pain_trigger"].get(text, text)
                            t_id = _trigger(G, can_attr)
                            relevance = label_to_float(attr.get("relevance_label", ""), "relevance") if isinstance(attr, dict) else 0.0
                            likelihood = label_to_float(attr.get("likelihood_label", ""), "likelihood") if isinstance(attr, dict) else 0.0
                            _upsert_edge(G, sp_id, "triggered_by", t_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                            touched.append(t_id)
                        print("Pain-Trigger Nodes and Edges Added")
                        print("Solving Pain - Trigger Nodes added")
                    print("Solving pain nodes completed")
                print("Felt in jobs nodes completed")
            print("Emission for Hop0 completed")

        return touched

    def _emit_hop_plus(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("Hop Plus Emission started...")
        gpt_outputs = payload["gpt_outputs"]
        print("Hop+ Emission Payload:\n", json.dumps(gpt_outputs, indent=2))
        product_id = payload["product_id"]
        _product(G, product_id)
        touched: List[str] = []

        for item in gpt_outputs or []:
            orig_pain_id = item.get("original_pain_id")
            if not orig_pain_id:
                continue
            if orig_pain_id not in G:
                # create placeholder if missing
                print("Original pain ID not found. Skipping entry but something is wrong.")
                continue

            for sj in item.get("felt_in_jobs", []) or []:
                print("Processing felt_in_jobs...")
                job_raw = sj.get("job_to_be_done","").strip()
                job_can = self.canon["job"].get(job_raw, {"canonical_label": job_raw})
                if isinstance(job_can, dict):
                    job_id = _job(G, job_can.get("canonical_label"))
                else:
                    job_id = _job(G, job_can)
                relevance = label_to_float(sj.get("relevance_label", ""), "relevance")
                likelihood = label_to_float(sj.get("likelihood_label", ""), "likelihood")
                touched.append(job_id)
                _upsert_edge(G, orig_pain_id, "felt_in", job_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                print("Pain-Job Nodes and Edges Added")
                

                for pr in sj.get("personas", []) or []:
                    print("Processing personas...")
                    key = {
                        "title": pr.get("title",""),
                        "department": pr.get("department",""),
                        "seniority": pr.get("seniority","")
                    }
                    canonical_node_id, variant_node_id, canonical_entry = self._resolve_persona_nodes(key)
                    persona_id = canonical_node_id or variant_node_id
                    if not persona_id:
                        pr_can = canonical_entry or key
                        persona_id = _persona(G, pr_can.get("title",""), pr_can.get("department",""), pr_can.get("seniority",""))
                    relevance = label_to_float(pr.get("relevance_label", ""), "relevance") if isinstance(pr, dict) else 0.0
                    likelihood = label_to_float(pr.get("likelihood_label", ""), "likelihood") if isinstance(pr, dict) else 0.0
                    touched.append(persona_id)
                    _upsert_edge(G, job_id, "performed_by", persona_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                print("Persona nodes added")

                for dp in sj.get("solving_pains", []) or []:
                    print("Processing solving pains...")
                    pain_raw = dp.get("pain","").strip()
                    pain_can = self.canon["pain"].get(pain_raw, {"canonical_label": pain_raw})
                    pain_source = dp.get("pain_source")
                    if isinstance(pain_can, dict):
                        dp_id = _pain(G, pain_can.get("canonical_label"), pain_source)
                    else:
                        dp_id = _pain(G, pain_can, pain_source)
                    touched.append(dp_id)
                    if dp_id == orig_pain_id:
                        print("Solving pain is the same as original pain. Skipping.")
                        continue
                    relevance = label_to_float(dp.get("relevance_label", ""), "relevance")
                    likelihood = label_to_float(dp.get("likelihood_label", ""), "likelihood")
                    _upsert_edge(G, job_id, "solves", dp_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                    

                    for metric in dp.get("perceived_metrics", []) or []:
                        text = metric.get("text", "").strip() if isinstance(metric, dict) else (metric or "").strip()
                        can_metric = self.canon["perceived_metric"].get(text, text)
                        m_id = _metric(G, can_metric)
                        relevance = label_to_float(metric.get("relevance_label", ""), "relevance") if isinstance(metric, dict) else 0.0
                        likelihood = label_to_float(metric.get("likelihood_label", ""), "likelihood") if isinstance(metric, dict) else 0.0
                        _upsert_edge(G, dp_id, "expressed_as", m_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                    print("Pain-Metric Nodes and Edges Added")

                    for attr in dp.get("pain_triggers", []) or []:
                        print("Processing pain triggers...")
                        text = attr.get("text", "").strip() if isinstance(attr, dict) else (attr or "").strip()
                        can_attr = self.canon["pain_trigger"].get(text, text)
                        t_id = _trigger(G, can_attr)
                        relevance = label_to_float(attr.get("relevance_label", ""), "relevance") if isinstance(attr, dict) else 0.0
                        likelihood = label_to_float(attr.get("likelihood_label", ""), "likelihood") if isinstance(attr, dict) else 0.0
                        _upsert_edge(G, dp_id, "triggered_by", t_id, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                        touched.append(t_id)
                    print("Pain-Trigger Nodes and Edges Added")
        print("Hop+ Emission completed")        

        return touched

    def _emit_zmot(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("ZMOT Emission started...")
        results = payload["results"]
        product_id = payload["product_id"]
        _product(G, product_id)
        touched: List[str] = []

        # Normalize input
        rows = []
        if isinstance(results, dict):
            rows = results.get("zmot") or results.get("zmot_matrix") or []
        elif isinstance(results, list):
            rows = results
        rows = rows or []

        canon_z = self.canon.get("zmot_event", {}) or {}
        canon_om = self.canon.get("observable_moment", {}) or {}
        canon_kw = self.canon.get("keyword", {}) or {}

        for item in rows:
            trig_id = (item.get("pain_trigger_id") or "").strip()
            if not trig_id or trig_id not in G:
                print(f"⚠️ Missing/unknown pain_trigger_id '{trig_id}'; skipping row.")
                continue
            touched.append(trig_id)

            for z in item.get("events", []) or []:
                ev_raw = (z.get("trigger_event") or "").strip()
                if not ev_raw:
                    continue
                ev_can = canon_z.get(ev_raw, {"canonical_label": ev_raw})
                ev_label = ev_can.get("canonical_label", ev_raw) if isinstance(ev_can, dict) else ev_can
                zmot_id = _zmot(G, ev_label)
                touched.append(zmot_id)

                # Overall edge (trigger → event): weight = max per-attribute boost
                max_boost, max_rel, max_lik = 0.0, 0.0, 0.0
                boosts = z.get("boosts", {}) or {}
                for dim, arr in boosts.items():
                    for ent in arr or []:
                        b = label_to_float(ent.get("boost_label",""), "boost")
                        r = label_to_float(ent.get("relevance_label",""), "relevance")
                        l = label_to_float(ent.get("likelihood_label",""), "likelihood")
                        if b > max_boost: max_boost = b
                        if r > max_rel:   max_rel = r
                        if l > max_lik:   max_lik = l

                _upsert_edge(G, trig_id, "accelerated_by", zmot_id,
                            weight=max_boost, attrs={"relevance": max_rel, "likelihood": max_lik, "boost": max_boost})

                # Per-attribute boosts: ZMOT → AttributeValue
                for dim, arr in boosts.items():
                    for ent in arr or []:
                        name = (ent.get("name") or "").strip()
                        if not name:
                            continue
                        av_id = _attribute_value(G, dim, name)
                        b = label_to_float(ent.get("boost_label",""), "boost")
                        r = label_to_float(ent.get("relevance_label",""), "relevance")
                        l = label_to_float(ent.get("likelihood_label",""), "likelihood")
                        _upsert_edge(G, zmot_id, "boosted_in", av_id,
                                    weight=b, attrs={"dimension": dim, "boost": b, "relevance": r, "likelihood": l})
                        touched.append(av_id)

                # Observable moments
                for om in z.get("observable_moments", []) or []:
                    om_raw = om.get("observable_moment")
                    if not om_raw:
                        continue
                    om_text = om_raw.get("text", "").strip() if isinstance(om_raw, dict) else (om_raw or "").strip()
                    if not om_text:
                        continue
                    om_can = canon_om.get(om_text, {"canonical_label": om_text})
                    om_label = om_can.get("canonical_label", om_text) if isinstance(om_can, dict) else om_can
                    om_id = _observable(G, om_label)
                    rel = label_to_float(om_raw.get("relevance_label",""), "relevance") if isinstance(om_raw, dict) else 0.0
                    lik = label_to_float(om_raw.get("likelihood_label",""), "likelihood") if isinstance(om_raw, dict) else 0.0
                    _upsert_edge(G, zmot_id, "observed_in", om_id, weight=rel, attrs={"relevance": rel, "likelihood": lik})

                # Keywords
                for kw in z.get("trigger_keywords", []) or []:
                    kw_raw = kw.get("keyword")
                    if not kw_raw:
                        continue
                    kw_text = kw_raw.get("text", "").strip() if isinstance(kw_raw, dict) else (kw_raw or "").strip()
                    kw_can = canon_kw.get(kw_text, {"canonical_label": kw_text})
                    kw_label = kw_can.get("canonical_label", kw_text) if isinstance(kw_can, dict) else kw_can
                    kw_id = _keyword(G, kw_label)
                    rel = label_to_float(kw_raw.get("relevance_label",""), "relevance") if isinstance(kw_raw, dict) else 0.0
                    lik = label_to_float(kw_raw.get("likelihood_label",""), "likelihood") if isinstance(kw_raw, dict) else 0.0
                    _upsert_edge(G, zmot_id, "associated_with", kw_id, weight=rel, attrs={"relevance": rel, "likelihood": lik})
                    touched.append(kw_id)

        print("ZMOT Emission completed")
        return list(dict.fromkeys(touched))



    def _emit_archetype(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("Archetype Emission started...")
        data = payload["results"]
        product_id = payload["product_id"]
        _product(G, product_id)
        touched: List[str] = []
        gpt_to_archid: Dict[str, str] = {}

        for a in data.get("archetypes", []) or []:
            print("Processing archetypes...")
            gpt_id = a.get("archetype_id")
            if not gpt_id:
                print("⚠️ Missing archetype_id on item; keeping node but not mapping.")
            # Use a canonical key (stringified dict)
            key_dict = {
                "industry": a.get("industry",""),
                "revenue_range": a.get("revenue_range",""),
                "employee_range": a.get("employee_range",""),
                "funding_stage": a.get("funding_stage",""),
                "geography": a.get("geography","")
            }
            key_str = json.dumps(key_dict, sort_keys=True)
            a_can = self.canon.get("archetype", {}).get(key_str, key_dict)
            arch_id = _archetype(G, a_can)

            # store provenance & JSON fields for debugging/analytics
            if gpt_id:
                G.nodes[arch_id]["gpt_archetype_id"] = gpt_id
            G.nodes[arch_id]["industry"] = a_can["industry"]
            G.nodes[arch_id]["revenue_range"] = a_can["revenue_range"]
            G.nodes[arch_id]["employee_range"] = a_can["employee_range"]
            G.nodes[arch_id]["funding_stage"] = a_can["funding_stage"]
            G.nodes[arch_id]["geography"] = a_can["geography"]

            gpt_to_archid[gpt_id] = arch_id


        for row in data.get("relevance_matrix", []) or []:
            
            print("Processing relevance matrix...")
            arch_ref = row.get("archetype_ref")
            if not arch_ref:
                print("Relevance matrix entry without archetype_ref. Skipping.")
                continue
            arch_node = gpt_to_archid.get(arch_ref)
            if not arch_node:
                print(f"Archetype reference {arch_ref} not found in gpt_to_archid mapping. Skipping.")
                continue
            for ts in row.get("trigger_scores", []) or []:
                pain_trigger_id = ts.get("pain_trigger_id")
                if not pain_trigger_id:
                    print("Trigger score entry without pain_trigger_id. Skipping.")
                    continue
                if pain_trigger_id not in G:
                    print(f"Pain trigger ID {pain_trigger_id} not found in graph. Skipping.")
                    continue
                relevance = label_to_float(ts.get("relevance_label", ""), "relevance") if isinstance(ts, dict) else 0.0
                likelihood = label_to_float(ts.get("likelihood_label", ""), "likelihood") if isinstance(ts, dict) else 0.0
                _upsert_edge(G, pain_trigger_id, "prevalent_in", arch_node, weight=relevance, attrs={"relevance": relevance, "likelihood": likelihood})
                touched.append(pain_trigger_id)
                

        print("Archetype Emission completed")
        return touched

    def ingest_trigger_attribute_scores(self, results: Dict[str, Any], product_id: str):
        """
        Ingest labels-only attribute relevance for pain_triggers:
        {
        "pain_triggers":[
            {
            "pain_trigger_id":"...",
            "industry":[{"name":"...", "relevance_label":"...", "likelihood_label":"..."}],
            "revenue_range":[...],
            "employee_range":[...],
            "funding_stage":[...],
            "geography":[...]
            }
        ],
        "notes":[...]
        }
        """
        print("Trigger Attribute Scores Ingestion started...")
        self._current_product_id = product_id
        self.plan.append(("trigger_attr_scores", {"results": results, "product_id": product_id}))
        print("Trigger Attribute Scores Ingestion complete.")

    def _emit_trigger_attr(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("Trigger×Attribute Emission started...")
        data = payload["results"] or {}
        product_id = payload["product_id"]
        _product(G, product_id)
        touched: List[str] = []

        rows = data.get("pain_triggers") if isinstance(data, dict) else data
        rows = rows or []
        dims = ["industry","revenue_range","employee_range","funding_stage","geography"]

        for row in rows:
            trig_id = row.get("pain_trigger_id","")
            if not trig_id or trig_id not in G:
                print(f"⚠️ Unknown pain_trigger_id {trig_id}; skipping.")
                continue
            touched.append(trig_id)

            for dim in dims:
                for item in row.get(dim, []) or []:
                    name = (item.get("name") or "").strip()
                    if not name:
                        continue
                    av_id = _attribute_value(G, dim, name)
                    rel = label_to_float(item.get("relevance","") or item.get("relevance_label",""), "relevance")
                    lik = label_to_float(item.get("likelihood","") or item.get("likelihood_label",""), "likelihood")
                    _upsert_edge(G, trig_id, "prevalent_in", av_id,
                                weight=rel, attrs={"dimension": dim, "relevance": rel, "likelihood": lik})
                    touched.append(av_id)

        print("Trigger×Attribute Emission completed")
        return list(dict.fromkeys(touched))


    def _emit_trigger_attr_scores(self, G: nx.DiGraph, payload: Dict[str, Any]) -> List[str]:
        print("Trigger Attribute Scores Emission started...")
        results = payload["results"]
        product_id = payload["product_id"]
        _product(G, product_id)
        touched: List[str] = []

        rows = results.get("pain_triggers") if isinstance(results, dict) else results
        rows = rows or []

        for row in rows:
            tid = (row.get("pain_trigger_id") or "").strip()
            if not tid or tid not in G:
                print(f"⚠️ Unknown or missing pain_trigger_id '{tid}'. Skipping.")
                continue

            # Store raw labeled matrices on the trigger node for downstream consumers
            # Also store numeric conveniences for quick scoring.
            matrices = {}
            numeric = {}
            for dim in ["industry","revenue_range","employee_range","funding_stage","geography"]:
                vals = row.get(dim) or []
                matrices[dim] = vals
                numeric[dim] = [
                    {
                        "name": v.get("name",""),
                        "relevance": label_to_float(v.get("relevance_label",""), "relevance"),
                        "likelihood": label_to_float(v.get("likelihood_label",""), "likelihood"),
                    }
                    for v in vals if isinstance(vals, list)
                ]

            G.nodes[tid]["attribute_scores_labeled"] = json.dumps(matrices)
            G.nodes[tid]["attribute_scores_numeric"] = json.dumps(numeric)
            touched.append(tid)

        print("Trigger Attribute Scores Emission completed")
        return touched
