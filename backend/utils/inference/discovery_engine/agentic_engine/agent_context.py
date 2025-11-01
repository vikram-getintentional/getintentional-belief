
from typing import Any, Dict, List, Optional, Set
import networkx as nx
from backend.utils.graph_base.schema import RAW_FIELDS_BY_TYPE


class AgentContext:
    def __init__(
        self,
        client=None,
        builder=None,
        product_summary="",
        domain="",
        industry="",
        max_depth=3,
        attributes_dict: Optional[dict] = None,   # ← add
    ):
        # LLM + graph builder
        if client is None:
            # reuse your centralized client
            from backend.utils.inference.openai_client import client as default_client
            client = default_client
        self.client = client

        if builder is None:
            from backend.utils.graph_base.agent_graph_builder import CanonManager
            builder = CanonManager()
        self.builder = builder

        # product context (optional; engine can also read from graph)
        self.product_summary = product_summary
        self.domain = domain
        self.industry = industry

        # attribute enums
        if attributes_dict is None:
            try:
                from backend.utils.graph_base.icp_catalog import ICP_CATALOG  
            except Exception:
                ICP_CATALOG = {}
            attributes_dict = ICP_CATALOG
        self.attributes_dict = attributes_dict

        # loop control
        self.max_depth = max_depth
        self.current_depth = 0

        # relevance threshold for pain traversal
        self.relevance_threshold = 0.01

        # work queues
        self.hop_plus_cache: List[str] = []
        self.zmot_cache: List[str] = []
        self.pain_source_cache: List[str] = []

        # visited sets
        self._visited_pains: Set[str] = set()
        self._visited_triggers: Set[str] = set()

        # logs
        self.inference_log: dict[str, dict] = {}

    # helpers
    def mark_pain_visited(self, pain_id: str): self._visited_pains.add(pain_id)
    def is_pain_visited(self, pain_id: str) -> bool: return pain_id in self._visited_pains
    def mark_trigger_visited(self, trig_id: str): self._visited_triggers.add(trig_id)
    def is_trigger_visited(self, trig_id: str) -> bool: return trig_id in self._visited_triggers
    def log(self, node_id: str, source: str, data: dict): self.inference_log[node_id] = {"source": source, "data": data}
    def enqueue_hop_plus_pain(self, pain_id): self.hop_plus_cache.append(pain_id)
    def enqueue_trigger_for_zmot(self, trig_id): self.zmot_cache.append(trig_id)
    def enqueue_pain_for_source(self, pain_id): self.pain_source_cache.append(pain_id)
    def visited_pains(self): return self._visited_pains
    def visited_triggers(self): return self._visited_triggers

def _ntype(G: nx.DiGraph, nid: str) -> str:
    return (G.nodes[nid].get("type") or G.nodes[nid].get("node_type") or "").lower()

def minimal_source_fields(G: nx.DiGraph, nid: str) -> Dict[str, Any]:
    """Return only the fields the LLM needs for reasoning, plus id/type."""
    t = _ntype(G, nid)
    keep = RAW_FIELDS_BY_TYPE.get(t, [])
    n = G.nodes[nid]
    slim = {k: n.get(k) for k in keep if k in n}
    slim["_node_id"] = nid
    slim["_type"] = t
    return slim

def already_linked_fields(G: nx.DiGraph, nid: str, target_type: str) -> List[Dict[str, Any]]:
    """Return raw field snapshots of already-linked targets of the desired type."""
    out: List[Dict[str, Any]] = []
    for _, v in G.out_edges(nid):
        tv = _ntype(G, v)
        if tv == "metric":  # normalize if you have both "metric" and "perceived_metric"
            tv = "perceived_metric"
        if tv == target_type:
            keep = RAW_FIELDS_BY_TYPE.get(target_type, [])
            tgt = G.nodes[v]
            out.append({k: tgt.get(k) for k in keep if k in tgt})
    return out

def product_pack(product_summary: str | None, domain: str | None, industry: str | None) -> Dict[str, Any]:
    """Stable wrapper used by prompts—keeps your prompt schema consistent."""
    return {
        "product_summary": product_summary or "",
        "domain": domain or "",
        "industry": industry or "",
    }




        