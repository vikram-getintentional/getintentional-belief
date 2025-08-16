
from typing import List, Set


class AgentContext:
    def __init__(self, client=None, builder=None, product_summary="", domain="", industry="",max_depth=3):
        # LLM + graph builder
        if client is None:
            # reuse your centralized client
            from backend.utils.inference.gpt_prompts.openai_client import client as default_client
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




        