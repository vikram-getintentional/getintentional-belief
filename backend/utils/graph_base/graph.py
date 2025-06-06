from typing import Dict, List, Set

class Graph:
    def __init__(
        self,
        node_registry=None,
        graph_edges=None,
        edge_weights=None
    ):
        self.pain_to_upstream_jobs: Dict[str, List[Dict]] = {}
        self.node_registry: Dict[tuple, str] = node_registry or {}
        self.edge_weights: Dict[tuple, float] = edge_weights or {}
        self.graph_edges: List[Dict] = graph_edges or []

    def get_node_id(self, node_type: str, value: str) -> int:
        """Return the node id for a given type and value, or None if not found."""
        return self.node_registry.get((node_type, value))
    
    def get_edge_weight(self, source_id: int, target_id: int) -> float:
        return self.edge_weights.get((source_id, target_id))

    def add_upstream_job(self, pain: str, job_info: Dict):
        if pain not in self.pain_to_upstream_jobs:
            self.pain_to_upstream_jobs[pain] = []
        self.pain_to_upstream_jobs[pain].append(job_info)

    def get_upstream_jobs(self, pain: str) -> List[Dict]:
        return self.pain_to_upstream_jobs.get(pain, [])
    
    def get_upstream_jobs_by_id(self, pain_id: int) -> List[Dict]:
        """
        Return all jobs where pain_id is the source and edge_type is 'addresses'.
        Output: List of dicts: {'job_id': ..., 'impact': ...}
        """
        results = []
        for edge in self.graph_edges:
            if (
                edge.get("source_id") == pain_id
                and edge.get("edge_type") == "addresses"
            ):
                results.append({
                    "job_id": edge.get("target_id"),
                    "impact": edge.get("weight", 1.0)
                })
        return results
    
# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.add_upstream_job("Inaccurate sales forecasts", {"job": "Oversee sales strategy"})
    print(g.get_upstream_jobs("Inaccurate sales forecasts"))