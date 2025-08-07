import networkx as nx


def get_zmot_root_job(product_subgraph: nx.DiGraph, zmot_node_id: str):
    """
    From a ZMOT node, return the job(s) that are triggered by it.
    Return the job with maximum temporal depth (most upstream cause).
    """
    job_candidates = [u for u, v, d in product_subgraph.in_edges(zmot_node_id, data=True) if d.get("type") == "triggered_by" and product_subgraph.nodes[u]["node_type"] == "job"]
    if not job_candidates:
        return None

    # Sort by depth (most negative is highest)
    job_candidates.sort(key=lambda n: product_subgraph.nodes[n].get("temporal_depth", 999))
    

    for job in job_candidates:
        print("Job:", job, "Depth:", product_subgraph.nodes[job].get("temporal_depth", "N/A"))
    return job_candidates[0] if job_candidates else None
