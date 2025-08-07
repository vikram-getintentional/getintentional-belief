
import networkx as nx

from backend.utils.inference.visual_analysis_engine.rcs_helpers import get_zmot_root_job
from backend.utils.inference.visual_analysis_engine.rcs_utils import assign_temporal_depths


def generate_rcs_from_zmot(product_subgraph, archetype_id, zmot_node_id):
    """
    Full RCS generation pipeline, anchored on the given ZMOT node.
    """
    depth = assign_temporal_depths(product_subgraph)

    root_triplets = get_zmot_root_job(product_subgraph, zmot_node_id)
    if not root_triplets:
        return {}
    print("Root Job for ZMOT:", root_triplets)

    return []



"""
we now have a beautiful loop in our graph like: 
source - node type - target
product node - is_icp - archetype node
archetype node - responds_to - zmot trigger event

And then we have
job_id - triggered_by - zmot  node
zmot node - observed_in - observable moment node
zmot node - associated_with - keyword node

And we already have
product node - offered_by - capability node
capability node - solves - pain node
pain node - adresses - job node
job node - performed_by - pain node

job node - solves - pain node

pain node - scales_with - pain_trigger node
pain node - expressed_as - perceived_metric node

1.
The first thing we want to do is - surface the most likely ICP Archetype
def get_top_archetype() =>
We do this by getting the most relevant Archetype Node in the graph.

1.5
We need to make this work for any combo of archetype attributes -
For a given mix of archetype attributes - we should be able to first get the "best match archetypes" from our graph.

2. 
Then for this archetype - we want to get the best-fit ZMOT Events
def get_best_match_zmots(archetype)
We do this by getting the top 5 highest edge-weight zmot events for this archetype

3. 
Then for a given archetype and zmot event we want to get the best match jobs, pains, personas, pain triggers and perceived metrics
def get_zmot_summary(archetype, zmot event) =>
get the highest edge weight job node for this archetype --> zmot event
get all pains of this job
get all pain triggers & metrics for this pain
get all personas of this job
return this along with relevance score for each node


"""

from backend.utils.graph_base.network_graph import get_edge_weight, get_node_by_id, get_node_id, get_nodes_list_ids, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data






def walk_forward_to_hop0(product_subgraph, start_job_id, hop0_pain_ids):
    """
    Walk from zmot root job toward hop0 pain nodes.
    Record nodes as 'problem realization' until we hit hop0.
    """
    visited = set()
    queue = [start_job_id]
    problem_realization = []

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        if node_id in hop0_pain_ids:
            continue  # Stop at discovery point
        node = get_node_by_id(product_subgraph, node_id)
        label = get_node_label(product_subgraph, node_id)

        problem_realization.append({
            "id": node_id,
            "depth": node.get("temporal_depth")
        })

        for _, tgt, d in product_subgraph.out_edges(node_id, data=True):
            if d.get("type") in ["solves", "addresses", "performed_by"]:
                queue.append(tgt)

    return problem_realization


def find_hop0_discovery_nodes(product_subgraph):
    """
    Identify the hop0 pain (directly solved by product capability) and its job.
    """
    discovery = []
    hop0_pain_id_set = set()
    hop0_job_id_set = set()
    capability_ids = get_nodes_list_ids(product_subgraph, "capability", {})
    
    for capability_id in capability_ids:
        pain_ids = get_target_nodes_by_source_and_type(product_subgraph, capability_id, "solves")
        if not pain_ids:
            print(f"No pains found for capability {capability_id}")
            continue
        hop0_pain_id_set.update(pain_ids)
    
    for pain_id in hop0_pain_id_set:
        job_ids = get_target_nodes_by_source_and_type(product_subgraph, pain_id, "addresses")
        hop0_job_id_set.update(job_ids)
    
    for p in hop0_pain_id_set:
        pain_node = get_node_by_id(product_subgraph, p)
        if not pain_node:
            continue
        discovery.append({
			"id": pain_node["id"],
			"depth": pain_node.get("temporal_depth")
		})
    
    for j in hop0_job_id_set:
        job_node = get_node_by_id(product_subgraph, j)
        if not job_node:
            continue
        discovery.append({
			"id": job_node["id"],
			"depth": job_node.get("temporal_depth")
		})
    
    return discovery



def find_implementation_capabilities(product_subgraph, discovery_node_ids):
    """
    Find capabilities that solve discovery pains
    """
    implementation = []
    for node_id in discovery_node_ids:
        node = get_node_by_id(product_subgraph, node_id)
        if node.get("node_type") != "pain":
            continue
        for u, v, d in product_subgraph.in_edges(node_id, data=True):
            if d.get("type") == "solves" and product_subgraph.nodes[u]["node_type"] == "capability":
                implementation.append({
                    "id": u,
                    "depth": product_subgraph.nodes[u].get("temporal_depth")
                })
    return implementation


def find_promised_land(product_subgraph, hop0_job_ids):
    """
    Traverse downstream from hop0 jobs to find 'enabled' jobs in Hop++.
    """
    promised_land = []
    visited = set()
    queue = []
    print("hop0 job ids: ", hop0_job_ids)
    for hop0_job_id in hop0_job_ids:
        j = get_node_by_id(product_subgraph, hop0_job_id)
        if j["node_type"] == "job":
            queue.append(j["id"])

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        for _, tgt, d in product_subgraph.out_edges(node_id, data=True):
            if d.get("type") in ["solves", "addresses"] and product_subgraph.nodes[tgt]["node_type"] in ["job", "pain"]:
                promised_land.append({
                    "id": tgt,
                    "depth": product_subgraph.nodes[tgt].get("temporal_depth")
                })
                queue.append(tgt)

    return promised_land




def reverse_case_study_trascriber(product_subgraph, reverse_case_study):
    rcs_output = {}
    for key, item in reverse_case_study.items():
        rcs_output[key] = {
            "id": item["id"],
            "depth": product_subgraph.nodes[item["id"]].get("temporal_depth")
        }
    
    return rcs_output