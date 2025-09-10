from backend.utils.graph_base.network_graph import get_cumulative_relevance, get_edge_attribute, get_edge_weight, get_node_by_id, get_node_subgraph_to_product, get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.rcs_generators.rcs_generator_engine import build_persona_adjacency_from_subgraph, generate_rcs
import networkx as nx
import numpy as np
from backend.utils.inference.rcs_generators.beliefs.machine import ProbBeliefMachine
from backend.utils.inference.rcs_generators.beliefs.states import BeliefState as S


def simulate_rcs(product_subgraph, archetype_id, zmot_id = None):
    print(f"Simulating RCS for archetype ID: {archetype_id} with zmot_id: {zmot_id}")
    archetype_node = get_node_by_id(product_subgraph, archetype_id)
    if not archetype_node:
        print(f"Working on base graph - without archetype node")
        archetype = {}
    else:
        archetype = {
            "industry": archetype_node.get("industry", ""),
            "revenue_range": archetype_node.get("revenue_range", ""),
            "employee_range": archetype_node.get("employee_range", ""),
            "funding_stage": archetype_node.get("funding_stage", ""),
            "geography": archetype_node.get("geography", ""),    
    }
    # ----------------------------
    # 1) Construct Archetype Graph
    # ----------------------------
    if zmot_id and zmot_id is not None:
        archetype_subgraph, report = generate_rcs(
            product_subgraph,
            archetype_id=archetype_id,
            zmot_id=zmot_id
        ) 
        print("Subgraph built with archetype:", archetype_id, " and ZMOT: ", zmot_id)
    else:
        archetype_subgraph, report = generate_rcs(
            product_subgraph,
            archetype_id=archetype_id
        )
        print(f"Archetype subgraph built for archetype ID: {archetype_id}")
    
     # ----------------------------
    # 2) Setup for Visualization
    # ----------------------------
    archetype_subgraph = _set_node_labels_positions(archetype_subgraph)
    print("Node labels set for visualization.")
    
    out_graph = export_graph_for_d3(archetype_subgraph)
    
    # ----------------------------
    # 3) Model Outputs
    # ----------------------------
    graph_win = archetype_subgraph.graph["win_likelihood"]
    for item in report["reverse_case_study"]["overview"]:
        id = item.get("id")
        node = get_node_by_id(archetype_subgraph, id)
        if node:
            item["label"] = node.get("label", "")
    for item in report["reverse_case_study"]["personas"]:
        id = item.get("id")
        node = get_node_by_id(archetype_subgraph, id)
        if node:
            item["label"] = node.get("label", "")
    for item in report["graph_concerns_summary"]:
        id = item.get("persona")
        node = get_node_by_id(archetype_subgraph, id)
        if node:
            persona_text = node.get("label", "")
        item["persona"] = persona_text

    report["graph_concerns_summary"] = convert_descriptions_to_labels(archetype_subgraph, report["graph_concerns_summary"])    

    



    output = {
        "graph_win_likelihood": graph_win,
        "top_N_personas": report["reverse_case_study"]["personas"][:5],
        "prioritized_10_concerns": report["graph_concerns_summary"][:10],
        "playbook": report["marketing_execution_plan"]

    }

    final_out = {
        "output": output,
        "graph": out_graph
    }

    return final_out


def _causal_timeline_builder(G: nx.DiGraph):
    timeline = []
    max_depth = G.graph.get("max_temporal_depth", 1)
    print("total jobs in subgraph:", len(get_nodes_list_ids(G, "job",{})))
    print("Starting timeline builder with max_depth:", max_depth)
    for d in range(int(max_depth), -1, -1):
        pains = [n for n, data in G.nodes(data=True)
                 if data.get("node_type") == "pain" and int(data.get("temporal_depth", 0)) == d]
        step = {"time_step": f"T-{d}", "events": []}
        for pain_id in pains:
            pain_desc = G.nodes[pain_id].get("description", "")
            solving_node_ids = get_source_nodes_by_target_and_type(G, pain_id, "solves")
            for node_id in solving_node_ids:
                node = get_node_by_id(G, node_id)
                if not node:
                    print("Node not found for ID:", node_id)
                    continue
                node_candidate_score = node.get("candidate_score", 0.0)
                
                if node.get("node_type") == "capability":
                    
                    capability_id = node_id
                    capability_desc = node.get("description", "")
                    capability_name = node.get("name", "")
                    step["events"].append({
                        "context": {"input_pain": pain_id, "pain_description": pain_desc, "likelihood": G.nodes[pain_id].get("cumulative_likelihood", 0.0), "relevance": G.nodes[pain_id].get("cumulative_relevance", 0.0)},
                        "action": {
                            "capability": capability_id,
                            "capability_name": capability_name,
                            "capability_description": capability_desc,
                            "candidate_score": node_candidate_score
                        },
                        "result": {}
                    })
                    
                    continue
                elif node.get("node_type") == "job":
                    
                    job_desc = node.get("description", "")
                    persona_ids = get_target_nodes_by_source_and_type(G, node_id, "performed_by")
                    personas = []
                    for persona_id in persona_ids:
                        persona_title = G.nodes[persona_id].get("title", "")
                        persona_candidate_score = G.nodes[persona_id].get("candidate_score", 0.0)
                        personas.append({
                            "id": persona_id,
                            "title": persona_title,
                            "candidate_score": persona_candidate_score,
                            "likelihood": G.nodes[persona_id].get("cumulative_likelihood", 0.0),
                            "relevance": G.nodes[persona_id].get("cumulative_relevance", 0.0)
                        })

                    next_pain_ids = get_source_nodes_by_target_and_type(G, node_id, "felt_in")
                    next_pains = []
                    for next_pain_id in next_pain_ids:
                        next_pain_desc = G.nodes[next_pain_id].get("description", "")
                        next_pains.append({
                            "id": next_pain_id,
                            "description": next_pain_desc,
                            "candidate_score": G.nodes[next_pain_id].get("candidate_score", 0.0),
                            "likelihood": G.nodes[next_pain_id].get("cumulative_likelihood", 0.0),
                            "relevance": G.nodes[next_pain_id].get("cumulative_relevance", 0.0)
                        })
                    step["events"].append({
                        "context": {"input_pain": pain_id, "pain_description": pain_desc, "likelihood": G.nodes[pain_id].get("cumulative_likelihood", 0.0), "relevance": G.nodes[pain_id].get("cumulative_relevance", 0.0)},
                        "action": {
                            "job": node_id,
                            "job_description": job_desc,
                            "job_candidate_score": node_candidate_score,
                            "likelihood": node.get("cumulative_likelihood", 0.0),
                            "relevance": node.get("cumulative_relevance", 0.0),
                            "personas": personas
                        },
                        "results": next_pains   
                    })
        if step["events"]:  # Only append if there are events
            timeline.append(step)
    return timeline





def export_graph_for_d3(G: nx.DiGraph):
    nodes = []
    for node_id, data in G.nodes(data=True):
        node = {"id": node_id}
        node.update(data)
        nodes.append(node)
    links = []
    for source, target, data in G.edges(data=True):
        link = {"source": source, "target": target}
        link.update(data)
        links.append(link)
    return {"nodes": nodes, "links": links}

def _set_node_labels_positions(G: nx.DiGraph):
    print("Setting node labels for visualization...")
    for node_id in G.nodes:
        node = get_node_by_id(G, node_id)
        node_type = node.get("node_type", "unknown")

        if node_type == "persona":
            title = node.get("title", "")
            department = node.get("department", "")
            seniority = node.get("seniority", "")
            node["label"] = f"{title} | {department} | {seniority}"
        elif node_type == "job":
            description = node.get("description", "")
            node["label"] = description if description else f'Job {node_id}'
        elif node_type == "pain":
            description = node.get("description", "")
            node["label"] = description if description else f'Pain {node_id}'
        elif node_type == "capability":
            cap_name = node.get("name", "")
            node["label"] = cap_name if cap_name else f'Capability {node_id}'
        elif node_type == "product":
            url = node.get("url", "")
            node["label"] = url if url else f'Product {node_id}'
        elif node_type == "pain_trigger":
            description = node.get("attribute", "")
            node["label"] = description if description else f'Pain Trigger {node_id}'
        elif node_type == "zmot_event":
            description = node.get("description", "")
            node["label"] = description if description else f'ZMOT Event {node_id}'
        elif node_type == "observable_moment":
            description = node.get("description", "")
            node["label"] = description if description else f'Observable Moment {node_id}'
        elif node_type == "keyword":
            keyword = node.get("keyword", "")
            node["label"] = keyword if keyword else f'Keyword {node_id}'
        elif node_type == "archetype":
            title = node.get("title", "")
            industry = node.get("industry", "")
            employee_range = node.get("employee_range", "")
            revenue_range = node.get("revenue_range", "")
            funding_stage = node.get("funding_stage", "")
            geography = node.get("geography", "")
            node["label"] = f"{title} | {industry} | {employee_range} | {revenue_range} | {funding_stage} | {geography}"
        elif node_type == "perceived_metric":
            description = node.get("metric", "")
            node["label"] = description if description else f'Perceived Metric {node_id}'
        else:
            description = node.get("description", "")
            node["label"] = description if description else f'Node {node_id}'
    return G

def id_list_to_labels(G, id_list):
    return [G.nodes[n].get("label", n) for n in id_list if n in G.nodes]

def convert_descriptions_to_labels(G, concerns):
    for c in concerns:
        desc = c.get("description", {})
        # Problem stage
        if "upstream_pains" in desc:
            desc["upstream_pains"] = id_list_to_labels(G, desc["upstream_pains"])
        if "pain_metrics" in desc:
            desc["pain_metrics"] = id_list_to_labels(G, desc["pain_metrics"])
        if "jobs" in desc:
            desc["jobs"] = id_list_to_labels(G, desc["jobs"])
        # Pain stage
        if "downstream_pains" in desc:
            desc["downstream_pains"] = id_list_to_labels(G, desc["downstream_pains"])
        # Solution stage
        if "solving_jobs" in desc:
            desc["solving_jobs"] = id_list_to_labels(G, desc["solving_jobs"])
        if "capabilities" in desc:
            desc["capabilities"] = id_list_to_labels(G, desc["capabilities"])
    return concerns