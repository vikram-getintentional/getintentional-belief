from backend.utils.graph_base.network_graph import get_cumulative_relevance, get_edge_attribute, get_edge_weight, get_node_by_id, get_node_subgraph_to_product, get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.rcs_generators.rcs_generator_engine import build_persona_adjacency_from_subgraph, generate_rcs, minimal_coalition, persona_scores, recommend_coalitions
import networkx as nx
import numpy as np
import numpy as np
from backend.utils.inference.rcs_generators.beliefs.machine import ProbBeliefMachine
from backend.utils.inference.rcs_generators.beliefs.states import BeliefState as S

def simulate_rcs(product_subgraph, archetype_id, zmot_id = None):
    print(f"Simulating RCS for archetype ID: {archetype_id} with zmot_id: {zmot_id}")
    if zmot_id and zmot_id is not None:
        archetype_subgraph, belief_bundle = generate_rcs(
            product_subgraph,
            archetype_id=archetype_id,
            zmot_id=zmot_id
        ) 
        print("Subgraph built with archetype:", archetype_id, " and ZMOT: ", zmot_id)
    else:
        archetype_subgraph, belief_bundle = generate_rcs(
            product_subgraph,
            archetype_id=archetype_id
        )
        print(f"Archetype subgraph built for archetype ID: {archetype_id}")
    
    # ----------------------------
    # 1) Construct Archetype for Output
    # ----------------------------
    archetype_node = get_node_by_id(product_subgraph, archetype_id)
    if not archetype_node:
        print(f"Error: Archetype node with ID {archetype_id} not found in the product subgraph.")
    archetype = {
        "industry": archetype_node.get("industry", ""),
        "revenue_range": archetype_node.get("revenue_range", ""),
        "employee_range": archetype_node.get("employee_range", ""),
        "funding_stage": archetype_node.get("funding_stage", ""),
        "geography": archetype_node.get("geography", ""),    
    }

    timeline_out = _causal_timeline_builder(archetype_subgraph)

    
    

    
    graph_win = archetype_subgraph.graph["win_likelihood"]    
    """
    What we want to do:
    From the rendered graph, get the temporal depth of each pain node. 
    Then present this as: 
    Starting from depth T-max_depth to T-0: At each time step - what key events happened? 
    - What pains were experienced? What jobs? What personas?
    - What pain did this next cause?
    The output should have:
    - At each time step: 
        - Context (what was the input pain if any)
        - Action (what was job performed, by which persona)
        - Result (what pain was felt next)
    - Until we reach T-0 (capability & product nodes)



    
    """

    #scores = persona_scores(archetype_subgraph)
    #print("Persona scores :", scores)
    #coalition = minimal_coalition(scores, threshold = 2/3)
    #print("Coalition choices :", coalition)

    #reco_coalition = recommend_coalitions(archetype_subgraph)
    #print("Recommended coalition :", reco_coalition)

    
    


    output = {
        "graph_win_likelihood": graph_win,
        "timeline": timeline_out,
        
        "pain_trigger_candidates": [],
    }

    return output


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
            print(f"At time T-{d}, processing pain: {pain_id} - {pain_desc}")
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
                        "context": {"input_pain": pain_id, "pain_description": pain_desc},
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
                    print("Found job node:", node_id, " - ", node.get("description", ""))
                    job_desc = node.get("description", "")
                    persona_ids = get_target_nodes_by_source_and_type(G, node_id, "performed_by")
                    personas = []
                    for persona_id in persona_ids:
                        persona_title = G.nodes[persona_id].get("title", "")
                        persona_candidate_score = G.nodes[persona_id].get("candidate_score", 0.0)
                        personas.append({
                            "id": persona_id,
                            "title": persona_title,
                            "candidate_score": persona_candidate_score
                        })

                    next_pain_ids = get_source_nodes_by_target_and_type(G, node_id, "felt_in")
                    next_pains = []
                    for next_pain_id in next_pain_ids:
                        next_pain_desc = G.nodes[next_pain_id].get("description", "")
                        next_pains.append({
                            "id": next_pain_id,
                            "description": next_pain_desc,
                            "candidate_score": G.nodes[next_pain_id].get("candidate_score", 0.0)
                        })
                    step["events"].append({
                        "context": {"input_pain": pain_id, "pain_description": pain_desc},
                        "action": {
                            "job": node_id,
                            "job_description": job_desc,
                            "job_candidate_score": node_candidate_score,
                            "personas": personas
                        },
                        "results": next_pains   
                    })
        if step["events"]:  # Only append if there are events
            timeline.append(step)
    return timeline


        