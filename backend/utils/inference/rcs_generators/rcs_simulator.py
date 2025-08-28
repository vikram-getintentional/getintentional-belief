from backend.utils.graph_base.network_graph import get_cumulative_relevance, get_edge_attribute, get_edge_weight, get_node_by_id, get_node_subgraph_to_product, get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import get_cumulative_relevance_data
from backend.utils.inference.rcs_generators.rcs_generator_engine import build_archetype_subgraph_with_temporal_depth, build_persona_adjacency_from_subgraph, generate_rcs, minimal_coalition, persona_scores, recommend_coalitions
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
    archetype_node = get_node_by_id(archetype_subgraph, archetype_id)
    archetype = {
        "industry": archetype_node.get("industry", ""),
        "revenue_range": archetype_node.get("revenue_range", ""),
        "employee_range": archetype_node.get("employee_range", ""),
        "funding_stage": archetype_node.get("funding_stage", ""),
        "geography": archetype_node.get("geography", ""),    
    }

    # ----------------------------
    # 2) Get all pain trigger canidates with candidate_score for archetype with zmot boost 
    # ----------------------------
    d = 0
    first_response_pain_family = []
    archetype_pain_trigger_ids = get_source_nodes_by_target_and_type(archetype_subgraph, archetype_id, "prevalent_in")
    print("Archetype pain triggers found")
    #----------------------
    # Softmax logic for baseline likelihoods of archetype on pain triggers
    #----------------------
    pain_trigger_candidates = []
    for pain_trigger_id in archetype_pain_trigger_ids:    
        pain_trigger_node = get_node_by_id(archetype_subgraph, pain_trigger_id)
        product_node_id = get_product_id_from_subgraph(archetype_subgraph)
        zmot_boost = 0
        if zmot_id is not None:
            zmot_node = get_node_by_id(archetype_subgraph, zmot_id)
            if not zmot_node:
                print("ZMOT node not found in archetype subgraph, skipping")
            zmot_boost = get_edge_attribute(archetype_subgraph, pain_trigger_id, zmot_id, "boost")
            zmot_weight = get_edge_attribute(archetype_subgraph, pain_trigger_id, zmot_id, "weight")
            print("Zmot Boost: ", zmot_boost, " Zmot Weight: ", zmot_weight)
        print("Zmot boost for pain trigger:", pain_trigger_id, " = ", zmot_boost)
        pain_trigger_attribute = pain_trigger_node.get("attribute", "UNKNOWN")
        pain_trigger_depth = pain_trigger_node.get("depth", 0)
        # Log odds multiplier for pain trigger boost...
        baseline_pain_trigger_likelihood = get_edge_attribute(archetype_subgraph, pain_trigger_id, archetype_id, "likelihood")* pain_trigger_depth
        print("Pain trigger likelihood without boost:", baseline_pain_trigger_likelihood)
        pain_trigger_likelihood = (1+ 10*zmot_boost)* baseline_pain_trigger_likelihood / (((1+ 10*zmot_boost)* baseline_pain_trigger_likelihood)+(1- baseline_pain_trigger_likelihood))
        #pain_trigger_likelihood = 1-((1 - baseline_pain_trigger_likelihood) * (1 - zmot_boost))
        print("Pain trigger likelihood with boost:", pain_trigger_likelihood)
        pain_trigger_relevance = get_cumulative_relevance_data(product_node_id, pain_trigger_id)
        candidate_score = pain_trigger_likelihood * pain_trigger_relevance
        
        caused_pain_ids = get_source_nodes_by_target_and_type(archetype_subgraph, pain_trigger_id, "triggered_by")
        pains_out = []
        for caused_pain_id in caused_pain_ids:
            caused_pain_node = get_node_by_id(archetype_subgraph, caused_pain_id)
            if not caused_pain_node:
                print("Caused pain node not found in archetype subgraph, skipping")
                continue
            caused_pain_description = caused_pain_node.get("description", "")
            pain_likelihood = get_edge_attribute(archetype_subgraph, caused_pain_id, pain_trigger_id, "likelihood")*pain_trigger_likelihood
            pain_relevance = get_cumulative_relevance_data(product_node_id, caused_pain_id)
            candidate_score = pain_likelihood * pain_relevance

            metrics = []
            caused_pain_metrics = get_target_nodes_by_source_and_type(archetype_subgraph, caused_pain_id, "expressed_as")
            for metric_id in caused_pain_metrics:
                metric_node = get_node_by_id(archetype_subgraph, metric_id)
                if not metric_node:
                    print("Metric node not found in archetype subgraph, skipping")
                    continue
                metric_description = metric_node.get("metric", "")
                metric_likelihood = get_edge_attribute(archetype_subgraph, caused_pain_id, metric_id, "likelihood") * pain_likelihood
                metric_relevance = get_cumulative_relevance_data(product_node_id, metric_id)
                candidate_score = metric_likelihood * metric_relevance

                                                     
                metrics.append({
                    "metric_id": metric_id,
                    "description": metric_description,
                    "likelihood": metric_likelihood,
                    "relevance": metric_relevance,
                    "candidate_score": candidate_score
                })

            jobs_out = []
            solving_job_ids = get_source_nodes_by_target_and_type(archetype_subgraph, caused_pain_id, "solves")
            for job_id in solving_job_ids:
                job_node = get_node_by_id(archetype_subgraph, job_id)
                if not job_node:
                    print("Job node not found in archetype subgraph, skipping")
                    continue
                if job_node.get("type", "") != "job":
                    print("Hit a cap node, skipping")
                    continue
                job_description = job_node.get("description", "")
                job_likelihood = get_edge_attribute(archetype_subgraph, job_id, caused_pain_id, "likelihood")*pain_likelihood
                job_relevance = get_cumulative_relevance_data(product_node_id, job_id)
                candidate_score = job_likelihood * job_relevance
                persona_ids = get_target_nodes_by_source_and_type(archetype_subgraph, job_id, "performed_by")
                personas_out = []
                for persona_id in persona_ids:
                    persona_node = get_node_by_id(archetype_subgraph, persona_id)
                    if not persona_node:
                        print("Persona node not found in archetype subgraph, skipping")
                        continue
                    persona_title = persona_node.get("title", "")
                    persona_department = persona_node.get("department", "")
                    persona_seniority = persona_node.get("seniority", "")
                    persona_likelihood = get_edge_attribute(archetype_subgraph, job_id, persona_id, "likelihood")*job_likelihood
                    persona_relevance = get_cumulative_relevance_data(product_node_id, persona_id)
                    candidate_score = persona_likelihood * persona_relevance
                    personas_out.append({
                        "persona_id": persona_id,
                        "title": persona_title,
                        "department": persona_department,
                        "seniority": persona_seniority,
                        "likelihood": persona_likelihood,
                        "relevance": persona_relevance,
                        "candidate_score": candidate_score
                    })
                personas_out = sorted(personas_out, key=lambda x: x["likelihood"], reverse=True)
                jobs_out.append({
                    "job_id": job_id,
                    "description": job_description,
                    "likelihood": job_likelihood,
                    "relevance": job_relevance,
                    "personas": personas_out,
                    "candidate_score": candidate_score
                })
            jobs_out = sorted(jobs_out, key=lambda x: x["likelihood"], reverse=True)
            pains_out.append({
                "pain_id": caused_pain_id,
                "description": caused_pain_description,
                "likelihood": pain_likelihood,
                "relevance": pain_relevance,
                "pain_candidate_score": candidate_score,
                "metrics": metrics,
                "jobs": jobs_out,
                
            })
        pains_out = sorted(pains_out, key=lambda x: x["likelihood"], reverse=True)
        pain_triggers_out = {
            "pain_trigger_id": pain_trigger_id,
            "pain_trigger": pain_trigger_attribute,
            "likelihood": pain_trigger_likelihood,
            "relevance": pain_trigger_relevance,
            "caused_pains": pains_out,
            "candidate_score": candidate_score

        }

        pain_trigger_candidates.append(pain_triggers_out)
    graph_win = archetype_subgraph.graph["win_likelihood"]
    scores = persona_scores(archetype_subgraph)
    print("Persona scores :", scores)
    coalition = minimal_coalition(scores, threshold = 2/3)
    print("Coalition choices :", coalition)

    reco_coalition = recommend_coalitions(archetype_subgraph)
    print("Recommended coalition :", reco_coalition)

    
    


    # ----------------------------
    """
    for a subgraph (the archetype subgraph) - we want to build the potential rcs event chain and best committe candidates.

    What is a rcs event chain?
    
    then with each node data's updated belief we want to generate the updated causal chains and best committee candidates.
    for eg: each pain trigger candidate - we want to generate the 
    """


    output = {
        "graph_win_likelihood": graph_win,
        "pain_trigger_candidates": pain_trigger_candidates,
    }

    return output


        