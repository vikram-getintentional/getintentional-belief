from backend.utils.graph_base.network_graph import get_edge_weight, get_node_by_id, get_nodes_list_ids, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.inference.rcs_generators.rcs_generator_engine import build_archetype_subgraph_with_temporal_depth


def simulate_rcs(product_subgraph, archetype_id, zmot_id = None):
    print(f"Simulating RCS for archetype ID: {archetype_id} with zmot_id: {zmot_id}")
    if zmot_id and zmot_id is not None:
        archetype_subgraph = build_archetype_subgraph_with_temporal_depth(
            product_subgraph,
            archetype_id=archetype_id,
            zmot_id=zmot_id
        ) 
    else:
        archetype_subgraph = build_archetype_subgraph_with_temporal_depth(
            product_subgraph,
            archetype_id=archetype_id
        )
    print(f"Archetype subgraph built with temporal depth for archetype ID: {archetype_id}")
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
    # 2) Generate first response pains/ persona/ jobs / metrics
    # ----------------------------
    d = 0
    first_response_pain_family = []
    archetype_pain_trigger_ids = get_source_nodes_by_target_and_type(archetype_subgraph, archetype_id, "prevalent_in")
    for pain_trigger_id in archetype_pain_trigger_ids:
        pain_set = {}
        metric = {}
        pain_trigger_res = {}
        pain_trigger_node = get_node_by_id(archetype_subgraph, pain_trigger_id)
        pain_trigger_depth = pain_trigger_node.get("depth")
        if pain_trigger_depth == 1:
            pain_trigger_attribute = pain_trigger_node.get("attribute", "UNKNOWN")
            pain_trigger_likelihood = get_edge_weight(archetype_subgraph, pain_trigger_id, archetype_id)
            
            pain_ids = get_source_nodes_by_target_and_type(archetype_subgraph, pain_trigger_id, "triggered_by")
            pain_sets = []
            for pain_id in pain_ids:
                pain_node = get_node_by_id(archetype_subgraph, pain_id)
                pain_depth = pain_node.get("depth", 0)
                if pain_depth == 1:
                    pain_description = pain_node.get("description", "UNKNOWN")
                    pain_likelihood = get_edge_weight(archetype_subgraph, pain_id, pain_trigger_id) * pain_trigger_likelihood
                    perceived_metric_ids = get_target_nodes_by_source_and_type(archetype_subgraph, pain_id, "expressed_as")
                    metrics = []
                    for perceived_metric_id in perceived_metric_ids:
                        perceived_metric_node = get_node_by_id(archetype_subgraph, perceived_metric_id)
                        metric_depth = perceived_metric_node.get("depth")
                        print(f"Processing perceived metric ID: {perceived_metric_id}, depth: {metric_depth}")
                        if metric_depth == 1:
                            metric_description = perceived_metric_node.get("metric", "UNKNOWN")
                            metric_likelihood = get_edge_weight(archetype_subgraph, pain_id, perceived_metric_id) * pain_likelihood
                            metric = {
                                "metric": metric_description,
                                "likelihood": metric_likelihood,
                                "depth": metric_depth
                            }
                            metrics.append(metric)
                    solving_jobs = []
                    solving_job_ids = get_source_nodes_by_target_and_type(archetype_subgraph, pain_id, "solves")
                    for solving_job_id in solving_job_ids:
                        solving_job_node = get_node_by_id(archetype_subgraph, solving_job_id)
                        if solving_job_node.get("type") != "job":
                            print("Hit a capability node. Skipping")
                            continue

                        solving_job_depth = solving_job_node.get("depth")
                        if solving_job_depth == 2:
                            solving_job_description = solving_job_node.get("description", "UNKNOWN")
                            solving_job_likelihood = get_edge_weight(archetype_subgraph, solving_job_id, pain_id) * pain_likelihood
                            persona_job_ids = get_target_nodes_by_source_and_type(archetype_subgraph, solving_job_id, "performed_by")
                            persona_jobs = []
                            for persona_job_id in persona_job_ids:
                                persona_job_node = get_node_by_id(archetype_subgraph, persona_job_id)
                                persona_job_likelihood = get_edge_weight(archetype_subgraph, solving_job_id, persona_job_id) * solving_job_likelihood
                                persona_job_depth = persona_job_node.get("depth")
                                persona_job = {
                                    "persona_title": persona_job_node.get("title", "UNKNOWN"),
                                    "persona_department": persona_job_node.get("department", "UNKNOWN"),
                                    "persona_seniority": persona_job_node.get("seniority", "UNKNOWN"),
                                    "likelihood": persona_job_likelihood,
                                    "depth": persona_job_depth
                                }
                                persona_jobs.append(persona_job)
                            solving_job = {
                                "job": solving_job_description,
                                "likelihood": solving_job_likelihood,
                                "depth": solving_job_depth,
                                "persona_jobs": persona_jobs
                            }
                            solving_jobs.append(solving_job)
                    pain_set = {
                        "pain": pain_description,
                        "likelihood": pain_likelihood,
                        "metrics": metrics,
                        "solving_jobs": solving_jobs
                    }
                    pain_sets.append(pain_set)
            pain_trigger_res = {
                "pain_trigger": pain_trigger_attribute,
                "trigger_likelihood": pain_trigger_likelihood,
                "trigger_depth": pain_trigger_depth,
                "pains": pain_sets
            }
            first_response_pain_family.append(pain_trigger_res)
        

        

    final_rcs = {
        "archetype": archetype,
        "first_response_pain_family": first_response_pain_family,
    }

    print(f"Final RCS generated for archetype ID: {archetype_id}")
    print("Archetype:", final_rcs["archetype"])
    print("First Response Pain Family:", final_rcs["first_response_pain_family"])



    
    return final_rcs