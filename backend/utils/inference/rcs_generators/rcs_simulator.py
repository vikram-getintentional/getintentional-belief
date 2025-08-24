from backend.utils.graph_base.network_graph import get_edge_attribute, get_edge_weight, get_node_by_id, get_nodes_list_ids, get_product_id_from_subgraph, get_source_nodes_by_target_and_type, get_target_nodes_by_source_and_type
from backend.utils.inference.rcs_generators.rcs_generator_engine import build_archetype_subgraph_with_temporal_depth, build_persona_adjacency_from_subgraph
import networkx as nx
import numpy as np
import numpy as np
from backend.utils.inference.rcs_generators.beliefs.machine import ProbBeliefMachine
from backend.utils.inference.rcs_generators.beliefs.states import BeliefState as S

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
    # 0) Add belief bundles
    # ----------------------------
    belief_bundle = archetype_subgraph.graph.get("belief_init", {})
    A_persona, persona_order = build_persona_adjacency_from_subgraph(archetype_subgraph)
    n = A_persona.shape[0]

    belief_summary = {
        "personas": persona_order,
        "activation_SOL": [],
        "activation_PR": [],
        "expected_score_SOL": 0.0,
        "expected_score_PR": 0.0,
        "marginal_lift_example": 0.0,
    }

    if n > 0:
        r = np.ones(n, dtype=float)  # uniform committee weight for now
        beta = float(belief_bundle.get("beta", 0.35))
        bm = ProbBeliefMachine(A=A_persona, r=r, beta=beta)

        # if the bundle has a π that matches, restore it (optional but nice)
        pi0 = belief_bundle.get("pi")
        if isinstance(pi0, list):
            pi0 = np.array(pi0, dtype=float)
            if pi0.shape == bm.pi.shape:
                bm.pi = pi0

        # baseline scores / activations
        belief_summary.update({
            "activation_SOL": bm.activation("SOL").tolist(),
            "activation_PR":  bm.activation("PR").tolist(),
            "expected_score_SOL": bm.expected_score("SOL"),
            "expected_score_PR":  bm.expected_score("PR"),
        })

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

    #----------------------
    # Softmax logic for baseline likelihoods of archetype on pain triggers
    #----------------------
    
    for pain_trigger_id in archetype_pain_trigger_ids:    
        pain_set = {}
        metric = {}
        pain_trigger_res = {}
        zmot_boost = 0
        pain_trigger_node = get_node_by_id(archetype_subgraph, pain_trigger_id)
        pain_trigger_depth = pain_trigger_node.get("depth")
        if pain_trigger_depth == 1:
            if zmot_id:
                print("Archetype ID:", archetype_id)
                print("ZMOT ID provided:", zmot_id)
                zmot_node = get_node_by_id(archetype_subgraph, zmot_id)
                if not zmot_node:
                    print("ZMOT node not found in archetype subgraph, skipping")
                zmot_boost = get_edge_attribute(archetype_subgraph, pain_trigger_id, zmot_id, "boost")
            print("Zmot boost for pain trigger:", pain_trigger_id, " = ", zmot_boost)
            pain_trigger_attribute = pain_trigger_node.get("attribute", "UNKNOWN")
            # Log odds multiplier for pain trigger boost...
            baseline_pain_trigger_likelihood = get_edge_weight(archetype_subgraph, pain_trigger_id, archetype_id)
            print("Pain trigger likelihood without boost:", baseline_pain_trigger_likelihood)
            pain_trigger_likelihood = (1+ 10*zmot_boost)* baseline_pain_trigger_likelihood / (((1+ 10*zmot_boost)* baseline_pain_trigger_likelihood)+(1- baseline_pain_trigger_likelihood))
            #pain_trigger_likelihood = 1-((1 - baseline_pain_trigger_likelihood) * (1 - zmot_boost))
            print("Pain trigger likelihood with boost:", pain_trigger_likelihood)

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
                        if metric_depth == 1:
                            metric_description = perceived_metric_node.get("metric", "UNKNOWN")
                            metric_likelihood = get_edge_weight(archetype_subgraph, pain_id, perceived_metric_id) * pain_likelihood
                            metric = {
                                "metric_id": perceived_metric_id,
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
                                    "persona_id": persona_job_id,
                                    "persona_title": persona_job_node.get("title", "UNKNOWN"),
                                    "persona_department": persona_job_node.get("department", "UNKNOWN"),
                                    "persona_seniority": persona_job_node.get("seniority", "UNKNOWN"),
                                    "likelihood": persona_job_likelihood,
                                    "depth": persona_job_depth
                                }
                                persona_jobs.append(persona_job)
                            solving_job = {
                                "job_id": solving_job_id,
                                "job": solving_job_description,
                                "likelihood": solving_job_likelihood,
                                "depth": solving_job_depth,
                                "persona_jobs": persona_jobs
                            }
                            solving_jobs.append(solving_job)
                    pain_set = {
                        "pain_id": pain_id,
                        "pain": pain_description,
                        "likelihood": pain_likelihood,
                        "metrics": metrics,
                        "solving_jobs": solving_jobs
                    }
                    pain_sets.append(pain_set)
            pain_trigger_res = {
                "pain_trigger_id": pain_trigger_id,
                "pain_trigger": pain_trigger_attribute,
                "trigger_likelihood": pain_trigger_likelihood,
                "trigger_depth": pain_trigger_depth,
                "pains": pain_sets
            }
            first_response_pain_family.append(pain_trigger_res)
    
        
        
    most_likely_pain_trigger = max(first_response_pain_family, key=lambda x: x["trigger_likelihood"], default=None)
    most_likely_pain = max(most_likely_pain_trigger.get("pains", []), key=lambda x: x["likelihood"], default=None) if most_likely_pain_trigger else None
    most_likely_job = max(most_likely_pain.get("solving_jobs", []), key=lambda x: x["likelihood"], default=None) if most_likely_pain else None
    most_likely_persona = max(most_likely_job.get("persona_jobs", []), key=lambda x: x["likelihood"], default=None) if most_likely_job else None

    most_likely_first_responder_path_ids = {
        "pain_trigger_id": most_likely_pain_trigger["pain_trigger_id"],
        "trigger_likelihood": most_likely_pain_trigger["trigger_likelihood"],
        "trigger_depth": most_likely_pain_trigger["trigger_depth"],
        "pain_id": most_likely_pain["pain_id"],
        "pain_likelihood": most_likely_pain["likelihood"],
        "perceived_metrics_id": [m["metric_id"] for m in most_likely_pain["metrics"]] if most_likely_pain else [],
        "job_id": most_likely_job["job_id"] if most_likely_job else "UNKNOWN",
        "job_likelihood": most_likely_job["likelihood"] if most_likely_job else 0,
        "persona_id": most_likely_persona["persona_id"] if most_likely_persona else "UNKNOWN",
    }

    # -----------------------------
    # 3) Lift from potential engagement
    # -----------------------------
    # --- Belief: next-best actions + ML persona what-if ---
    belief_next_best = []
    belief_mlp_lift = 0.0
    if n > 0:  # bm exists
        # Top-k actions across all personas
        belief_next_best = recommend_next_actions(bm, persona_order, k=5, mode="SOL")

        # If your “most likely persona” is in the persona_order, compute its what-if lift
        mlp_id = most_likely_persona["persona_id"] if most_likely_persona else None
        if mlp_id and mlp_id in persona_order:
            belief_mlp_lift = marginal_lift_for_persona(bm, persona_order, mlp_id, mode="SOL")

    #-----------------------------
    # 4) Causal Temporal Path Reconstruction
    #-----------------------------
    job_id = most_likely_first_responder_path_ids.get("job_id")
    product_id = get_product_id_from_subgraph(archetype_subgraph)
    if not job_id or not product_id:
        print("No job or product ID found, cannot reconstruct causal path.")
        return None
    causal_paths = list(nx.all_simple_paths(archetype_subgraph, source=product_id, target=job_id))
    causal_paths = [list(reversed(path)) for path in causal_paths] 
    causal_chains = []
    for path in causal_paths:
        path_likelihood = most_likely_first_responder_path_ids.get("job_likelihood", 0)
        causal_chain = []
        while path:
            node_id = path.pop(0)
            node = get_node_by_id(archetype_subgraph, node_id)
            if not node:
                print(f"Node with ID {node_id} not found in subgraph, skipping.")
                continue
            node_content = _node_content(archetype_subgraph, node_id)

            node = get_node_by_id(archetype_subgraph, node_id)
            node_depth = node.get("depth", 0)
            node_likelihood = path_likelihood
            path_likelihood *= get_edge_weight(archetype_subgraph, path[0], node_id) if path else 1
            causal_chain.append({
                "node_content": node_content,
                "node_likelihood": node_likelihood,
                "node_depth": node_depth,
            })
            if node.get("type")== "product":
                continue
        causal_chains.append(causal_chain)


    most_likely_first_responder_data = {
        "pain_trigger": _node_content(archetype_subgraph, most_likely_first_responder_path_ids["pain_trigger_id"]),
        "trigger_likelihood": most_likely_first_responder_path_ids["trigger_likelihood"],
        "trigger_depth": most_likely_first_responder_path_ids["trigger_depth"],
        "pain": _node_content(archetype_subgraph, most_likely_first_responder_path_ids["pain_id"]),
        "pain_likelihood": most_likely_first_responder_path_ids["pain_likelihood"],
        "perceived_metrics": [
                                _node_content(archetype_subgraph, metric_id)
                                for metric_id in most_likely_first_responder_path_ids["perceived_metrics_id"]
                            ] if most_likely_first_responder_path_ids["perceived_metrics_id"] else [],
        "job": _node_content(archetype_subgraph, most_likely_first_responder_path_ids["job_id"]),
        "job_likelihood": most_likely_first_responder_path_ids["job_likelihood"],
        "persona": _node_content(archetype_subgraph, most_likely_persona["persona_id"]) if most_likely_persona else "UNKNOWN",

    }

    first_response_pain_family = sorted(first_response_pain_family, key=lambda x: x["trigger_likelihood"], reverse=True)
    first_response_pain_family_data = []
    for trigger_res in first_response_pain_family:
            first_response_pain_family_data.append({
                "pain_trigger": trigger_res["pain_trigger"],
                "trigger_likelihood": trigger_res["trigger_likelihood"],
                "trigger_depth": trigger_res["trigger_depth"],
            })


    final_rcs = {
        "archetype": archetype,
        
        "first_response_pain_family": first_response_pain_family_data,
        "most_likely_first_response_path": most_likely_first_responder_data,
        "causal_chains": causal_chains,
        "belief": {
            "personas": persona_order,
            "activation_SOL": belief_summary.get("activation_SOL", []),
            "activation_PR":  belief_summary.get("activation_PR", []),
            "expected_score_SOL": belief_summary.get("expected_score_SOL", 0.0),
            "expected_score_PR":  belief_summary.get("expected_score_PR", 0.0),
            "marginal_lift_example": belief_mlp_lift,
            "next_best_actions": belief_next_best,
        },
    }

    print(f"Final RCS generated for archetype ID: {archetype_id}")



    
    return final_rcs

def _node_content(archetype_subgraph, node_id):
    node = get_node_by_id(archetype_subgraph, node_id)
    if not node:
        return "Node not found"
    node_type = node.get("type", "")
    if node_type == "job":
        content = {
            "description": node.get("description", ""),
        }
    elif node_type == "persona":
        content = {
            "title": node.get("title", ""),
            "department": node.get("department", ""),
            "seniority": node.get("seniority", "")
        }
    elif node_type == "pain":
        content = {
            "description": node.get("description", ""),
        }
    elif node_type == "capability":
        content = {
            "name": node.get("name", ""),
            "description": node.get("description", ""),
        }
    elif node_type == "product":
        content = {
            "product_url": node.get("url", ""),
        }
    elif node_type == "metric":
        content = {
            "metric": node.get("metric", ""),
        }
    elif node_type == "pain_trigger":
        content = {
            "attribute": node.get("attribute", ""),
        }
    elif node_type == "archetype":
        content = {
            "industry": node.get("industry", ""),
            "revenue_range": node.get("revenue_range", ""),
            "employee_range": node.get("employee_range", ""),
            "funding_stage": node.get("funding_stage", ""),
            "geography": node.get("geography", "")
        }
    elif node_type == "zmot":
        content = {
            "description": node.get("description", ""),
        }
    else:
        content = {
            "type": node_type,
            "attributes": {k: v for k, v in node.items() if k not in ["id", "type"]}
        }
    return content
    
def _current_stage(bm, i: int) -> S:
    # argmax over the 6-state distribution
    return S(int(bm.pi[i].argmax()))

def _next_stage(s: S) -> S | None:
    order = [S.Unaware, S.ProblemRealisation, S.PainRealisation, S.Discovery, S.Barriers, S.Implementation]
    try:
        j = order.index(s)
        return order[j+1] if j+1 < len(order) else None
    except ValueError:
        return None

def marginal_lift_for_persona(bm, persona_order: list[str], persona_id: str, mode="SOL") -> float:
    if persona_id not in persona_order:
        return 0.0
    i = persona_order.index(persona_id)
    s_from = _current_stage(bm, i)
    s_to = _next_stage(s_from)
    if s_to is None:
        return 0.0  # already at terminal state
    return float(bm.marginal_lift_if_forced(i, s_from, s_to, mode=mode))


def recommend_next_actions(bm, persona_order: list[str], k: int = 5, mode: str = "SOL"):
    items = []
    for pid in persona_order:
        i = persona_order.index(pid)
        s_from = _current_stage(bm, i)
        s_to = _next_stage(s_from)
        if s_to is None:
            continue
        dS = float(bm.marginal_lift_if_forced(i, s_from, s_to, mode=mode))
        items.append({
            "persona_id": pid,
            "from": s_from.name,
            "to": s_to.name,
            "delta_score": dS
        })
    items.sort(key=lambda x: x["delta_score"], reverse=True)
    return items[:k]
