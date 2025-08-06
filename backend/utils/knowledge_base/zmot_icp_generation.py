from typing import Dict, List, Any
from backend.utils.graph_base.network_graph import calculate_cumulative_relevance, calculate_soft_or_relevance, get_node_by_id, get_node_id, get_nodes_list_ids, get_target_nodes_by_source_and_type
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data, get_cumulative_relevance_data
import networkx as nx


def get_zmot_icp_relevance(sub_graph: nx.DiGraph, threshold = 0.0) -> list[dict]:
    print("Starting relevance computation - at this point centrality & cum relevance should be set")
    personas = []
    product_id = get_node_id(sub_graph, "product", {})
    relevance_nodes = calculate_soft_or_relevance(sub_graph)
    cumulative_relevance = {item["node_id"]: item["relevance"] for item in relevance_nodes}
    add_or_update_cumulative_relevance_data(product_id, cumulative_relevance)

    # Do looping for ICP Mapping first

    """
    First lets get the list of ICP nodes and their relevance.
    ICP nodes are of type: icp_industry, icp_revenue, icp_employees, icp_funding_stage, icp_geography
    We will get the relevance for each of these nodes and then aggregate them as combinations of single industry, revenue, employees, funding stage, geography.
    The output will be a sorted list of these combinations with their relevance scores.
    """
    archetypes = []
    archetype_ids = get_nodes_list_ids(sub_graph, "archetype", {})
    for archetype_id in archetype_ids:
        relevance = get_cumulative_relevance_data(product_id, archetype_id)
        if relevance > threshold:
            archetype_node = get_node_by_id(sub_graph, archetype_id)
            industry = archetype_node.get("industry", "").strip().lower()
            revenue = archetype_node.get("revenue_range", "").strip().lower()
            employees = archetype_node.get("employee_range", "").strip().lower()
            funding_stage = archetype_node.get("funding_stage", "").strip().lower()
            geography = archetype_node.get("geography", "").strip().lower()
            archetypes.append({
                "industry": industry,
                "revenue": revenue,
                "employees": employees,
                "funding_stage": funding_stage,
                "geography": geography,
                "relevance": relevance
            })
    print("Archetypes found:", archetypes)

    
    zmot = []
    zmot_triggerevent_ids = get_nodes_list_ids(sub_graph, "zmot_triggerevents", {})
    for zmot_triggerevent_id in zmot_triggerevent_ids:
        observable_moments = []
        trigger_keywords = []
        trigger_event_node = get_node_by_id(sub_graph, zmot_triggerevent_id)
        trigger_relevance = get_cumulative_relevance_data(product_id, zmot_triggerevent_id)
        zmot_triggerevent = trigger_event_node.get("trigger_event", "").strip().lower()
        if trigger_relevance < threshold:
            continue
        observable_moment_ids = get_target_nodes_by_source_and_type(sub_graph, zmot_triggerevent_id, "observed_in")
        trigger_keyword_ids = get_target_nodes_by_source_and_type(sub_graph, zmot_triggerevent_id, "associated_with")
        for observable_moment_id in observable_moment_ids:
            obs_relevance = get_cumulative_relevance_data(product_id, observable_moment_id)
            obs_label = get_node_by_id(sub_graph, observable_moment_id).get("observable_moment", "").strip().lower()
            if obs_relevance < threshold:
                continue
            observable_moments.append((obs_label, obs_relevance))
        for trigger_keyword_id in trigger_keyword_ids:
            kw_relevance = get_cumulative_relevance_data(product_id, trigger_keyword_id)
            kw_label = get_node_by_id(sub_graph, trigger_keyword_id).get("keyword", "").strip().lower()
            if kw_relevance < threshold:
                continue
            trigger_keywords.append((kw_label, kw_relevance))
        zmot.append({
            "trigger_event": zmot_triggerevent,
            "trigger_event_relevance": trigger_relevance,
            "observable_moments": list(observable_moments),
            "trigger_keywords": list(trigger_keywords)
        })
        
        print("Current zmot list:", zmot)

    icp_zmot_result = {
        "icp": archetypes,
        "zmot": zmot
    }

    return icp_zmot_result