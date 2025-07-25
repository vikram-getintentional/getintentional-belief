from typing import Dict, List, Any
from backend.utils.graph_base.graph import Graph
from backend.utils.graph_base.graph_utils.aggregate_persona_cards import aggregate_persona_cards
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data, get_cumulative_relevance_data


def get_zmot_icp_relevance(sub_graph: Graph, threshold = 0.3) -> list[dict]:
    print("Starting relevance computation - at this point centrality & cum relevance should be set")
    personas = []
    product_id = sub_graph.get_node_id("product",{})
    relevance_nodes = sub_graph.calculate_cumulative_relevance()
    #add_or_update_cumulative_relevance_data(product_id, relevance_nodes)

    # Do looping for ICP Mapping first

    """
    First lets get the list of ICP nodes and their relevance.
    ICP nodes are of type: icp_industry, icp_revenue, icp_employees, icp_funding_stage, icp_geography
    We will get the relevance for each of these nodes and then aggregate them as combinations of single industry, revenue, employees, funding stage, geography.
    The output will be a sorted list of these combinations with their relevance scores.
    """
    industry = set()
    revenue = set()
    employees = set()
    funding_stage = set()
    geography = set()

    icp_industry_node_ids = sub_graph.get_nodes_list_ids("icp_industry", {})
    for icp_industry_node_id in icp_industry_node_ids:
        relevance = get_cumulative_relevance_data(product_id, icp_industry_node_id)
        label = sub_graph.get_node_by_id(icp_industry_node_id).get("industry", "").strip().lower()
        if relevance > threshold:
            industry.add((label, relevance))
    icp_revenue_node_ids = sub_graph.get_nodes_list_ids("icp_revenue", {})
    for icp_revenue_node_id in icp_revenue_node_ids:
        relevance = get_cumulative_relevance_data(product_id, icp_revenue_node_id)
        label = sub_graph.get_node_by_id(icp_revenue_node_id).get("revenue", "").strip().lower()
        if relevance > threshold:
            revenue.add((label, relevance))
    icp_employees_node_ids = sub_graph.get_nodes_list_ids("icp_employees", {})
    for icp_employees_node_id in icp_employees_node_ids:
        relevance = get_cumulative_relevance_data(product_id, icp_employees_node_id)
        label = sub_graph.get_node_by_id(icp_employees_node_id).get("employees", "").strip().lower()
        if relevance > threshold:
            employees.add((label, relevance))
    icp_funding_stage_node_ids = sub_graph.get_nodes_list_ids("icp_funding_stage", {})
    for icp_funding_stage_node_id in icp_funding_stage_node_ids:
        relevance = get_cumulative_relevance_data(product_id, icp_funding_stage_node_id)
        label = sub_graph.get_node_by_id(icp_funding_stage_node_id).get("funding_stage", "").strip().lower()
        if relevance > threshold:
            funding_stage.add((label, relevance))
    icp_geography_node_ids = sub_graph.get_nodes_list_ids("icp_geography", {})
    for icp_geography_node_id in icp_geography_node_ids:
        relevance = get_cumulative_relevance_data(product_id, icp_geography_node_id)
        label = sub_graph.get_node_by_id(icp_geography_node_id).get("geography", "").strip().lower()
        if relevance > threshold:
            geography.add((label, relevance))

    
    zmot = []
    zmot_triggerevents = sub_graph.get_nodes_list_ids("zmot_triggerevents", {})
    for zmot_triggerevent in zmot_triggerevents:
        observable_moments = []
        trigger_keywords = []
        trigger_relevance = get_cumulative_relevance_data(product_id, zmot_triggerevent)
        trigger_label = sub_graph.get_node_by_id(zmot_triggerevent).get("trigger_event", "").strip().lower()
        if trigger_relevance > threshold:
            zmot_observable_moments = sub_graph.get_target_nodes_by_source_and_type(
                zmot_triggerevent, "zmot_observable_moment")
            for zmot_observablemoment in zmot_observable_moments:
                obs_relevance = get_cumulative_relevance_data(product_id, zmot_observablemoment)
                obs_label = sub_graph.get_node_by_id(zmot_observablemoment).get("observable_moment", "").strip().lower()
                zmot_trigger_keywords = sub_graph.get_target_nodes_by_source_and_type(
                    zmot_observablemoment, "zmot_keyword")
                for zmot_trigger_keyword in zmot_trigger_keywords:
                    kw_relevance = get_cumulative_relevance_data(product_id, zmot_trigger_keyword)
                    kw_label = sub_graph.get_node_by_id(zmot_trigger_keyword).get("keyword", "").strip().lower()
                    trigger_keywords.append((kw_label, kw_relevance))
                    print("ZMOT trigger keyword data:", trigger_keywords)
                observable_moments.append((obs_label, obs_relevance, trigger_keywords))

            zmot.append({
                "trigger_event": trigger_label,
                "trigger_event_relevance": trigger_relevance,
                "observable_moments": list(observable_moments),

            })

            print("Current zmot list:", zmot)

    icp_zmot_result = {
        "icp": {
            "industries": industry,
            "revenues": revenue,
            "employees": employees,
            "funding_stages": funding_stage,
            "geographies": geography
        },
        "zmot": zmot
    }

    return icp_zmot_result

