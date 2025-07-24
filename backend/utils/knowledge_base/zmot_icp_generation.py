from typing import Dict, List, Any
from backend.utils.graph_base.graph import Graph
from backend.utils.graph_base.graph_utils.aggregate_persona_cards import aggregate_persona_cards
from backend.utils.graph_base.relevance.cumulative_relevance_manager import add_or_update_cumulative_relevance_data, get_cumulative_relevance_data


def get_zmot_icp_relevance(sub_graph: Graph) -> list[dict]:
    print("Starting relevance computation - at this point centrality & cum relevance should be set")
    personas = []
    product_id = sub_graph.get_node_id("product",{})


    relevance_nodes = sub_graph.calculate_cumulative_relevance()
    add_or_update_cumulative_relevance_data(product_id, relevance_nodes)

    # Do looping for ICP Mapping first

    """
    First lets get the list of ICP nodes and their relevance.
    ICP nodes are of type: icp_industry, icp_revenue, icp_employees, icp_funding_stage, icp_geography
    We will get the relevance for each of these nodes and then aggregate them as combinations of single industry, revenue, employees, funding stage, geography.
    The output will be a sorted list of these combinations with their relevance scores.
    """

    icp_node_ids = sub_graph.get_nodes_list_ids("icp_industry", {})
    icp_node_ids += sub_graph.get_nodes_list_ids("icp_revenue", {})
    icp_node_ids += sub_graph.get_nodes_list_ids("icp_employees", {})
    icp_node_ids += sub_graph.get_nodes_list_ids("icp_funding_stage", {})
    icp_node_ids += sub_graph.get_nodes_list_ids("icp_geography", {})

    print("Found ICP Nodes:", icp_node_ids)

    result = []
    return result

