from backend.utils.graph_base.network_graph import get_node_by_id, get_target_nodes_by_source_and_type
from backend.utils.inference.visual_analysis_engine.rcs_utils import get_best_match_zmots_for_archetype
import networkx as nx


def assumed_zmot_sample(product_subgraph: nx.DiGraph, archetype_id, threshold=0.2, top_n=5):
    """
    This is a temp testing function to assume a zmot event has occurred.
    It takes in the best match zmot events output from the previous function 
    and returns a single zmot event + observable moment + trigger keyword.

    """
    best_match_zmots = get_best_match_zmots_for_archetype(product_subgraph, archetype_id, threshold, top_n)
    if not best_match_zmots:
        print("No ZMOTs found for archetype")
        return None
    zmot_event_id = best_match_zmots[0].get("id")
    
    return zmot_event_id