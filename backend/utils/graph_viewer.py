#This code reads the node and edge data from json files and builds a visual graph representation for internal debugging and testing.
import json
from pathlib import Path
from graphviz import Digraph
from backend.utils.graph_base.graph_utils.json_store import load_json
GRAPH_PATH = Path("backend/utils/graph_base/graph_data")
EDGE_PATH = GRAPH_PATH / "graph_edges.json" 
CAPABILITY_PATH = GRAPH_PATH / "capability_nodes.json"
PERSONA_PATH = GRAPH_PATH / "persona_nodes.json"
JOB_PATH = GRAPH_PATH / "job_nodes.json"
PAIN_PATH = GRAPH_PATH / "pain_nodes.json"
def visualize_graph():
    # Load nodes and edges
    edges = load_json(EDGE_PATH)
    capabilities = load_json(CAPABILITY_PATH)
    personas = load_json(PERSONA_PATH)
    jobs = load_json(JOB_PATH)
    pains = load_json(PAIN_PATH)

    # Create a directed graph
    dot = Digraph(comment='Graph Visualization')

    # Add nodes
    for cap in capabilities:
        dot.node(cap['id'], cap['value'], shape='box', color='lightblue')
    
    for persona in personas:
        dot.node(persona['id'], persona['value'], shape='ellipse', color='lightgreen')
    
    for job in jobs:
        dot.node(job['id'], job['value'], shape='diamond', color='lightyellow')
    
    for pain in pains:
        dot.node(pain['id'], pain['value'], shape='hexagon', color='lightcoral')

    # Add edges
    for edge in edges:
        dot.edge(edge['source'], edge['target'], label=edge['type'])

    # Render the graph to a file
    dot.render('graph_visualization', format='png', cleanup=True)
    print("Graph visualization saved as 'graph_visualization.png'")
    return dot
    # Return the graph object for further manipulation if needed
if __name__ == "__main__":
    visualize_graph()
    # You can also save the graph to a file or display it in a Jupyter notebook
    # dot.view()  # Uncomment to view the graph directly if you have Graphviz installed
#                             pain_triggers.append(trigger_value["value"])
#
#                 # --- END NEW ---

