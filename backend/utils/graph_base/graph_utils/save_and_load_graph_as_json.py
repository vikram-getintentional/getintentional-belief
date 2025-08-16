"""
IMPORTANT: This is a dev file to save graph as JSON each time we hit update_graph.
It is not smart or optimized - it just dumps the current state of the graph.
We should migrate this flow to Neo4j or Dgraph soon.
"""
import json
import os
import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_JSON_DIR = os.path.join(BASE_DIR, "graph_base", "graph_data", "network_json")
 

def save_graph_as_json(G, product_id):
    # Ensure the directory exists
    os.makedirs(GRAPH_JSON_DIR, exist_ok=True)
    print("Directory found. Saving graph to:", GRAPH_JSON_DIR)
    # Always save in the designated directory
    filename = f"{product_id}_graph.json"
    full_path = os.path.join(GRAPH_JSON_DIR, filename)
    print("Full file path:", full_path)

    data = {
        "graph": dict(G.graph),
        "nodes": [
            {"id": n, **G.nodes[n]} for n in G.nodes
        ],
        "edges": [
            {"source": u, "target": v, **G[u][v]} for u, v in G.edges
        ]
    }
    with open(full_path, "w") as f:
        json.dump(data, f, indent=2)
        print("Writing to file:")
    print("Graph saved successfully to:", full_path)


def load_graph_from_json(product_id):
    # Path to saved graph JSON
    filename = f"{product_id}_graph.json"
    full_path = os.path.join(GRAPH_JSON_DIR, filename)
    if not os.path.exists(full_path):
        print(f"Graph file does not exist: {full_path}")
        return nx.DiGraph() 

    G = nx.DiGraph()
    with open(full_path, "r") as f:
        data = json.load(f)
        G = nx.DiGraph()
        # Restore graph attributes
        if "graph" in data:
            G.graph.update(data["graph"])
        # Restore nodes and edges as before
        for node in data["nodes"]:
            node_id = node.pop("id")
            G.add_node(node_id, **node)
        for edge in data["edges"]:
            source = edge.pop("source")
            target = edge.pop("target")
            G.add_edge(source, target, **edge)
    return G