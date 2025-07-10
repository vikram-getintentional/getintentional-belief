# graph_loader.py
import os
import json

def load_graph_from_folder(folder_path: str):
    node_registry = {}
    graph_edges = []
    edge_weights = {}
    
    print("Current working directory:", os.getcwd())
    print("GRAPH_DATA_PATH:", folder_path)
    print("Exists?", os.path.exists(folder_path))

    for fname in os.listdir(folder_path):
        if fname.endswith("_nodes.json"):
            node_type = fname.replace("_nodes.json", "")
            
            with open(os.path.join(folder_path, fname), "r") as f:
                nodes = json.load(f)
                for node in nodes:
                    if node_type == "persona":
                        value = (node.get("title", "").strip().lower(),
                                 node.get("seniority", "").strip().lower(),
                                 node.get("department", "").strip().lower())
                    elif node_type == "pain":
                        value = node.get("text", "").strip().lower()
                    elif node_type == "job":
                        value = node.get("description", "").strip().lower()
                    elif node_type == "capability":
                        value = (node.get("name", "").strip().lower(),
                                 node.get("description", "").strip().lower())
                    elif node_type == "scaling_factor":
                        value = node.get("description", "").strip().lower()
                    elif node_type == "product":
                        value = (node.get("summary", "").strip().lower(),
                                 node.get("company_id", "").strip().lower(),
                                 node.get("url", "").strip().lower(),
                                 node.get("plg_flag", ""))
                    else:
                        value = node.get("id")
                    node["node_type"] = node_type
                    node_registry[node["id"]] = node

    # Load edges
    edges_path = os.path.join(folder_path, "graph_edges.json")
    if os.path.exists(edges_path):
        with open(edges_path, "r") as f:
            graph_edges = json.load(f)
            for edge in graph_edges:
                edge_weights[(edge["source"], edge["target"])] = edge.get("weight", 1.0)

    # Load cumulative relevance if present
    cumulative_relevance_path = os.path.join(folder_path, "cumulative_relevance.json")
    if os.path.exists(cumulative_relevance_path):
        with open(cumulative_relevance_path, "r") as f:
            cumulative_relevance = json.load(f)
    else:
        cumulative_relevance = {}
    print("Graph loaded with nodes:", len(node_registry), "and edges:", len(graph_edges))
    return node_registry, graph_edges, edge_weights, cumulative_relevance



