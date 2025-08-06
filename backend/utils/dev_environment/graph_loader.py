# graph_loader.py
import os
import json


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DATA_PATH = os.path.join(BASE_DIR, "utils", "graph_base", "graph_data")



def load_product_graph_from_folder(product_id: str, folder_path: str = GRAPH_DATA_PATH):
    node_registry = {}
    graph_edges = []
    edge_weights = {}
    print("load_product_graph_from_folder running")
    print("Current working directory:", os.getcwd())
    print("GRAPH_DATA_PATH:", folder_path)
    print("Exists?", os.path.exists(folder_path))

    # Load edges for the given product_id only and add list of unique node IDs in graph
    edges_path = os.path.join(folder_path, "graph_edges.json")
    unique_node_ids = set()
    if os.path.exists(edges_path):
        with open(edges_path, "r") as f:
            edges_data = json.load(f)
            # edges_data is now a dict: {product_id: [edges]}
            edges_list = edges_data.get(product_id, [])
            for edge in edges_list:
                if edge["source"] in [None, "null"] or edge["target"] in [None, "null"]:
                    print(f"Edge {edge} has null source or target. Skipping this edge.")
                    continue
                graph_edges.append(edge)
                edge_weights[(edge["source"], edge["target"])] = edge.get("weight", 1.0)
                unique_node_ids.add(edge["source"])
                unique_node_ids.add(edge["target"])

    # Load nodes for each unique node ID found in edges
    for fname in os.listdir(folder_path):
        if fname.endswith("_nodes.json"):
            print("Processing file:", fname)
            node_type = fname.replace("_nodes.json", "")
            print("Node type:", node_type)
            
            with open(os.path.join(folder_path, fname), "r") as f:
                nodes = json.load(f)

                # Special handling for zmot nodes
                if node_type == "zmot":
                    for zmot_key in ["TriggerEvents", "ObservableMoments", "Keywords"]:
                        for node in nodes.get(zmot_key, []):
                            if node.get("id") not in unique_node_ids:
                                print("Node not in unique_node_ids, skipping:", node)
                                continue
                            node["node_type"] = f"zmot_{zmot_key.lower()}"
                            if zmot_key == "TriggerEvents":
                                node["value"] = node.get("trigger_event", "").strip().lower()
                            elif zmot_key == "ObservableMoments":
                                node["value"] = node.get("observable_moment", "").strip().lower()
                            elif zmot_key == "Keywords":
                                node["value"] = node.get("keyword", "").strip().lower()
                            node_registry[node["id"]] = node
                    continue  # skip generic block

                # Special handling for company nodes
                

                for node in nodes:
                    if node.get("id") not in unique_node_ids:
                        continue
                    if node_type == "product":
                        value = (
                            node.get("summary", "").strip().lower(),
                            node.get("url", "").strip().lower(),
                            node.get("plg_flag")
                        )
                    elif node_type == "persona":
                        value = (node.get("title", "").strip().lower(),
                                 node.get("department", "").strip().lower(),
                                 node.get("seniority", "").strip().lower())
                    elif node_type == "pain":
                        value = node.get("text", "").strip().lower()
                    elif node_type == "job":
                        value = node.get("description", "").strip().lower()
                    elif node_type == "capability":
                        value = (node.get("name", "").strip().lower(),
                                 node.get("description", "").strip().lower())
                    elif node_type == "pain_trigger":
                        value = (
                            node.get("attribute", "").strip().lower(),
                            node.get("dimension", "").strip().lower(),
                            node.get("direction", "").strip().lower()
                        )
                    elif node_type == "archetype":
                        value = (
                            node.get("industry", "").strip().lower(),
                            node.get("revenue_range", "").strip().lower(),
                            node.get("employee_range", "").strip().lower(),
                            node.get("funding_stage", "").strip().lower(),
                            node.get("geography", "").strip().lower()
                        )
                    
    
                    else:
                        value = node.get("id")
                        print("Setting unknown node type:", node_type, "with value:", value)
                    node["node_type"] = node_type
                    node["value"] = value
                    node_registry[node["id"]] = node

    
    print("Graph loaded with nodes:", len(node_registry), "and edges:", len(graph_edges))
    
    return node_registry, graph_edges, edge_weights


def load_product_nodes_from_folder(folder_path: str = GRAPH_DATA_PATH):
    node_registry = {}
    
    print("Current working directory:", os.getcwd())
    print("GRAPH_DATA_PATH:", folder_path)
    print("Exists?", os.path.exists(folder_path))

    # Load nodes for the given product_id only
    nodes_path = os.path.join(folder_path, "product_nodes.json")
    if os.path.exists(nodes_path):
        with open(nodes_path, "r") as f:
            nodes = json.load(f)
            # if no nodes found, return empty dict
            if not nodes:
                print("No product nodes found in", nodes_path)
                return []
            for node in nodes:
                node_registry[node["id"]] = node

    print("Product nodes loaded:", len(node_registry))
    return node_registry




"""
def load_graph_from_folder(folder_path: str = GRAPH_DATA_PATH):
    node_registry = {}
    graph_edges = []
    edge_weights = {}
    
    print("Current working directory:", os.getcwd())
    print("GRAPH_DATA_PATH:", folder_path)
    print("Exists?", os.path.exists(folder_path))

    for fname in os.listdir(folder_path):
        if fname.endswith("_nodes.json"):
            print("Processing file:", fname)
            node_type = fname.replace("_nodes.json", "").strip().lower()
            print("Node type:", node_type)
            
            with open(os.path.join(folder_path, fname), "r") as f:
                nodes = json.load(f)

                # Special handling for zmot nodes
                if node_type == "zmot":
                    print("Processing zmot nodes:")
                    for zmot_key in ["TriggerEvents", "ObservableMoments", "Keywords"]:
                        for node in nodes.get(zmot_key, []):
                            node["node_type"] = f"zmot_{zmot_key.lower()}"
                            if zmot_key == "TriggerEvents":
                                node["value"] = node.get("trigger_event", "").strip().lower()
                            elif zmot_key == "ObservableMoments":
                                node["value"] = node.get("observable_moment", "").strip().lower()
                            elif zmot_key == "Keywords":
                                node["value"] = node.get("keyword", "").strip().lower()
                            node_registry[node["id"]] = node
                            print("Processed zmot node:", node, "with type", node["node_type"])
                    continue  # skip generic block

                for node in nodes:
                    if node_type == "product":
                        value = (
                            node.get("summary", "").strip().lower(),
                            node.get("url", "").strip().lower(),
                            node.get("plg_flag")
                        )
                    elif node_type == "persona":
                        value = (node.get("title", "").strip().lower(),
                                 node.get("department", "").strip().lower(),
                                 node.get("seniority", "").strip().lower())
                    elif node_type == "pain":
                        value = node.get("text", "").strip().lower()
                    elif node_type == "job":
                        value = node.get("description", "").strip().lower()
                    elif node_type == "capability":
                        value = (node.get("name", "").strip().lower(),
                                 node.get("description", "").strip().lower())
                    elif node_type == "pain_trigger":
                        value = (
                            node.get("attribute", "").strip().lower(),
                            node.get("dimension", "").strip().lower(),
                            node.get("direction", "").strip().lower()
                        )
                    elif node_type == "archetype":
                        print("Processing archetype node:", node)
                        value = (
                            node.get("industry", "").strip().lower(),
                            node.get("revenue_range", "").strip().lower(),
                            node.get("employee_range", "").strip().lower(),
                            node.get("funding_stage", "").strip().lower(),
                            node.get("geography", "").strip().lower()
                        )
                                
                    else:
                        value = node.get("id")
                    node["node_type"] = node_type
                    node["value"] = value
                    node_registry[node["id"]] = node
                    

    # Load edges
    edges_path = os.path.join(folder_path, "graph_edges.json")
    if os.path.exists(edges_path):
        with open(edges_path, "r") as f:
            graph_edges = json.load(f)
            for edge in graph_edges:
                # Add a check to ensure source and target are not null, and source and target IDs exist in the node registry
                if edge["source"] == "null" or edge["source"] is None or edge["target"] == "null" or edge["target"] is None:
                    print(f"Edge {edge} has null source or target. Skipping this edge.")
                    continue
                edge_weights[(edge["source"], edge["target"])] = edge.get("weight", 1.0)
    print("Graph loaded with nodes:", len(node_registry), "and edges:", len(graph_edges))
    return node_registry, graph_edges, edge_weights


"""