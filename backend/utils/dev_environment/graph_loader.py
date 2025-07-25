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
            print("Processing file:", fname)
            node_type = fname.replace("_nodes.json", "")
            print("Node type:", node_type)
            
            with open(os.path.join(folder_path, fname), "r") as f:
                nodes = json.load(f)

                # Special handling for zmot nodes
                if node_type == "zmot":
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
                    continue  # skip generic block

                # Special handling for company nodes
                if node_type == "company":
                    for company_key in ["industry", "revenue", "employees", "funding_stage", "geography"]:
                        for node in nodes.get(company_key, []):
                            node["node_type"] = f"icp_{company_key.lower()}"
                            if company_key == "industry":
                                node["value"] = node.get("industry", "").strip().lower()
                            elif company_key == "revenue":
                                node["value"] = node.get("revenue", "").strip().lower()
                            elif company_key == "employees":
                                node["value"] = node.get("employees", "").strip().lower()
                            elif company_key == "funding_stage":
                                node["value"] = node.get("funding_stage", "").strip().lower()
                            elif company_key == "geography":
                                node["value"] = node.get("geography", "").strip().lower()
                            node_registry[node["id"]] = node
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
                                
                    else:
                        value = node.get("id")
                        print("Setting unknown node type:", node_type, "with value:", value)
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
                if edge["source"] is "null" or edge["source"] is None or edge["target"] is "null" or edge["target"] is None:
                    print(f"Edge {edge} has null source or target. Skipping this edge.")
                    continue
                edge_weights[(edge["source"], edge["target"])] = edge.get("weight", 1.0)
    print("Graph loaded with nodes:", len(node_registry), "and edges:", len(graph_edges))
    return node_registry, graph_edges, edge_weights