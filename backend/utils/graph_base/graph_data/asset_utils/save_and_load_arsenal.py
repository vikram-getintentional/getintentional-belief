"""
IMPORTANT: This is a dev file to save graph as JSON each time we hit update_graph.
It is not smart or optimized - it just dumps the current state of the graph.
We should migrate this flow to Neo4j or Dgraph soon.
"""
import json
import os
import networkx as nx

from backend.utils.knowledge_base.arsenal.arsenal_models import Asset, Channel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARSENAL_JSON_DIR = os.path.join(BASE_DIR, "graph_data", "arsenal_json")


def save_asset_arsenal_as_json(assets, product_id):
    os.makedirs(ARSENAL_JSON_DIR, exist_ok=True)
    filename = "assets.json"
    full_path = os.path.join(ARSENAL_JSON_DIR, filename)

    # Load existing data if present
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # Convert Asset objects to dicts
    data = [a.__dict__ if hasattr(a, "__dict__") else dict(a) for a in assets]
    all_data[product_id] = data

    with open(full_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Assets Arsenal for {product_id} saved successfully to: {full_path}")

def save_channel_arsenal_as_json(channels, product_id):
    os.makedirs(ARSENAL_JSON_DIR, exist_ok=True)
    filename = "channels.json"
    full_path = os.path.join(ARSENAL_JSON_DIR, filename)

    # Load existing data if present
    if os.path.exists(full_path):
        with open(full_path, "r") as f:
            all_data = json.load(f)
    else:
        all_data = {}

    # Convert Channel objects to dicts
    data = [c.__dict__ if hasattr(c, "__dict__") else dict(c) for c in channels]
    all_data[product_id] = data

    with open(full_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Channels Arsenal for {product_id} saved successfully to: {full_path}")

def load_assets_from_arsenal_json(product_id):
    filename = "assets.json"
    full_path = os.path.join(ARSENAL_JSON_DIR, filename)
    if not os.path.exists(full_path):
        print(f"Arsenal file does not exist: {full_path}")
        return []
    with open(full_path, "r") as f:
        all_data = json.load(f)
    if product_id in all_data:
        assets = all_data[product_id]
        
    else:
        print(f"No assets found for product_id {product_id}")
        assets = []

    return assets

def load_channels_from_arsenal_json(product_id):
    filename = "channels.json"
    full_path = os.path.join(ARSENAL_JSON_DIR, filename)
    if not os.path.exists(full_path):
        print(f"Arsenal file does not exist: {full_path}")
        return []
    with open(full_path, "r") as f:
        all_data = json.load(f)
    if product_id in all_data:
        channels = all_data[product_id]
        
    else:
        print(f"No channels found for product_id {product_id}")
        channels = []

    return channels
