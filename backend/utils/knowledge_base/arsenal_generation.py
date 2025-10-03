import os
from typing import List
import uuid
import json
from backend.utils.graph_base.graph_data.asset_utils.save_and_load_arsenal import load_assets_from_arsenal_json, load_channels_from_arsenal_json
from backend.utils.graph_base.graph_utils.json_store import load_json, save_json
from backend.utils.knowledge_base.arsenal.arsenal_models import Asset, Channel





def get_all_arsenals(product_id):
    """
    Gets all existing arsenals from assets and channels json and returns a dict as arsenal{assets[],channels[]}
    """
    
    print("Getting Arsenal for product", product_id)
    assets = load_assets_from_arsenal_json(product_id)
    channels = load_channels_from_arsenal_json(product_id)
    ASSETS: List[Asset] = load_assets_from_arsenal_json(product_id)
    print("Loaded", len(ASSETS), "assets", "for product", product_id)
    CHANNELS: List[Channel] = load_channels_from_arsenal_json(product_id)
    print("Loaded", len(CHANNELS), "channels", "for product", product_id)
    output = {
        "assets": ASSETS,
        "channels": CHANNELS
    }
    print(f"Loaded Arsenal for product {product_id}: {output}")
    return output
