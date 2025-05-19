import json
import os

def load_canonical_map(path, fallback_value=None):
    print(f"Loading canonical map from: {path}")
    if not os.path.exists(path):
        return fallback_value or {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error loading JSON from {path}: {e}")
        return fallback_value or {}


def load_embeddings(entity_type: str) -> dict:
    """
    Load embeddings for a specific entity type from a JSON file.

    Args:
        entity_type (str): The type of entity (e.g., "persona", "job", "pain").

    Returns:
        dict: A dictionary of embeddings, or an empty dictionary if the file is missing or invalid.
    """
    path = f"backend/utils/knowledge_base/canonical_maps/canonical_embeddings/canonical_{entity_type}_embedding.json"
    if not os.path.exists(path):
        print(f"⚠️ Embeddings file not found: {path}. Returning an empty dictionary.")
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)
            print(f"✅ Loaded embeddings from {path}.")
            return data
    except json.JSONDecodeError as e:
        print(f"❌ Error loading embeddings from {path}: {e}. Returning an empty dictionary.")
        return {}

import json
import os

def save_canonical_map(canonical_map: dict, file_path: str):
    # Load existing data if the file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: {file_path} is corrupted. Initializing as an empty dictionary.")
                existing_data = {}
    else:
        existing_data = {}

    # Check for overwriting keys
    overlapping_keys = set(existing_data.keys()) & set(canonical_map.keys())
    if overlapping_keys:
        print(f"⚠️ Warning: Overwriting keys in {file_path}")

    # Merge the new data with the existing data
    existing_data.update(canonical_map)

    # Write the merged data back to the file
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(f"✅ Canonical map saved to {file_path}")


def save_embeddings(entity_type: str, new_data: dict):
    path = f"backend/utils/knowledge_base/canonical_maps/canonical_embeddings/{entity_type}_embeddings.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Load existing embeddings
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: {path} is corrupted. Initializing as an empty dictionary.")
            existing_data = {}
    else:
        existing_data = {}

    # Check for overwriting keys
    overlapping_keys = set(existing_data.keys()) & set(new_data.keys())
    if overlapping_keys:
        print(f"⚠️ Warning: Overwriting keys in {path}")

    # Update with new embeddings
    existing_data.update(new_data)

    # Save back to file
    with open(path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(f"✅ Embeddings saved to {path}")
