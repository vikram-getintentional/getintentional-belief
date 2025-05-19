import os
import uuid

from backend.utils.embedding.embed_utils import get_embedding
from backend.utils.knowledge_base.canonical_maps.canonical_loader import load_embeddings, save_embeddings
import json
from pathlib import Path
from backend.utils.embedding.embed_utils import get_embedding


import numpy as np

CANONICAL_MAP_PATH = Path("backend/utils/knowledge_base/canonical_maps/")
EMBEDDING_OUTPUT_PATH = CANONICAL_MAP_PATH / "canonical_embeddings/"
CANONICAL_FILE = CANONICAL_MAP_PATH / "persona_to_canonical.json"
OUTPUT_FILE = EMBEDDING_OUTPUT_PATH / "canonical_persona_embeddings.json"


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def get_best_match_from_embedding(text, canonical_map, entity_type="job", threshold=0.5):
    embedding = get_embedding(text.strip().lower())
    embeddings = load_embeddings(entity_type)

    best_id = None
    best_label = None
    best_score = -1

    for cid, cinfo in embeddings.items():
        sim = cosine_similarity(embedding, cinfo["embedding"])
        if sim > best_score:
            best_score = sim
            best_id = cid
            best_label = cinfo["representative"]

    if best_score >= threshold:
        return best_label

    # Create new cluster
    canonical_id = f"{entity_type}_{uuid.uuid4().hex[:8]}"
    canonical_label = text
    print(f"🆕 Creating new {entity_type} cluster: {canonical_label} → {canonical_id}")

    # Save to canonical map
    canonical_map[text.lower()] = canonical_label
    with open(CANONICAL_MAP_PATH / f"{entity_type}_to_canonical.json", "w") as f:
        json.dump(canonical_map, f, indent=2)

    # Save to embedding store
    embeddings[canonical_id] = {
        "representative": canonical_label,
        "embedding": embedding
    }
    save_embeddings(entity_type, embeddings)

    return canonical_label




def update_persona_clusters():
    print("📦 Starting persona embedding refresh")

    # Load canonical persona map
    if CANONICAL_FILE.exists():
        with open(CANONICAL_FILE, "r") as f:
            persona_map = json.load(f)
            print(f"✅ Loaded {len(persona_map)} personas from canonical map")
    else:
        print("⚠️ Persona map doesn't exist. Cannot proceed without it.")
        persona_map = {}

    embeddings = {}

    for key, val in persona_map.items():
        # Either use full string or canonical cluster label
        raw_text = key if isinstance(key, str) else val
        text = raw_text.strip()

        if not text:
            continue

        emb = get_embedding(text)
        if emb:
            embeddings[val] = {
                "representative": text,
                "embedding": emb
            }
        else:
            print(f"❌ Failed to embed: {text}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(embeddings, f, indent=2)

    print(f"✅ Saved {len(embeddings)} persona embeddings to {OUTPUT_FILE}")



