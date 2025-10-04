# regenerate_personas.py

from backend.utils.knowledge_base.canonical_maps.update_clusters import update_persona_clusters

if __name__ == "__main__":
    update_persona_clusters()
    print("✅ Persona canonicalization and embeddings regenerated.")
