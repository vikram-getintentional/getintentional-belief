from sentence_transformers import SentenceTransformer

from backend.utils.knowledge_base.canonical_maps.canonical_loader import save_embeddings

# Load once and reuse (guarded for offline environments)
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as exc:
    print("⚠️ SentenceTransformer model load failed; embeddings disabled:", exc)
    model = None

def get_embedding(text: str) -> list[float]:
    if model is None:
        return []
    try:
        embedding = model.encode([text], convert_to_numpy=True)[0]
        return embedding.tolist()
    except Exception as e:
        print(f"❌ Local embedding failed for: {text[:50]}...", e)
        return []


def generate_embeddings(texts: list[str]) -> dict:
    if model is None:
        print("⚠️ Embedding model unavailable; skipping generation.")
        return {}
    print(f"Generating embeddings  for texts")
    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
        print(f"Generated embeddings")
        return {text: embedding.tolist() for text, embedding in zip(texts, embeddings)}
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        return {}
    

def generate_and_save_embeddings(texts: list[str], entity_type: str) -> dict:
    """
    Generate embeddings for a list of texts and save them to a file.

    Args:
        texts (list[str]): A list of texts to generate embeddings for.
        entity_type (str): The type of entity (e.g., "persona", "job", "pain").

    Returns:
        dict: A dictionary of generated embeddings.
    """
    if not texts:
        print(f"⚠️ No texts provided for {entity_type} embedding generation.")
        return {}

    # Generate embeddings
    embeddings = generate_embeddings(texts)
    if not embeddings:
        print(f"❌ Failed to generate embeddings for {entity_type}.")
        return {}

    # Save embeddings
    save_embeddings(entity_type, embeddings)

    # Return the generated embeddings
    return embeddings
