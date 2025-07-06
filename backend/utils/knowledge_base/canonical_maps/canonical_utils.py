from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def cluster_items(embeddings: dict, distance_threshold=0.5, entity_type="generic") -> dict:
    """
    Cluster items based on their embeddings.

    Args:
        embeddings (dict): A dictionary where keys are items (e.g., pains, jobs, personas)
                           and values are their embeddings.
        distance_threshold (float): The threshold for clustering.
        entity_type (str): The type of entity being clustered (e.g., "pain", "job", "persona").

    Returns:
        dict: A dictionary where keys are cluster IDs and values are lists of items in each cluster.
    """
    if not embeddings:
        print(f"⚠️ No embeddings provided for {entity_type}. Returning empty clusters.")
        return {}

    items = list(embeddings.keys())
    embedding_vectors = np.array(list(embeddings.values()))

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embedding_vectors)

    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Automatically determine the number of clusters
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average"
    )
    cluster_labels = clustering.fit_predict(1 - similarity_matrix)  # Use 1 - similarity for distance

    # Group items by cluster
    clustered_items = {}
    for item, cluster_id in zip(items, cluster_labels):
        if cluster_id not in clustered_items:
            clustered_items[cluster_id] = []
        clustered_items[cluster_id].append(item)

    return clustered_items


def assign_canonical_labels(clustered_items: dict, selection_strategy="shortest") -> dict:
    """
    Assign canonical labels to clusters.

    Args:
        clustered_items (dict): A dictionary where keys are cluster IDs and values are lists of items.
        selection_strategy (str): The strategy for selecting the canonical label ("shortest", "first", etc.).

    Returns:
        dict: A dictionary where keys are items and values are their canonical labels.
    """
    canonical_map = {}
    for cluster_id, items in clustered_items.items():
        if selection_strategy == "shortest":
            canonical_label = min(items, key=len)
        elif selection_strategy == "first":
            canonical_label = items[0]
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")

        for item in items:
            canonical_map[item] = canonical_label

    return canonical_map


