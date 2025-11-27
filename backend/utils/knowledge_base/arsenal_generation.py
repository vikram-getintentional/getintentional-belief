from backend.database import get_db
from backend.utils.knowledge_base.arsenal.service import serialize_arsenal_library


def get_all_arsenals(product_id: str):
    """
    Return the structured arsenal library for a product.

    This consolidates assets, channels, and any recorded impact evidence
    so the API layer can serve it directly without touching JSON files.
    """
    print("Getting Arsenal for product", product_id)
    db = next(get_db())
    try:
        library = serialize_arsenal_library(db, product_id=product_id)
        print(
            f"Loaded {len(library.get('assets', []))} assets and "
            f"{len(library.get('channels', []))} channels for product {product_id}"
        )
        return library
    finally:
        db.close()
