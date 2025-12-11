from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import get_db
from backend.utils.crm_management.channel_attribution import attribute_channels_for_engagements
from backend.utils.crm_management.engagement_service import _ensure_engagement_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute engagement events to canonical channels.")
    parser.add_argument("--product-id", dest="product_id", help="Optional product_id (default: all products).")
    args = parser.parse_args()
    db = next(get_db())
    try:
        _ensure_engagement_columns()
        updated = attribute_channels_for_engagements(db, product_id=args.product_id)
        target = f"product {args.product_id}" if args.product_id else "all products"
        print(f"Attributed channels for {target}; updated {updated} engagement rows.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
