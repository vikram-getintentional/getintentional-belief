# scripts/rebuild_arsenal_from_engagements.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import get_db
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.engagement_service import (
    _attach_arsenal_references,
    _ensure_engagement_columns,
)


def rebuild_arsenal(product_id: str | None = None) -> None:
    _ensure_engagement_columns()
    db = next(get_db())
    created_assets: set[str] = set()
    created_channels: set[str] = set()
    try:
        query = db.query(TargetAccountEngagement).order_by(TargetAccountEngagement.timestamp_dt.asc())
        if product_id:
            query = query.filter(TargetAccountEngagement.product_id == product_id)
        engagements = query.all()
        for row in engagements:
            asset_tracker = {"created": set(), "pending": set()}
            channel_tracker = {"created": set(), "pending": set()}
            _attach_arsenal_references(
                db,
                product_id=row.product_id,
                engagement_payload=row.payload or {},
                engagement_row=row,
                asset_tracker=asset_tracker,
                channel_tracker=channel_tracker,
                account_meta=None,
            )
            created_assets.update(asset_tracker["created"])
            created_channels.update(channel_tracker["created"])
        db.commit()
        print(f"Rebuilt arsenal entries. Assets created: {len(created_assets)}, channels created: {len(created_channels)}")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild Arsenal catalog from engagements.")
    parser.add_argument("--product-id", dest="product_id", help="Optional product_id to scope rebuild.")
    args = parser.parse_args()
    rebuild_arsenal(product_id=args.product_id)
