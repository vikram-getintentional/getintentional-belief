# scripts/rebuild_arsenal_from_engagements.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import get_db
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.engagement_service import (
    _attach_arsenal_references,
    _ensure_engagement_columns,
)


def _normalize_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
    return {}


def _normalize_account_meta(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, dict):
                return decoded
        except Exception:
            pass
    return None


def rebuild_arsenal(
    product_id: str | None = None,
    *,
    batch_size: int = 500,
    dry_run: bool = False,
) -> None:
    _ensure_engagement_columns()
    db = next(get_db())
    created_assets: set[str] = set()
    created_channels: set[str] = set()
    processed = 0
    try:
        query = db.query(TargetAccountEngagement).order_by(TargetAccountEngagement.timestamp_dt.asc())
        if product_id:
            query = query.filter(TargetAccountEngagement.product_id == product_id)
        query = query.yield_per(batch_size)
        for row in query:
            asset_tracker = {"created": set(), "pending": set()}
            channel_tracker = {"created": set(), "pending": set()}
            payload = _normalize_payload(row.payload)
            account_meta = _normalize_account_meta(row.account_meta)
            _attach_arsenal_references(
                db,
                product_id=row.product_id,
                engagement_payload=payload,
                engagement_row=row,
                asset_tracker=asset_tracker,
                channel_tracker=channel_tracker,
                account_meta=account_meta,
            )
            created_assets.update(asset_tracker["created"])
            created_channels.update(channel_tracker["created"])
            processed += 1
            if processed % batch_size == 0:
                db.flush()
        if dry_run:
            db.rollback()
            print("Dry-run mode; no changes were persisted.")
        else:
            db.commit()
        print(
            f"Rebuilt arsenal entries for {processed} engagements "
            f"({len(created_assets)} unique assets, {len(created_channels)} unique channels created or updated)."
        )
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild Arsenal catalog from existing engagements in the database."
    )
    parser.add_argument("--product-id", dest="product_id", help="Optional product_id to scope rebuild.")
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=500,
        help="How many engagements to fetch per database batch.",
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Walk engagements without committing the resulting metadata.",
    )
    args = parser.parse_args()
    rebuild_arsenal(product_id=args.product_id, batch_size=args.batch_size, dry_run=args.dry_run)
