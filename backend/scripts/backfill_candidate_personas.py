# backend/scripts/backfill_candidate_personas.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.database import get_db
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.persona_normalization import normalize_persona_label


def backfill_candidate_labels(product_id: str | None = None, commit_every: int = 1000) -> None:
    """
    Recompute candidate_persona_label for existing TargetAccountEngagement rows.

    This script walks the engagements table, normalizes actor title/department/seniority
    with the same helper the ingestion path uses, and updates the column in place.
    """
    db = next(get_db())
    total_rows = 0
    updated_rows = 0
    normalized_rows = 0
    try:
        query = db.query(TargetAccountEngagement).order_by(TargetAccountEngagement.timestamp_dt.asc())
        if product_id:
            query = query.filter(TargetAccountEngagement.product_id == product_id)

        for row in query.yield_per(commit_every):
            total_rows += 1
            new_label = normalize_persona_label(
                row.actor_title,
                row.actor_department,
                row.actor_seniority,
            )
            existing_label = row.candidate_persona_label

            if new_label:
                normalized_rows += 1

            if new_label != existing_label:
                row.candidate_persona_label = new_label
                updated_rows += 1

            if total_rows % commit_every == 0:
                db.commit()

        db.commit()
        print(
            f"Processed {total_rows} engagements "
            f"(updated {updated_rows}, normalized labels found for {normalized_rows})."
        )
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill candidate_persona_label for existing engagements."
    )
    parser.add_argument(
        "--product-id",
        dest="product_id",
        help="Optional product_id to scope the backfill.",
    )
    parser.add_argument(
        "--commit-every",
        dest="commit_every",
        type=int,
        default=1000,
        help="Commit interval when iterating through rows (default: 1000).",
    )
    args = parser.parse_args()
    backfill_candidate_labels(product_id=args.product_id, commit_every=args.commit_every)
