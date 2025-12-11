#!/usr/bin/env python3
"""
Backfill script to ensure every SHM episode carries account metadata.

Usage:
  python backend/scripts/backfill_episode_account_meta.py --product-id <product_id>
"""
from __future__ import annotations

import argparse
from typing import Optional

from backend.database import SessionLocal
from backend.super_models.shm.episode import ShmEpisode
from backend.utils.inference.belief_manager.journey.shm_service import (
    resolve_episode_account_meta,
)


def backfill(product_id: Optional[str] = None) -> None:
    session = SessionLocal()
    updated = 0
    processed = 0
    try:
        query = session.query(ShmEpisode)
        if product_id:
            query = query.filter(ShmEpisode.product_id == product_id)

        for episode in query.yield_per(250):
            processed += 1
            existing_meta = (
                episode.account_meta if isinstance(episode.account_meta, dict) else {}
            )
            resolved_meta, meta_source = resolve_episode_account_meta(
                session,
                product_id=episode.product_id,
                account_id=episode.account_id,
                provided_meta=existing_meta,
            )
            if not resolved_meta:
                continue
            if existing_meta != resolved_meta or meta_source:
                episode.account_meta = resolved_meta
                metrics = episode.metrics or {}
                if meta_source:
                    metrics["account_meta_source"] = meta_source
                episode.metrics = metrics
                updated += 1
        session.commit()
    finally:
        session.close()
    print(f"Processed {processed} episodes; updated {updated}.")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill SHM episodes with account metadata."
    )
    parser.add_argument(
        "--product-id",
        dest="product_id",
        help="Optional product ID to scope the backfill.",
    )
    args = parser.parse_args()
    backfill(product_id=args.product_id)


if __name__ == "__main__":
    main()
