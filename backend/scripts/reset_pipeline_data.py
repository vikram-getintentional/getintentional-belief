from __future__ import annotations

import argparse
from typing import Optional

from sqlalchemy import text

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backend.database import SessionLocal
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.knowledge_base.arsenal.db_models import (
    ArsenalAsset,
    ArsenalChannel,
    AssetChannelImpact,
)


TARGET_ENGAGEMENT_TABLE = "target_account_engagements"
ARSENAL_ASSET_TABLE = "arsenal_assets"
ARSENAL_CHANNEL_TABLE = "arsenal_channels"
ARSENAL_IMPACT_TABLE = "asset_channel_impacts"
SHM_EPISODE_TABLE = "shm_episodes_meta"
SHM_STEP_TABLE = "shm_episode_steps"
SHM_UPDATE_TABLE = "shm_learning_updates"


def _delete_all(session, table_name: str) -> int:
    res = session.execute(text(f"DELETE FROM {table_name}"))
    return res.rowcount if hasattr(res, "rowcount") else 0


def purge_engagements(session) -> int:
    return session.query(TargetAccountEngagement).delete(synchronize_session=False)


def purge_incomplete_assets(session) -> int:
    asset_ids = [
        row.id
        for row in session.query(ArsenalAsset.id)
        .filter(ArsenalAsset.metadata_complete.is_(False))
        .all()
    ]
    channel_ids = [
        row.id
        for row in session.query(ArsenalChannel.id)
        .filter(ArsenalChannel.metadata_complete.is_(False))
        .all()
    ]

    impacts_deleted = 0
    if asset_ids or channel_ids:
        impacts_deleted = (
            session.query(AssetChannelImpact)
            .filter(
                AssetChannelImpact.asset_id.in_(asset_ids or [""]),
                AssetChannelImpact.channel_id.in_(channel_ids or [""]),
            )
            .delete(synchronize_session=False)
        )
    assets_deleted = (
        session.query(ArsenalAsset)
        .filter(ArsenalAsset.id.in_(asset_ids or [""]))
        .delete(synchronize_session=False)
    )
    channels_deleted = (
        session.query(ArsenalChannel)
        .filter(ArsenalChannel.id.in_(channel_ids or [""]))
        .delete(synchronize_session=False)
    )
    return impacts_deleted + assets_deleted + channels_deleted


def purge_all_assets_and_channels(session) -> int:
    impacts = _delete_all(session, ARSENAL_IMPACT_TABLE)
    assets = _delete_all(session, ARSENAL_ASSET_TABLE)
    channels = _delete_all(session, ARSENAL_CHANNEL_TABLE)
    return impacts + assets + channels


def purge_shm(session) -> int:
    updates = _delete_all(session, SHM_UPDATE_TABLE)
    steps = _delete_all(session, SHM_STEP_TABLE)
    episodes = _delete_all(session, SHM_EPISODE_TABLE)
    return updates + steps + episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset engagement + SHM pipeline data.")
    parser.add_argument(
        "--purge-assets",
        action="store_true",
        help="Delete ALL arsenal assets/channels (not only incomplete ones).",
    )
    parser.add_argument(
        "--purge-incomplete-assets",
        action="store_true",
        help="Remove only assets/channels with missing metadata.",
    )
    parser.add_argument(
        "--purge-engagements",
        action="store_true",
        help="Delete all rows in target_account_engagements.",
    )
    parser.add_argument(
        "--purge-shm",
        action="store_true",
        help="Delete all SHM episodes, steps, and learning updates.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )

    args = parser.parse_args()

    if not any(
        [
            args.purge_assets,
            args.purge_incomplete_assets,
            args.purge_engagements,
            args.purge_shm,
        ]
    ):
        print("Nothing to do. Specify at least one --purge flag.")
        return

    if not args.yes:
        print("This will permanently remove data from the local SQLite database.")
        print("Flags:")
        print("  purge engagements:", args.purge_engagements)
        print("  purge assets (all):", args.purge_assets)
        print("  purge assets (incomplete only):", args.purge_incomplete_assets)
        print("  purge shm episodes:", args.purge_shm)
        confirmation = input("Type 'yes' to continue: ").strip().lower()
        if confirmation not in {"y", "yes"}:
            print("Aborted.")
            return

    session = SessionLocal()
    try:
        if args.purge_engagements:
            deleted = purge_engagements(session)
            print(f"Deleted {deleted} engagement rows.")

        if args.purge_assets:
            deleted = purge_all_assets_and_channels(session)
            print(f"Deleted {deleted} arsenal asset/channel rows.")
        elif args.purge_incomplete_assets:
            deleted = purge_incomplete_assets(session)
            print(f"Deleted {deleted} incomplete arsenal rows (impacts + assets + channels).")

        if args.purge_shm:
            deleted = purge_shm(session)
            print(f"Deleted {deleted} SHM rows (episodes + steps + updates).")

        session.commit()
        print("Done.")
    except Exception as exc:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
