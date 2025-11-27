from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from backend.database import SessionLocal
from backend.utils.knowledge_base.arsenal.db_models import ArsenalAsset, ArsenalChannel
from backend.utils.knowledge_base.arsenal.service import (
    get_or_create_asset,
    get_or_create_channel,
    update_asset_metadata,
    update_channel_metadata,
)


ASSET_CATEGORY_MAP: Dict[str, str] = {
    "webinar": "webinar",
    "blog": "blog_post",
    "blog_post": "blog_post",
    "blog_article": "blog_post",
    "article": "blog_post",
    "report": "report",
    "whitepaper": "report",
    "ebook": "ebook",
    "case_study": "case_study",
    "case-study": "case_study",
    "case study": "case_study",
    "demo": "demo",
    "product_demo": "demo",
    "press": "news_article",
    "pr": "news_article",
    "landing_page": "landing_page",
    "offer_landing": "landing_page",
    "calculator": "calculator",
    "community_event": "event",
    "event": "event",
    "email_sequence": "email_sequence",
    "documentation": "playbook",
    "academy": "playbook",
    "video_channel": "toolkit",
    "partner_page": "playbook",
    "community": "event",
    "newsletter": "email_sequence",
}


CHANNEL_TYPE_MAP: Dict[str, str] = {
    "email": "email",
    "owned": "website",
    "sales_outreach": "sales_outreach",
    "webinar_platform": "webinar_platform",
    "webinar": "webinar_platform",
    "paid_social": "paid_social",
    "paid_search": "paid_search",
    "retargeting": "display",
    "display": "display",
    "field_event": "field_event",
    "event": "field_event",
    "community": "community",
    "partner": "partner",
    "earned": "website",
    "video": "website",
    "audio": "website",
    "social": "organic_social",
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def heuristic_time_to_consume(days: Any) -> str | None:
    try:
        value = float(days)
    except (TypeError, ValueError):
        return None
    if value <= 7:
        return "short"
    if value <= 21:
        return "medium"
    return "long"


def infer_delivery(channel_type: str | None) -> str | None:
    if not channel_type:
        return None
    if channel_type in {"paid_social", "paid_search", "display"}:
        return "paid"
    if channel_type in {"email", "sales_outreach", "webinar_platform", "website", "organic_social"}:
        return "owned"
    if channel_type in {"community", "field_event"}:
        return "owned"
    if channel_type in {"partner"}:
        return "earned"
    return None


def load_assets(session, assets_json: Dict[str, Any]) -> Tuple[int, int]:
    inserted = 0
    skipped = 0

    for product_id, assets in assets_json.items():
        for asset in assets:
            name = asset.get("name")
            if not name:
                skipped += 1
                continue

            slug = ArsenalAsset.slug_for(f"{product_id}-{name}")

            db_asset, created = get_or_create_asset(
                session,
                product_id=product_id,
                name=name,
                defaults={"slug": slug},
            )

            format_raw = (asset.get("format") or "").strip().lower()
            category = ASSET_CATEGORY_MAP.get(format_raw)
            time_to_consume = heuristic_time_to_consume(asset.get("time_to_effect_days"))

            description_parts = []
            if asset.get("stage_fit"):
                description_parts.append(f"Stages: {', '.join(asset['stage_fit'])}")
            if asset.get("concern_tags"):
                description_parts.append(f"Concerns: {', '.join(asset['concern_tags'])}")
            if asset.get("persona_fit"):
                description_parts.append(f"Personas: {', '.join(asset['persona_fit'])}")
            evidence = asset.get("evidence")
            if evidence:
                description_parts.append(f"Evidence: {evidence}")
            description = "\n".join(description_parts) if description_parts else None

            update_asset_metadata(
                session,
                asset_id=db_asset.id,
                patch={
                    "category": category,
                    "content_type": None,
                    "time_to_consume": time_to_consume,
                    "depth": None,
                    "description": description,
                    "notes": asset.get("notes") or "",
                },
            )

            if created:
                inserted += 1
            else:
                skipped += 1

    return inserted, skipped


def load_channels(session, channels_json: Dict[str, Any]) -> Tuple[int, int]:
    inserted = 0
    skipped = 0

    for product_id, channels in channels_json.items():
        for channel in channels:
            name = channel.get("name")
            if not name:
                skipped += 1
                continue

            slug = ArsenalChannel.slug_for(f"{product_id}-{name}")

            db_channel, created = get_or_create_channel(
                session,
                product_id=product_id,
                name=name,
                defaults={"slug": slug},
            )

            type_raw = (channel.get("type") or "").strip().lower()
            channel_type = CHANNEL_TYPE_MAP.get(type_raw)
            delivery = infer_delivery(channel_type)

            reach = channel.get("reach_score")
            try:
                reach_value = float(reach) if reach is not None else None
            except (TypeError, ValueError):
                reach_value = None

            update_channel_metadata(
                session,
                channel_id=db_channel.id,
                patch={
                    "channel_type": channel_type,
                    "delivery_mode": delivery,
                    "reach_score_estimate": reach_value,
                    "notes": channel.get("cadence_hint") or "",
                },
            )

            if created:
                inserted += 1
            else:
                skipped += 1

    return inserted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill arsenal tables from JSON files.")
    parser.add_argument(
        "--assets",
        default="backend/utils/graph_base/graph_data/arsenal_json/assets.json",
        help="Path to assets JSON dump",
    )
    parser.add_argument(
        "--channels",
        default="backend/utils/graph_base/graph_data/arsenal_json/channels.json",
        help="Path to channels JSON dump",
    )
    args = parser.parse_args()

    if not os.path.exists(args.assets):
        raise SystemExit(f"Assets JSON not found at {args.assets}")
    if not os.path.exists(args.channels):
        raise SystemExit(f"Channels JSON not found at {args.channels}")

    assets_json = load_json(args.assets)
    channels_json = load_json(args.channels)

    session = SessionLocal()
    try:
        asset_inserted, asset_skipped = load_assets(session, assets_json)
        channel_inserted, channel_skipped = load_channels(session, channels_json)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    print(f"Assets inserted: {asset_inserted}, skipped (already existed): {asset_skipped}")
    print(f"Channels inserted: {channel_inserted}, skipped (already existed): {channel_skipped}")


if __name__ == "__main__":
    main()
