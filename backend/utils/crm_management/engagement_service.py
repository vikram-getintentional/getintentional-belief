# backend/utils/crm_management/engagement_service.py
from typing import List, Dict, Any, Optional, Tuple, Set
from sqlalchemy import text
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from backend.database import get_db
from backend.utils.crm_management.engagement_models import TargetAccountEngagement
from backend.utils.crm_management.target_account_models_dto import TargetAccount
from backend.utils.crm_management.target_account_manager import get_account_by_id  # your ORM model
from backend.utils.knowledge_base.arsenal.db_models import ArsenalAsset, ArsenalChannel
from backend.utils.knowledge_base.arsenal.service import (
    get_or_create_asset,
    get_or_create_channel,
)
import traceback

def _to_conf(x) -> int:
    try:
        v = float(x if x is not None else 1.0)
    except Exception:
        v = 1.0
    v = max(0.0, min(1.0, v))
    return int(round(v * 100))

def _to_dt_z(ts: str) -> datetime:
    # accepts "...Z" or "...+00:00"
    ts = (ts or "").strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)

def _now_z():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = value.strip()
    return text or None

def _resolve_asset_labels(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns a tuple of (display_name, slug_seed) for asset creation.
    """
    raw_label = (
        payload.get("asset_label")
        or payload.get("asset_name")
        or payload.get("asset_title")
        or payload.get("raw_activity")
        or payload.get("asset_id")
    )
    raw_label = raw_label.strip() if isinstance(raw_label, str) else raw_label
    slug_seed = payload.get("asset_id") or raw_label
    slug_seed = slug_seed.strip() if isinstance(slug_seed, str) else slug_seed
    return raw_label, slug_seed

def _resolve_channel_label(payload: Dict[str, Any]) -> Optional[str]:
    return _normalize_text(payload.get("channel") or payload.get("source"))

def _attach_arsenal_references(
    db: Session,
    *,
    product_id: str,
    engagement_payload: Dict[str, Any],
    engagement_row: TargetAccountEngagement,
    asset_tracker: Dict[str, Set[str]],
    channel_tracker: Dict[str, Set[str]],
) -> None:
    asset_name, asset_slug_seed = _resolve_asset_labels(engagement_payload)
    asset_identifier = engagement_payload.get("asset_id")
    if asset_name or asset_identifier:
        defaults = {}
        if asset_slug_seed:
            defaults["slug"] = ArsenalAsset.slug_for(str(asset_slug_seed))
        defaults["created_from_engagement_id"] = getattr(engagement_row, "id", None)
        defaults.setdefault("description", engagement_payload.get("raw_activity"))
        asset, created = get_or_create_asset(
            db,
            product_id=product_id,
            name=asset_name or asset_identifier or "Asset",
            defaults=defaults,
            explicit_id=asset_identifier.strip() if isinstance(asset_identifier, str) and asset_identifier.strip() else None,
        )
        if engagement_row.asset_id != asset.id:
            engagement_row.asset_id = asset.id
        if asset_name and asset.name in (None, "", asset.id):
            asset.name = asset_name
        if not asset.description and engagement_payload.get("raw_activity"):
            asset.description = engagement_payload.get("raw_activity")
        asset.update_metadata_status()
        if created:
            asset_tracker["created"].add(asset.id)
        if not asset.metadata_complete:
            asset_tracker["pending"].add(asset.id)
        else:
            asset_tracker["pending"].discard(asset.id)
        if not asset.created_from_engagement_id and getattr(engagement_row, "id", None):
            asset.created_from_engagement_id = engagement_row.id
        db.add(asset)

    channel_name = _resolve_channel_label(engagement_payload)
    if channel_name:
        defaults = {
            "created_from_engagement_id": getattr(engagement_row, "id", None),
        }
        channel_identifier = f"channel:{ArsenalChannel.slug_for(channel_name)}"
        channel, created = get_or_create_channel(
            db,
            product_id=product_id,
            name=channel_name,
            defaults=defaults,
            explicit_id=channel_identifier,
        )
        if created:
            channel_tracker["created"].add(channel.id)
        if not channel.metadata_complete:
            channel_tracker["pending"].add(channel.id)
        else:
            channel_tracker["pending"].discard(channel.id)
        if not channel.created_from_engagement_id and getattr(engagement_row, "id", None):
            channel.created_from_engagement_id = engagement_row.id
        if not channel.notes and engagement_payload.get("raw_activity"):
            channel.notes = engagement_payload.get("raw_activity")
        channel.update_metadata_status()
        db.add(channel)
        # retain original friendly channel label in engagement row for UI
def save_and_update_account_engagements(product_id: str, engagements: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Persist a batch of engagements (append-only).
    Updates per-account denormalized counters (optional) on TargetAccount if those fields exist.
    """
    print("Saving engagements for product_id:", product_id, " with engagements:", engagements)
    db: Session = next(get_db())
    created = 0
    touched: set[str] = set()
    asset_tracker = {"created": set(), "pending": set()}
    channel_tracker = {"created": set(), "pending": set()}

    try:
        # preload accounts for this product
        
        for e in engagements or []:
            acc_id = e.get("account_id") or e.get("target_account_id")
            if not acc_id:
                print("Skipping engagement with missing account_id:", e)
                continue
            account = get_account_by_id(product_id, acc_id)
            if not account:
                print("Skipping engagement for unknown account_id:", acc_id)
                continue
            actor = e.get("actor") or {}
            ts_raw = (e.get("timestamp") or _now_z()).strip()
            row = TargetAccountEngagement(
                product_id=product_id,
                target_account_id=acc_id,
                timestamp=ts_raw,                 # raw string (optional)
                timestamp_dt=_to_dt_z(ts_raw),    # proper aware datetime
                source=(e.get("source") or "other"),
                channel=e.get("channel"),
                inferred=bool(e.get("inferred", False)),
                actor_name=(actor.get("name") or None),
                actor_title=(actor.get("title") or "").strip(),
                actor_department=(actor.get("department") or "").strip(),
                actor_seniority=(actor.get("seniority") or "").strip(),
                actor_confidence=_to_conf(actor.get("confidence")),
                raw_activity=(e.get("raw_activity") or ""),
                asset_id=e.get("asset_id"),
                payload=e,
            )
            print("Adding engagement row to db:", row)
            db.add(row)
            db.flush()
            _attach_arsenal_references(
                db,
                product_id=product_id,
                engagement_payload=e,
                engagement_row=row,
                asset_tracker=asset_tracker,
                channel_tracker=channel_tracker,
            )
            created += 1
            touched.add(acc_id)
            print("Engagement added to db")

        """
        # Optional: update denormalized counters if you added them to TargetAccount
        if touched and hasattr(TargetAccount, "engagement_count"):
            
            for acc_id in touched:
                count = db.query(TAE).filter(TAE.target_account_id == acc_id).count()
                last_ts = db.query(TAE.timestamp).filter(TAE.target_account_id == acc_id)\
                           .order_by(TAE.timestamp.desc()).first()
                acct = get_account_by_id(acc_id)
                acct.engagement_count = count
                if hasattr(acct, "last_engagement_at"):
                    acct.last_engagement_at = last_ts[0] if last_ts else None
        """
        db.commit()
        print("Engagements committed to db")
        print("Created engagements:", created)
        print("Touched accounts:", list(touched))
        return {
            "created": created,
            "touched_accounts": list(touched),
            "arsenal_assets_created": sorted(asset_tracker["created"]),
            "arsenal_channels_created": sorted(channel_tracker["created"]),
            "arsenal_assets_missing_meta": sorted(asset_tracker["pending"]),
            "arsenal_channels_missing_meta": sorted(channel_tracker["pending"]),
        }
    except Exception:
        db.rollback()
        print("❌ Commit failed:", repr(e))
        print(traceback.format_exc())
        raise
    finally:
        db.close()

def get_account_engagements(product_id: str, account_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    account = get_account_by_id(product_id, account_id)
    if not account:
        print("Account not found for engagements fetch:", account_id)
        return []
    db: Session = next(get_db())
    try:
        q = (db.query(TargetAccountEngagement)
               .filter(TargetAccountEngagement.product_id == product_id,
                       TargetAccountEngagement.target_account_id == account_id)
               .order_by(TargetAccountEngagement.timestamp.desc())
               .limit(limit))
        rows = q.all()
        out = []
        for r in rows:
            out.append({
                "id": r.id,
                "account_id": r.target_account_id,
                "timestamp": r.timestamp,
                "source": r.source,
                "channel": r.channel,
                "inferred": r.inferred,
                "actor": {
                    "name": r.actor_name,
                    "title": r.actor_title,
                    "department": r.actor_department,
                    "seniority": r.actor_seniority,
                    "confidence": round((r.actor_confidence or 100) / 100, 2),
                },
                "raw_activity": r.raw_activity,
                "asset_id": r.asset_id,
                "payload": r.payload,
            })
        return out
    finally:
        db.close()


def _infer_seniority_from_title(title: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in ["chief", "cxo", "cfo", "ceo", "coo", "cto", "cmo", "vp", "vice president", "head", "director"]):
        return "Executive"
    if any(k in t for k in ["manager", "lead", "owner"]):
        return "Manager"
    if "senior" in t or "sr" in t:
        return "Senior"
    return "Operator"


def _parse_ts(ts: str, default_dt: datetime) -> datetime:
    try:
        # accept "...Z" or "+00:00"
        if ts and isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        pass
    return default_dt


def save_engagements_bulk(product_id: str, account_id: str, engagements: List[Dict[str, Any]]) -> int:
    """
    OVERWRITE MODE:
      - Deletes ALL existing rows for (product_id, account_id)
      - Inserts the provided engagements fresh
      - If engagements == [], the account is left with 0 rows

    Expected engagement shape (per item):
    {
      "account_id": "...",                  # ignored/overwritten by `account_id` arg
      "timestamp": "2025-01-01T10:00:00Z",
      "actor": {
        "name": "...",
        "title": "...",
        "department": "...",
        "seniority": "...",                 # optional; inferred if missing
        "confidence": 1.0                   # optional; 0..1
      },
      "channel": "webinar|email|...",
      "source": "marketing|sales|cs|product|other",
      "raw_activity": "...",
      "asset_id": "optional-id",
      "inferred": false
    }
    """
    print("Saving bulk engagements for product_id:", product_id, " account_id:", account_id, " engagements:", engagements)
    db: Session = next(get_db())
    now = datetime.now(timezone.utc)
    asset_tracker = {"created": set(), "pending": set()}
    channel_tracker = {"created": set(), "pending": set()}

    try:
        # Guard: target account must exist for this product
        if not get_account_by_id(product_id, account_id):
            print(f"[warn] account_id {account_id} not found for product {product_id}; skipping")
            return 0

        # 1) Wipe existing rows
        print("Wiping existing engagements for account:", account_id)
        (
            db.query(TargetAccountEngagement)
              .filter(
                  TargetAccountEngagement.product_id == product_id,
                  TargetAccountEngagement.target_account_id == account_id,
              )
              .delete(synchronize_session=False)
        )

        # 2) Insert provided payload
        inserted = 0
        seen_ids = set()  # dedupe within the payload
        print("cleared db rows, now inserting new engagements")

        for e in engagements or []:
            print("Inserting engagement:", e)

            ts_raw: str = (e.get("timestamp") or "").strip()
            actor = e.get("actor") or {}
            name = (actor.get("name") or "").strip() or None
            title = (actor.get("title") or actor.get("role") or "").strip()   # tolerate legacy 'role'
            dept  = (actor.get("department") or "").strip()
            snr   = (actor.get("seniority") or "").strip() or _infer_seniority_from_title(title)
            conf  = actor.get("confidence", 1.0)

            raw_activity = (e.get("raw_activity") or "").strip()
            if not title or not dept or not raw_activity or not ts_raw:
                # skip malformed entry
                print("[skip] missing one of required fields: title/department/raw_activity/timestamp")
                continue

            ts_dt = _parse_ts(ts_raw, now)

            # dedupe within payload (since we wiped the DB already)
            dedupe_key = f"{account_id}|{ts_dt.isoformat()}|{name}|{title}|{dept}|{raw_activity}"
            if dedupe_key in seen_ids:
                print("[dedupe] skipping duplicate payload row:", dedupe_key)
                continue
            seen_ids.add(dedupe_key)

            print("Data to insert:", {
                "product_id": product_id,
                "target_account_id": account_id,
                "timestamp": ts_raw,
                "timestamp_dt": ts_dt,
                "source": (e.get("source") or "other"),
                "channel": (e.get("channel") or None),
                "inferred": bool(e.get("inferred") or False),
                "actor_name": name,
                "actor_title": title,
                "actor_department": dept,
                "actor_seniority": snr,
                "actor_confidence": int(max(0.0, min(1.0, float(conf))) * 100),
                "raw_activity": raw_activity,
                "asset_id": (e.get("asset_id") or None),
                "payload": e,
            })

            row = TargetAccountEngagement(
                product_id=product_id,
                target_account_id=account_id,
                timestamp=ts_raw,
                timestamp_dt=ts_dt,
                source=(e.get("source") or "other"),
                channel=(e.get("channel") or None),
                inferred=bool(e.get("inferred") or False),

                actor_name=name,
                actor_title=title,
                actor_department=dept,
                actor_seniority=snr,
                actor_confidence=int(max(0.0, min(1.0, float(conf))) * 100),

                raw_activity=raw_activity,
                asset_id=(e.get("asset_id") or None),
                payload=e,
            )
            db.add(row)
            db.flush()
            _attach_arsenal_references(
                db,
                product_id=product_id,
                engagement_payload=e,
                engagement_row=row,
                asset_tracker=asset_tracker,
                channel_tracker=channel_tracker,
            )
            inserted += 1
            print("Inserted engagement row:", row.id if hasattr(row, "id") else "<pending>")

        db.commit()
        if asset_tracker["created"] or channel_tracker["created"]:
            print(
                "[arsenal] bulk load created assets:",
                list(asset_tracker["created"]),
                "channels:",
                list(channel_tracker["created"]),
            )
        if asset_tracker["pending"] or channel_tracker["pending"]:
            print(
                "[arsenal] bulk load pending metadata assets:",
                list(asset_tracker["pending"]),
                "channels:",
                list(channel_tracker["pending"]),
            )
        return inserted

    except Exception as ex:
        print("❌ Bulk save failed with exception:", repr(ex))
        try:
            cols = db.execute(text("PRAGMA table_info(target_account_engagements);")).fetchall()
            print("[debug] target_account_engagements columns:", cols)
        except Exception as ex2:
            print("[debug] failed to introspect columns:", repr(ex2))
        db.rollback()
        raise
    finally:
        db.close()
