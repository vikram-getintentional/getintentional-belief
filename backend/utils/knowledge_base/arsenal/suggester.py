# suggester.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math

try:
    from backend.utils.knowledge_base.arsenal.execution_arsenal_repository import (
        load_assets_from_arsenal_json,
        load_channels_from_arsenal_json,
    )
except Exception:
    # fallback path
    from backend.utils.execution_arsenal_repository import (  # type: ignore
        load_assets_from_arsenal_json,
        load_channels_from_arsenal_json,
    )

# --- helpers ---
def _tag_hit(qtags: List[str] | None, tags: List[str] | None) -> float:
    if not qtags or not tags:
        return 0.0
    q = set(t.strip().lower() for t in qtags if t)
    t = set(t.strip().lower() for t in tags if t)
    return len(q & t) / max(1, len(q))

def _soft_match(q: str, text: str) -> float:
    if not q or not text:
        return 0.0
    q = q.lower()
    text = text.lower()
    if q == text:
        return 1.0
    if q in text:
        return min(0.95, 0.6 + 0.35 * (len(q) / max(1, len(text))))
    return 0.0

def _concern_asset_fit(concern: Dict[str, Any], asset: Dict[str, Any]) -> float:
    # primary: tag overlap; secondary: text match on labels
    tag_fit = _tag_hit(concern.get("tags"), (asset.get("tags") or []))
    label_fit = max(
        _soft_match(concern.get("concern_label", ""), asset.get("name", "")),
        _soft_match(concern.get("label", ""), asset.get("name", "")),
    )
    return max(tag_fit, label_fit, 0.1)  # floor so we can rank

def _persona_channel_fit(persona: str, channel: Dict[str, Any]) -> float:
    # use channel.persona_fit if available else default heuristics
    pf = channel.get("persona_fit", {})
    if pf and persona in pf:
        return float(pf[persona])
    tags = channel.get("tags") or []
    # lightweight defaults
    if "email" in " ".join(tags).lower():
        return 0.6
    if "linkedin" in " ".join(tags).lower():
        return 0.65
    return 0.5

def suggest_asset_channel_combos(
    product_id: str,
    concerns: List[Dict[str, Any]],
    personas: List[str],
    limit_per_concern: int = 3,
) -> List[Dict[str, Any]]:
    """Return a ranked list of {asset, channel, fitScore, engagementScore, concernsAddressed}"""
    assets = load_assets_from_arsenal_json(product_id)
    channels = load_channels_from_arsenal_json(product_id)

    out: List[Dict[str, Any]] = []
    if not assets or not channels:
        return out

    for c in concerns or []:
        # pick top-N assets for this concern
        ranked_assets = sorted(
            assets,
            key=lambda a: _concern_asset_fit(c, a),
            reverse=True,
        )[:limit_per_concern]

        for a in ranked_assets:
            for p in personas or []:
                # pick the single best channel for persona for each asset
                best_ch = max(channels, key=lambda ch: _persona_channel_fit(p, ch))
                asset_fit = _concern_asset_fit(c, a)
                channel_fit = _persona_channel_fit(p, best_ch)
                out.append({
                    "asset": a.get("name") or a.get("id") or "Asset",
                    "assetId": a.get("id"),
                    "channel": best_ch.get("name") or best_ch.get("id") or "Channel",
                    "channelId": best_ch.get("id"),
                    "fitScore": round(asset_fit, 3),
                    "engagementScore": round(channel_fit, 3),
                    "concernsAddressed": [c.get("concern_label") or c.get("label") or ""],
                })

    # rank combos by combined fitness first, then engagement
    out.sort(key=lambda r: (r["fitScore"] * 0.7 + r["engagementScore"] * 0.3), reverse=True)
    # mildly dedup by (asset, channel, concern label)
    seen = set()
    deduped = []
    for r in out:
        key = (r["asset"], r["channel"], tuple(r["concernsAddressed"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped
