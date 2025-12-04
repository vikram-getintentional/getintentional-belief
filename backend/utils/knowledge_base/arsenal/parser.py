from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


def _normalize_text(value: Optional[str]) -> Optional[str]:
  """
  Shared whitespace normalizer used by the engagement parser. Keeps this module
  decoupled from callers so we can reuse it inside ingestion as well as
  interactive tooling/tests.
  """
  if value is None:
    return None
  text = value.strip()
  return text or None


def _title_case(value: str) -> str:
  return " ".join(
    segment.capitalize() for segment in value.replace("_", " ").split() if segment
  )


def _extract_labeled_value(text: Optional[str], label: str) -> Optional[str]:
  if not text:
    return None
  pattern = re.compile(rf"{label}\s*[:\-]\s*(?P<value>[^|]+)", re.IGNORECASE)
  match = pattern.search(text)
  if not match:
    return None
  value = match.group("value").strip(" -—;")
  return value or None


def _extract_topic_from_text(text: Optional[str], keyword: str) -> Optional[str]:
  if not text:
    return None
  lower = text.lower()
  idx = lower.find(keyword)
  if idx == -1:
    return None
  remainder = text[idx + len(keyword):]
  remainder = re.split(r"[|;/]", remainder, maxsplit=1)[0]
  remainder = remainder.strip(" :-—")
  return remainder or None


def _extract_quoted_title(*texts: Optional[str]) -> Optional[str]:
  for text in texts:
    if not text:
      continue
    match = re.search(r"['\"]([^'\"]{3,})['\"]", text)
    if match:
      title = match.group(1).strip()
      if len(title) >= 3:
        return title
  return None


def _extract_action_and_activity(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
  raw_activity = _normalize_text(payload.get("raw_activity"))
  action = _normalize_text(payload.get("action"))
  activity = _normalize_text(payload.get("activity"))
  if raw_activity:
    action = action or _extract_labeled_value(raw_activity, "action")
    activity = activity or _extract_labeled_value(raw_activity, "activity")
  return action, activity, raw_activity


class ParsedField:
  __slots__ = ("label", "confidence", "source")

  def __init__(self, label: Optional[str], confidence: float, source: str):
    self.label = label
    self.confidence = confidence
    self.source = source

  def as_dict(self) -> Dict[str, Any]:
    return {"label": self.label, "confidence": self.confidence, "source": self.source}


@dataclass
class ParsedEngagementActivity:
  asset_name: ParsedField
  asset_slug_seed: Optional[str]
  asset_category: ParsedField
  channel_label: ParsedField
  activity_label: ParsedField
  parser_version: str = "multi_head_v1"

  def derived_metadata(self) -> Dict[str, Any]:
    return {
      "parser_version": self.parser_version,
      "asset_name": self.asset_name.as_dict(),
      "asset_category": self.asset_category.as_dict(),
      "channel": self.channel_label.as_dict(),
      "activity": self.activity_label.as_dict(),
    }


class MultiHeadEngagementParser:
  """
  Lightweight parser that imitates the multi-head classifier + retriever
  pipeline for now. It centralizes heuristics so we can swap its internal
  implementation with a model-backed service later without touching callers.
  """

  ACTIVITY_HINT_LIBRARY: Tuple[Dict[str, Any], ...] = (
    {
      "keywords": ("webinar", "virtual event", "on-demand webinar"),
      "asset_prefix": "Webinar",
      "asset_category": "webinar",
      "channel_label": "Webinar Platform",
      "activity": "attended webinar",
      "confidence": 0.82,
    },
    {
      "keywords": ("workshop", "roundtable", "fireside chat"),
      "asset_prefix": "Event",
      "asset_category": "event",
      "channel_label": "Field / Community Event",
      "activity": "joined workshop",
      "confidence": 0.78,
    },
    {
      "keywords": ("whitepaper", "white paper"),
      "asset_prefix": "Whitepaper",
      "asset_category": "report",
      "channel_label": "Website / Resource Hub",
      "activity": "downloaded asset",
      "confidence": 0.7,
    },
    {
      "keywords": ("ebook", "e-book"),
      "asset_prefix": "Ebook",
      "asset_category": "ebook",
      "channel_label": "Website / Resource Hub",
      "activity": "downloaded asset",
      "confidence": 0.7,
    },
    {
      "keywords": ("case study", "customer story"),
      "asset_prefix": "Case Study",
      "asset_category": "case_study",
      "channel_label": "Website / Resource Hub",
      "activity": "viewed reference",
      "confidence": 0.65,
    },
    {
      "keywords": ("blog", "article", "post"),
      "asset_prefix": "Blog",
      "asset_category": "blog_post",
      "channel_label": "Website / Resource Hub",
      "activity": "read article",
      "confidence": 0.6,
    },
    {
      "keywords": ("email", "newsletter"),
      "asset_prefix": "Email",
      "asset_category": "email_sequence",
      "channel_label": "Email",
      "activity": "engaged email",
      "confidence": 0.58,
    },
    {
      "keywords": ("demo", "product demo"),
      "asset_prefix": "Product Demo",
      "asset_category": "demo",
      "channel_label": "Product Demo",
      "activity": "joined demo",
      "confidence": 0.74,
    },
    {
      "keywords": ("sales call", "discovery call", "intro call"),
      "asset_prefix": "Sales Call",
      "asset_category": "sales_call",
      "channel_label": "SDR / Sales Outbound",
      "activity": "joined call",
      "confidence": 0.72,
    },
  )

  def __init__(self) -> None:
    pass

  def parse(self, payload: Dict[str, Any]) -> ParsedEngagementActivity:
    action, activity, raw_activity = _extract_action_and_activity(payload)
    inferred = self._infer_from_hints(activity, raw_activity)
    asset_name = inferred.get("asset_name")
    slug_seed = inferred.get("asset_slug_seed", asset_name)
    asset_category = inferred.get("asset_category")
    channel_label = inferred.get("channel_label")
    activity_label = inferred.get("activity_label") or self._fallback_activity_label(
      action, activity
    )
    confidence = inferred.get("confidence", 0.42)

    quoted_title = _extract_quoted_title(activity, raw_activity)
    if not asset_name and quoted_title:
      asset_name = quoted_title
      slug_seed = quoted_title
      confidence = max(confidence, 0.55)

    if not asset_name:
      if activity:
        base_label = activity
        if action:
          base_label = f"{_title_case(action)} {activity}"
        asset_name = base_label
        slug_seed = base_label
      elif action and raw_activity and len(raw_activity) <= 160:
        base_label = f"{_title_case(action)} — {raw_activity}"
        asset_name = base_label
        slug_seed = base_label
      else:
        asset_name = payload.get("asset_label") or payload.get("asset_name") or "Engagement Asset"
        slug_seed = asset_name
        confidence = 0.35

    if not asset_category and inferred.get("asset_category_from_payload"):
      asset_category = inferred["asset_category_from_payload"]

    return ParsedEngagementActivity(
      asset_name=ParsedField(asset_name, confidence, inferred.get("source", "heuristic")),
      asset_slug_seed=slug_seed,
      asset_category=ParsedField(asset_category, confidence, inferred.get("source", "heuristic")),
      channel_label=ParsedField(channel_label, confidence, inferred.get("source", "heuristic")),
      activity_label=ParsedField(activity_label, 0.5 if activity_label else 0.2, "heuristic"),
    )

  def _fallback_activity_label(self, action: Optional[str], activity: Optional[str]) -> Optional[str]:
    if activity and action:
      return f"{action.strip()} {activity.strip()}".strip()
    if activity:
      return activity
    if action:
      return action
    return None

  def _infer_from_hints(
    self,
    activity: Optional[str],
    raw_activity: Optional[str],
  ) -> Dict[str, Any]:
    search_space_parts: Sequence[str] = [part for part in (activity, raw_activity) if part]
    search_space = " | ".join(search_space_parts).lower()
    result: Dict[str, Any] = {}
    for hint in self.ACTIVITY_HINT_LIBRARY:
      matched_keyword = next((kw for kw in hint["keywords"] if kw in search_space), None)
      if not matched_keyword:
        continue
      topic = (
        _extract_topic_from_text(activity, matched_keyword)
        or _extract_topic_from_text(raw_activity, matched_keyword)
      )
      display_name = hint["asset_prefix"]
      if topic:
        display_name = f"{hint['asset_prefix']} — {topic}"
      result.update(
        {
          "asset_name": display_name,
          "asset_slug_seed": display_name,
          "asset_category": hint.get("asset_category"),
          "channel_label": hint.get("channel_label"),
          "activity_label": hint.get("activity"),
          "confidence": hint.get("confidence", 0.65),
          "source": "multi_head_classifier",
        }
      )
      break
    if not result:
      payload_log = activity or raw_activity
      if not payload_log:
        return result
      lowered = payload_log.lower()
      if "blog" in lowered or "article" in lowered:
        result["asset_category_from_payload"] = "blog_post"
      elif "report" in lowered or "research" in lowered:
        result["asset_category_from_payload"] = "report"
    return result


_PARSER = MultiHeadEngagementParser()


def parse_engagement_activity(payload: Dict[str, Any]) -> ParsedEngagementActivity:
  """
  Public helper that callers can use without instantiating the parser manually.
  """
  return _PARSER.parse(payload or {})
