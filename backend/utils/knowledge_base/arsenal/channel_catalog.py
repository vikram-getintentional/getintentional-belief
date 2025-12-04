from __future__ import annotations

from typing import Dict, Optional, Tuple

from backend.utils.knowledge_base.arsenal.db_models import ChannelDelivery, ChannelType


CANONICAL_CHANNELS: Dict[str, Dict[str, object]] = {
    "email": {
        "slug": "email",
        "name": "Email",
        "channel_type_label": "Email",
        "channel_type": ChannelType.EMAIL,
        "delivery_mode": ChannelDelivery.OWNED,
    },
    "sdr_outbound": {
        "slug": "sdr_outbound",
        "name": "SDR / Sales Outbound",
        "channel_type_label": "SDR / Sales Outbound",
        "channel_type": ChannelType.SALES_OUTREACH,
        "delivery_mode": ChannelDelivery.OWNED,
    },
    "webinar_platform": {
        "slug": "webinar_platform",
        "name": "Webinar Platform",
        "channel_type_label": "Webinar Platform",
        "channel_type": ChannelType.WEBINAR_PLATFORM,
        "delivery_mode": ChannelDelivery.OWNED,
    },
    "field_event": {
        "slug": "field_event",
        "name": "Field / Community Event",
        "channel_type_label": "Field / Community Event",
        "channel_type": ChannelType.FIELD_EVENT,
        "delivery_mode": ChannelDelivery.OWNED,
    },
    "website": {
        "slug": "website",
        "name": "Website / Resource Hub",
        "channel_type_label": "Website / Resource Hub",
        "channel_type": ChannelType.WEBSITE,
        "delivery_mode": ChannelDelivery.OWNED,
    },
    "paid_social_linkedin": {
        "slug": "paid_social_linkedin",
        "name": "Paid Social — LinkedIn",
        "channel_type_label": "Paid Social — LinkedIn",
        "channel_type": ChannelType.PAID_SOCIAL,
        "delivery_mode": ChannelDelivery.PAID,
    },
    "organic_social": {
        "slug": "organic_social",
        "name": "Organic Social",
        "channel_type_label": "Organic Social",
        "channel_type": ChannelType.ORGANIC_SOCIAL,
        "delivery_mode": ChannelDelivery.EARNED,
    },
    "community": {
        "slug": "community",
        "name": "Community / Slack",
        "channel_type_label": "Community / Slack",
        "channel_type": ChannelType.COMMUNITY,
        "delivery_mode": ChannelDelivery.EARNED,
    },
    "in_app": {
        "slug": "in_app",
        "name": "In-App",
        "channel_type_label": "In-App",
        "channel_type": ChannelType.IN_APP,
        "delivery_mode": ChannelDelivery.IN_PRODUCT,
    },
    "direct_mail": {
        "slug": "direct_mail",
        "name": "Direct Mail / Gifting",
        "channel_type_label": "Direct Mail / Gifting",
        "channel_type": ChannelType.DIRECT_MAIL,
        "delivery_mode": ChannelDelivery.PAID,
    },
    "partner": {
        "slug": "partner",
        "name": "Partner Co-Marketing",
        "channel_type_label": "Partner Co-Marketing",
        "channel_type": ChannelType.PARTNER,
        "delivery_mode": ChannelDelivery.EARNED,
    },
}

CHANNEL_SYNONYMS: Dict[str, str] = {
    "newsletter": "email",
    "drip_email": "email",
    "nurture_email": "email",
    "sales_call": "sdr_outbound",
    "sales_outreach": "sdr_outbound",
    "outbound_call": "sdr_outbound",
    "phone_call": "sdr_outbound",
    "call": "sdr_outbound",
    "webinar": "webinar_platform",
    "virtual_event": "webinar_platform",
    "community_event": "field_event",
    "field_event": "field_event",
    "event": "field_event",
    "download": "website",
    "content download": "website",
    "landing_page": "website",
    "website": "website",
    "paid_social": "paid_social_linkedin",
    "linkedin_paid": "paid_social_linkedin",
    "linkedin": "organic_social",
    "slack": "community",
    "community": "community",
}


def resolve_canonical_channel(label: Optional[str]) -> Tuple[str, Dict[str, object]]:
    """
    Returns (slug, metadata dict). If the label does not match the canon,
    slug will be slugified version of the label and metadata will only contain
    the inferred display name.
    """
    normalized = (label or "").strip().lower()
    slug = CHANNEL_SYNONYMS.get(normalized) or normalized
    if slug in CANONICAL_CHANNELS:
        return slug, CANONICAL_CHANNELS[slug]
    # fall back to slug derived from label, but keep display name
    fallback_name = label.strip() if isinstance(label, str) else "Custom Channel"
    slug = normalized or "custom-channel"
    return slug, {
        "slug": slug,
        "name": fallback_name,
        "channel_type_label": fallback_name,
        "channel_type": None,
        "delivery_mode": None,
    }
