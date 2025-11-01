# stage_playbooks.py
from __future__ import annotations
from typing import Dict, List

# Simple, data-only defaults that suggester can use as priors.
STAGE_RECOMMENDATIONS: Dict[str, Dict[str, List[str]]] = {
    "pre_zmot": {
        "assets": ["Narrative memo", "Pattern-break post", "Teaser video"],
        "channels": ["LinkedIn Ads", "Founder Post", "PR / Earned"],
    },
    "zmot": {
        "assets": ["Trigger-based email", "Observability thread", "Problem story"],
        "channels": ["Email Nurture", "LinkedIn Ads", "Community"],
    },
    "problem_realization": {
        "assets": ["ROI one-pager", "Pain quant cheat-sheet", "Assessment quiz"],
        "channels": ["Outbound + SDR", "Email Nurture", "Partner Webinar"],
    },
    "discovery": {
        "assets": ["Reverse case study", "Deep dive demo", "Technical explainer"],
        "channels": ["Sales Assist", "Webinar", "Docs Hub"],
    },
    "barriers": {
        "assets": ["Security pack", "Implementation plan", "Champion deck"],
        "channels": ["Sales Assist", "Security Review", "Proof Workshop"],
    },
    "implementation": {
        "assets": ["Onboarding guide", "Runbook", "Success plan"],
        "channels": ["CSM Enablement", "Product Tours", "Help Center"],
    },
}
