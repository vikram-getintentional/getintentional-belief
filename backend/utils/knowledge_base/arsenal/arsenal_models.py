from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

Persona = Literal[
    "CFO", "CTO", "VPEng", "CIO", "OpsMgr", "SecurityLead",
    "FinanceOps", "DataLeader", "Procurement"
]
Stage = Literal["problem", "pain", "solution"]
Purpose = Literal["awareness", "pain_validation", "solution_proof"]
AssetFormat = Literal[
    "whitepaper", "case_study", "roi_deck", "tco_report", "checklist",
    "benchmark", "tech_deep_dive", "architecture", "blog_series", "webinar",
    "roundtable", "calculator", "email_sequence", "landing_page", "one_pager",
    "datasheet", "demo_video", "comparison_guide", "poc_offer", "analyst_report"
]
ChannelType = Literal[
    "email", "sales_outreach", "linkedin_ads", "google_display", "retargeting",
    "search", "webinar_platform", "industry_newsletter", "community",
    "website", "field_event", "sdr_call", "partner"
]

STAGE_TO_PURPOSE: Dict[Stage, Purpose] = {
    "problem":  "awareness",
    "pain":     "pain_validation",
    "solution": "solution_proof",
}

@dataclass
class Asset:
    id: str
    name: str
    format: AssetFormat
    stages: List[Stage]
    personas: List[Persona]
    purpose: Purpose
    evidence_strength: float
    production_days: int
    est_cost: float
    evergreen: bool = True
    industry_tags: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class Channel:
    id: str
    name: str
    type: ChannelType
    reach_score: float
    cost_index: float
    lead_days: int
    persona_fit: Dict[Persona, float]
    stage_fit: Dict[Stage, float]
    targeting: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class Play:
    asset_id: str
    channel_id: str
    score: float
    rationale: str