from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CanonicalEntry:
    id: str
    description: str
    keywords: List[str]
    base_score: float = 0.8


@dataclass(frozen=True)
class CanonicalVocabulary:
    pains: List[CanonicalEntry]
    jobs: List[CanonicalEntry]
    zmots: List[CanonicalEntry]
    aspirations: List[CanonicalEntry]
    emotions: List[CanonicalEntry]
    belief_levels: List[str]


def load_canonical_vocab() -> CanonicalVocabulary:
    pains = [
        CanonicalEntry(
            id="pain:sku_profit_unknown",
            description="Unclear SKU profitability",
            keywords=["profit", "margin", "sku", "profitability"],
        ),
        CanonicalEntry(
            id="pain:channel_profitability_reporting",
            description="Lack of visibility into channel profitability",
            keywords=["channel profitability", "channel profits", "channel reporting"],
        ),
    ]
    jobs = [
        CanonicalEntry(
            id="job:channel_profitability_reporting",
            description="Build reliable reporting on channel profitability",
            keywords=["reporting", "dashboard", "analysis", "visibility"],
        )
    ]
    zmots = [
        CanonicalEntry(
            id="zmot:scaling_without_visibility",
            description="Scaling without channel visibility",
            keywords=["scaling", "blind", "unknown performance"],
        )
    ]
    aspirations = [
        CanonicalEntry(
            id="asp:invest_in_channels_confidently",
            description="Invest in channels confidently",
            keywords=["confident", "invest in channels", "scaling safely"],
        )
    ]
    emotions = [
        CanonicalEntry(
            id="emotion:anxiety_about_scaling",
            description="Anxiety about scaling",
            keywords=["nervous", "anxious", "worried", "concerned"],
        )
    ]
    belief_levels = ["low", "medium", "high"]
    return CanonicalVocabulary(
        pains=pains,
        jobs=jobs,
        zmots=zmots,
        aspirations=aspirations,
        emotions=emotions,
        belief_levels=belief_levels,
    )
