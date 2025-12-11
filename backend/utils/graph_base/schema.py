# backend/utils/graph_base/schema.py
NODE_TYPES = {
    "product",
    "capability",
    "pain",
    "job",
    "persona",
    "canonical_persona",
    "persona_variant",
    "perceived_metric",
    "pain_trigger",
    "zmot_event",
    "observable_moment",
    "keyword",
    "archetype",
}

EDGES = {
    "OFFERS": "offers",
    "SOLVES": "solves",                 # capability -> pain, job -> pain
    "FELT_IN": "felt_in",               # pain -> job
    "PERFORMED_BY": "performed_by",     # job -> persona
    "EXPRESSED_AS": "expressed_as",     # pain -> perceived_metric
    "TRIGGERED_BY": "triggered_by",     # pain -> pain_trigger
    "ACCELERATED_BY": "accelerated_by", # pain_trigger -> zmot_event
    "OBSERVED_IN": "observed_in",       # zmot_event -> observable_moment
    "ASSOCIATED_WITH": "associated_with", # zmot_event -> keyword
    "PREVALENT_IN": "prevalent_in",     # pain_trigger -> archetype
    "RELEVANT_TO": "relevant_to",       # archetype -> zmot_event
    "HAS_VARIANT": "has_variant",       # canonical_persona -> persona_variant
    "VARIANT_OF": "variant_of",         # persona_variant -> canonical_persona
}

# === Frontier expansion contracts (add-only) ===

# What successor types each node type should produce in a one-hop expand
NEXT_HOPS: dict[str, list[str]] = {
    "capability": ["pain"],
    "pain": ["job", "perceived_metric", "pain_trigger"],
    "pain_trigger": ["attribute_value", "zmot_event"],
    "attribute_value": ["zmot_event"],  # optional fan-out
    "zmot_event": ["observable_moment", "keyword"],
    "job": ["persona", "canonical_persona", "pain"],  # canonical personas replace legacy personas
    "persona": [],
    "canonical_persona": ["persona_variant", "pain"],
    "persona_variant": [],
    "perceived_metric": [],
    "observable_moment": [],
    "keyword": [],
}

# Minimal raw/canonical fields you want to pass to the LLM per type
RAW_FIELDS_BY_TYPE: dict[str, list[str]] = {
    "capability": ["name", "description"],
    "pain": ["description", "pain_source"],
    "job": ["description"],
    "persona": ["title", "department", "seniority", "linkedin_profiles"],
    "canonical_persona": [
        "label",
        "description",
        "core_jobs",
        "supporting_jobs",
        "core_pains",
        "example_titles",
        "typical_departments",
    ],
    "persona_variant": [
        "title",
        "department",
        "seniority",
        "team_context",
        "crm_person_ids",
    ],
    "perceived_metric": ["metric"],
    "metric": ["metric"],
    "pain_trigger": ["attribute"],
    "attribute_value": ["dimension", "name"],
    "zmot_event": ["event"],
    "observable_moment": ["text"],
    "keyword": ["text"],
}
