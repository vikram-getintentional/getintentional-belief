# backend/utils/graph_base/schema.py
NODE_TYPES = {
    "product", "capability", "pain", "job", "persona",
    "perceived_metric", "pain_trigger", "zmot_event",
    "observable_moment", "keyword", "archetype"
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
}

# === Frontier expansion contracts (add-only) ===

# What successor types each node type should produce in a one-hop expand
NEXT_HOPS: dict[str, list[str]] = {
    "capability": ["pain"],
    "pain": ["job", "perceived_metric", "pain_trigger"],
    "pain_trigger": ["attribute_value", "zmot_event"],
    "attribute_value": ["zmot_event"],  # optional fan-out
    "zmot_event": ["observable_moment", "keyword"],
    "job": ["persona", "pain"],         # "pain" here = solves
    "persona": [],
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
    "perceived_metric": ["metric"],
    "metric": ["metric"],
    "pain_trigger": ["attribute"],
    "attribute_value": ["dimension", "name"],
    "zmot_event": ["event"],
    "observable_moment": ["text"],
    "keyword": ["text"],
}
