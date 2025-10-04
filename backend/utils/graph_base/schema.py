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