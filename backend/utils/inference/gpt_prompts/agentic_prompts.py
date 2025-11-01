# agentic_prompts.py
# Build-only helpers. DO NOT call the LLM here.
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json


def _fmt(obj: Any) -> str:
    # Compact but readable JSON for inclusion inside prompts
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), indent=2)


LABEL_GUIDE = """
Label guides (use exactly these strings; DO NOT return numbers):
- relevance_label: Critical | Core | Supportive | Ancillary | Out-of-scope
  Meaning: how contextually relevant TARGET is to SOURCE (how expected TARGET is, given SOURCE).
- likelihood_label: Essential | Expected | Common | Rare | Unlikely
  Meaning: how likely SOURCE is to occur given TARGET occurs.

Special:
- boost_label: Very High | High | Medium | Low | Negligible
  Include ONLY for pain_trigger → zmot_event (how much the ZMOT accelerates the trigger).
""".strip()


# ---------------------------------------------------------------------------
# HOP 0: capability -> pains -> felt_in (jobs/personas) (+ metrics, triggers)
# ---------------------------------------------------------------------------
def build_hop0_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    capability_ids: List[str],
    capability_context: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    For each product capability, map pains, felt_in jobs, personas, perceived_metrics, and pain_triggers.
    All edges MUST include relevance_label and likelihood_label (labels only, not numbers).
    """
    caps_payload = capability_context if capability_context else capability_ids

    return f"""
        You are an expert in B2B job architecture.

        {LABEL_GUIDE}

        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Capabilities: {_fmt(caps_payload)}

        Density & coverage rules:
        - Prefer many-to-many mappings; avoid one-to-one unless truly unique.
        - Dedupe/merge near-synonyms; keep causal/temporal order; avoid cycles.
        - Each capability → ≥3 pains; each pain → ≥2 jobs; each job → 2–3 personas and 2–3 solving pains.

        Instructions:
        1) For each capability, infer ≥3 concrete pains it directly solves.
        For each pain include:
        - pain (one sentence, specific workflow inefficiency)
        - relevance_label (to capability) & likelihood_label (pain drives usage of capability)
        - perceived_metrics: ≥2 items with relevance_label & likelihood_label
        - pain_triggers: 3–5 normalized, lemmatized nouns (no units/values), each with relevance_label & likelihood_label
        - felt_in_jobs: 2–3 upstream jobs where this pain is experienced
            For each job:
            - job_to_be_done (one sentence)
            - relevance_label (pain relevance to job) & likelihood_label (job likely to yield pain)
            - personas: 2–3 typical titles with department, seniority, and both labels
            - solving_pains: 2–3 upstream business pains (distinct, earlier-in-time than the job and original pain), each with:
                - pain (one sentence), relevance_label, likelihood_label
                - perceived_metrics: ≥2 with labels
                - pain_triggers: 3–5 normalized nouns with labels

        Hard constraints:
        - Treat each capability independently.
        - Keep language organization-realistic; be specific; no vague filler.
        - All results must be relevant to the given product/domain/industry.
        - Return STRICT JSON array only. No commentary.

        Return STRICT JSON array:
        [{{
        "capability_id": "<must match an input capability_id>",
        "pains": [{{
            "pain": "string",
            "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
            "likelihood_label": "Essential|Expected|Common|Rare|Unlikely",
            "pain_triggers": [{{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
            "perceived_metrics": [{{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
            "felt_in_jobs": [{{
            "job_to_be_done":"string",
            "relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope",
            "likelihood_label":"Essential|Expected|Common|Rare|Unlikely",
            "personas":[{{"title":"string","department":"string","seniority":"Junior|Operator|Manager|Senior|Executive","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
            "solving_pains":[{{
                "pain":"string",
                "relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope",
                "likelihood_label":"Essential|Expected|Common|Rare|Unlikely",
                "pain_triggers":[{{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
                "perceived_metrics":[{{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}]
            }}]
            }}]
        }}]
        }}]
        ONLY JSON.
        """.strip()


# -------------------------------------------------------
# Pain source classifier: terminal vs non-terminal anchor
# -------------------------------------------------------
def build_pain_source_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    pain_contexts: List[Dict[str, Any]],
) -> str:
    """
    Classify pains as "terminal" (external/strategic) vs "non-terminal" (internal/operational).
    """
    return f"""
        You classify whether a business pain is "terminal" or "non-terminal" to the organization.

        Definitions (concise):
        - non-terminal: internal/operational; has plausible upstream jobs where pain is felt; day-to-day workflow issue.
        - terminal: external or strategic (market, customers, regulators) OR major top-down directive (IPO, M&A, market entry) not tied to routine operations.

        Heuristics:
        - Default to non-terminal unless you are confident there are no plausible upstream jobs.
        - Sudden, externally-triggered pains → likely terminal; gradual, process-bound pains → likely non-terminal.

        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Pains: {_fmt(pain_contexts)}

        Rules:
        - Treat each pain independently. Keep language organization-realistic.
        - Return STRICT JSON only. No commentary.

        Return STRICT JSON array:
        [{{
        "pain_id": "<from input>",
        "pain_source":"terminal|non-terminal"
        }}]
        ONLY JSON.
        """.strip()


# ---------------------------------------------------------------------------
# HOP+: infer felt_in (jobs/personas) & upstream solving pains for each pain
# ---------------------------------------------------------------------------
def build_hop_plus_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    upstream_pain_contexts: List[Dict[str, Any]],
) -> str:
    """
    For given organizational pains, infer where each pain is felt (jobs/personas)
    and the upstream pains those jobs exist to solve.
    All edges MUST include relevance_label and likelihood_label.
    """

    return f"""
        You are an expert in B2B org design for {domain}.

        {LABEL_GUIDE}

        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Internal pains (with anchors): {_fmt(upstream_pain_contexts)}

        Coverage & quality:
        - Prefer many-to-many; dedupe near-synonyms; keep causal/temporal order; avoid cycles.
        - For each input pain: 1–3 felt_in_jobs; each job: ≥1 persona; each job: 1–3 upstream solving pains (distinct, causally earlier than the job and original pain).

        Instructions per pain:
        1) felt_in_jobs (1–3): upstream jobs where this pain is experienced.
        For each job: job_to_be_done, relevance_label, likelihood_label.
        2) Personas (per job, 1–3): title, department, seniority, relevance_label, likelihood_label.
        3) Solving pains (per job, 1–3): concrete upstream organizational inefficiencies (not paraphrases), each with:
        - pain (one sentence), relevance_label, likelihood_label
        - pain_triggers: 3–5 normalized nouns with labels
        - perceived_metrics: ≥2 with labels

        Guardrails:
        - Use only provided context; keep responses tightly relevant to product/domain/industry.
        - Return STRICT JSON only. No commentary.

        Return STRICT JSON array:
        [{{
        "original_pain_id":"<from input>",
        "felt_in_jobs":[{{
            "job_to_be_done":"string",
            "relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope",
            "likelihood_label":"Essential|Expected|Common|Rare|Unlikely",
            "personas":[{{"title":"string","department":"string","seniority":"Junior|Operator|Manager|Senior|Executive","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
            "solving_pains":[{{
            "pain":"string",
            "relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope",
            "likelihood_label":"Essential|Expected|Common|Rare|Unlikely",
            "pain_triggers":[{{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
            "perceived_metrics":[{{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}]
            }}]
        }}]
        }}]
        ONLY JSON.
        """.strip()


# ---------------------------------------------------------------------------
# Archetype ATTRIBUTES relevance matrix (no archetype nodes)
# ---------------------------------------------------------------------------
def build_archetypes_relevance_matrix(
    product_summary: str,
    domain: str,
    industry: str,
    trigger_contexts: List[Dict[str, Any]],
    attributes_dict: Dict[str, List[str]],
    top_n_industries: int = 5,
    allow_industry_extras: bool = True,
    max_extra_industries: int = 5,
) -> str:
    """
    Returns a prompt asking the LLM to estimate LABELS (no numerics) for
    PainTrigger × {industry, revenue_range, employee_range, funding_stage, geography}.
    """

    # Validate presence of required enums
    required_keys = ["revenue_range", "employee_range", "funding_stage", "geography"]
    for k in required_keys:
        if k not in attributes_dict or not attributes_dict[k]:
            raise ValueError(f"attributes_dict must include non-empty list for '{k}'")
    
    # Compose guardrail text for industry
    industry_rule = (
        f"• industry (3–5 items) — use provided list if non-empty; "
        f"you MAY add up to {max_extra_industries} additional industries not in the list if they are a strong fit; "
        f"set is_new=true on those extras"
        if allow_industry_extras else
        f"• industry (3–5 items) — use provided list if non-empty; else infer top-{top_n_industries}"
    )

    return f"""
        You are a senior B2B go-to-market analyst.
        Return ONLY valid JSON. No prose. No markdown.

        {LABEL_GUIDE}

        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Pain triggers (use IDs exactly): {_fmt(trigger_contexts)}
        - Permissible attribute values (MUST select from these lists exactly;  do not invent values EXCEPT for industry extras as noted below):
          {_fmt(attributes_dict)}

        Task:
        For EACH pain trigger, estimate RELEVANCE and LIKELIHOOD LABELS across:
        • {industry_rule}
        • revenue_range (score ALL provided options)
        • employee_range (score ALL provided options)
        • funding_stage (score ALL provided options)
        • geography (score ALL provided options)

        Return STRICT JSON object:
        {{
          "pain_triggers": [
            {{
              "pain_trigger_id": "<id from input>",
              "industry":       [{{"name":"string","relevance":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood":"Essential|Expected|Common|Rare|Unlikely","is_new":true}}],
              "revenue_range":  [{{"name":"string","relevance":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood":"Essential|Expected|Common|Rare|Unlikely"}}],
              "employee_range": [{{"name":"string","relevance":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood":"Essential|Expected|Common|Rare|Unlikely"}}],
              "funding_stage":  [{{"name":"string","relevance":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood":"Essential|Expected|Common|Rare|Unlikely"}}],
              "geography":      [{{"name":"string","relevance":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood":"Essential|Expected|Common|Rare|Unlikely"}}]
            }}
          ],
          "notes": ["optional, concise"]
        }}
        ONLY JSON.
    """.strip()



# ---------------------------------------------------------------------------
# ZMOT discovery per pain trigger with per-attribute boosts (labels only)
# ---------------------------------------------------------------------------
def build_zmot_for_triggers_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    trigger_contexts: List[Dict[str, Any]],
    attributes_dict: Dict[str, List[str]],
    top_n_industries: int = 5,
    max_events_per_trigger: int = 5,
) -> str:
    """
    Returns a prompt to extract ZMOT events per pain trigger, with:
      • observable_moments & trigger_keywords (with relevance/likelihood labels),
      • a FULL per-dimension boosts array using LABELS ONLY,
      • and a REQUIRED boost_label on pain_trigger → zmot_event edges.
    """

    required_keys = ["revenue_range", "employee_range", "funding_stage", "geography"]
    for k in required_keys:
        if k not in attributes_dict or not attributes_dict[k]:
            raise ValueError(f"attributes_dict must include non-empty list for '{k}'")

    return f"""
        You are an expert in B2B org design and external triggers in {domain}/{industry}.
        Return ONLY valid JSON. No prose. No markdown.

        {LABEL_GUIDE}

        Context:
          - Product value proposition: {product_summary}
          - Product domain: {domain}
          - Product industry: {industry}
          - Pain triggers (use IDs exactly): {_fmt(trigger_contexts)}
          - Permissible attribute values (MUST use exactly these values): {_fmt(attributes_dict)}

        Task:
        For EACH pain trigger, infer up to {max_events_per_trigger} specific, observable ZMOT events
        that accelerate or intensify the trigger.

        Requirements:
          - Keep events concrete and externally observable (pricing change, audit, layoffs, M&A, region entry, vendor deprecation, leadership change, etc.).
          - Provide 3–6 observable_moments and 6–12 trigger_keywords per event (each with relevance_label & likelihood_label).
          - For **boosts**:
              · industry: 3–5 items — use provided list or infer top-{top_n_industries} if empty
              · revenue_range: score ALL provided values
              · employee_range: score ALL provided values
              · funding_stage: score ALL provided values
              · geography: score ALL provided values
          - event_id = slug(trigger_event) (lowercase, hyphenated).
          - Reuse event_id across triggers if the name matches exactly.

        Return STRICT JSON only:
        {{
          "zmot": [
            {{
              "pain_trigger_id": "<id from input>",
              "events": [
                {{
                  "event_id": "string-slug",
                  "trigger_event": "short, specific noun phrase",
                  "edge_scores": {{
                    "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
                    "likelihood_label": "Essential|Expected|Common|Rare|Unlikely",
                    "boost_label": "Very High|High|Medium|Low|Negligible"
                  }},
                  "observable_moments": [
                    {{"observable_moment": {{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}}}
                  ],
                  "trigger_keywords": [
                    {{"keyword": {{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}}}
                  ],
                  "boosts": {{
                    "industry":      [{{"name":"string","boost_label":"Very High|High|Medium|Low|Negligible","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
                    "revenue_range": [{{"name":"string","boost_label":"Very High|High|Medium|Low|Negligible","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
                    "employee_range":[{{"name":"string","boost_label":"Very High|High|Medium|Low|Negligible","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
                    "funding_stage": [{{"name":"string","boost_label":"Very High|High|Medium|Low|Negligible","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
                    "geography":     [{{"name":"string","boost_label":"Very High|High|Medium|Low|Negligible","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}]
                  }}
                }}
              ]
            }}
          ],
          "notes": ["optional, concise"]
        }}
        ONLY JSON.
    """.strip()


# ---------------------------------------------------------------------------
# Single Node Expansion Prompts (for agentic frontier expansion)
# ---------------------------------------------------------------------------

def _records_from_contexts(
    expansion_node_ids: Optional[List[str]],
    expansion_node_contexts: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if expansion_node_contexts:
        return expansion_node_contexts
    # Fallback: minimal records if only IDs are provided
    records: List[Dict[str, Any]] = []
    for sid in expansion_node_ids or []:
        records.append({
            "source_id": sid,
            "source": {"id": sid},          # caller should prefer contexts
            "already_linked": [],
            "need_fields": []
        })
    return records


# Capability -> Pain
def build_capability_expansion_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    expansion_node_ids: Optional[List[str]] = None,
    expansion_node_contexts: Optional[List[Dict[str, Any]]] = None,
    max_items_per_source: int = 5,
    target_type: Optional[str] = None,
) -> str:
    records = _records_from_contexts(expansion_node_ids, expansion_node_contexts)
    return f"""
You are expanding a product graph one hop. Return STRICT JSON only. No prose, no markdown.

{LABEL_GUIDE}

Context:
- Product value proposition: {product_summary}
- Product domain: {domain}
- Product industry: {industry}

Task:
For each capability, propose pains it directly solves or unlocks.
- Use concise, concrete phrasing for pain.description.
- Optionally set pain_source: "terminal" | "non-terminal".
- Avoid any targets present in 'already_linked'.
- Include relevance_label (capability→pain) and likelihood_label (pain→capability usage).
- Max items per source: {max_items_per_source}.

Schema:
items: [
  {{
    "source_id": "<capability_id>",
    "targets": {{
      "pain": {{
        "proposals": [{{"description":"string","pain_source":"terminal|non-terminal",
                        "relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope",
                        "likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}],
        "evidence": ["optional why strings"]
      }}
    }}
  }}
]

Records (inputs):
{_fmt(records)}

Return JSON object:
{{"items":[...]}}
""".strip()


# Pain -> (Job, PainTrigger, PerceivedMetric)
def build_pain_expansion_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    expansion_node_ids: Optional[List[str]] = None,
    expansion_node_contexts: Optional[List[Dict[str, Any]]] = None,
    max_items_per_source: int = 5,
    target_type: Optional[str] = None,
) -> str:
    records = _records_from_contexts(expansion_node_ids, expansion_node_contexts)
    return f"""
You are expanding a product graph one hop. Return STRICT JSON only. No prose.

{LABEL_GUIDE}

Context:
- Product value proposition: {product_summary}
- Product domain: {domain}
- Product industry: {industry}

Task:
From each pain, propose:
1) job            — where the pain is felt operationally (short actionable description).
2) pain_trigger   — observable attribute/condition that causes or worsens the pain (attribute field).
3) perceived_metric — how the pain is expressed/monitored (metric field).
Avoid 'already_linked'. Max items per source: {max_items_per_source}.
Every proposed edge MUST include relevance_label and likelihood_label.

Schema:
items: [
  {{
    "source_id": "<pain_id>",
    "targets": {{
      "job":            {{"proposals":[{{"description":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}], "evidence":["..."]}},
      "pain_trigger":   {{"proposals":[{{"attribute":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}], "evidence":["..."]}},
      "perceived_metric":{{"proposals":[{{"metric":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}], "evidence":["..."]}}
    }}
  }}
]

Records (inputs):
{_fmt(records)}

Return JSON object:
{{"items":[...]}}
""".strip()


# Job -> (Persona, Pain [solves-path])
def build_job_expansion_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    expansion_node_ids: Optional[List[str]] = None,
    expansion_node_contexts: Optional[List[Dict[str, Any]]] = None,
    max_items_per_source: int = 5,
    target_type: Optional[str] = None,  # if "pain", caller is forcing solves-path UX
) -> str:
    records = _records_from_contexts(expansion_node_ids, expansion_node_contexts)
    return f"""
You are expanding a product graph one hop. Return STRICT JSON only. No prose.

{LABEL_GUIDE}

Context:
- Product value proposition: {product_summary}
- Product domain: {domain}
- Product industry: {industry}

Task:
From each job, propose:
1) persona — likely owner/performer of this job (title, department, seniority; optional linkedin_profiles[] of {{url,bio}}) WITH relevance_label & likelihood_label.
2) pain    — pains this job directly solves (business wording, optional pain_source) WITH relevance_label & likelihood_label.
Avoid 'already_linked'. Max items per source: {max_items_per_source}.

Schema:
items: [
  {{
    "source_id": "<job_id>",
    "targets": {{
      "persona": {{"proposals":[{{"title":"string","department":"string","seniority":"Junior|Operator|Manager|Senior|Executive","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely","linkedin_profiles":[{{"url":"string","bio":"string"}}]}}], "evidence":["..."]}},
      "pain":    {{"proposals":[{{"description":"string","pain_source":"terminal|non-terminal","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}], "evidence":["..."]}}
    }}
  }}
]

Records (inputs):
{_fmt(records)}

Return JSON object:
{{"items":[...]}}
""".strip()


# PainTrigger -> (AttributeValue, ZMOT Event)
def build_pain_trigger_expansion_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    expansion_node_ids: Optional[List[str]] = None,
    expansion_node_contexts: Optional[List[Dict[str, Any]]] = None,
    max_items_per_source: int = 5,
    target_type: Optional[str] = None,
) -> str:
    records = _records_from_contexts(expansion_node_ids, expansion_node_contexts)
    return f"""
You are expanding a product graph one hop. Return STRICT JSON only.

{LABEL_GUIDE}

Context:
- Product value proposition: {product_summary}
- Product domain: {domain}
- Product industry: {industry}

Task:
From each pain_trigger, propose:
1) attribute_value — {{dimension:"industry|employee_range|revenue_range|funding_stage|geography", name:"string"}} WITH relevance_label & likelihood_label.
2) zmot_event     — {{event:"short specific name"}} WITH relevance_label, likelihood_label, AND boost_label (REQUIRED).
Avoid 'already_linked'. Max items per source: {max_items_per_source}.

Schema:
items: [
  {{
    "source_id": "<pain_trigger_id>",
    "targets": {{
      "attribute_value": {{"proposals":[{{"dimension":"string","name":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}], "evidence":["..."]}},
      "zmot_event":      {{"proposals":[{{"event":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely","boost_label":"Very High|High|Medium|Low|Negligible"}}], "evidence":["..."]}}
    }}
  }}
]

Records (inputs):
{_fmt(records)}

Return JSON object:
{{"items":[...]}}
""".strip()


# AttributeValue -> ZMOT Event
def build_attribute_value_expansion_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    expansion_node_ids: Optional[List[str]] = None,
    expansion_node_contexts: Optional[List[Dict[str, Any]]] = None,
    max_items_per_source: int = 5,
    target_type: Optional[str] = None,
) -> str:
    records = _records_from_contexts(expansion_node_ids, expansion_node_contexts)
    return f"""
You are expanding a product graph one hop. Return STRICT JSON only.

{LABEL_GUIDE}

Context:
- Product value proposition: {product_summary}
- Product domain: {domain}
- Product industry: {industry}

Task:
For each attribute_value (segment), propose relevant ZMOT events as {{ "event": "..." }} WITH relevance_label & likelihood_label.
Avoid 'already_linked'. Max items per source: {max_items_per_source}.
(Do NOT include boost_label here; boost applies only to pain_trigger → zmot_event.)

Schema:
items: [
  {{
    "source_id": "<attribute_value_id>",
    "targets": {{
      "zmot_event": {{"proposals":[{{"event":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}], "evidence":["..."]}}
    }}
  }}
]

Records (inputs):
{_fmt(records)}

Return JSON object:
{{"items":[...]}}
""".strip()


# ZMOT Event -> (ObservableMoment, Keyword)
def build_zmot_event_expansion_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    expansion_node_ids: Optional[List[str]] = None,
    expansion_node_contexts: Optional[List[Dict[str, Any]]] = None,
    max_items_per_source: int = 5,
    target_type: Optional[str] = None,
) -> str:
    records = _records_from_contexts(expansion_node_ids, expansion_node_contexts)
    return f"""
You are expanding a product graph one hop. Return STRICT JSON only.

{LABEL_GUIDE}

Context:
- Product value proposition: {product_summary}
- Product domain: {domain}
- Product industry: {industry}

Task:
From each zmot_event, propose:
1) observable_moment — external signals (press, filings, job posts, release notes, audits, etc.) as {{text:"..."}}
2) keyword           — search/discovery phrases as {{text:"..."}}
Each proposed edge MUST include relevance_label & likelihood_label.
Avoid 'already_linked'. Max items per source: {max_items_per_source}.

Schema:
items: [
  {{
    "source_id": "<zmot_event_id>",
    "targets": {{
      "observable_moment": {{"proposals":[{{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}], "evidence":["..."]}},
      "keyword":           {{"proposals":[{{"text":"string","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope","likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}}], "evidence":["..."]}}
    }}
  }}
]

Records (inputs):
{_fmt(records)}

Return JSON object:
{{"items":[...]}}
""".strip()
