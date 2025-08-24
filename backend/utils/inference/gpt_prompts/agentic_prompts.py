# agentic_prompts.py
# Build-only helpers. DO NOT call the LLM here.
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json


def _fmt(obj: Any) -> str:
    # Compact but readable JSON for inclusion inside prompts
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), indent=2)


def build_hop0_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    capability_ids: List[str],
    capability_context: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Hop0: capability -> pains -> felt_in (jobs/personas) (+ metrics, triggers)
    If capability_context is provided, pass a list of objects like:
      { "capability_id": "...", "name": "...", "description": "..." }
    Otherwise we pass just the IDs to keep tokens low.
    """
    caps_payload = capability_context if capability_context else capability_ids

    return f"""
        You are an expert in B2B job architecture. For each product capability, map pains, felt_in jobs, personas, perceived_metrics, and pain_triggers.

        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Capabilities: {_fmt(caps_payload)}

        Labeling guides (apply to all outputs as directed):
        - Relevance label: How central is the source node to resolving or enabling the target node?
        One of: {"Critical","Core","Supportive","Ancillary","Out-of-scope"}
        • Critical: Without the source, the target cannot be achieved.
        • Core: The source directly enables the target in most workflows; removing it would significantly weaken the connection.
        • Supportive: The source contributes to the target but is not sufficient on its own.
        • Ancillary: The source may only help in edge cases or indirectly.
        • Out-of-scope: The source does not materially affect the target.

        - Likelihood label: How expected is it that the source would be the solution or enabler for the target?
        One of: {"Essential","Expected","Common","Rare","Unlikely"}
        • Essential: The source is almost always the solution/enabler for the target.
        • Expected: Frequently expected as a solution/enabler; omission would surprise users.
        • Common: Commonly expected, but other solutions/enablers exist.
        • Rare: Rarely used or expected only in special cases.
        • Unlikely: Unlikely to be chosen as a solution/enabler for the target.

        Instructions:
        1) For each capability, infer 1–3 *concrete pains* it directly solves.
        IMPORTANT:
            Each pain must be an actual inefficiency or friction a persona faces while performing a job. Think of a pain as the reason why a persona was unable to sufficiently, efficiently, or effectively deliver the job’s outcomes.
        For each pain include:
        1a) the pain description (string) as a one-sentence description of a specific business pain or workflow inefficiency that blocks a persona's ability to perform a job.
        1b) Relevance & Likelihood Labels: For each pain as the target and the corresponding capability as the source provide a relevance label and a likelihood label.

        2) pain_triggers: list 3–5 attributes (normalized, lemmatized nouns only) that, if scaled/changed, make this pain worse.
            - Evaluate each "pain trigger" as a response to "what must increase, scale, or significantly change for this pain to become worse?"
            - Only return the attribute term — no units, direction, or values.
            - Be specific in your response
            - Ensure that the returned output is normalized and lemmatized
            For example, a pain in "difficulty managing data pipelines" gets worse with "analytics data sources".
            Only return: "analytics data sources" as the pain trigger attribute. 
            - For each pain trigger as the target and the corresponding pain as the source provide a relevance label and a likelihood label.
        - perceived_metrics: list ≥2 measurable indicators of the pain
            for example "data pipeline latency", "data quality issues", "data processing costs"
            for each perceived metric as the target and the corresponding pain as the source provide a relevance label and a likelihood label.
        3) felt_in_jobs: For each pain infer 1–3 jobs-to-be-done that this pain is typically felt in.
        IMPORTANT: 
            The job must be the specific task or business objective that a persona performs during which this pain is felt. 
            The job must be causally upstream of the pain, meaning the pain is experienced while performing this job. If this job did not occur, this pain may never be experienced or noticed.
            The existence of this pain directly degrades the effectiveness or efficiency of this job to deliver its outcomes.
            The job must be temporally upstream of this pain, meaning the pain is only felt during or after the job is performed.
            The job language must be something that you might find in the typical job description for a persona in an organization.
        For each job include:
        - job_to_be_done (string): A one sentence description of the specific job that is performed by a persona during which this pain is felt.
        - for each job_to_be_done as the target and the corresponding pain as the source provide a relevance label and a likelihood label.
        
        4) For each job, infer 1-3 personas who would typically be responsible for performing this job in an organization.
        Ensure that each persona is an actual "job title" that exists in an organization, and the job to be done usually is part of their job description or responsibilities.
        For each persona include:
                 - "title",
                 - "department",
                 - "seniority" : "Junior|Operator|Manager|Senior|Executive",
                 - provide a relevance label and a likelihood label for the persona as the target and the corresponding job as the source.
       
        5) For each job, infer 1-3 "solving pains" as the business pains, workflow inefficiencies or organizational issues that this job exists to solve.
        IMPORTANT:
            Each pain must be an actual business inefficiency or workflow friction that occurs in the organization.
            The "solving pain" must occur causally earlier to the job meaning the job is a response, resolution or delegation to this pain.
            The solving pain exists independent of the job, meaning the pain is not just a rephrasing of the job description. The job exists in the organization wholly or in part only to solve this pain.
            The solving pain must be temporally earlier to the job, meaning the pain is experienced first, and the job is performed or delegated as a response to this pain.
            The solving pain remains unsolved or gets worse if the job is not performed well or does not deliver its intended outcomes.
            The solving pain must NOT equal, paraphrase, or be the canonical equivalent of the job description or the original pain.
        For each solving pain include:
                - pain description (string) as a one-sentence description of a specific business pain or workflow inefficiency that is solved by this job_to_be_done.
                - for each solving pain as the target and the corresponding job as the source provide a relevance label and a likelihood label.
                - perceived_metrics: list ≥2 "metrics"
                - pain_triggers: list 2–4 "attributes"
        Rules:
        1. Treat each capability independently. Do not let the context of one output bleed into another.
        2. Ensure that the output includes all capabilities provided in the input, at least 1 pain per capability, 1 job per pain, and 1 solving pain per job.
        3. Ensure all responses are in the context of a typical organization that could potentially benefit from the provided product summary, domain and industry.
        4. IMPORTANT: Avoid cycles and rewording in pains, jobs, and solving pains. Ensure distinct responses with clear causal/temporal order.
        5. Use precise and specific, domain-relevant language; avoid vague terms (“optimize”, “improve process”) without specifics.
        6. IMPORTANT: Ensure all results are relevant and with context of the provided product summary, domain and industry. Do not hallucinate or return results that aare irrelevant or questionable in the given context.

        Return STRICT JSON array:
        [{{
            "capability_id": must be exactly one of the input capability_ids provided in context,
            "pains": [{{
            "pain": "string",
            "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
            "likelihood_label": "Essential|Expected|Common|Rare|Unlikely",
            "pain_triggers": [{{"text": "string", "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"}},...],
            "perceived_metrics": [{{"text": "string", "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"}},...],
            "felt_in_jobs": [{{
                "job_to_be_done": "string",
                "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
                "likelihood_label": "Essential|Expected|Common|Rare|Unlikely",
                "personas": [{{
                    "title":"string",
                    "department":"string",
                    "seniority":"Junior|Operator|Manager|Senior|Executive",
                    "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
                    "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"
                    }},...],
                    
                "solving_pains": [{{
                    "pain":"string",
                    "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
                    "likelihood_label": "Essential|Expected|Common|Rare|Unlikely",
                    "pain_triggers": [{{"text": "string", "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"}},...],
                    "perceived_metrics": [{{"text": "string", "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"}},...],
                    }},...]
                }},...]
            }},...]
        }}]
        ONLY return JSON. No commentary.
        """.strip()


def build_pain_source_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    pain_contexts: List[Dict[str, Any]],  # enriched: [{"pain_id","pain_text", ...}]
) -> str:
    """
    Classify pains as internal/external with compact anchors.
    pain_contexts item example:
      {
        "pain_id": "...",
        "pain_text": "string",
        "felt_in_jobs": [{"job_to_be_done":"...", "importance":0.9}],
        "personas": [{"title":"...","department":"...","seniority":"..."}]
      }
    """
    return f"""
        You classify whether a business pain experienced in the context of a workflow or process is "terminal" or "non-terminal" to the organization.

        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Pains: {_fmt(pain_contexts)}

        Definitions:
        - non-terminal: caused by internal workflow/organizational issues; often stemming from an upstream or cross-team delegation from inside org; usually has a plausible upstream "job-to-be-done" that this pain is experienced during.
        - terminal: driven by outside forces (market, customers, regulators) not controllable by org, or by significant strategic directive not directly tied to day-to-day operations.

        Rules:
        - Treat each pain independently.
        - Default to "Non-Terminal" unless you can confidently determine that there are no upstream jobs that this pain is experienced in.
            An upstream job is a job-to-be-done in the organization by a persona with same or higher seniority as the persona in the pain_context, and is causally and temporally earlier to the pain.
        - A terminal pain is likely clearly triggered by an outside event or condition that is not under the organization's control such as a market event, competitor activities, regulatory changes etc.
        - A terminal pain can also be triggered by significant strategic event or activity that is driven by the organization but is not directly related to the organization's day to day processes or operations - such as a funding event, a new leadership hire, management change, new market expansion, etc.

        - For example, you may ask
            - Is this pain the result of, or experienced while performing a job-to-be-done in the context of the organization? (Likely Non-Terminal)
            - Does this pain independently require a response, or is it due to a larger pain like "Customer Experience", "Regulatory Fines", "Revenue Loss" or "Legal Exposure" getting impacted by it? (If there is a likely larger pain, non-terminal) 
            - Is this pain caused by the typical day-to-day operations of the organization? (Likely Non-Terminal)
            - Is this pain purely a response to an external event or condition that is not under the organization's direct control? (Likely Terminal)
            - Would this pain exist if the organization continued in status quo? (Likely Non-Terminal)
            - Is the onset of this pain sudden or gradual - over a period of time? (Sudden onset may indicate an external trigger)
            - Is this pain localized to a specific persona, team or department, or felt across multiple teams and layers in the organization? (Localized may indicate internal, widespread may indicate external) 
        
        Few short examples:
        - "New regulatory requirements" - terminal (Since this is completely outside the org's control and typically is some persona's job to monitor and respond to)
        - "Website outage" - non-terminal (Not a terminal pain in isolation - has a larger consequence on Customer Experience or Loss of Revenue)
        - "Delayed data-driven decisions" - non-terminal (This causes an upstream pain like "CXOs making incorrect strategic decisions", or managers being "Slow to respond")
        - "Revenue loss due to market downturn" - terminal (This is a direct response to an external market condition that is not under the org's control)
        - "Compliance readiness for IPO" - terminal (This is a strategic directive that is not directly tied to day-to-day operations, but is a response to an external event like IPO)
        - "Market expansion into APAC" - terminal (This is a strategic initiative that is driven by external market opportunities and is not tied to day-to-day operations)


        - If uncertainty in your answer > 40% return "Non-Terminal" by default.

        IMPORTANT:
        - Ensure each response is in the context of a typical organization that could potentially benefit from the provided product summary, domain and industry.

        Return STRICT JSON array:
        [{{
            "pain_id": must be exactly one of the input pain_ids provided in context,
            "pain_source":"terminal|non-terminal"}}]
        ONLY JSON. No commentary.
        """.strip()


def build_hop_plus_prompt(
    product_summary: str,
    domain: str,
    industry: str,
    upstream_pain_contexts: List[Dict[str, Any]],
) -> str:
    """
    Hop0: capability -> pains -> felt_in (jobs/personas) (+ metrics, triggers)
    If capability_context is provided, pass a list of objects like:
      { "capability_id": "...", "name": "...", "description": "..." }
    Otherwise we pass just the IDs to keep tokens low.
    """
    

    return f"""
        You are an expert industry analyst in B2B organization design, with specific expertise in {domain}. For the given set of organizational pains, infer where each pain is felt (jobs) and the independent upstream pains those jobs exist to solve.
        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Internal pains (with anchors): {_fmt(upstream_pain_contexts)}

        Labeling guides (apply to all outputs as directed):
        - Relevance label: How central is the source node to resolving or enabling the target node?
        One of: {{"Critical","Core","Supportive","Ancillary","Out-of-scope"}}
        • Critical: Without the source, the target cannot be achieved.
        • Core: The source directly enables the target in most workflows; removing it would significantly weaken the connection.
        • Supportive: The source contributes to the target but is not sufficient on its own.
        • Ancillary: The source may only help in edge cases or indirectly.
        • Out-of-scope: The source does not materially affect the target.

        - Likelihood label: How expected is it that the source would be the solution or enabler for the target?
        One of: {{"Essential","Expected","Common","Rare","Unlikely"}}
        • Essential: The source is almost always the solution/enabler for the target.
        • Expected: Frequently expected as a solution/enabler; omission would surprise users.
        • Common: Commonly expected, but other solutions/enablers exist.
        • Rare: Rarely used or expected only in special cases.
        • Unlikely: Unlikely to be chosen as a solution/enabler for the target.

        Instructions:
        For each pain in the input, do the following:
        1) felt_in_jobs: infer 1–3 jobs where this pain is experienced.
        IMPORTANT: 
            The job must be the specific task or business objective that a persona performs during which this pain is felt. 
            The job must be causally upstream of the pain, meaning the pain is experienced while performing this job. If this job did not occur, this pain may never be experienced or noticed.
            The existence of this pain directly degrades the effectiveness or efficiency of this job to deliver its outcomes.
            The job must be temporally upstream of this pain, meaning the pain is only felt during or after the job is performed.
            The job language must be something that you might find in the typical job description for a persona in an organization.
            Use the provided context about the pain, including what jobs solve this pain, and which personas perform those jobs as anchors. Do not restate them. Infer the upstream job where this pain is experienced.
            Remember not to just rephrase the downstream job description that solves for this pain, but to infer the upstream job that this pain is experienced during.
        For each job include:
        - job_to_be_done (string): A one sentence description of the specific job that is performed by a persona during which this pain is experienced.
        - for each job_to_be_done as the target and the corresponding pain as the source provide a relevance label and a likelihood label.
        
        2) Why–What–Fail reasoning (for each job):
            a. Why does this job exist? (independent of the original pain; job exists as a standing responsibility)
            b. What upstream pain does it solve? (exists even if the job is done perfectly; the reason the job exists)
            c. What happens if this job fails or never occurs? (connects the upstream pain to the original pain)
        
        3) Use the upstream pain from 2b as the job’s solving pain (1-3 solving pains per job). It must:
           - be causally/temporally earlier than the job and earlier than the original pain,
           - NOT equal, paraphrase, or be the canonical equivalent of the original pain or the job wording,
           - be a concrete organizational inefficiency/friction (not a downstream effect like “lost revenue”).
         For each solving pain include(from 3):
            - pain description (string) as a one-sentence description of a specific business pain or workflow inefficiency that is solved by this job_to_be_done.
            IMPORTANT:
                Each pain must be an actual business inefficiency or workflow friction that occurs in the organization.
                The "solving pain" must occur causally earlier to the job meaning the job is a response, resolution or delegation to this pain.
                The solving pain exists independent of the job, meaning the pain is not just a rephrasing of the job description. The job exists in the organization wholly or in part only to solve this pain.
                The solving pain must be temporally earlier to the job, meaning the pain is experienced first, and the job is performed or delegated as a response to this pain.
                The solving pain remains unsolved or gets worse if the job is not performed well or does not deliver its intended outcomes.
                The solving pain must not be a rephrasing of the job description or the original pain. It must be a distinct causally and temporally upstream business pain. 
            - for each solving pain as the target and the corresponding job as the source provide a relevance label and a likelihood label.
            - pain_triggers: list 3-5 attributes (normalized, lemmatized nouns only) that, if scaled/changed, make this pain worse.
                - Evaluate each "pain trigger" as a response to "what must increase, scale, or significantly change for this pain to become worse?"
                - Only return the attribute term — no units, direction, or values.
                - Be specific in your response
                - Ensure that the returned output is normalized and lemmatized
                For example, a pain in "difficulty managing data pipelines" gets worse with "analytics data sources".
                Only return: "analytics data sources" as the pain trigger attribute. 
                - For each pain trigger as the target and the corresponding pain as the source provide a relevance label and a likelihood label.
            - perceived_metrics: list ≥2 measurable indicators of the pain
                for example "data pipeline latency", "data quality issues", "data processing costs"
                = for each perceived metric as the target and the corresponding pain as the source provide a relevance label and a likelihood label.
        
        4) For each job, infer 1-3 personas who would typically be responsible for performing this job in an organization.
        Ensure that each persona is an actual "job title" that exists in an organization, and the job to be done usually is part of their job description or responsibilities.
        For each persona include:
                 - "title",
                 - "department",
                 - "seniority" : "Junior|Operator|Manager|Senior|Executive",
                 - provide a relevance label and a likelihood label for the persona as the target and the corresponding job as the source.
       
                 
        Rules:
        1. IMPORTANT: Only use the provided context and ensure all responses are tightly relevant to the product summary, domain, and industry inputs. If this is not a strongly plausible context or you find your confidence < 40% return an empty array [].
        2. Treat each input pain independently. Do not let context or output of one pain bleed into another.
        3. Ensure that the output includes all pains provided in the input, at least 1 "felt_in" job per pain, 1 persona per job, and 1 solving pain per job.
        4. Ensure all responses are in the context of a typical organization that could potentially benefit from the provided product summary, domain and industry.
        5. IMPORTANT: Avoid cycles and rewording in pains, jobs, and solving pains. Ensure distinct responses with clear causal/temporal order.
        6. Use precise and specific language that is typically used in an organization. Do not use generic or vague terms.

         Return STRICT JSON array:
        [{{
        "original_pain_id":must be exactly one of the input pain_ids provided in context,
        "felt_in_jobs":
            [{{
            "job_to_be_done":"string",
            "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
            "likelihood_label": "Essential|Expected|Common|Rare|Unlikely",
            "personas":
                [{{"title":"string",
                "department":"string",
                "seniority":"Junior|Operator|Manager|Senior|Executive",
                "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
                "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"
                }}...],
            "solving_pains":[{{
                "pain":"string",
                "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
                "likelihood_label": "Essential|Expected|Common|Rare|Unlikely",
                "pain_triggers": [{{"text":"string", "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"}},...],
                "perceived_metrics": [{{"text":"string", "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"}},...],
                }},...]
            }}...],
        }}]
        ONLY JSON. No commentary.
        """.strip()



def build_archetypes_relevance_matrix(
    product_summary: str,
    domain: str,
    industry: str,
    trigger_contexts: List[Dict[str, Any]],
) -> str:
    return f"""
        You are an expert analyst in B2B organization design with deep understanding of {domain} and {industry}.
        You are given pain_triggers that make organizational pains worse. Your tasks:
        • Propose ideal-customer organizational archetypes that typically experience these triggers (baseline prevalence).
        • Identify concrete internal/external events (ZMOTs) that accelerate each trigger (boost).

        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Pain triggers with anchors: {_fmt(trigger_contexts)}
        
        Labeling guides (apply to all outputs as directed):
        - Relevance label: How central is the source node to resolving or enabling the target node?
        One of: {"Critical","Core","Supportive","Ancillary","Out-of-scope"}
        • Critical: Without the source, the target cannot be achieved.
        • Core: The source directly enables the target in most workflows; removing it would significantly weaken the connection.
        • Supportive: The source contributes to the target but is not sufficient on its own.
        • Ancillary: The source may only help in edge cases or indirectly.
        • Out-of-scope: The source does not materially affect the target.

        - Likelihood label: How expected is it that the source would be the solution or enabler for the target?
        One of: {"Essential","Expected","Common","Rare","Unlikely"}
        • Essential: The source is almost always the solution/enabler for the target.
        • Expected: Frequently expected as a solution/enabler; omission would surprise users.
        • Common: Commonly expected, but other solutions/enablers exist.
        • Rare: Rarely used or expected only in special cases.
        • Unlikely: Unlikely to be chosen as a solution/enabler for the target.
        

        Instructions:
        1) Ensure all responses are in tight context of the provided product summary, domain and industry. If this is not a strongly plausible context or you find your confidence < 40% return an empty archetypes array and an empty relevance_matrix array.
        1) Internally cluster triggers into coherent themes to avoid duplication and to ground events. (Do NOT return these clusters.)
        2) Infer a candidate set of 8-12 organizational archetypes that typically experience these triggers, ensuring each input trigger is represented in at least one archetype.
        For each archetype, include:
           - industry: such as "SaaS", "FinTech", "Healthcare", "Manufacturing", "Retail", etc.
           - revenue_range: "<$10M" | "$10–50M" | "$50–200M" | "$200M–$1B" | ">$1B"
           - employee_range: "<50" | "50–200" | "200–1k" | "1k–5k" | ">5k"
           - funding_stage: "bootstrapped" | "seed" | "Series A" | "Series B" | "growth" | "public"
           - geography: "North America", "Europe", "Asia-Pacific", "Latin America", "Middle East & Africa", etc.
        3) Build a **full relevance matrix**: for each archetype, providing a Relevance and Likelihood label for every input trigger (dense, no omissions).
        4) Keep language specific and organization-realistic; deduplicate near-synonyms; do not invent trigger IDs.

        Guardrails:
        - Use only provided pain_trigger_ids; do not invent IDs.
        - Be specific; avoid vague terms. Deduplicate near-synonyms.
        - Keep language organization-realistic for the given domain/industry.

        Return STRICT JSON object:
        {{
        "archetypes": [
            {{
            "archetype_id":"arch_1",
            "industry":"string",
            "revenue_range":"<$10M|$10–50M|$50–200M|$200M–$1B|>$1B",
            "employee_range":"<50|50–200|200–1k|1k–5k|>5k",
            "funding_stage":"bootstrapped|seed|Series A|Series B|growth|public",
            "geography":"string"
            }}
        ],
        "relevance_matrix": [
            {{
            "archetype_ref":"arch_1",
            "trigger_scores":[
                {{"pain_trigger_id":"<id from input>","relevance_label":"Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label":"Essential|Expected|Common|Rare|Unlikely"}},...  # one for each input trigger
            ]
            }}
        ],
        "notes":[
            "one-liners for key assumptions or deduping decisions"
        ]
        }}
        ONLY JSON. No commentary.
        """.strip()



def build_zmot_for_triggers_prompt(product_summary: str,
    domain: str,
    industry: str,
    archetype_contexts: List[Dict[str, Any]],
    trigger_contexts: List[Dict[str, Any]],
) -> str:
    return f"""
        You are an expert in B2B Organization Structure and how they respond to external triggers with deep understanding of {domain} and {industry}. 
        You are tasked with identifying the external events that accelerate organizational pain triggers.

        Context:
        - Product value proposition: {product_summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Organizational Archetypes: {_fmt(archetype_contexts)}
        - Pain triggers with anchors: {_fmt(trigger_contexts)}

        Labeling guides (apply to all outputs as directed):
        - Relevance label: How central is the source node to resolving or enabling the target node?
        One of: {"Critical","Core","Supportive","Ancillary","Out-of-scope"}
        • Critical: Without the source, the target cannot be achieved.
        • Core: The source directly enables the target in most workflows; removing it would significantly weaken the connection.
        • Supportive: The source contributes to the target but is not sufficient on its own.
        • Ancillary: The source may only help in edge cases or indirectly.
        • Out-of-scope: The source does not materially affect the target.

        - Likelihood label: How expected is it that the source would be the solution or enabler for the target?
        One of: {"Essential","Expected","Common","Rare","Unlikely"}
        • Essential: The source is almost always the solution/enabler for the target.
        • Expected: Frequently expected as a solution/enabler; omission would surprise users.
        • Common: Commonly expected, but other solutions/enablers exist.
        • Rare: Rarely used or expected only in special cases.
        • Unlikely: Unlikely to be chosen as a solution/enabler for the target.

        - Boost label: How strongly does the occurance of this event accelerate or intensify the pain trigger for this archetype?
        One of: {"Very High", "High","Medium","Low","Negligible"}
        • Very High: This event always significantly accelerates or intensifies the pain trigger for this archetype, and requires immediate attention.
        • High: This event often significantly accelerates or intensifies the pain trigger for this archetype, and should be monitored closely.
        • Medium: This event sometimes accelerates or intensifies the pain trigger for this archetype, and should be monitored periodically.
        • Low: This event rarely accelerates or intensifies the pain trigger for this archetype, and can be monitored infrequently.
        • Negligible: This event does not materially accelerate or intensify the pain trigger for this archetype, and does not require monitoring.

        For each (archetype × pain_trigger) pair, identify **(minimum) 3 to (utmost) 5** specific, discrete external events that would significantly accelerate or intensify the given pain trigger for that archetype.

        For each external event include:
        - trigger_event: short, specific description of the event (e.g., "leadership change", "pricing overhaul", "market entry — APAC", "regulatory change", "compliance audit", "IPO readiness", "funding round", "merger announcement", "customer dissatisfaction", "employee churn", etc.)
        - include an archetype relevance label and an archetype likelihood label for the trigger_event as the target and the corresponding archetype as the source.
        - include a pain_trigger relevance label and a pain_trigger likelihood label for the trigger_event as the target and the corresponding pain_trigger as the source.
        - For each trigger event include a boost label describing How strongly the occurance of this event accelerates the pain trigger for this archetype 
        - observable_moments: 3–6 concrete sources or proxy signals (free-form; allow niche communities, forums, datasets, etc.).
                   • For each trigger_event ask "what externally observable information can signal or help infer the occurrence of this event?" 
                   • Example (get creative here): 
                        app store reviews/ software review sites to infer "customer satisfaction", 
                        engineering blog posts to infer "engineering culture" or "tech stack changes", 
                        job postings of AEs/ sales executives in specific regions to infer "market expansion"
                        subreddits, forums or communities discussions to infer specific concerns
                        glassdoor reviews to infer "employee churn" or "company culture"
            - For each observable moment include a relevance label and liklihood label for the observable_moment as the target and the corresponding trigger_event as the source.
        - trigger_keywords: 6–12 normalized, lemmatized, lowercase terms to look for in an observable moment that indicate the occurrence of the event
                        each trigger keyword should be a word or short phrase - no units/direction; brand terms only if essential.
            - For each trigger keyword include a relevance label and liklihood label for the trigger_keyword as the target and the corresponding trigger_event as the source.

        
        Guardrails:
        - IMPORTANT: Ensure all responses are in tight context of the provided product summary, domain and industry. If this is not a strongly plausible context or you find your confidence < 40% return empty results.
        - Do not invent archetype IDs or trigger IDs — use exactly those given.
        - Make events specific to both the archetype and trigger context.
        - Ensure all answers are in the context of a typical organization that could potentially benefit from the provided product summary, domain, and industry.
        - Avoid vague, generic events (e.g., "market change") — tie them to the archetype’s domain/industry.

        Optimization goal (global across all pairs):
        - Build the smallest possible pool of distinct trigger_events that collectively cover as many (archetype × pain_trigger) pairs as possible.
        - Prefer events that plausibly accelerate multiple triggers and/or apply to multiple archetypes.
        - Do NOT invent events: if no relevant event exists for a pair, omit that pair from coverage and state why in notes.

        Event normalization & reuse:
        - Normalize trigger_event as a short, lowercased noun phrase; deduplicate near-synonyms.
        - Assign a stable event_id = slug(trigger_event).
        - Reuse the same event_id across all pairs where applicable; do not create duplicates.

        Coverage rules:
        - Attempt coverage for every pain_trigger; partial coverage is acceptable if justified.
        - Cap per-pair at 3–5 events; prioritize events by (global_coverage_rank, boost_score, match_score).
        - Provide a global "events" list and then reference by event_id in the per-pair matrix.     
              
        Return STRICT JSON only:
        {{
          "zmot_matrix": [
            {{
              "archetype_id": "string",
              "original_pain_trigger_id": "string",
              "zmot_triggers": [
                {{
                  "trigger_event": "string",
                  "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope",
                  "likelihood_label": "Essential|Expected|Common|Rare|Unlikely",
                  "boost_label": "Very High|High|Medium|Low|Negligible",
                  "observable_moments": [
                    {{"observable_moment": {{"text": "string", "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"}}}}
                  ],
                  "trigger_keywords": [
                    {{"keyword": {{"text": "string", "relevance_label": "Critical|Core|Supportive|Ancillary|Out-of-scope", "likelihood_label": "Essential|Expected|Common|Rare|Unlikely"}}}}]
                }}
              ]
            }}
          ]
        }}
        ONLY JSON — no commentary.
        """.strip()





        