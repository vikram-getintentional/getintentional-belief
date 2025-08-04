import json
from backend.utils.inference.gpt_prompts.openai_client import client  # uses our centralized OpenAI client



def run_internal_pain_discovery(summary, domain, industry, job_sets: list[dict[str, any]]) -> list[dict[str, any]]:
    """
    Discover internal upstream pains that cause a given job to exist.
    Returns a list of upstream pains, triggers, metrics, severity scores, and upstream jobs.
    """
    # Inputs:
    # - job: "Generate MRR/ARR reports"
    # - pain: "Delayed reporting and audit friction"
    # - persona: "Revenue Ops Lead"
    # - product_summary: 2-3 line product value summary
    
    prompt = f"""
        You are a business reasoning expert. You are given the following job-to-be-done and pains in the context of this product:

        - Product Summary: {summary}
        - Product Domain: {domain}
        - Product Industry: {industry}
        - Jobs and Pains = json.dumps(job_sets, indent=2)


        Your task is to reason upstream: why does this job exist?

        For each pain (1–3) that could have caused this job to exist:
        - pain: a specific upstream pain this job resolves
        - is_external: true or false

        If is_external == false (internal pain):
        - pain_trigger: {{
            attribute: (e.g. "pipeline changes"),
            dimension: (e.g. "frequency"),
            direction: (e.g. "increase")
        }}
        - pain_expression: list of 2–3 observable internal metrics or indicators
        - severity: float from 0.0 to 1.0 (how fully the job solves this pain)
        - upstream_jobs: list of {{
            description: a job where this pain is directly experienced,
            impact: float from 0.0 to 1.0
        }}

        If is_external == true (external trigger):
        - external_reason: Why this pain is caused by something outside the org (e.g. market shift, funding round, regulation)
        - Leave upstream_jobs, triggers, and metrics blank

        Return as structured JSON in the following format:

        - original_job_id: string
        - pains: list of 
        [
            - pain: string
            - is_external: false
            - pain_trigger: list of
                - attribute: string
                - dimension: string
                - direction: string
            - pain_expression: list of strings
            - severity: float
            - upstream_jobs: list of
                - description: string
                - impact: float
        ]
        [
            - pain: string
            - is_external: true
            - external_reason: string
        ]

        IMPORTANT: Return only valid JSON, with no comments or extra text.

        """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You create upstream business impact of a personas jobs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        # Parse and return the JSON output from OpenAI
        return json.loads(content)
    except Exception as e:
        print("❌ run_internal_pain_discovery:", e)
        return []

def infer_upstream_jobs_and_personas(summary, domain, industry, job_sets: list[dict[str, any]]) -> list:
    """
    For a given internal pain, infer upstream jobs and personas responsible for solving it.
    """
    prompt = f"""
        You are analyzing an internal organizational pain.

        Context:
        - Product Summary: {summary}
        - Product Domain: {domain}
        - Product Industry: {industry}
        - Jobs and Pains = json.dumps(job_sets, indent=2)

        For each upstream job that is performed to address this pain:
        - job_description
        - impact_score (float 0.0–1.0): how much this job helps resolve the pain
        - personas: list of {{
            title,
            department,
            seniority (Junior | Operator | Manager | Senior | Executive),
            job_importance (0.0–1.0)
        }}

        Return as structured JSON in the following format:

        - original_pain_id: string
        - upstream_jobs: list of
        [
            - job_description: string
            - impact_score: float
            - personas: list of
                - title: string
                - department: string
                - seniority: string
                - job_importance: float
        ]
        IMPORTANT: Return only valid JSON, with no comments or extra text.
        IMPORTANT: Use only realistic, clearly defined jobs and persona roles. Do not invent exotic titles unless required by the domain. All capabilities should return at least one pain with structured jobs and personas.
        IMPORTANT: For each job, do not simply repeat jobs or pains from the input list.
        For each job in the input list, process it independently and return a separate JSON object using the job_id as the key. Do not let the context of one job influence the others.

        """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You analyze internal pains and infer upstream jobs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        # Parse and return the JSON output from OpenAI
        return json.loads(content)
    except Exception as e:
        print("❌ Error in infer_upstream_jobs_and_personas:", e)
        return []
    
def infer_external_trigger_for_pain(summary, domain, industry, job_sets: list[dict[str, any]]) -> dict:
    """
    For a pain that was externally caused, infer ICP archetypes and ZMOT triggers.
    """
    prompt = f"""
        You are analyzing a business pain faced by a persona in an organization that was caused by external events or market shifts.

        - Product Summary: {summary}
        - Product Domain: {domain}
        - Product Industry: {industry}
        - Jobs and Pains = json.dumps(job_sets, indent=2)

        Step 1: List 3–5 Ideal Customer Profile (ICP) archetypes most likely to experience this pain trigger.
        For each archetype, list:
        - industry
        - revenue (ARR bucket)
        - employees
        - funding_stage
        - geographies
        - For each field, include a match_score (0.0–1.0)

        Step 2: For each ICP archetype, list 3–5 likely ZMOT trigger events that would cause this pain.
        Each ZMOT trigger must include:
        - trigger_event (e.g. "Funding Round", "Compliance Audit")
        - match_score (0.0–1.0)
        - observable_moments: 2-3 observable moments that indicate this trigger is happening
            observable_moment: string
            match_score: float (0.0–1.0)
        - trigger_keywords: list of trigger keywords to look for in the observable moment that indicate that this event occured
            trigger_keyword: string
            match_score: float (0.0–1.0)

        Return structured JSON in the following format:
        - original_pain_id: string
        - icp_archetypes: list of
            - industry: {{string (e.g. "Retail"), match score: float (0.0–1.0)}}
            - revenue: {{string (e.g. "1M-10M", "10M-100M"), match score: float (0.0–1.0)}}
            - employees: {{string (e.g. "1-10", "11-50"), match score: float (0.0–1.0)}}
            - funding_stage: {{string (e.g. "Seed", "Series A"), match score: float (0.0–1.0)}}
            - geographies: {{list of strings, match score: float (0.0–1.0)}}
        - zmot_events: list of
            - trigger_event: string
            - trigger_event_match_score: float (0.0–1.0)
            - observable_moments: list of
                - observable_moment: string
                - match_score: float (0.0–1.0)
            - trigger_keywords: list of
                - trigger_keyword: string
                - match_score: float (0.0–1.0)
        Use only realistic, clearly defined pain triggers, events, observable moments and keywords. 
        IMPORTANT: Return ONLY the JSON array, with no explanation, formatting, or trailing commas. Ensure all arrays and objects are properly closed.
        IMPORTANT: For each list (pain_triggers, icp_archetypes, zmot_events), you MUST provide at least 3-5 items. Do not return only one item for any list. If you cannot find 3 strong matches, look for slightly poorer matches.
        IMPORTANT: Each ICP Archetype, pain trigger, ZMOT event, and keyword must be tightly linked to the context of the product’s value proposition. Do not generalize — reference domain-specific signals wherever possible.
        Be brutally specific with the returned values and match scores. 


        """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You create upstream business impact of a personas jobs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        # Parse and return the JSON output from OpenAI
        return json.loads(content)
    except Exception as e:
        print("❌ Error in get_upstream_triplets:", e)
        return []
    

