import json
from backend.utils.inference.gpt_prompts.openai_client import client  # uses our centralized OpenAI client
import re

def extract_json(text):
    # Extract the first JSON array or object from the text
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return text  # fallback

def infer_pain_triggers_zmot_icp(summary, jobs_pains):
    """
    Given a list of pains and jobs to be done that they are experienced in, this function queries OpenAI to produce a mapping of pain triggers, ideal customer profile (ICP) archetypes, and relevant external events (ZMOTs).
    """
    jobs_pains_json = json.dumps(jobs_pains, indent=2)

    prompt = f"""
You are an expert in business design and job architecture. You are tasked with identifiying the ideal target customers for a product and the specific trigger events that drive these customers.
Given:
- A summary value proposition of the product or service,
- A list of pains that are solved by the capabilities of this product,
- A list of jobs to be done by various personas in an organization where these pains are experienced,

For each pain, do the following:
1. Specify at least 3-5 pain triggers as the attribute that must scale for this pain to become intolerable in the format of:
      - attribute: the real-world metric or variable (e.g., "Number of support tickets")
      - dimension: one of ["volume", "complexity", "frequency", "compliance", etc.]
      - direction: one of ["Increase", "Decrease", "Change"] (choose from: volume, frequency, complexity, or describe the trigger in plain terms).
2. For each pain trigger include:
    - A list of at least 3-5 Ideal Customer Profile (ICP) organization archetypes most likely to experience these pain triggers as:
        - industry: the industry or sector (e.g., "Healthcare", "Finance", "Retail")
        - Revenue: revenue of the company in ARR (0-1M | 1-10M | 10-100M | 100M-500M | 500M - 1B | 1B+),
        - Employees: number of employees in this company type (1-10 | 11-50 | 51-200 | 201-500 | 501-1000 | 1000-5000 | 5000-10000 | 10000+),
        - Funding Stage: funding stages (Pre-Seed | Seed | Series A | Series B | Series C+ | Public),
        - Geographies: locations or market presence of this company archetype (North America | Europe | Asia | South America | Africa | Australia),
        - match_score: float (0.0-1.0) indicating how likely this archetype is to experience this pain trigger.
        A match score of 0.7-1 indicates very likely, 0.3-0.6 indicates a moderate likelihood, and 0.0-0.2 indicates it is unlikely to experience this pain trigger.
        Return each value in the ICP archetype as list of all possible values.
        For example, if both "healthcare" and "finance" are valid industries, return them as ["Healthcare", "Finance"].
    - A list of at least 3-5 external events (ZMOTs) that likely caused or accelerated these internal conditions in these organizations as:
        - trigger_event: The Event such as Recent funding round, market expansion, new product launch, leadership change, security breach, compliance failure, etc.
        - observable_moment: At least 3 Specific externally observable data points or event sources for each ZMOT trigger event that indicate the event such as News articles; press releases; social media post/ discussions; job postings; interviews in podcasts, webinars, etc.; Status pages; G2 reviews; glassdoor reviews; SEC filings; and any other pertinent publicly available data that might indicate this.
        - trigger_keywords: Specific keywords or short phrases in the observable moment that indicates the event such as "funding", "DDoS attack", "Local sales team hiring", "leadership change", etc.
        - match_score: float (0.0-1.0) indicating how likely this event is to trigger the pain in the ICP archetype.
        A match score of 0.7-1 indicates very likely, 0.3-0.6 indicates a moderate likelihood, and 0.0-0.2 indicates it is unlikely to trigger this pain.

The input is a list of jobs and pains in the following format:
{{
  "pain_id": "string",
  "pain": "pain text",
  "jobs": [
    "job description 1",
    "job description 2",
    ...
  ]
}}
        
Return your output as a JSON array, one entry per pain, with this structure:
  {{
    "pain_id": "string",
    "pain_triggers": [
      {{
        "attribute": "string",
        "dimension": "string",
        "direction": "string",
        "icp_archetypes": [
          {{
            "industries": ["string", ...],
            "revenues": ["string", ...],
            "employees": ["string", ...],
            "funding_stages": ["string", ...],
            "geographies": ["string", ...],
            "icp_match_score": 0.9
          }},
          {{
            "industries": ["string", ...],
            "revenues": ["string", ...],
            "employees": ["string", ...],
            "funding_stages": ["string", ...],
            "geographies": ["string", ...],
            "icp_match_score": 0.3
          }},
          ...
        ],
        "zmot_events": [
          {{
            "trigger_event": "string",
            "observable_moments": ["string", ...],
            "trigger_keywords": ["string", ...],
            "zmot_match_score": 0.8
          }},
          {{
            "trigger_event": "string",
            "observable_moments": ["string", ...],
            "trigger_keywords": ["string", ...],
            "zmot_match_score": 0.6
          }},
          ...
        ]
      }},
      {{
        "attribute": "string",
        "dimension": "string",
        "direction": "string",
        "icp_archetypes": [
          {{
            "industries": ["string", ...],
            "revenues": ["string", ...],
            "employees": ["string", ...],
            "funding_stages": ["string", ...],
            "geographies": ["string", ...],
            "icp_match_score": 0.9
          }},
          {{
            "industries": ["string", ...],
            "revenues": ["string", ...],
            "employees": ["string", ...],
            "funding_stages": ["string", ...],
            "geographies": ["string", ...],
            "icp_match_score": 0.3
          }},
          ...
        ],
        "zmot_events": [
          {{
            "trigger_event": "string",
            "observable_moments": ["string", ...],
            "trigger_keywords": ["string", ...],
            "zmot_match_score": 0.8
          }},
          {{
            "trigger_event": "string",
            "observable_moments": ["string", ...],
            "trigger_keywords": ["string", ...],
            "zmot_match_score": 0.6
          }},
          ...
        ]
      }},
      ...

    ]
  }}

Use only realistic, clearly defined jobs and persona roles. Do not invent exotic titles unless required by the domain. All capabilities should return at least one pain with structured jobs and personas.
IMPORTANT: Return ONLY the JSON array, with no explanation or formatting.
IMPORTANT: For each list (pain_triggers, icp_archetypes, zmot_events), you MUST provide at least 3-5 items. Do not return only one item for any list. If you cannot find 3 strong matches, look for slightly poorer matches.


Summary:
{summary}
Jobs and Pains::
{jobs_pains_json}
"""
    print("🔍 Job-Pains passed to OpenAI:", jobs_pains)
    print("🧠 Prompt being sent:\n", prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You create detailed business impact mappings from Jobs to Be Done."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content
        print("🧠 OpenAI raw response:", response.choices[0].message.content)
        # Parse and return the JSON output from OpenAI
        json_str = extract_json(content)
        print("🧠 OpenAI parsed JSON:", json_str)
        return json.loads(json_str)
    except Exception as e:
        print("❌ infer_pain_triggers_zmot_icp:", e)
        return []
