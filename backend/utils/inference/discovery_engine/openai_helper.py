import os
import json
from openai import OpenAI
from dotenv import load_dotenv


# 🔐 Load environment and OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 🚀 Extract summary and capabilities from website scrape
def extract_summary_and_capabilities(text: str, plg_cta_found: bool = False, footer_features: list[str] = None) -> dict:
    print("Starting gpt prompt to extract summary and capabilities...")
    footer_list = "\n".join(footer_features or [])
    prompt = f"""
      You're a neutral technology analyst. Only based on the following inputs from the website, your job is to infer the core value proposition and key capabilities of a product or service.

      Input:

      Website content: json.dumps({text[:6000]})
      Footer features: json.dumps({footer_list})

      1. Value Proposition: 2-3 sentences. 
        The value proposition should be plain English sentences without jargon, unsubstantiated marketing claims or superlatives.
        The value proposition should be in a language that a neutral prospective customer would use to describe the product or service.
        Derive your result only from the provided text, not from external knowledge or assumptions.
      2. Domain: Infer the product's primary domain (e.g., "B2B SaaS", "E-commerce", "Healthcare").
      3. Industry: Infer the product's primary industry (e.g., "Finance", "Education", "Retail").
      4. List 5–8 key capabilities offered by the product. 
        For each capability describe a "name", "description", "relevance_label", and "likelihood label".
        The name should be a short, clear label (e.g., "Customer Support Automation").
        The description should be a concise, plain English explanation of what the capability does.
        Avoid brand/company-specific terms; focus on functionality. Use industry terms only if widely understood.
      5. For each capability provide a label describing its relevance to the product's value proposition between:{{"Critical","Core","Supportive","Ancillary","Out-of-scope"}} 
          Use the following inference criteria:
              1) Counterfactual: Would the product still be useful if this capability were removed? (Higher relevance if removing it would significantly reduce the product's value)
              2) Coverage: How often is this capability likely to be used in a typical user workflow? (Higher relevance if it is a common, essential part of user workflows)
              3) Substitutability: Is there another capability offered by this product that would still solve the same user flows if this capability did not exist? (Higher relevance if there are no substitutes for this capability within the product)
          Use the following labelling guide for your inference output:
            - Critical: Without it, the product stops being what it is.
            - Core: Strongly shapes the main value prop, used often.
            - Supportive: Helps but isn’t central.
            - Ancillary: Edge-case or rarely used.
            - Out-of-scope: Doesn’t really belong to this product.

      6. For each capability provide a label describing how likely or expected this capability is given the product category and domain as:{{"Essential","Expected","Common","Rare","Unique"}} 
          Use the following inference criteria:
              1) Counterfactual: In the market/category, would this capability be expected as part of the offering?
              2) Coverage (industry-level): How frequently do competitors / peer products include it?
              3) Substitutability (external): If missing, could customers easily find it in adjacent tools?
          Use the following labelling guide for your inference output:
            - Essential: Any product in this category would absolutely be expected to have it.
            - Expected: Most peers have it; omission would surprise users.
            - Common: Nice-to-have, common but not universal.
            - Rare: Rarely expected in this category.
            - Unique: Would be surprising/odd for this category.
      

      Use all available context, including main site content and footer feature hints.

      Respond ONLY in this JSON format:
      {{
        "summary": "...",
        "domain": "string",  # e.g., "B2B SaaS", "E-commerce", etc.
        "industry": "string",  # e.g., "Healthcare", "Finance", etc.
        "capabilities": [
          {{
            "name": "...",
            "description": "...",
            "relevance": "{{Critical, Core, Supportive, Ancillary, Out-of-scope}}",  # Relevance label
            "likelihood": "{{Essential, Expected, Common, Rare, Unique}}"  # Likelihood label
          }},
          ...
        ]
      }}

    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract marketing summaries from website text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print("❌ Error parsing or requesting OpenAI:", e)
        return {"summary": "Could not analyze site", "capabilities": []}
    


# 🚀 Extract summary and capabilities from website scrape
def validate_summary_and_persona_samples(website_text: str, review_samples: list[str], inferred_value_prop: str, inferred_capabilities: list[dict]) -> dict:
    prompt = f"""
      You're a neutral technology . Only based on detailed case studies and external reviews of a product or service, your job is to Infer the types of personas, their pains, jobs to be done, and trigger events that cause an organization to use this product.
      Input:
      - Case Studies Text: json.dumps({website_text[:6000]})
      - Review Samples: json.dumps({review_samples[:6000]})
      - Inferred Value Proposition: "{inferred_value_prop}"
      - Inferred Capabilities: {json.dumps(inferred_capabilities)}

      1. Expected Value Proposition: Based on the provided case studies and reviews, summarize the product's core value proposition in 2-3 words.
      2. Value Proposition Match: How well does the inferred value proposition in the input align with the expected value proposition from the case studies and reviews? Choose one:{{"Excellent", "Good", "Fair", "Poor"}}.
      3. Inferred Capabilities: List capabilities that are strongly supported by case studies or explicitly mentioned in reviews. For each capability, provide a "name" and "description".
      4. Potential Personas: Identify 3-5 key user personas from the provided case studies and reviews that use or derive value from the product.
        For each persona, provide:
        - Title (e.g., "Marketing Manager")
        - Department (e.g., "Marketing")
        - Seniority (e.g., "Mid-level")
        - Evidence snippet from a case study or review that supports this persona.
        - Match Score: How well does this persona align with the inferred value proposition? Choose one:{{"Excellent", "Good", "Fair", "Poor"}}.
      5. Pain Points: Identify 3-5 specific problems or challenges that users face, as described in the case studies and reviews.
        For each pain point, provide:
        - Description of the pain point (e.g., "Difficulty managing customer data")
        - Evidence snippet from a case study or review that supports this pain point.
        - Match Score: How well does this pain point align with the inferred value proposition? Choose one:{{"Excellent", "Good", "Fair", "Poor"}}.
      6. Jobs-to-be-Done: Identify 3-5 key tasks or goals that the product enabled the user to perform or unblock.
        For each job, provide:
        - Description of the job (e.g., "Automating email marketing campaigns")
        - Evidence snippet from a case study or review that supports this job.
        - Match Score: How well does this job align with the inferred value proposition? Choose one:{{"Excellent", "Good", "Fair", "Poor"}}.
      7. Trigger Events: Identify 3-5 external events or changes that led a customer to the need for this product.
        For each trigger event, provide:
        - Description of the event (e.g., "New compliance regulations")
        - Evidence snippet from a case study or review that supports this trigger event.
        - Match Score: How well does this trigger event align with the inferred value proposition? Choose one:{{"Excellent", "Good", "Fair", "Poor"}}.
      RULES:
        Use all available context provided. 
        Ensure all results are aligned with the inferred value proposition and capabilities.
        Do not invent or assume information not supported by the text.



      Respond ONLY in this JSON format:
      {{
        "expected_value_proposition": "...",
        "value_proposition_match": "{{Excellent, Good, Fair, Poor}}",
        "inferred_capabilities": [{{"name": "...", "description": "..."}}...],
        "personas": [{{"title": "...", "department": "...", "seniority": "...", "evidence": "...", "match_score":"..."}}...,],
        "pains": [{{"pain": "...", "evidence": "...", "match_score":"..."}}...],
        "jobs_to_be_done": [{{"job": "...", "evidence": "...", "match_score":"..."}}...],
        "trigger_events": [{{"event": "...", "evidence": "...", "match_score":"..."}}...]
      }}

    """
    print("Starting gpt prompt to validate summary and persona samples with prompt...")
    print(json.dumps(prompt))
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract marketing summaries from website text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print("❌ Error parsing or requesting Validation cycle:", e)
        return {"summary": "Could not analyze site", "capabilities": []}



"""
You are a neutral B2B technology analyst. Your role is to infer the core value proposition and key capabilities from the scraped website content of a product or service.
(homepage, product, features, solutions, pricing, tour/overview, customers, case studies, testimonials)
and from review platforms (G2, Capterra).

INPUTS - Payload includes:
- Key Product Pages: {_json_lite(product_content, 7000)}
- Key Vocabulary across website: {_json_lite(key_vocab, 22000)}
- PLG heuristic: {json.dumps(plg_heuristic)}

RULES
- Do not invent. If an item is not supported by the text, return an empty list [] or null.
- Use plain English; avoid jargon and unsubstantiated superlatives.
- Cap list sizes to keep precision: max 8 capabilities; max 3 personas, 5 pains, 5 jobs, 5 trigger events.
- Every customer-derived item (personas/pains/jobs/triggers) must include a short evidence snippet from a case study/review/testimonial.
- If evidence is weak or ambiguous, omit the item.
- Strictly grounduse and derive insights from only the provided text. Do not make assumptions or use any other external data, or draw inferences that do not align with the input texts.

Now infer the following only using the provided text as input:
1) Value Proposition:
  Summarize the product’s core value in 2-3 sentences. Describe this as a neutral expert in the product's domain and B2B software products.
  The value proposition should be plain English sentences without jargon, unsubstantiated marketing claims or superlatives.
  Derive an initial hypothesis from the homepage content and important phrases, and then validate it against product pages, case studies and other provided content.
  Provide a single concise value proposition statement that aligns with the context provided. 

2) Core Capabilities (5–8):
  Infer 5-8 core functional capabilities of the product.
  For each capability describe a "name", "description", and "relevance_label".
  The name should be a short, clear label (e.g., "Customer Support Automation").
  The description should be a concise, plain English explanation of what the capability does.
  Avoid brand/company-specific terms; focus on functionality. Use industry terms only if widely understood.
  Relevance label ∈ {{"Critical","Core","Supportive","Ancillary","Out-of-scope"}} using:
    1) Counterfactual: Would the product still be useful if this capability were removed?
    2) Coverage: How often is this capability likely to be used in a typical user workflow?
    3) Substitutability: Is there another capability offered by this product that would still solve the same user flows if this capability did not exist?

4) PLG Pathway: Determine if the product offers a self-serve path (evidence of “Sign up”, “Free trial”, “Get started”).
  Return has_plg_path true/false and the strongest exact phrase as evidence (or null).

5) Domain: Infer the product's primary domain (e.g., "B2B SaaS", "E-commerce", "Healthcare").

6) Industry: Best-fit vertical (e.g., "Finance", "Retail", "Healthcare").

7) Customer Insights (from case studies/testimonials/reviews only):
  - Personas: title, department, seniority if available, plus evidence snippet.
  - Pains: specific problems in customer voice, plus evidence snippet.
  - Jobs-to-be-done: tasks users needed help with, plus evidence snippet.
  - Trigger events: external events leading to product need (e.g., funding round, compliance change), plus evidence snippet.
  If data is insufficient for any category, return [].

OUTPUT (STRICT JSON ONLY; no markdown). If a field is unknown, use null or []:
{{
  "value_proposition": "...",
  "capabilities": [
    {{"name":"...", "description":"...", "relevance_label":"Critical/Core/Supportive/Ancillary/Out-of-scope"}}
  ],
  "plg": {{"has_plg_path": true/false, "evidence": "exact phrase or null"}},
  "domain": "...",
  "industry": "...",
  "customer_insights": {{
    "personas": [{{"title":"...", "department":"...", "seniority":"...", "evidence":"..."}}]],
    "pains": [{{"pain":"...", "evidence":"..."}}]],
    "jobs_to_be_done": [{{"job":"...", "evidence":"..."}}]],
    "trigger_events": [{{"event":"...", "evidence":"..."}}]]
  }}
}}
"""