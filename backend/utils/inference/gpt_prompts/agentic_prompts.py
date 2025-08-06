import json
from backend.utils.inference.gpt_prompts.openai_client import client  # uses our centralized OpenAI client

import re

def infer_pain_and_source(summary, domain, industry, job_sets: list[dict[str, any]]) -> list[dict[str, any]]:
    """
    Given a set of triplets of persona, job description, pain description, department, and job title, 
    this function queries OpenAI to infer whether a job has internal or external upstream pains.
    """
    job_list_json = json.dumps(job_sets, indent=2)
    prompt = f"""

    You are given a list of Jobs To Be Done (JTBD) in an organization along with the personas doing them, and pains they face while performing this job. 
    Now answer the following in the context of the product summary, domain, and industry provided below.
    Given:
    - Product value proposition: {summary}
    - Product domain: {domain}
    - Product industry: {industry}
    - Jobs to be done, and pains in the context of this product: {job_list_json}

    For each job, Infer whether the job exists as a response to an internal pain or an external event.
        Internal: it is caused by a workflow inefficiency, process bottleneck, or organizational issue faced by another persona in the same or different department while performing the job.
        External: it is caused by market conditions, customer demands, or other external factors that cannot be directly controlled by the organization.
    Infer a job as External only if the following conditions are met:
    - The nature of a job is clearly external-facing, such as directly managing investor, shareholder, partner, or customer relationships; or relying heavily on external inputs such as legal, compliance, regulatory requirements, etc.
    - The job does not have a strong internal organizational connection or issue
    - The job is typically not a delegation from another senior persona or complementary department.
    Return your result for each job in the following format:
    - original_job_id: string
    - pain_source: string (either "internal" or "external")
      ---
      
      IMPORTANT: Return ONLY the JSON array, with no explanation or formatting for every capability in the list above. Do not skip any job.
      For each job in the input list, process it independently and return a separate JSON object using the job_id as the key. Do not let the context of one job influence the others.
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
        return extract_json(content)
    except Exception as e:
        print("❌ Error in get_upstream_triplets:", e)
        return []
    

def infer_upstream_for_internal_jobs(summary, domain, industry, job_sets: list[dict[str, any]]) -> list[dict[str, any]]:
    """
    Given a set of triplets of persona, job description, pain description, department, and job title, 
    this function queries OpenAI to produce a mapping of upstream pains, jobs and personas tied to each original job id.
    """
    job_list_json = json.dumps(job_sets, indent=2)
    prompt = f"""

    You are given a list of Jobs To Be Done (JTBD) in an organization along with the personas doing them, and pains they face while performing this job. 
    Now answer the following in the context of the product summary, domain, and industry provided below.

    - Product value proposition: {summary}
    - Product domain: {domain}
    - Product industry: {industry}
    - Jobs to be done, and pains in the context of this product: {job_list_json}

    Do the following:
    1. List 1-3 upstream business pains that this job exists to solve.
    - Upstream pains should be internal to the organization, faced by another persona in the same or different department.
    - The upstream pain should be a specific workflow inefficiency or business problem that is directly addressed by the job.
    - The pain should be causally upstream of the job, meaning that the job exists to resolve this upstream pain.
    - The pain should be temporally upstream of the job, meaning that it is experienced before the job is performed.
    - You may think of the current job as a delegation to resolve this upstream pain, for example from a senior persona or from a complementary department.
    - Avoid creating cycles or loops with the input jobs/pains unless there is a strong, unavoidable business logic reason.
    
    2. For each pain, assign a Severity Score as a float between 0.0 and 1.0 that reflects how directly and completely it is solved by the job.
    - Severity scores of 0.7-1.0 indicate that the pain is a completely solved by the job.
    - Scores of 0.3–0.6 indicate that the pain is only partially solved by this job, and other jobs may be necessary to completely solve it. (e.g. same persona, downstream workflow, or shared pain).
    - Use 0.0 only if the pain has no meaningful connection to the job.
    3. For each pain specify at least 3-5 pain triggers in the context of the product domain and industry as the attribute that must scale in the context of this product's domain for this pain to become intolerable in the format of:
      - attribute: the real-world metric or variable (e.g., "Number of support tickets")
      - dimension: one of ["volume", "complexity", "frequency", "compliance", etc.]
      - direction: one of ["Increase", "Decrease", "Change"] (choose from: volume, frequency, complexity, or describe the trigger in plain terms).
    4. For each pain specify at least 2 pain_perceived_metrics in the context of the product domain and industry as the metric or data point that indicates the pain is being felt in the organization.
    You may think of the perceived metrics as a data point that a persona performing the job might present to their manager or executive to indicate that this pain is being felt in the organization.
    5. For each pain, List 1–3 upstream jobs during which this pain is typically experienced. 
       You may infer that these jobs as a Job to Be Done in the context of the product, that is directly blocked by this pain, and improved when this pain is solved.
    6. For each job provide a score (0.0 to 1.0) indicating how directly the job is impacted by the pain. 0.7-1.0 indicates this pain always occurs in this job, 0.3-0.6 indicates this pain is common but not always present, 0.1-0.2 indicates this pain is rarely felt in this job, and 0.0 indicates this job is not affected by this pain.
    7. For each job, provide a list of personas responsible for that job, each with:
          - title
          - department
          - seniority (one of: Junior, Operator, Manager, Senior, Executive)
    8. For each persona provide a "job importance score" (0.0-1.0) indicating how critical this job is to the persona's role. 0.7-1.0 indicates this job is essential, 0.3-0.6 indicates it is important but not critical, and 0.1-0.2 indicates it is a minor task or responsibility.
          
      ---

     Return your output as a JSON array, one entry per capability, with this structure:
      - original_job_id: string
      - pains: list of
          - pain: string
          - severity: float
          - pain_triggers: list of
                - attribute: string
                - dimension: string
                - direction: string
        - pain_perceived_metrics: list of
            - metric: string
          - jobs: list of
              - description: string
              - impact: float
              - personas: list of
                  - job_importance: float
                  - title: string
                  - department: string
                  - seniority: string

      Use only realistic, clearly defined jobs and persona roles. Do not invent exotic titles unless required by the domain. All capabilities should return at least one pain with structured jobs and personas.
      IMPORTANT: Return ONLY the JSON array, with no explanation or formatting for every capability in the list above. Do not skip any job.
      For each job you must infer at least 1 plausible upstream pain and job that are NOT already present in the input list. Use your knowledge of typical business processes in this domain and industry to hypothesize what could be upstream, even if it is not explicitly mentioned. Avoid simply repeating jobs or pains from the input unless absolutely necessary.
      IMPORTANT: For each job, do not simply repeat jobs or pains from the input list.
      For each job in the input list, process it independently and return a separate JSON object using the job_id as the key. Do not let the context of one job influence the others.
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
        return extract_json(content)
    except Exception as e:
        print("❌ Error in get_upstream_triplets:", e)
        return []


def infer_zmot_for_external_jobs(summary, domain, industry, jobs_pains):
    """
    Given a list of pains and jobs to be done that they are experienced in, 
    this function queries OpenAI to produce a mapping of ZMOTs
    """
    jobs_pains_json = json.dumps(jobs_pains, indent=2)

    prompt = f"""
    You are an expert in business design and job architecture. You are tasked with identifiying the ideal target customers for a product and the specific trigger events that drive these customers.
    Given:
    - Product value proposition: {summary}
    - Product domain: {domain}
    - Product industry: {industry}
    - Jobs to be done, and pains in the context of this product: {jobs_pains_json}

    For each job, do the following:
    1. Specify at least 3-5 external trigger events (ZMOTs) in the context of the product domain and industry that likely caused or accelerated the need for this job to be done.
    - A ZMOT is an external event such as a market change, customer demand, regulatory change, industry trend, or other external factor that drives the need for this job.
    - The ZMOT should be a specific, measurable event that is relevant to the product's domain and industry.
    - The ZMOT should be causally related to the job, meaning that it is a trigger that leads to the job being performed.
    - The ZMOT should be temporally related to the job, meaning that it occurs before or during the time the job is performed.
    2. For each ZMOT trigger, provide a match score as a float (0.0-1.0) indicating how likely this event is to trigger this job in the organization.
    3. For each ZMOT include at least 3 Observable Moments - as Specific externally observable data points or event sources to infer the occurance of this eventsuch as News articles; press releases; social media post/ discussions; job postings; interviews in podcasts, webinars, Website Status pages; G2 reviews; glassdoor reviews; SEC filings; and any other pertinent trackable data that might indicate this.
            - For each observable moment provide a match_score as a float (0.0-1.0) indicating how likely this observable moment is to indicate the trigger_event.
    4. For each trigger event include trigger_keywords as specific keywords or short phrases in the observable moment that can be used to identify the event in a search or analysis context.
            - Use terms and phrases specific to the product context or industry. Avoid generic business jargon unless absolutely necessary.
            - For each trigger_keywords provide a match_score as a float (0.0-1.0) indicating how likely this keyword is to appear when the trigger_event occurs.
     
    The input is a list of jobs to be done, the pains they solve, and personas solving them in the following format:
    - job_id: string
    - job_description: string
    - pains: list of pain descriptions
    - personas performing job: list of dictionaries with keys:
        - persona title: string
        - persona department: string
        - persona seniority: string (one of: Junior, Operator, Manager, Senior, Executive)

            
    Return your output as a JSON array, one entry per job, with this structure:
    - original_job_id: string
    - zmot_triggers: list of
        - trigger_event: string
        - match_score: float (0.0-1.0)
        - observable_moments: list of
            - observable_moment: string
            - match_score: float (0.0-1.0)
        - trigger_keywords: list of
            - keyword: string
            - match_score: float (0.0-1.0)

    Use only realistic, clearly defined pain triggers, events, observable moments and keywords. 
    IMPORTANT: Return ONLY the JSON array, with no explanation, formatting, or trailing commas. Ensure all arrays and objects are properly closed.
    IMPORTANT: For each list, you MUST provide at least 3-5 items. Do not return only one item for any list. If you cannot find 3 strong matches, look for slightly poorer matches.
    IMPORTANT: Each ZMOT event, and keyword must be tightly linked to the context of the product’s value proposition. Do not generalize — reference domain-specific signals wherever possible.
    Be brutally specific with the returned values and match scores. 

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
        return extract_json(json_str)
    except Exception as e:
        print("❌ infer_pain_triggers_zmot:", e)
        return []
    

    
def infer_persona_job_pain_from_capabilities(summary, domain, industry, capabilities):
    """
    Given a product's details and its capabilities, this function queries OpenAI to produce a mapping of pains, jobs and personas, and specific relevance scores corresponding to this pain-capability.
    """
    capabilities_json = json.dumps(capabilities, indent=2)

    prompt = f"""
    You are an expert in business design and job architecture.
    Given:
    - Product value proposition: {summary}
    - Product domain: {domain}
    - Product industry: {industry}
    - Product Capabilities: {capabilities_json}
    
    For each capability, do the following in the context of the product, its domain and industry:
    1. List 1–3 business pains this capability directly solves. These should be specific workflow inefficiencies or failure modes.
    2. For each pain, assign a relevance score to every capability in the list, even if that capability is only indirectly related or shares an overlapping job or data dependency.
    - The relevance score must be a float between 0.0 and 1.0.
    - The array must be the same length as the list of capabilities, aligned by order.
    - Directly related capabilities should have scores between 0.7–1.0.
    - Indirectly related ones (e.g. same persona, downstream workflow, or shared pain) should have scores between 0.1–0.6.
    - Use 0.0 if the capability has no meaningful connection to the pain.
    3. For each pain specify at least 3-5 pain triggers in the context of the product domain and industry as the attribute that must scale in the context of this product's domain for this pain to become intolerable in the format of:
      - attribute: the real-world metric or variable (e.g., "Number of support tickets")
      - dimension: one of ["volume", "complexity", "frequency", "compliance", etc.]
      - direction: one of ["Increase", "Decrease", "Change"] (choose from: volume, frequency, complexity, or describe the trigger in plain terms).
    4. For each pain specify at least 2 pain_perceived_metrics in the context of the product domain and industry as the metric or data point that indicates the pain is being felt in the organization.
    You may think of the perceived metrics as a data point that a persona performing the job might present to their manager or executive to indicate that this pain is being felt in the organization.
    5. For each pain, List 1–3 upstream jobs during which this pain is typically experienced. 
    You may infer that these jobs as a Job to Be Done in the context of the product, that is directly blocked by this pain, and improved when this pain is solved.
    6. For each job provide a score (0.0 to 1.0) indicating how directly the job is impacted by the pain. 0.7-1.0 indicates this pain always occurs in this job, 0.3-0.6 indicates this pain is common but not always present, 0.1-0.2 indicates this pain is rarely felt in this job, and 0.0 indicates this job is not affected by this pain.
    7. For each job, provide a list of personas responsible for that job, each with:
        - title
        - department
        - seniority (one of: Junior, Operator, Manager, Senior, Executive)
    8. For each persona provide a "job importance score" (0.0-1.0) indicating how critical this job is to the persona's role. 0.7-1.0 indicates this job is essential, 0.3-0.6 indicates it is important but not critical, and 0.1-0.2 indicates it is a minor task or responsibility.
    
    Return your output as a JSON array, one entry per capability, with this structure:
    - capability_id: string
    - capability: string
    - pains: list of
        - pain: string
        - relevance: array of floats
        - pain_triggers: list of
            - attribute: string
            - dimension: string
            - direction: string
        - pain_perceived_metrics: list of
            - metric: string
        - jobs: list of
            - description: string
            - impact: float
            - personas: list of
                - job_importance: float
                - title: string
                - department: string
                - seniority: string

    Use only realistic, clearly defined jobs and persona roles. Do not invent exotic titles unless required by the domain. All capabilities should return at least one pain with structured jobs and personas.
    IMPORTANT: Return ONLY the JSON array, with no explanation or formatting.
    Return a JSON entry for every capability in the list above. Do not skip any capability.

    
    """
    print("🔍 Capabilities passed to OpenAI:", capabilities)
    print("🧠 Prompt being sent:\n", prompt)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You create detailed business impact mappings from product capabilities."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        print("🧠 OpenAI raw response:", response.choices[0].message.content)
        # Parse and return the JSON output from OpenAI
        return extract_json(content)
    except Exception as e:
        print("❌ Error in infer_persona_job_pain_from_capabilities:", e)
        return []

def infer_icp(summary, domain, industry, capabilities_mapping):
    """
    Given a set of capabilties mapping including Hop0 jobs, pains, and capabilities
    this function queries OpenAI to produce a mapping of ICP Archetypes typically solved by this product.
    """
    capabilities_mapping_json = json.dumps(capabilities_mapping, indent=2)
    prompt = f"""
        You are a B2B market segmentation expert. You are given a list of product capabilities, primary pains, and jobs to be done (JTBD) within an organization, along with the product's domain and industry.

        Your task is to identify 5–10 **Ideal Customer Profile (ICP) archetypes** — distinct organizational types that are most likely to experience these pains and perform these jobs, given the product context.

        Given:
        - Product value proposition: {summary}
        - Product domain: {domain}
        - Product industry: {industry}
        - Capabilities, jobs, and pains (JSON): {capabilities_mapping_json}

        For each ICP archetype:
        - Choose **exactly one** value for each of the following attributes:
            - `industry` (e.g., "Healthcare", "Finance", "Retail")
            - `revenue_range` (e.g., "0-1M", "1-10M", "10-100M", "100-500M", "500M-1B", "1B+")
            - `employee_range` (e.g., "1-10", "11-50", "51-200", "201-500", "501-1000", "1000-5000", "5000-10000", "10000+")
            - `funding_stage` (e.g., "Pre-Seed", "Seed", "Series A", "Series B", "Series C+", "Public")
            - `geography` (e.g., "North America", "Europe", "Asia", "South America", "Africa", "Australia")

        - For each archetype, provide a single `match_score` (float between 0.0–1.0) that reflects how well this organizational type aligns with the provided jobs and pains. Use this scale:
            - 0.7–1.0: Very likely fit
            - 0.4–0.6: Moderate fit
            - 0.0–0.3: Unlikely

        Return your answer as a **strict JSON array**, where each entry is a full ICP archetype. Use this structure:

        ```json
        [
        {{
            "industry": "Healthcare",
            "revenue_range": "100-500M",
            "employee_range": "1000-5000",
            "funding_stage": "Series C+",
            "geography": "North America",
            "match_score": 0.87
        }},
        ...
        ]
        IMPORTANT:

        Do not return multiple values per attribute. Each archetype must use only one value per attribute.
        Do not include any commentary or explanation.
        Return only the JSON array.
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
        return extract_json(content)
    except Exception as e:
        print("❌ Error in get_upstream_triplets:", e)
        return []
    
    

def extract_json(text):
    # Extract the first JSON array or object from the text
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        return match.group(1)
    return text  # fallback