import json
from backend.utils.inference.openai_client import client  # uses our centralized OpenAI client

def infer_persona_job_pain_from_capabilities(summary, capabilities):
    """
    Given a product summary and its capabilities, this function queries OpenAI to produce a mapping of pains, jobs and personas, and specific relevance scores corresponding to this pain-capability.
    """
    capabilities_json = json.dumps(capabilities, indent=2)

    prompt = f"""
You are an expert in business design and job architecture.
Given:
- A product summary
- A list of product capabilities
For each capability, do the following:
1. List 1–3 business pains this capability directly solves. These should be specific workflow inefficiencies or failure modes.
2. For each pain, assign a relevance score to every capability in the list, even if that capability is only indirectly related or shares an overlapping job or data dependency.
- The relevance score must be a float between 0.0 and 1.0.
- The array must be the same length as the list of capabilities, aligned by order.
- Directly related capabilities should have scores between 0.7–1.0.
- Indirectly related ones (e.g. same persona, downstream workflow, or shared pain) should have scores between 0.1–0.6.
- Use 0.0 if the capability has no meaningful connection to the pain.

3. For each pain Specify what attribute must scale for this pain to become intolerable in the format of:
      - attribute: the real-world metric or variable (e.g., "Number of support tickets")
      - dimension: one of ["volume", "complexity", "frequency", "compliance", etc.]
      - direction: one of ["Increase", "Decrease", "Change"] (choose from: volume, frequency, complexity, or describe the trigger in plain terms).
4. List 1–3 jobs that are directly blocked or improved when this pain is solved in the context of the capability and product summary.
5. For each job provide a score (0.0 to 1.0) indicating how directly the job is impacted by the pain. 0.7-1.0 indicates this pain always occurs in this job, 0.3-0.6 indicates this pain is common but not always present, 0.1-0.2 indicates this pain is rarely felt in this job, and 0.0 indicates this job is not affected by this pain.
6. For each job, provide a list of personas responsible for that job, each with:
      - title
      - department
      - seniority (one of: Junior, Operator, Manager, Senior, Executive)
7. For each persona provide a "job importance score" (0.0-1.0) indicating how critical this job is to the persona's role. 0.7-1.0 indicates this job is essential, 0.3-0.6 indicates it is important but not critical, and 0.1-0.2 indicates it is a minor task or responsibility.

Return your output as a JSON array, one entry per capability, with this structure:
- capability_id: string
- capability: string
- pains: list of
    - pain: string
    - relevance: array of floats
    - pain_trigger: object
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

Summary:
{summary}

Capabilities:
{capabilities_json}
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
        return json.loads(content)
    except Exception as e:
        print("❌ Error in infer_persona_job_pain_from_capabilities:", e)
        return []
