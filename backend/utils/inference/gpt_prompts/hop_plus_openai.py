import json
from backend.utils.inference.gpt_prompts.openai_client import client  # uses our centralized OpenAI client

#Hop_plus OpenAI Prompt function to get upstream persona-job-pain triplets that are dependent on the current job being performed well
# In the openAI output include the original job as "original_job" and original persona as "original_persona.title+department+seniority"


def get_upstream_triplets(triplet: dict) -> list[dict[str, any]]:
    """
    Given a set of triplets of persona, job description, pain description, department, and job title, this function queries OpenAI to produce a mapping of upstream pains, jobs and personas tied to each original job id.
    """
    job_list_json = json.dumps(triplet, indent=2)
    prompt = f"""

    You are given a list of persona-job-pain triplets, each with a relevance score.

    For each triplet, treat the persona as the actor, performing the specified job, and experiencing the specified pain (with the given relevance).
    Then do the following:
    1. List 1–3 upstream business pains directly resulting from the failure of this persona to perform their job well. These should be specific workflow inefficiencies or failure modes.
    2. For each pain, assign a relevance score as:
    - The relevance score must be a float between 0.0 and 1.0.
    - The array must be the same length as the list of capabilities, aligned by order.
    - Directly related capabilities should have scores between 0.7–1.0.
    - Indirectly related ones (e.g. same persona, downstream workflow, or shared pain) should have scores between 0.1–0.6.
    - Use 0.0 if the capability has no meaningful connection to the pain.
    3. List 1–3 jobs that are directly blocked or improved when this pain is solved in the context of the capability and product summary.
    4. For each job provide a score (0.0 to 1.0) indicating how directly the job is impacted by the pain. 0.7-1.0 indicates this pain always occurs in this job, 0.3-0.6 indicates this pain is common but not always present, 0.1-0.2 indicates this pain is rarely felt in this job, and 0.0 indicates this job is not affected by this pain.
    5. For each job, provide a list of personas responsible for that job, each with:
          - title
          - department
          - seniority (one of: Junior, Operator, Manager, Senior, Executive)
    6. For each persona provide a "job importance score" (0.0-1.0) indicating how critical this job is to the persona's role. 0.7-1.0 indicates this job is essential, 0.3-0.6 indicates it is important but not critical, and 0.1-0.2 indicates it is a minor task or responsibility.
          
      ---

      Input Jobs:
      {{
      {job_list_json}
      }}

      ---

     Return your output as a JSON array, one entry per capability, with this structure:
      - original job_id: string
      - pains: list of
          - pain: string
          - relevance: array of floats
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
