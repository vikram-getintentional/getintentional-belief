import json
from backend.utils.inference.gpt_prompts.openai_client import client  # uses our centralized OpenAI client

#Hop_plus OpenAI Prompt function to get upstream persona-job-pain triplets that are dependent on the current job being performed well
# In the openAI output include the original job as "original_job" and original persona as "original_persona.title+department+seniority"


def get_upstream_triplets(summary, job_sets: list[dict[str, any]]) -> list[dict[str, any]]:
    """
    Given a set of triplets of persona, job description, pain description, department, and job title, this function queries OpenAI to produce a mapping of upstream pains, jobs and personas tied to each original job id.
    """
    job_list_json = json.dumps(job_sets, indent=2)
    prompt = f"""

    You are given a list of Jobs to be done in an organization along with the personas typically doing them, and the pains faced while performing this job. 
    Now answer the following in the context of the product summary provided below:

    For each job, treat the personas as actors performing the specified job, and experiencing the specified pains.
    Then do the following:
    1. List 1–3 upstream business pains internal to the organization resulting from the failure of this persona to perform their job well. These should be specific workflow inefficiencies or failure modes experienced by a persona while performing a job within the organization.
    2. For each pain, assign a Severity Score as a float between 0.0 and 1.0 that reflects how much it is impacted by the job's failure.
    - Severity scores of 0.7-1.0 indicate that the pain is a direct and complete result of this job's failure.
    - Scores of 0.3–0.6 indicate that the pain is only partially caused by this job's failure (e.g. same persona, downstream workflow, or shared pain).
    - Use 0.0 if the capability has no meaningful connection to the pain.
    3. For each pain, List 1–3 jobs that are directly blocked or improved when this pain is solved in the product summary.
    4. For each job provide a score (0.0 to 1.0) indicating how directly the job is impacted by the pain. 0.7-1.0 indicates this pain always occurs in this job, 0.3-0.6 indicates this pain is common but not always present, 0.1-0.2 indicates this pain is rarely felt in this job, and 0.0 indicates this job is not affected by this pain.
    5. For each job, provide a list of personas responsible for that job, each with:
          - title
          - department
          - seniority (one of: Junior, Operator, Manager, Senior, Executive)
    6. For each persona provide a "job importance score" (0.0-1.0) indicating how critical this job is to the persona's role. 0.7-1.0 indicates this job is essential, 0.3-0.6 indicates it is important but not critical, and 0.1-0.2 indicates it is a minor task or responsibility.
    7. Only if no reliable internal upstream pains can be identified, or if the job is clearly the result of an external factor, mention the pain_source as external and leave the rest of the results empty.
          
      ---

      Input Jobs:
      {{
      {job_list_json}
      }}

      ---

     Return your output as a JSON array, one entry per capability, with this structure:
      - original_job_id: string
      - pain_source: string (either "internal" or "external")
      - pains: list of
          - pain: string
          - severity: float
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
