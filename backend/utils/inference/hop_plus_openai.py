import json
from backend.utils.inference.openai_client import client  # uses our centralized OpenAI client

#Hop_plus OpenAI Prompt function to get upstream persona-job-pain triplets that are dependent on the current job being performed well
# In the openAI output include the original job as "original_job" and original persona as "original_persona.title+department+seniority"


def get_upstream_triplets(triplet: dict) -> list[dict[str, any]]:
    """
    Given a set of triplets of persona, job description, pain description, department, and job title, this function queries OpenAI to produce a mapping of upstream pains, jobs and personas tied to each original job id.
    """
    job_list_json = json.dumps(triplet, indent=2)
    prompt = f"""

    You are given a list of persona-job-pain triplets, each with a relevance score.

    For each triplet, do the following:
    1. Treat the persona as the actor, performing the specified job, and experiencing the specified pain (with the given relevance).
    2. Identify 1–3 **upstream business pains** directly resulting from the failure of this persona to perform their job well.
      - Each pain must describe a **workflow bottleneck**, **data unavailability**, or **preceding task failure** that prevents this job from being done well.
      - The pain must be **causally upstream** — i.e., if this pain exists, the current job is essential.
      - The pain should not be a vague concern — it must be a **specific, operational dependency**.
    3. For each upstream pain, provide:
      - **impact**: A float between 0.0 and 1.0 indicating the degree to which the original job's failure drives this pain.
      - an impact score of 0.7-1.0 indicates not performing this job well causes debilitating pain upstream, 0.3-0.6 indicates partial pain that can be mitigated with workarounds, and 0.1-0.2 indicates weak or non-critical pains that can be lived with.
      - **pain_trigger**: An object with:
        - attribute: the real-world metric or variable (e.g., "Number of support tickets")
        - dimension: one of ["volume", "complexity", "frequency", "compliance", etc.]
        - direction: one of ["Increase", "Decrease", "Change"]
    4. For each upstream pain, list 1–2 **upstream jobs to be done** where this pain is typically experienced.
        - Each job should be a **specific task or responsibility** in a business process that describes a "job to be done".
        - Avoid abstract statements. Use job phrases like “Prepare monthly financial reports” or “Maintain lead scoring logic”.

    5. For each job, list 1–2 responsible personas with:
        - title
        - department
        - seniority (one of: Junior, Operator, Manager, Senior, Executive)
          
      ---

      Input Jobs:
      {{
      {job_list_json}
      }}

      ---

      Return the result in the following JSON format:
      [
        {{
          "original_job_id": "string",
          "upstream_pains": [
            {{
              "pain": "string",
              "impact": float,
              "pain_trigger": {{
                "attribute": "string",
                "dimension": "string",
                "direction": "Increase" | "Decrease" | "Change"
              }},
              "upstream_jobs": [
                {{
                  "description": "string",
                  "personas": [
                    {{
                      "title": "string",
                      "department": "string",
                      "seniority": "Junior" | "Operator" | "Manager" | "Senior" | "Executive"
                    }}
                  ]
                }}
              ]
            }}
          ]
        }}
      ]

      All output must be realistic, based on actual workflows and organizational roles. Do not invent exotic personas or vague responsibilities.
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
