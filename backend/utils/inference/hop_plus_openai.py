import json
from backend.utils.inference.openai_client import client  # uses our centralized OpenAI client

#Hop_plus OpenAI Prompt function to get upstream persona-job-pain triplets that are dependent on the current job being performed well
# In the openAI output include the original job as "original_job" and original persona as "original_persona.title+department+seniority"


def get_upstream_triplets(persona: dict,job: str) -> list[dict[str, any]]:
    """
    Given a persona, job description, pain description, department, and job title, this function queries OpenAI to produce a mapping in the following structure:

    [
      {
        "persona": {
          "title": "...",
          "department": "...",
          "seniority": "..."
        },
        "job": "...",
        "impact":0.5,
        "pain": "...",
        "pain_trigger": "Volume of incoming leads"
      },
      ...
    ]
    """

    prompt = f"""
      You are a {persona['title']} in the {persona['department']} team at {persona['seniority']} level with an original job – {job}.
      If you fail to do this job well, what upstream pain does it create, for whom, when performing what job?

      Provide a structured upstream impact mapping that shows:
      - For each original job, list the upstream business pain experienced by the failure of this job
      - For each upstream pain, include how much does failing this job impact this upstream pain (Scale: 0.0 to 1.0)
      - For each pain what attribute must scale in volume, frequency or complexity for this pain to become intolerable.
      - For each upstream pain list the upstream jobs that are blocked by this pain, or directly improved if this pain is alleviated.
      - For each upstream job, list the persona responsible for that job (include title, department, and seniority).
      

      Return your output in JSON format as a list of entries:
      [
      {{
          "original_job": "{job}",
          "dependent_pains": [{{
              "pain": "...",
              "pain_impact": "0.5",
              "pain_trigger": "Volume of incoming leads",
              "dependent_jobs": [{{
                  "description": "Job description that is blocked or affected",
                  "dependent_personas": [
                      {{
                          "title": "Job holder title",
                          "department": "Department",
                          "seniority": "Seniority level (must be one of: Junior, Operator, Manager, Senior, Executive)"
                      }},
                      ...
                  ]
              }}],
          }}]
      }},
      ...
      ]
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
