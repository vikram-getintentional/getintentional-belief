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
        You are a {persona['title']} in the {persona['department']} team at {persona['seniority']} level with a {job} job.
        If you fail to do this job well, what pain does it create, for whom, when performing what job?

        For each upstream impact, provide:
        - The persona affected
        - Their job
        - The pain they would experience
        - How much does failing the original job impact this pain (Scale: 0.0 to 1.0)
        - What attribute must scale in volume, frequency or complexity for this new pain to become intolerable

        Return your output in JSON format as a list of entries:

        [
        {{
            "dependent_persona": {{
            "title": "...",
            "department": "...",
            "seniority": "..."
            }},
            "dependent_job": "...",
            "dependent_pain": "...",
            "dependent_pain_impact": "0.5",
            "dependent_pain_trigger": "Volume of incoming leads"
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
