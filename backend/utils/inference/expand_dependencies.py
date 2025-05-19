import os
import json
from openai import OpenAI
from dotenv import load_dotenv

client = OpenAI()

def expand_dependent_personas(primary_persona, job, pain):
    prompt = f"""
You are the {primary_persona['title']} in the {primary_persona['department']} team.

You are responsible for the job: "{job}" and are currently experiencing the pain: "{pain}".

Tell us:
1. Who in your company relies on your successful execution of this job and is directly impacted if you fail to deliver on this job?
2. What job are they trying to do?
3. What pain do they face if your job isn't done well?

Respond in this JSON format:
[
  {{
    "dependent_persona": {{
      "title": "...",
      "department": "...",
      "seniority": "..."
    }},
    "dependent_job": "...",
    "dependent_pain": "..."
  }},
  ...
]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You infer org-level job dependencies from persona pain."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print("❌ GPT Step 2 error:", e)
        return []
