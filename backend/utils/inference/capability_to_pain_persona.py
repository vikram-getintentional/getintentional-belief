import json
from backend.utils.inference.openai_client import client  # uses our centralized OpenAI client

def infer_persona_job_pain_from_capabilities(summary, capabilities):
    """
    Given a product summary and its capabilities, this function queries OpenAI to produce a mapping in the following structure:

    [
      {
        "capability": "Automated forecasting",
        "description": "Enables accurate revenue prediction by automating the forecasting process.",
        "pains": [
          {
            "pain": "Manual spreadsheet forecasting causes delays and errors",
            "relevance": "0.85",
            "pain_trigger": "Increase in volume of incoming leads",
            "jobs": [
              {
                "description": "Forecast revenue across regions",
                "personas": [
                  {
                    "title": "VP of Sales",
                    "department": "Sales",
                    "seniority": "Senior Management"
                  },
                  {
                    "title": "Head of Sales Operations",
                    "department": "Sales",
                    "seniority": "Mid Management"
                  }
                ]
              }
            ]
          }
        ]
      },
      ...
    ]
    """
    capabilities_json = json.dumps(capabilities, indent=2)

    prompt = f"""
You are an expert in business design and job architecture.

For the given product summary and a list of product capabilities, provide a structured mapping that shows:
1. For each capability:
   - The workflow or business pains that this capability solves.
   - For each pain, rate the relevance of each capability in the list to this pain from 1.0 (directly highly relevant) to 0.5 (indirectly supports or partially resolves pain) to 0.0 (not relevant).
   - For each pain what attribute must scale in volume, frequency or complexity for this pain to become intolerable.
   - For each pain, list the jobs that are blocked or directly improved when this pain is alleviated.
   - For each job, list the personas responsible for that job (include title, department, and seniority).

Return your output in JSON format as a list of entries:
[
  {{
    "capability": "Capability Name",
    "pains": [
      {{
        "pain": "Description of the pain",
        "relevance": ["0.85", "0.45", "1.0"],
        "pain_trigger": "Increase in volume of incoming leads",
        "jobs": [
          {{
            "description": "Job description that is blocked or affected",
            "personas": [
              {{
                "title": "Job holder title",
                "department": "Department",
                "seniority": "Seniority level (must be one of: Junior, Operator, Manager, Senior, Executive)"
              }},
              ...
            ]
          }},
          ...
        ]
      }},
      ...
    ]
  }},
  ...
]

Summary:
{summary}

Capabilities:
{capabilities_json}
print("🔍 Capabilities passed to OpenAI:", capabilities)
print("🧠 Prompt being sent:\n", prompt)

"""

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
