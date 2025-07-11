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
1. List 1–3 business pains this capability solves. These should be specific workflow inefficiencies or failure modes.
- Each pain must be a clear, actionable description of a problem that the capability addresses.
2. For each pain, assign a relevance score to every capability in the list, even if that capability is only indirectly related or shares an overlapping job or data dependency.
- The relevance score must be a float between 0.0 and 1.0.
- The array must be the same length as the list of capabilities, aligned by order.
- Directly related capabilities should have scores between 0.7–1.0.
- Indirectly related ones (e.g. same persona, downstream workflow, or shared pain) should have scores between 0.1–0.6.
- Only use 0.0 if the capability has no meaningful connection to the pain.

3. For each pain specify what attribute must scale for this pain to become intolerable. The pain trigger should include: 
- Trigger attribute :What must increase or change for this pain to become intolerable,
- Dimension: 'volume', 'complexity', 'frequency', 'compliance', etc.,
- Direction: Increase, Decrease, or Change.
- Example: "Increasing Volume of incoming support tickets" > "Incoming support tickets", "Volume", "Increase". "Increasing Complexity of data processing tasks" > "Data processing", "Complexity", "Increase".
4. List 1–3 jobs that are directly blocked or improved when this pain is solved.
   For each job, provide a list of personas responsible for that job, each with:
      - title
      - department
      - seniority (one of: Junior, Operator, Manager, Senior, Executive)

Return your output in JSON format as a list of entries:
[
  {{
    "capability_id": "string",
    "capability": "string",
    "pains": [
      {{
        "pain": "string",
        "relevance": [0.8, 0.5, 1.0],
        "pain_trigger": {{
          "attribute": "Incoming support tickets",
          "dimension": "Volume",
          "direction": "Increase"
        }},
        "jobs": [
          {{
            "description": "Resolve incoming customer tickets in under 24 hours",
            "personas": [
              {{
                "title": "Customer Support Executive",
                "department": "Support",
                "seniority": "Operator"
              }},
              {{
                "title": "Support Team Lead",
                "department": "Support",
                "seniority": "Manager"
              }}
            ]
          }}
        ]
      }}
    ]
  }}
]

Use only realistic, clearly defined jobs and persona roles. Do not invent exotic titles unless required by the domain. All capabilities should return at least one pain with structured jobs and personas.

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
