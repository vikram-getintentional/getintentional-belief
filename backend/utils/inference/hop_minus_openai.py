import json
from backend.utils.inference.openai_client import client  # uses our centralized OpenAI client

#Hop_plus OpenAI Prompt function to get upstream persona-job-pain triplets that are dependent on the current job being performed well
# In the openAI output include the original job as "original_job" and original persona as "original_persona.title+department+seniority"


def get_downstream_triplets(persona: dict,job: str, pain: str) -> list[dict[str, any]]:
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
      You are a {persona['title']} in the {persona['department']} team at {persona['seniority']} level with an original job – {job} and original pain – {pain}.
      In the absence of a dedicated tool or solution, what downstream jobs would you require to collaborate with or delegate to, in order to resolve this pain? For whom?

      For each of the following pains, identify downstream dependencies as part of a causal graph traversal.

        1. For each input pain, list 1–3 downstream jobs to be done that are required to resolve this pain. 
            - Avoid abstract statements. Use job phrases like “Prepare monthly financial reports” or “Maintain lead scoring logic”.
            - Each job must describe a **workflow bottleneck**, **data unavailability**, or **preceding task failure** that prevents this job from being done well.
            - For each job, provide a **impact** score to indicate how severly the original pain is impacted by this job.
                - Use 0.7–1.0 for direct blockers, 0.3–0.6 for partial blockers or degraded context, and 0.1–0.2 for weak signals or related but non-critical issues.
        2. For each job, list 1–3 business pains that block or degrade this job.
            - Each pain should be a **specific task or responsibility** in a business process that describes a "job to be done".
            - The pain must be **causally upstream** — i.e., if this pain exists, the current job will be delayed, done poorly, or skipped.
            - The pain should not be a vague concern — it must be a **specific, operational dependency**.

        3. For each pain, provide:
            - **impact**: A float between 0.0 and 1.0 indicating how severely this pain affects the input job.
                - Use 0.7–1.0 for direct blockers
                - 0.3–0.6 for partial blockers or degraded context
                - 0.1–0.2 for weak signals or related but non-critical issues
            - A **pain_trigger** that describes when this pain becomes intolerable:
                - attribute: the real-world metric or variable (e.g., "Number of support tickets")
                - dimension: one of ["volume", "complexity", "frequency", "compliance", etc.]
                - direction: one of ["Increase", "Decrease", "Change"]

        4. For each upstream job, list 1–2 personas responsible, with:
            - title
            - department
            - seniority (one of: Junior, Operator, Manager, Senior, Executive)

      ---

      Input Pains:
      {{
      {pain_list_json}
      }}

      ---

      Return the result in the following JSON format:
      [
        {{
          "original_pain_id": "string",
          "downstream_jobs": [
            {{
              "job": "string",
              "impact": float,
              pains: [
                {{
                    "pain": "string",
                    "impact": float,
                    "pain_trigger": {{
                      "attribute": "string",
                      "dimension": "string",
                      "direction": "Increase" | "Decrease" | "Change"
                    }},
                personas: [
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
