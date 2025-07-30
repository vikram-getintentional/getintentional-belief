import os
import json
from openai import OpenAI
from dotenv import load_dotenv


# 🔐 Load environment and OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 🚀 Extract summary and capabilities from website scrape
def extract_summary_and_capabilities(text: str, plg_cta_found: bool = False, footer_features: list[str] = None) -> dict:
    footer_list = "\n".join(footer_features or [])
    prompt = f"""
You're a neutral technology analyst. Based on this website content:

1. Summarize a concise 2–3 sentence value proposition.
2. List 5–8 key capabilities. For each, include a short, human-readable description.
- Capabilities should be clearly defined in the context of the product as understood by a target user.
- Avoid product-specific jargon or vague brand terms that don't convey clear functionality.
3. For each capability score its relevance to the product's value proposition on a scale of 0.0 to 1.0.

Use all available context, including main site content and footer feature hints.

Respond ONLY in this JSON format:
{{
  "summary": "...",
  "capabilities": [
    {{
      "name": "...",
      "description": "...",
      "coreness": 0.8  # Relevance score from 0.0 to 1.0
    }},
    ...
  ]
}}

---

Website content:
{text[:6000]}

---

Footer features:
{footer_list}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract marketing summaries from website text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print("❌ Error parsing or requesting OpenAI:", e)
        return {"summary": "Could not analyze site", "capabilities": []}

