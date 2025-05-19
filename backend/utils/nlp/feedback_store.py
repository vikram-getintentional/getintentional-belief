import json
import os
from pathlib import Path

FEEDBACK_FILE = Path("backend/utils/knowledge_base/persona_jobs_pains.json")

def save_feedback(persona, new_pains: list[str]):
    if not FEEDBACK_FILE.exists():
        print("⚠️ persona_jobs_pains.json file not found.")
        return

    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)
    
    # Handle if persona is a string
    if isinstance(persona, str):
        persona_name = persona
        persona_dict = None
    elif isinstance(persona, dict):
        persona_name = persona.get("name")
        persona_dict = persona
    else:
        print("⚠️ Invalid persona type. Skipping.")
        return
    
    if not persona_name:
        print("⚠️ Persona has no name. Skipping.")
        return

    # Look for existing persona by name
    existing = next((p for p in data["personas"] if p["name"] == persona_name), None)

    if existing:
        # Add only new pains
        added = 0
        for pain in new_pains:
            if pain not in existing["pains"]:
                existing["pains"].append(pain)
                added += 1
        if added:
            print(f"✅ Added {added} new pain(s) to {persona_name}")
    elif persona_dict:
        # Add the full persona with provided pains
        persona_to_add = {
            "name": persona_dict.get("name"),
            "department": persona_dict.get("department", ""),
            "seniority": persona_dict.get("seniority", ""),
            "responsibilities": persona_dict.get("responsibilities", ""),
            "pains": list(set(new_pains))  # Remove dupes
        }
        data["personas"].append(persona_to_add)
        print(f"✅ Added new persona: {persona_name}")
    else:
        print(f"⚠️ Skipping unknown persona without full data: {persona_name}")
        return

    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Feedback saved for {persona_name}")

def load_feedback():
    if FEEDBACK_FILE.exists():
        with FEEDBACK_FILE.open() as f:
            return json.load(f)
    return {}
