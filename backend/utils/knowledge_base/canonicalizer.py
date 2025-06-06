import uuid
from pathlib import Path
from backend.utils.knowledge_base.canonical_maps.canonical_loader import save_canonical_map
from backend.utils.knowledge_base.canonical_maps.canonical_utils import assign_canonical_labels, cluster_items
from backend.utils.embedding.embed_utils import generate_and_save_embeddings

# File paths
CANONICAL_MAP_PATH = Path("backend/utils/knowledge_base/canonical_maps/")
EMBEDDING_OUTPUT_PATH = CANONICAL_MAP_PATH / "canonical_embeddings/"

# Map Paths
JOB_CANONICAL_MAP_PATH = CANONICAL_MAP_PATH / "job_to_canonical.json"
PAIN_CANONICAL_MAP_PATH = CANONICAL_MAP_PATH / "pain_to_canonical.json"
PERSONA_CANONICAL_MAP_PATH = CANONICAL_MAP_PATH / "persona_to_canonical.json"

# Preloaded embeddings
try:
    from backend.utils.knowledge_base.canonical_maps.canonical_loader import load_embeddings
    JOB_EMBEDDINGS = load_embeddings("job")
    PAIN_EMBEDDINGS = load_embeddings("pain")
    PERSONA_EMBEDDINGS = load_embeddings("persona")
except Exception as e:
    print(f"⚠️ Embedding preload failed: {e}")
    JOB_EMBEDDINGS = {}
    PAIN_EMBEDDINGS = {}
    PERSONA_EMBEDDINGS = {}

# Utility
def generate_new_id(prefix="pain"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def canonicalize_job(jobs: list[str]) -> dict:
    print("Starting job canonicalization")
    # Step 0: Dedupes
    jobs = list(set(jobs))
    if not jobs:
        print("⚠️ No jobs provided for canonicalization.")
        return {}

    # Step 1: Generate embeddings
    print(f"🔍 Generating embeddings for {len(jobs)} jobs.")
    job_embeddings = generate_and_save_embeddings(jobs, "job")

    # Handle case with only one embedding
    if len(job_embeddings) == 1:
        print("⚠️ Only one job provided. Skipping clustering.")
        canonical_map = {jobs[0]: jobs[0]}
        save_canonical_map(canonical_map, JOB_CANONICAL_MAP_PATH)
        print(f"✅ Job canonical map saved to {JOB_CANONICAL_MAP_PATH}")
        return canonical_map

    # Step 2: Cluster jobs
    clustered_jobs = cluster_items(job_embeddings)
    print(f"🔍 Clustered jobs into {len(clustered_jobs)} clusters.")

    # Step 3: Assign canonical labels
    canonical_map = assign_canonical_labels(clustered_jobs)

    # Step 4: Save the canonical map
    save_canonical_map(canonical_map, JOB_CANONICAL_MAP_PATH)
    print(f"✅ Job canonical map saved to {JOB_CANONICAL_MAP_PATH}")

    return canonical_map

def canonicalize_pain(pains: list[str]) -> dict:
    print("Starting pain canonicalization")
    # Step 0: Dedupes
    pains = list(set(pains))
    
    if not pains:
        print("⚠️ No pains provided for canonicalization.")
        return {}

    # Step 1: Generate embeddings
    print(f"🔍 Generating embeddings for {len(pains)} pains.")
    pain_embeddings = generate_and_save_embeddings(pains, "pain")
    
    # Handle case with only one embedding
    if len(pain_embeddings) == 1:
        print("⚠️ Only one pain provided. Skipping clustering.")
        canonical_map = {pains[0]: pains[0]}
        save_canonical_map(canonical_map, PAIN_CANONICAL_MAP_PATH)
        print(f"✅ Pain canonical map saved to {PAIN_CANONICAL_MAP_PATH}")
        return canonical_map

    # Step 2: Cluster pains
    clustered_pains = cluster_items(pain_embeddings)
    print(f"🔍 Clustered pains into {len(clustered_pains)} clusters.")

    # Step 3: Assign canonical labels
    canonical_map = assign_canonical_labels(clustered_pains)

    # Step 4: Save the canonical map
    save_canonical_map(canonical_map, PAIN_CANONICAL_MAP_PATH)
    print(f"✅ Pain canonical map saved to {PAIN_CANONICAL_MAP_PATH}")

    return canonical_map

def canonicalize_pain_trigger(pain_triggers: list[str]) -> dict:
    print("Starting pain trigger canonicalization")
    # Step 0: Dedupes
    pain_triggers = list(set(pain_triggers))
    
    if not pain_triggers:
        print("⚠️ No pain triggers provided for canonicalization.")
        return {}

    # Step 1: Generate embeddings
    print(f"🔍 Generating embeddings for {len(pain_triggers)} pain triggers.")
    pain_trigger_embeddings = generate_and_save_embeddings(pain_triggers, "pain_trigger")
    
    # Handle case with only one embedding
    if len(pain_trigger_embeddings) == 1:
        print("⚠️ Only one pain trigger provided. Skipping clustering.")
        canonical_map = {pain_triggers[0]: pain_triggers[0]}
        # Optionally save to a canonical map file if you want
        return canonical_map

    # Step 2: Cluster pain triggers
    clustered_triggers = cluster_items(pain_trigger_embeddings)
    print(f"🔍 Clustered pain triggers into {len(clustered_triggers)} clusters.")

    # Step 3: Assign canonical labels
    canonical_map = assign_canonical_labels(clustered_triggers)

    # Step 4: Optionally save the canonical map
    save_canonical_map(canonical_map, CANONICAL_MAP_PATH / "pain_trigger_to_canonical.json")
    print(f"✅ Pain trigger canonical map saved to {CANONICAL_MAP_PATH / 'pain_trigger_to_canonical.json'}")

    return canonical_map

def canonicalize_persona(personas: list[dict]) -> dict:
    print("Starting persona canonicalization with input")

    # Step 0: Dedupes
    seen = set()
    deduplicated_personas = []
    for persona in personas:
        print("Processing persona:", persona, "of type", type(persona))
        persona_tuple = tuple(sorted(persona.items()))  # Convert dict to a sorted tuple of key-value pairs
        if persona_tuple not in seen:
            seen.add(persona_tuple)
            deduplicated_personas.append(persona)
    personas = deduplicated_personas
    print("Deduplicated personas")

    if not personas:
        print("⚠️ No personas provided for canonicalization.")
        return {}

    # Step 1: Generate embeddings for persona titles
    print(f"🔍 Generating embeddings for {len(personas)} personas.")
    persona_texts = [f"{p['title']}|{p['department']}|{p['seniority']}" for p in personas]
    persona_embeddings = generate_and_save_embeddings(persona_texts, "persona")

    # Handle case with only one embedding
    if len(persona_embeddings) == 1:
        print("⚠️ Only one persona provided. Skipping clustering.")
        canonical_map = {persona_texts[0]: persona_texts[0]}  # Map the raw persona text to itself
        save_canonical_map(canonical_map, PERSONA_CANONICAL_MAP_PATH)
        print(f"✅ Persona canonical map saved to {PERSONA_CANONICAL_MAP_PATH}")
        return canonical_map

    # Step 2: Cluster personas
    clustered_personas = cluster_items(persona_embeddings)
    print(f"🔍 Clustered personas into {len(clustered_personas)} clusters.")

    # Step 3: Assign canonical labels
    canonical_map = assign_canonical_labels(clustered_personas)

    # Step 4: Map raw personas to canonical personas
    persona_canonical_map = {}
    for persona, persona_text in zip(personas, persona_texts):
        canonical_persona_text = canonical_map[persona_text]
        canonical_persona_parts = canonical_persona_text.split("|")
        canonical_persona = {
            "title": canonical_persona_parts[0],
            "department": canonical_persona_parts[1],
            "seniority": canonical_persona_parts[2],
        }
        # Use the stringified persona dictionary as the key
        persona_canonical_map[str(persona)] = canonical_persona
    # Step 5: Save the canonical map
    save_canonical_map(persona_canonical_map, PERSONA_CANONICAL_MAP_PATH)
    print(f"✅ Persona canonical map saved to {PERSONA_CANONICAL_MAP_PATH}")

    return persona_canonical_map

if __name__ == "__main__":
    # Example input data
    pains = [
        "difficulty in managing subscriptions and ensuring no missed payments.",
        "manual management of subscriptions leading to missed payments.",
        "inflexible pricing structures that don't reflect usage.",
        "pricing structures that don't align with customer usage."
    ]

    jobs = [
        "subscription management and billing",
        "pricing strategy and management",
        "customer retention and growth"
    ]

    personas = [
        {"title": "Billing Manager", "department": "Finance", "seniority": "Mid-Senior level"},
        {"title": "Pricing Analyst", "department": "Marketing", "seniority": "Mid level"},
        {"title": "Customer Success Manager", "department": "Customer Success", "seniority": "Senior level"}
    ]

    # Canonicalize pains
    canonicalize_pain(pains)

    # Canonicalize jobs
    canonicalize_job(jobs)

    # Canonicalize personas
    canonicalize_persona(personas)