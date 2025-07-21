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
    jobs = list(set(jobs))
    if not jobs:
        print("⚠️ No jobs provided for canonicalization.")
        return {}

    job_embeddings = generate_and_save_embeddings(jobs, "job")

    if len(job_embeddings) == 1:
        print("⚠️ Only one job provided. Skipping clustering.")
        canonical_map = {jobs[0]: jobs[0]}
        save_canonical_map(canonical_map, JOB_CANONICAL_MAP_PATH)
        return canonical_map

    clustered_jobs = cluster_items(job_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_jobs)
    canonical_map = {job: canonical_label_map[job] for job in jobs}
    save_canonical_map(canonical_map, JOB_CANONICAL_MAP_PATH)
    return canonical_map

def canonicalize_pain(pains: list[str]) -> dict:
    pains = list(set(pains))
    if not pains:
        print("⚠️ No pains provided for canonicalization.")
        return {}

    pain_embeddings = generate_and_save_embeddings(pains, "pain")

    if len(pain_embeddings) == 1:
        print("⚠️ Only one pain provided. Skipping clustering.")
        canonical_map = {pains[0]: pains[0]}
        save_canonical_map(canonical_map, PAIN_CANONICAL_MAP_PATH)
        return canonical_map

    clustered_pains = cluster_items(pain_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_pains)
    canonical_map = {pain: canonical_label_map[pain] for pain in pains}
    save_canonical_map(canonical_map, PAIN_CANONICAL_MAP_PATH)
    return canonical_map

def canonicalize_attributes(attributes: list[str]) -> dict:
    attributes = list(set(attributes))
    if not attributes:
        print("⚠️ No attributes provided for canonicalization.")
        return {}

    attribute_embeddings = generate_and_save_embeddings(attributes, "attribute")

    if len(attribute_embeddings) == 1:
        print("⚠️ Only one attribute provided. Skipping clustering.")
        canonical_map = {attributes[0]: attributes[0]}
        return canonical_map

    clustered_attributes = cluster_items(attribute_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_attributes)
    canonical_map = {attribute: canonical_label_map[attribute] for attribute in attributes}
    return canonical_map

def canonicalize_pain_trigger(pain_triggers: list[dict]) -> dict:
    if not pain_triggers:
        print("⚠️ No pain triggers provided for canonicalization.")
        return {}
    attribute_cache = set()
    for trigger in pain_triggers:
        attribute_cache.add(trigger['attribute'].strip().lower())
    canonical_attributes = canonicalize_attributes(list(attribute_cache))

    # Build 1:1 mapping for each pain trigger in input
    pain_trigger_canonical_map = {}
    for orig in pain_triggers:
        canonical_trigger = {
            "attribute": canonical_attributes.get(orig["attribute"].strip().lower(), orig["attribute"]),
            "dimension": orig.get("dimension", "").strip().lower(),
            "direction": orig.get("direction", "").strip().lower()
        }
        pain_trigger_canonical_map[str(orig)] = canonical_trigger

    save_canonical_map(pain_trigger_canonical_map, CANONICAL_MAP_PATH / "pain_trigger_to_canonical.json")
    return pain_trigger_canonical_map

def canonicalize_titles(titles: list[str]) -> dict:
    titles = list(set(titles))
    if not titles:
        print("⚠️ No titles provided for canonicalization.")
        return {}

    title_embeddings = generate_and_save_embeddings(titles, "title")

    if len(title_embeddings) == 1:
        print("⚠️ Only one title provided. Skipping clustering.")
        canonical_map = {titles[0]: titles[0]}
        return canonical_map

    clustered_titles = cluster_items(title_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_titles)
    canonical_map = {title: canonical_label_map[title] for title in titles}
    return canonical_map

def canonicalize_departments(departments: list[str]) -> dict:
    departments = list(set(departments))
    if not departments:
        print("⚠️ No departments provided for canonicalization.")
        return {}

    department_embeddings = generate_and_save_embeddings(departments, "department")

    if len(department_embeddings) == 1:
        print("⚠️ Only one department provided. Skipping clustering.")
        canonical_map = {departments[0]: departments[0]}
        return canonical_map

    clustered_departments = cluster_items(department_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_departments)
    canonical_map = {department: canonical_label_map[department] for department in departments}
    return canonical_map


def canonicalize_persona(personas: list[dict]) -> dict:
    # Updated logic by splitting personas into title, dept, seniority for canon work
    if not personas:
        print("⚠️ No personas provided for canonicalization.")
        return {}
    title_cache = set()
    dept_cache = set()
    for p in personas:
        title_cache.add(p['title'].strip().lower())
        dept_cache.add(p['department'].strip().lower())
    canonical_titles = canonicalize_titles(list(title_cache))
    canonical_depts = canonicalize_departments(list(dept_cache))
    

    SENIORITY_NORMALIZATION = {
        "intern": "Junior",
        "junior": "Junior",
        "associate": "Operator",
        "mid level": "Operator",
        "mid-senior level": "Manager",
        "senior level": "Senior",
        "lead": "Senior",
        "director": "Executive",
        "vp": "Executive",
        "c-level": "Executive",
    }
    
    def normalize_seniority(seniority: str) -> str:
        if not seniority:
            return "Operator"
        return SENIORITY_NORMALIZATION.get(seniority.strip().lower(), "Operator")

    # Build 1:1 mapping for each persona in input
    persona_canonical_map = {}
    for orig in personas:
        canonical_persona = {
            "title": canonical_titles.get(orig["title"].strip().lower(), orig["title"]),
            "department": canonical_depts.get(orig["department"].strip().lower(), orig["department"]),
            "seniority": normalize_seniority(orig.get("seniority", "")),
        }
        persona_canonical_map[str(orig)] = canonical_persona
 
    save_canonical_map(persona_canonical_map, PERSONA_CANONICAL_MAP_PATH)
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