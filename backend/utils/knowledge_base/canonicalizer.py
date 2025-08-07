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
PAIN_TRIGGER_CANONICAL_MAP_PATH = CANONICAL_MAP_PATH / "pain_trigger_to_canonical.json"
TRIGGER_EVENT_CANONICAL_MAP_PATH = CANONICAL_MAP_PATH / "trigger_event_to_canonical.json"
OBSERVABLE_MOMENT_CANONICAL_MAP_PATH = CANONICAL_MAP_PATH / "observable_moment_to_canonical.json"
METRIC_CANONICAL_MAP_PATH = CANONICAL_MAP_PATH / "metric_to_canonical.json"
KEYWORD_CANONICAL_MAP_PATH = CANONICAL_MAP_PATH / "keyword_to_canonical.json"


# Preloaded embeddings
try:
    from backend.utils.knowledge_base.canonical_maps.canonical_loader import load_embeddings
    JOB_EMBEDDINGS = load_embeddings("job")
    PAIN_EMBEDDINGS = load_embeddings("pain")
    PERSONA_EMBEDDINGS = load_embeddings("persona")
    ATTRIBUTE_EMBEDDINGS = load_embeddings("attribute")
    DEPARTMENT_EMBEDDINGS = load_embeddings("department")
    PAIN_TRIGGER_EMBEDDINGS = load_embeddings("pain_trigger")
    OBSERVABLE_MOMENT_EMBEDDINGS = load_embeddings("observable_moment")
    TITLE_EMBEDDINGS = load_embeddings("title")
    TRIGGER_EVENT_EMBEDDINGS = load_embeddings("trigger_event")
    PERCEIVED_METRIC_EMBEDDINGS = load_embeddings("perceived_metric")
    KEYWORD_EMBEDDINGS = load_embeddings("keyword")

except Exception as e:
    print(f"⚠️ Embedding preload failed: {e}")
    JOB_EMBEDDINGS = {}
    PAIN_EMBEDDINGS = {}
    PERSONA_EMBEDDINGS = {}
    KEYWORD_EMBEDDINGS = {}
    ATTRIBUTE_EMBEDDINGS = {}
    DEPARTMENT_EMBEDDINGS = {}
    PAIN_TRIGGER_EMBEDDINGS = {}
    OBSERVABLE_MOMENT_EMBEDDINGS = {}
    TITLE_EMBEDDINGS = {}
    TRIGGER_EVENT_EMBEDDINGS = {}
    PERCEIVED_METRIC_EMBEDDINGS = {}
    

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

def canonicalize_perceived_metric(perceived_metrics: list[str]) -> dict:
    perceived_metrics = list(set(perceived_metrics))
    if not perceived_metrics:
        print("⚠️ No perceived metrics provided for canonicalization.")
        return {}

    perceived_metric_embeddings = generate_and_save_embeddings(perceived_metrics, "perceived_metric")

    if len(perceived_metric_embeddings) == 1:
        print("⚠️ Only one perceived metric provided. Skipping clustering.")
        canonical_map = {perceived_metrics[0]: perceived_metrics[0]}
        save_canonical_map(canonical_map, METRIC_CANONICAL_MAP_PATH)
        return canonical_map

    clustered_perceived_metrics = cluster_items(perceived_metric_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_perceived_metrics)
    canonical_map = {perceived_metric: canonical_label_map[perceived_metric] for perceived_metric in perceived_metrics}
    save_canonical_map(canonical_map, METRIC_CANONICAL_MAP_PATH)
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
        orig_attr = orig["attribute"].strip().lower()
        canonical_attr = canonical_attributes.get(orig_attr, orig["attribute"].strip())
        canonical_trigger = {
            "attribute": canonical_attr,
            "dimension": orig.get("dimension", "").strip().lower(),
            "direction": orig.get("direction", "").strip().lower()
        }
        # Use the RAW key for mapping, just like other canonicalizers
        raw_key = f"{orig['attribute'].strip().lower()}|{canonical_trigger['dimension']}|{canonical_trigger['direction']}"
        pain_trigger_canonical_map[raw_key] = canonical_trigger


    save_canonical_map(pain_trigger_canonical_map, PAIN_TRIGGER_CANONICAL_MAP_PATH)
    return pain_trigger_canonical_map

def canonicalize_trigger_events(trigger_events: list[str]) -> dict:
    trigger_events = list(set(trigger_events))
    if not trigger_events:
        print("⚠️ No trigger events provided for canonicalization.")
        return {}

    trigger_event_embeddings = generate_and_save_embeddings(trigger_events, "trigger_event")

    if len(trigger_event_embeddings) == 1:
        print("⚠️ Only one trigger event provided. Skipping clustering.")
        canonical_map = {trigger_events[0]: trigger_events[0]}
        save_canonical_map(canonical_map, TRIGGER_EVENT_CANONICAL_MAP_PATH)
        return canonical_map

    clustered_trigger_events = cluster_items(trigger_event_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_trigger_events)
    canonical_map = {trigger_event: canonical_label_map[trigger_event] for trigger_event in trigger_events}
    print(f"Canonical map for trigger events: {canonical_map}")
    save_canonical_map(canonical_map, TRIGGER_EVENT_CANONICAL_MAP_PATH)
    return canonical_map

def canonicalize_observable_moments(observable_moments: list[str]) -> dict:
    observable_moments = list(set(observable_moments))
    if not observable_moments:
        print("⚠️ No observable moments provided for canonicalization.")
        return {}

    observable_moment_embeddings = generate_and_save_embeddings(observable_moments, "observable_moment")

    if len(observable_moment_embeddings) == 1:
        print("⚠️ Only one observable moment provided. Skipping clustering.")
        canonical_map = {observable_moments[0]: observable_moments[0]}
        save_canonical_map(canonical_map, OBSERVABLE_MOMENT_CANONICAL_MAP_PATH)
        return canonical_map

    clustered_observable_moments = cluster_items(observable_moment_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_observable_moments)
    canonical_map = {observable_moment: canonical_label_map[observable_moment] for observable_moment in observable_moments}
    print(f"Canonical map for observable moments: {canonical_map}")
    save_canonical_map(canonical_map, OBSERVABLE_MOMENT_CANONICAL_MAP_PATH)
    return canonical_map

def canonicalize_keywords(keywords: list[str]) -> dict:
    keywords = list(set(keywords))
    if not keywords:
        print("⚠️ No keywords provided for canonicalization.")
        return {}

    keyword_embeddings = generate_and_save_embeddings(keywords, "keyword")

    if len(keyword_embeddings) == 1:
        print("⚠️ Only one keyword provided. Skipping clustering.")
        canonical_map = {keywords[0]: keywords[0]}
        save_canonical_map(canonical_map, KEYWORD_CANONICAL_MAP_PATH)
        return canonical_map

    clustered_keywords = cluster_items(keyword_embeddings)
    canonical_label_map = assign_canonical_labels(clustered_keywords)
    canonical_map = {keyword: canonical_label_map[keyword] for keyword in keywords}
    save_canonical_map(canonical_map, KEYWORD_CANONICAL_MAP_PATH)
    return canonical_map

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