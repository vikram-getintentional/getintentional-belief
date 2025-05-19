import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from backend.utils.graph_base.graph_utils.json_store import load_json
from backend.utils.knowledge_base.canonicalizer import canonicalize_persona
from backend.utils.graph_base.edges.edge_manager import add_edge

# Paths
GRAPH_PATH = Path("backend/utils/graph_base/graph_data")
PERSONA_PATH = GRAPH_PATH / "persona_nodes.json"
JOB_PATH = GRAPH_PATH / "job_nodes.json"
PAIN_PATH = GRAPH_PATH / "pain_nodes.json"
EDGE_PATH = GRAPH_PATH / "graph_edges.json"

# Load canonical maps
CANONICAL_PATH = Path("backend/utils/knowledge_base/canonical_maps")
from backend.utils.knowledge_base.canonical_maps.canonical_loader import load_canonical_map

JOB_CANONICAL_MAP = load_canonical_map(CANONICAL_PATH / "job_to_canonical.json")
PAIN_CANONICAL_MAP = load_canonical_map(CANONICAL_PATH / "pain_to_canonical.json")

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_graph_persona_combos():
    print("running load graph persona combos")
    personas = load_json(PERSONA_PATH)
    jobs = load_json(JOB_PATH)
    pains = load_json(PAIN_PATH)
    edges = load_json(EDGE_PATH)

    
    
    job_map = {j["id"]: j["description"] for j in jobs}
    pain_map = {p["id"]: p["text"] for p in pains}

    

    # Collect personas for canonicalization
    persona_cache = [
        {"title": persona["title"], "department": persona.get("department", ""), "seniority": persona.get("seniority", "")}
        for persona in personas
    ]
    canonical_personas = canonicalize_persona(persona_cache)
    

    combo_entries = []

    for persona in personas:
        pid = persona["id"]
        persona_name = persona["title"]
        dept = persona.get("department", "")
        seniority = persona.get("seniority", "")

        # Canonicalize persona
        raw_persona = {"title": persona_name, "department": dept, "seniority": seniority}
        raw_persona_key = tuple(sorted(raw_persona.items()))  # Option 2: Use tuple representation
        canonical_persona = canonical_personas.get(raw_persona_key, raw_persona)
        
        # Jobs linked to persona
        job_edges = [e for e in edges if e["target"] == pid and e["type"] == "performed_by"]
        for je in job_edges:
            job_id = je["source"]
            job_text = job_map.get(job_id, "")

            canonical_job = JOB_CANONICAL_MAP.get(job_text, job_text)

            # Pains linked to job
            pain_edges = [e for e in edges if e["target"] == job_id and e["type"] == "addresses"]
            for pe in pain_edges:
                pain_id = pe["source"]
                pain_text = pain_map.get(pain_id, "")

                canonical_pain = PAIN_CANONICAL_MAP.get(pain_text, pain_text)

                combo_entries.append({
                    "persona": canonical_persona["title"],
                    "department": canonical_persona["department"],
                    "seniority": canonical_persona["seniority"],
                    "original_job": job_text,
                    "original_pain": pain_text,
                    "canonical_job": canonical_job,
                    "canonical_pain": canonical_pain,
                    "combo_text": f"{canonical_job}. Pain: {canonical_pain}",
                    "source": "openai"
                })
    return combo_entries


def match_capabilities_to_canonical_personas(capabilities: list[dict], threshold=0.7):
    if not capabilities:
        print("❌ No capabilities provided.")
        return []

    combo_entries = load_graph_persona_combos()
    if not combo_entries:
        print("❌ No persona/job/pain graph data found.")
        return []

    cap_texts = [f'{c["name"]}: {c["description"]}' for c in capabilities if c.get("name") and c.get("description")]
    cap_vecs = model.encode(cap_texts)

    combo_texts = [e["combo_text"] for e in combo_entries]
    combo_vecs = model.encode(combo_texts)

    sim_matrix = cosine_similarity(cap_vecs, combo_vecs)  # shape: (cap, combos)

    # Compute max similarity per combo to find best matching capability
    max_relevance = sim_matrix.max(axis=0)
    best_cap_indices = sim_matrix.argmax(axis=0)

    for i, entry in enumerate(combo_entries):
        best_cap_idx = best_cap_indices[i]
        entry["capability"] = capabilities[best_cap_idx]["name"]

        # Extract the similarity score for this capability-pain combination
        similarity_relevance = max_relevance[i]

        # Determine the weight for the edge
        if entry.get("source") == "openai":
            weight = float(max(similarity_relevance, 0.75))  # OpenAI responses get a minimum weight of 0.75
        else:
            weight = float(similarity_relevance)  # Use the similarity score directly for non-OpenAI sources
        # Add the edge to the graph with the calculated weight
        add_edge(
            source_id=entry["canonical_pain"],
            target_id=capabilities[best_cap_idx]["name"],
            edge_type="solves",
            weight=weight
        )

        # Assign the score to the entry for ranking purposes
        entry["relevance"] = round(float(similarity_relevance), 3)
        
    results = [e for e in combo_entries if e["relevance"] >= threshold]
    results.sort(key=lambda x: x["relevance"], reverse=True)
    for entry in results:
        persona = entry.get("persona", "Unknown Persona")
        if isinstance(persona, str):
            # Convert string persona to dictionary format
            entry["persona"] = {
                "title": persona,
                "department": entry.get("department", "Unknown Department"),
                "seniority": entry.get("seniority", "Unknown Seniority")
            }
    
    return results


def extract_top_personas(match_results: list[dict], threshold=0.05):
    seen = {}
    for entry in match_results:
        name = entry["persona"]
        relevance = entry["relevance"]
        if relevance >= threshold:
            seen[name] = max(seen.get(name, 0), relevance)
    return sorted([{"persona": k, "relevance": v} for k, v in seen.items()], key=lambda x: x["relevance"], reverse=True)
