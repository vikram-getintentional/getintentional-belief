import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os, sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.load_utils import load_json

# === File Paths ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PERSONA_PATH = os.path.join(BASE_DIR, "utils/knowledge_base/persona_jobs_pains.json")
PAIN_EMB_PATH = os.path.join(BASE_DIR, "utils/knowledge_base/canonical_maps/canonical_pain_embeddings.json")
JOB_EMB_PATH = os.path.join(BASE_DIR, "utils/knowledge_base/canonical_maps/canonical_job_embeddings.json")

# === Config ===
TOP_N_MATCHES = 3
MAX_DEPTH = 5
CUMULATIVE_THRESHOLD = 0.3

def load_embeddings(path):
    raw = load_json(path)
    fixed = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "embedding" in v:
            fixed[v["representative"]] = np.array(v["embedding"], dtype=float)
    return fixed

def get_seniority_rank(seniority):
    rank_map = {
        "CXO": 5,
        "Executive": 4,
        "Head Level": 3,
        "Director Level": 3,
        "Senior-level": 2,
        "Mid-level": 1,
        "Entry-level": 0
    }
    return rank_map.get(seniority, 1)

def combine_embeddings(pain_emb, job_emb):
    return np.concatenate([pain_emb, job_emb])

def recursive_chain(current_path, current_embedding, persona_pool, cumulative_score, results, depth):
    if depth > MAX_DEPTH or cumulative_score < CUMULATIVE_THRESHOLD:
        return

    similarities = []
    for p in persona_pool:
        pname = p["name"]
        seniority = p.get("seniority", "Mid-level")
        for pain in p["pains"]:
            pain_emb = pain_embeddings.get(pain)
            job_emb = job_embeddings.get(p["responsibilities"])
            if pain_emb is None or job_emb is None:
                continue
            emb = combine_embeddings(pain_emb, job_emb)
            score = cosine_similarity([current_embedding], [emb])[0][0]
            similarities.append({
                "persona": pname,
                "pain": pain,
                "job": p["responsibilities"],
                "score": float(round(score, 4)),
                "seniority": seniority,
                "embedding": emb
            })

    similarities.sort(key=lambda x: x["score"], reverse=True)
    top_matches = similarities[:TOP_N_MATCHES]

    for match in top_matches:
        new_path = current_path + [{
            "persona": match["persona"],
            "pain": match["pain"],
            "score": match["score"],
            "cumulative_score": round(cumulative_score * match["score"], 4),
            "hop": depth + 1
        }]
        results.append(new_path)

        # Remove this persona from next recursion to avoid loops
        reduced_pool = [p for p in persona_pool if p["name"] != match["persona"]]

        # Recurse further
        recursive_chain(
            new_path,
            match["embedding"],
            reduced_pool,
            cumulative_score * match["score"],
            results,
            depth + 1
        )

if __name__ == "__main__":
    print("starting this")

    personas = load_json(PERSONA_PATH)["personas"]
    pain_embeddings = load_embeddings(PAIN_EMB_PATH)
    job_embeddings = load_embeddings(JOB_EMB_PATH)

    # === Seed ===
    seed_persona = "Digital Marketing Specialist"
    seed_pain = "Difficulty in capturing and tracking prospective buyer intent signals from various channels."

    seed_entry = next((p for p in personas if p["name"] == seed_persona), None)
    if not seed_entry:
        raise ValueError(f"Seed persona '{seed_persona}' not found")

    seed_job = seed_entry["responsibilities"]
    seed_pain_emb = pain_embeddings.get(seed_pain)
    seed_job_emb = job_embeddings.get(seed_job)

    if seed_pain_emb is None or seed_job_emb is None:
        raise ValueError("Missing seed embedding")

    seed_embedding = combine_embeddings(seed_pain_emb, seed_job_emb)

    # === Begin Traversal ===
    print(f"Signal Map for: {seed_persona} | Pain: {seed_pain}")
    seed_path = [{
        "persona": seed_persona,
        "pain": seed_pain,
        "score": 1.0,
        "cumulative_score": 1.0,
        "hop": 0
    }]
    reduced_personas = [p for p in personas if p["name"] != seed_persona]
    results = []

    recursive_chain(
        seed_path,
        seed_embedding,
        reduced_personas,
        cumulative_score=1.0,
        results=results,
        depth=0
    )

    # === Print Paths ===
    for path in results:
        print("\n--- New Path ---")
        for step in path:
            hop = step["hop"]
            persona = step["persona"]
            pain = step["pain"][:50]
            score = step["score"]
            cumulative = step["cumulative_score"]
            print(f"Hop {hop}: {persona:30} | {pain:50} | score: {score:.3f} | cumulative: {cumulative:.3f}")

    # === Aggregate Scores by Persona ===
    persona_scores = defaultdict(lambda: {"count": 0, "cumulative_score": 0.0})

    for path in results:
        for step in path:
            name = step["persona"]
            persona_scores[name]["count"] += 1
            persona_scores[name]["cumulative_score"] += step["cumulative_score"]

    ranked_personas = sorted(
        [(p, v["count"], v["cumulative_score"]) for p, v in persona_scores.items()],
        key=lambda x: x[2],
        reverse=True
    )

    print("\n=== Aggregated Persona Influence Map ===")
    for i, (persona, count, score_sum) in enumerate(ranked_personas, 1):
        print(f"{i:2}. {persona:30} | Appearances: {count:<2} | Cumulative Score: {score_sum:.3f}")
