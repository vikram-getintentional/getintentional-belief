from backend.utils.inference.latent_pressure_engine import build_latent_pressure_chain

if __name__ == "__main__":
    seed_persona = "VP Sales"
    seed_pain = "Struggles with low pipeline visibility"

    chain = build_latent_pressure_chain(seed_persona, seed_pain)

    for step in chain:
        print(f"{step['persona']:20} | {step['canonical_pain']:50} | score: {step['score']}")
