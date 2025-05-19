def persona_relevance_score(belief_results: list[dict]) -> dict:
    """
    Compute the total relevance score for each persona by summing
    the belief scores across all matched pains.

    Args:
        belief_results (list[dict]): Results from capability-to-pain matching,
            each entry contains "persona" and "score".

    Returns:
        dict: Mapping of persona name → total relevance score
    """
    persona_scores = {}

    for entry in belief_results:
        persona = entry["persona"]
        score = entry.get("score", 0)

        if persona in persona_scores:
            persona_scores[persona] += score
        else:
            persona_scores[persona] = score

    # Optional: round scores for readability
    return {p: round(s, 3) for p, s in persona_scores.items()}
