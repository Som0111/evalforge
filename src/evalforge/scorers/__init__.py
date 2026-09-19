"""Scorer registry: runs all rule-based and embedding scorers (Phase 1.3)."""
from evalforge.scorers.embedding import relevance_to_input, semantic_similarity
from evalforge.scorers.rule_based import format_score, keyword_overlap, length_score


def score_all(output: str, input: str, reference: str | None = None) -> dict[str, float]:
    scores = {
        "length": length_score(output),
        "keyword_overlap": keyword_overlap(output, input),
        "format": format_score(output),
        "relevance": relevance_to_input(output, input),
    }
    if reference is not None:
        scores["semantic_sim"] = semantic_similarity(output, reference)
    return scores
