"""Semantic similarity via sentence-transformers (Phase 1.2)."""
import functools

from sentence_transformers import SentenceTransformer

from evalforge.config import EMBEDDING_MODEL


@functools.lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Lazy + cached: loads once on first score, so importing evalforge stays cheap.
    return SentenceTransformer(EMBEDDING_MODEL)


def semantic_similarity(output: str, reference: str) -> float:
    if not (isinstance(output, str) and isinstance(reference, str) and output.strip() and reference.strip()):
        return 0.0
    a, b = _model().encode([output, reference], normalize_embeddings=True)
    return min(1.0, max(0.0, float(a @ b)))  # negative cosine has no meaning for a quality score


def relevance_to_input(output: str, input: str) -> float:
    return semantic_similarity(output, input)
