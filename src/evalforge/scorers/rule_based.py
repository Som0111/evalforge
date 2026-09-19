"""Length, keyword overlap, and format checks (Phase 1.1). All return floats in [0, 1]; never raise."""
import functools
import re

from sklearn.feature_extraction.text import TfidfVectorizer

TRUNCATION_MARKERS = ("[TRUNCATED]", "...", "…")
REPEATED_TRIGRAMS_FOR_ZERO = 5


def _safe(fn):
    """Bad input (non-str, empty, vectorizer failure) scores 0.0 instead of raising."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return min(1.0, max(0.0, float(fn(*args, **kwargs))))
        except Exception:  # noqa: BLE001 - spec: scorers never raise on bad input
            return 0.0
    return wrapper


@_safe
def length_score(output: str, min_words: int = 10, max_words: int = 80) -> float:
    n = len(output.split())
    if n < min_words:
        return n / min_words
    if n > max_words:
        return 1 - (n - max_words) / max_words  # hits 0.0 at 2x max_words
    return 1.0


@_safe
def keyword_overlap(output: str, input: str, top_n: int = 5) -> float:
    # Sentences/lines are the TF-IDF "documents" so IDF is meaningful for a single input.
    docs = [d for d in re.split(r"[.\n]+", input) if d.strip()]
    vec = TfidfVectorizer(stop_words="english")
    weights = vec.fit_transform(docs).sum(axis=0).A1
    terms = vec.get_feature_names_out()
    keywords = {terms[i] for i in weights.argsort()[::-1][:top_n]}
    if not keywords:
        return 0.0
    return len(keywords & set(vec.build_analyzer()(output))) / len(keywords)


@_safe
def format_score(output: str) -> float:
    if not output.strip():
        return 0.0
    if any(m in output for m in TRUNCATION_MARKERS) or "�" in output:
        return 0.0
    if any(ord(c) < 32 and c not in "\n\r\t" for c in output):
        return 0.0
    words = re.findall(r"\w+", output.lower())
    trigrams = list(zip(words, words[1:], words[2:]))
    repeats = len(trigrams) - len(set(trigrams))
    return 1 - repeats / REPEATED_TRIGRAMS_FOR_ZERO
