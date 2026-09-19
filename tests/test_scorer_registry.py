from evalforge.scorers import score_all

INPUT = "Nine tests failed in dicttoolz. The merge function returns wrong values with a custom factory."
OUTPUT = "The merge function in dicttoolz likely returns wrong values when a custom factory is used."
BASE_KEYS = {"length", "keyword_overlap", "format", "relevance"}


def test_score_all_returns_all_keys_as_unit_floats():
    scores = score_all(OUTPUT, INPUT, reference=OUTPUT)
    assert set(scores) == BASE_KEYS | {"semantic_sim"}
    for v in scores.values():
        assert isinstance(v, float) and 0.0 <= v <= 1.0


def test_no_reference_skips_semantic_sim():
    assert set(score_all(OUTPUT, INPUT)) == BASE_KEYS
