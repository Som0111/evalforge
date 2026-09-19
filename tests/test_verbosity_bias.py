from evalforge.bias import detect_verbosity_bias

DIMS = ("faithfulness", "relevance", "coherence", "conciseness")


def rec(words, score):
    return {"output": " ".join(["w"] * words), "judge_label": dict.fromkeys(DIMS, score)}


def test_longer_outputs_always_scoring_higher_is_significant():
    data = [rec(60 + i, 3) for i in range(10)] + [rec(5 + i, 1) for i in range(10)]
    result = detect_verbosity_bias(data)
    assert result["pearson_r"] > 0.9
    assert result["significant"] is True


def test_length_and_score_independent_is_not_significant():
    # Full factorial of lengths x scores: correlation is exactly zero by construction.
    data = [rec(w, s) for w in (10, 20, 30, 40) for s in (1, 2, 3)]
    result = detect_verbosity_bias(data)
    assert abs(result["pearson_r"]) < 1e-9
    assert result["significant"] is False


def test_records_without_judge_scores_are_skipped():
    data = [rec(60 + i, 3) for i in range(5)] + [rec(5 + i, 1) for i in range(5)]
    data.append({"output": "x y z", "judge_label": dict.fromkeys(DIMS)})
    assert detect_verbosity_bias(data)["n"] == 10


def test_undefined_correlation_returns_not_significant():
    assert detect_verbosity_bias([rec(10, 3)] * 5)["significant"] is False
    assert detect_verbosity_bias([])["significant"] is False
