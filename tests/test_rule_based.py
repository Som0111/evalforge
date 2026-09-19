import pytest

from evalforge.scorers.rule_based import format_score, keyword_overlap, length_score

INPUT = (
    "Nine tests failed together in toolz dicttoolz. The merge function returns wrong values. "
    "Failures involve merge factory arguments and mapping types."
)


def test_length_score_in_range_and_too_short():
    assert length_score(" ".join(["word"] * 20)) == 1.0
    assert length_score("just three words") < 0.5


def test_length_score_decays_when_too_long():
    assert length_score(" ".join(["word"] * 120)) == pytest.approx(0.5)
    assert length_score(" ".join(["word"] * 500)) == 0.0


def test_keyword_overlap_high_when_output_uses_input_keywords():
    output = "Tests in toolz dicttoolz failed because the merge function returns wrong values."
    assert keyword_overlap(output, INPUT) > 0.5


def test_keyword_overlap_low_for_unrelated_output():
    assert keyword_overlap("Quarterly revenue declined in the retail segment.", INPUT) < 0.2


def test_format_score_zero_for_repeated_trigrams():
    assert format_score("the cat sat on the mat " * 4) == 0.0


def test_format_score_zero_for_truncation():
    assert format_score("The merge helper drops keys when the factory is...") == 0.0
    assert format_score("The merge helper drops keys [TRUNCATED]") == 0.0


def test_format_score_clean_text_is_one():
    assert format_score("The merge helper drops duplicate keys when a custom factory is passed.") == 1.0


@pytest.mark.parametrize("bad", [None, 123, "", "   "])
def test_bad_input_scores_zero_without_raising(bad):
    assert length_score(bad) == 0.0
    assert keyword_overlap(bad, INPUT) == 0.0
    assert keyword_overlap("text", bad) == 0.0
    assert format_score(bad) == 0.0
