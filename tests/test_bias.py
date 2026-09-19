import pytest

from evalforge.bias import detect_position_bias

PAIRS = [(f"short {i}", f"a much longer output number {i}") for i in range(12)]


def test_judge_that_always_picks_first_slot_flips_every_pair():
    result = detect_position_bias(PAIRS, lambda first, second: 0)
    assert result["flip_rate"] == 1.0
    assert result["flips"] == result["total_pairs"] == 12
    assert result["significant"] is True


def test_judge_that_picks_same_output_never_flips():
    prefer_longer = lambda first, second: 0 if len(first) >= len(second) else 1
    result = detect_position_bias(PAIRS, prefer_longer)
    assert result["flip_rate"] == 0.0
    assert result["significant"] is False


def test_fewer_than_ten_pairs_raises():
    with pytest.raises(ValueError):
        detect_position_bias(PAIRS[:9], lambda first, second: 0)
