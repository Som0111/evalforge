import json
import random

import pytest

from evalforge.calibration import agreement_report, compute_kappa

DIMS = ("faithfulness", "relevance", "coherence", "conciseness")


def record(human, judge):
    return {"human_label": dict.fromkeys(DIMS, human), "judge_label": dict.fromkeys(DIMS, judge)}


def test_identical_lists_give_kappa_one():
    assert compute_kappa([1, 2, 3, 2, 1, 3], [1, 2, 3, 2, 1, 3]) == 1.0


def test_identical_constant_lists_are_perfect_not_nan():
    assert compute_kappa([3, 3, 3], [3, 3, 3]) == 1.0


def test_random_disagreement_is_near_zero():
    rng = random.Random(0)
    human = [rng.randint(1, 3) for _ in range(300)]
    judge = [rng.randint(1, 3) for _ in range(300)]
    assert compute_kappa(human, judge) == pytest.approx(0.0, abs=0.2)


def test_none_judge_labels_are_filtered_before_computing():
    assert compute_kappa([1, 2, 3, 1], [1, None, 3, 1]) == 1.0
    assert compute_kappa([1, 2], [None, None]) is None


def test_constant_judge_against_varied_human_is_zero():
    assert compute_kappa([1, 2, 3, 2], [3, 3, 3, 3]) == 0.0


def test_quadratic_weighting_penalises_far_misses_more():
    near = compute_kappa([1, 2, 3, 1, 2, 3], [2, 3, 3, 1, 2, 2])
    far = compute_kappa([1, 2, 3, 1, 2, 3], [3, 2, 1, 1, 2, 3])
    assert far < near


def test_agreement_report_schema_and_file(tmp_path):
    records = [record(h, j) for h, j in [(1, 1), (2, 2), (3, 3), (2, 3), (1, 1)]] + [record(2, None)]
    path = tmp_path / "cal.json"
    report = agreement_report(records, path)
    assert set(DIMS) | {"weighted_avg"} <= set(report)
    assert report["details"]["faithfulness"]["n"] == 5  # the None record is excluded
    assert json.loads(path.read_text()) == report
    assert 0 < report["weighted_avg"] <= 1


def test_agreement_report_with_no_judged_records(tmp_path):
    report = agreement_report([record(2, None)], tmp_path / "cal.json")
    assert report["weighted_avg"] is None
