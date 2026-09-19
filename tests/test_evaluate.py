import json

from evalforge.evaluate import run_full_eval

DIMS = ("faithfulness", "relevance", "coherence", "conciseness")


def golden(path, n=4):
    labels = [(1, 2, 3, 2), (2, 2, 3, 3), (3, 3, 1, 2), (1, 1, 2, 3)]
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            rec = {"id": f"cb_{i}", "input": f"in {i}", "output": "word " * (i + 1),
                   "human_label": dict(zip(DIMS, labels[i], strict=True))}
            f.write(json.dumps(rec) + "\n")


def fake_judge_runner(dataset):
    # The judge "fails" on the last record: it is absent from the returned list.
    return [{**r, "judge_label": dict(r["human_label"])} for r in dataset[:-1]]


def fake_score(output, input):
    return {"length": 0.5, "keyword_overlap": 0.25, "format": 1.0, "relevance": 0.75}


def test_report_written_with_expected_schema_and_coverage(tmp_path):
    golden(tmp_path / "g.jsonl")
    out = tmp_path / "eval.json"
    report = run_full_eval(tmp_path / "g.jsonl", out, tmp_path / "cal.json",
                           judge_runner=fake_judge_runner, score_fn=fake_score)

    assert json.loads(out.read_text()) == report
    assert (tmp_path / "cal.json").exists()
    assert set(report) == {"dataset_size", "scorer_averages", "calibration", "bias", "judge_coverage", "judge_prompt_version", "judge_model"}
    assert report["dataset_size"] == 4
    assert report["judge_coverage"] == 0.75
    assert report["scorer_averages"] == fake_score("", "")
    assert report["calibration"]["weighted_avg"] == 1.0  # judge copied the human labels
    assert "pearson_r" in report["bias"]["verbosity"]
