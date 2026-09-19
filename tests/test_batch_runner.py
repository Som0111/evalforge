import json
import logging

from evalforge.scorers import llm_judge

GOOD = {"faithfulness": 3, "relevance": 2, "coherence": 3, "conciseness": 2}
NONE = dict.fromkeys(GOOD)


def records(n=3):
    return [{"id": f"cb_{i:03d}", "input": f"in{i}", "output": f"out{i}", "human_label": GOOD} for i in range(n)]


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_all_records_get_judge_label_and_version(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_judge, "judge", lambda i, o, t=None: GOOD)
    out = tmp_path / "judge.jsonl"
    result = llm_judge.run_judge_on_dataset(records(3), out_path=out)
    assert len(result) == 3
    assert all(r["judge_label"] == GOOD and r["judge_prompt_version"] == "v1" for r in result)
    assert all("human_label" in r for r in result)
    assert len(read(out)) == 3


def test_all_none_is_skipped_and_logged_rest_processed(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(llm_judge, "judge", lambda i, o, t=None: NONE if i == "in1" else GOOD)
    out = tmp_path / "judge.jsonl"
    with caplog.at_level(logging.WARNING):
        result = llm_judge.run_judge_on_dataset(records(3), out_path=out)
    assert [r["id"] for r in result] == ["cb_000", "cb_002"]
    assert len(read(out)) == 2
    assert "cb_001" in caplog.text


def test_rerun_resumes_and_retries_skipped(monkeypatch, tmp_path):
    out = tmp_path / "judge.jsonl"
    monkeypatch.setattr(llm_judge, "judge", lambda i, o, t=None: NONE if i == "in1" else GOOD)
    llm_judge.run_judge_on_dataset(records(3), out_path=out)

    seen = []

    def judge_second_run(i, o, t=None):
        seen.append(i)
        return GOOD

    monkeypatch.setattr(llm_judge, "judge", judge_second_run)
    result = llm_judge.run_judge_on_dataset(records(3), out_path=out)
    assert seen == ["in1"]  # only the previously skipped record hits the API again
    assert len(result) == 3
    assert len(read(out)) == 3
