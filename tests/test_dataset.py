import json

import pytest

from evalforge.dataset import GoldenDataset


def make_record(rid="cb_001", **overrides):
    record = {
        "id": rid,
        "input": "some failure cluster description",
        "output": "some LLM-generated summary",
        "human_label": {"faithfulness": 3, "relevance": 2, "coherence": 3, "conciseness": 2},
    }
    record.update(overrides)
    return record


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)


def test_validate_rejects_record_missing_human_label():
    record = make_record()
    del record["human_label"]
    with pytest.raises(ValueError):
        GoldenDataset.validate(record)


@pytest.mark.parametrize("bad_score", [0, 4])
def test_validate_rejects_score_out_of_range(bad_score):
    record = make_record(human_label={"faithfulness": bad_score, "relevance": 2, "coherence": 3, "conciseness": 2})
    with pytest.raises(ValueError):
        GoldenDataset.validate(record)


def test_split_is_deterministic():
    records = [make_record(rid=f"cb_{i:03d}") for i in range(20)]
    train_a, test_a = GoldenDataset.split(records, train_ratio=0.8, seed=42)
    train_b, test_b = GoldenDataset.split(records, train_ratio=0.8, seed=42)
    assert [r["id"] for r in train_a] == [r["id"] for r in train_b]
    assert [r["id"] for r in test_a] == [r["id"] for r in test_b]
    assert len(train_a) == 16
    assert len(test_a) == 4


def test_load_reads_all_records_with_correct_types(tmp_path):
    records = [make_record(rid=f"cb_{i:03d}") for i in range(20)]
    path = tmp_path / "golden.jsonl"
    write_jsonl(path, records)

    loaded = GoldenDataset.load(path)

    assert len(loaded) == 20
    for record in loaded:
        assert isinstance(record["id"], str)
        assert isinstance(record["input"], str)
        assert isinstance(record["output"], str)
        assert isinstance(record["human_label"]["faithfulness"], int)
