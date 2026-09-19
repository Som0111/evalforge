"""Golden dataset loader and validator (Phase 0.3)."""
import json
from pathlib import Path

from evalforge.config import SCORE_MAX, SCORE_MIN

REQUIRED_DIMENSIONS = ("faithfulness", "relevance", "coherence", "conciseness")
REQUIRED_FIELDS = ("id", "input", "output", "human_label")


class GoldenDataset:
    @staticmethod
    def load(path: str | Path) -> list[dict]:
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                GoldenDataset.validate(record)
                records.append(record)
        return records

    @staticmethod
    def validate(record: dict) -> None:
        for field in REQUIRED_FIELDS:
            if field not in record:
                raise ValueError(f"record {record.get('id', '?')} missing required field: {field}")

        for field in ("id", "input", "output"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"record {record.get('id', '?')} has empty or non-string '{field}'")

        human_label = record["human_label"]
        for dim in REQUIRED_DIMENSIONS:
            if dim not in human_label:
                raise ValueError(f"record {record['id']} human_label missing dimension: {dim}")
            score = human_label[dim]
            if not isinstance(score, int) or not (SCORE_MIN <= score <= SCORE_MAX):
                raise ValueError(
                    f"record {record['id']} human_label['{dim}']={score!r} out of range "
                    f"[{SCORE_MIN}, {SCORE_MAX}]"
                )

    @staticmethod
    def split(records: list[dict], train_ratio: float = 0.8, seed: int = 42) -> tuple[list[dict], list[dict]]:
        indices = list(range(len(records)))
        rng_state = seed
        # Deterministic Fisher-Yates using a simple LCG so the split needs no
        # extra dependency and is reproducible across Python versions.
        for i in range(len(indices) - 1, 0, -1):
            rng_state = (rng_state * 1103515245 + 12345) & 0x7FFFFFFF
            j = rng_state % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]

        split_point = round(len(records) * train_ratio)
        train_idx = indices[:split_point]
        test_idx = indices[split_point:]
        return [records[i] for i in train_idx], [records[i] for i in test_idx]
