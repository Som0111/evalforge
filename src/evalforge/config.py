"""Paths, model names, and thresholds. The only place magic strings live."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
GOLDEN_DIR = DATA_DIR / "golden"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = ROOT_DIR / "reports"

GOLDEN_DATASET_PATH = GOLDEN_DIR / "ci_brain_summaries.jsonl"
RAW_DATASET_PATH = RAW_DIR / "ci_brain_raw.jsonl"
JUDGE_OUTPUTS_PATH = REPORTS_DIR / "judge_outputs.jsonl"
CALIBRATION_REPORT_PATH = REPORTS_DIR / "calibration_report.json"
EVAL_REPORT_PATH = REPORTS_DIR / "eval_report.json"

# Deliberately not gemini-3.5-flash (the model that wrote the summaries) to limit self-preference bias.
JUDGE_MODEL = "gemini-3.1-flash-lite"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SCORE_MIN = 1
SCORE_MAX = 3

# Kappa < 0.4 fails the CI eval gate (see .github/workflows/ci.yml, Phase 5).
MIN_WEIGHTED_KAPPA = 0.4

# A flip rate above this is considered a significant position bias (Phase 3.2).
POSITION_BIAS_FLIP_THRESHOLD = 0.3
