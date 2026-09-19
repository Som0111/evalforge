"""Run full eval pipeline, write report (Phase 3.4)."""
import functools
import json
import logging
import statistics
import sys

from evalforge.bias import detect_verbosity_bias
from evalforge.calibration import agreement_report
from evalforge.config import (
    CALIBRATION_REPORT_PATH,
    EVAL_REPORT_PATH,
    GOLDEN_DATASET_PATH,
    MIN_WEIGHTED_KAPPA,
    versioned_path,
)
from evalforge.dataset import GoldenDataset
from evalforge.scorers import score_all

log = logging.getLogger(__name__)


def _current_model() -> str:
    from evalforge.scorers import llm_judge  # needs GEMINI_API_KEY; imported lazily

    return llm_judge.JUDGE_MODEL


def run_full_eval(
    golden_dataset_path=GOLDEN_DATASET_PATH,
    output_path=EVAL_REPORT_PATH,
    calibration_path=CALIBRATION_REPORT_PATH,
    judge_runner=None,
    score_fn=score_all,
    prompt_version="v1",
    judge_model=None,
) -> dict:
    if judge_runner is None:
        from evalforge.scorers.llm_judge import (
            run_judge_on_dataset,  # needs GEMINI_API_KEY; imported lazily
        )

        judge_runner = functools.partial(run_judge_on_dataset, prompt_version=prompt_version)
    model = judge_model or _current_model()

    dataset = GoldenDataset.load(golden_dataset_path)

    per_record = [score_fn(r["output"], r["input"]) for r in dataset]
    scorer_averages = {k: statistics.fmean(s[k] for s in per_record) for k in per_record[0]}

    judged = judge_runner(dataset)  # records the judge could not score are absent
    report = {
        "dataset_size": len(dataset),
        "scorer_averages": scorer_averages,
        "calibration": agreement_report(judged, calibration_path),
        "bias": {"verbosity": detect_verbosity_bias(judged)},
        "judge_coverage": len(judged) / len(dataset),
        "judge_prompt_version": prompt_version,
        "judge_model": model,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    kappa = report["calibration"]["weighted_avg"]
    if kappa is not None and kappa < MIN_WEIGHTED_KAPPA:
        log.warning("weighted kappa %.2f is below the %.2f gate: the judge prompt needs revision", kappa, MIN_WEIGHTED_KAPPA)
    return report


if __name__ == "__main__":
    from evalforge.scorers import llm_judge  # needs GEMINI_API_KEY

    version = sys.argv[1] if len(sys.argv) > 1 else "v1"  # `python -m evalforge.evaluate v2 [model]`
    if len(sys.argv) > 2:
        llm_judge.JUDGE_MODEL = sys.argv[2]
    model = llm_judge.JUDGE_MODEL
    logging.basicConfig(level=logging.WARNING)
    print(json.dumps(run_full_eval(
        output_path=versioned_path(EVAL_REPORT_PATH, version, model),
        calibration_path=versioned_path(CALIBRATION_REPORT_PATH, version, model),
        prompt_version=version,
    ), indent=2))
