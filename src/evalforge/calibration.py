"""Judge vs human agreement, Cohen's Kappa (Phase 3.1)."""
import json
import math

from sklearn.metrics import cohen_kappa_score

from evalforge.config import CALIBRATION_REPORT_PATH
from evalforge.dataset import REQUIRED_DIMENSIONS

# Kappa interpretation: < 0.2 poor, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, > 0.8 near-perfect.
# Labels are ordinal (1-3), so kappa is quadratic-weighted: a 1-vs-3 miss costs more than 2-vs-3.
KAPPA_WEIGHTS = "quadratic"


def compute_kappa(human_labels: list[int], judge_labels: list[int | None]) -> float | None:
    """Quadratic-weighted kappa over the pairs where the judge produced a score.

    Returns None when fewer than 2 usable pairs exist. Kappa is undefined (0/0) when both raters
    are constant; identical constants are perfect agreement, so that case returns 1.0.
    """
    pairs = [(h, j) for h, j in zip(human_labels, judge_labels, strict=True) if j is not None]
    if len(pairs) < 2:
        return None
    human, judge = zip(*pairs, strict=True)
    if human == judge:
        return 1.0
    kappa = cohen_kappa_score(human, judge, weights=KAPPA_WEIGHTS, labels=[1, 2, 3])
    return 0.0 if math.isnan(kappa) else float(kappa)


def agreement_report(records: list[dict], path=CALIBRATION_REPORT_PATH) -> dict:
    """Kappa per dimension plus an n-weighted average; also writes the report to `path`."""
    per_dim, details = {}, {}
    for dim in REQUIRED_DIMENSIONS:
        human = [r["human_label"][dim] for r in records]
        judge = [r["judge_label"][dim] for r in records]
        per_dim[dim] = compute_kappa(human, judge)
        used = [(h, j) for h, j in zip(human, judge, strict=True) if j is not None]
        details[dim] = {
            "n": len(used),
            "exact_agreement": sum(h == j for h, j in used) / len(used) if used else None,
        }

    # Weight each dimension by how many pairs it had, so a dimension with few judged records counts less.
    weighted = [(k, details[d]["n"]) for d, k in per_dim.items() if k is not None]
    total = sum(n for _, n in weighted)
    weighted_avg = sum(k * n for k, n in weighted) / total if total else None

    report = {**per_dim, "weighted_avg": weighted_avg, "kappa_weights": KAPPA_WEIGHTS, "details": details}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
