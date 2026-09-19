"""Position bias and verbosity bias detection (Phase 3.2, 3.3)."""
import math

from scipy.stats import pearsonr

from evalforge.config import POSITION_BIAS_FLIP_THRESHOLD
from evalforge.dataset import REQUIRED_DIMENSIONS

MIN_PAIRS = 10
VERBOSITY_MIN_R = 0.3
VERBOSITY_MAX_P = 0.05


def detect_position_bias(pairs: list[tuple[str, str]], judge_fn) -> dict:
    """Show each pair in both orders. `judge_fn(first, second)` returns 0 or 1, the position it prefers.

    A judge with no position bias prefers the same *output* in both orders. If it prefers the same
    *position* instead (e.g. always the first slot), the preferred output changed: a flip.
    """
    if len(pairs) < MIN_PAIRS:
        raise ValueError(f"need at least {MIN_PAIRS} pairs for a meaningful flip rate, got {len(pairs)}")
    flips = 0
    for a, b in pairs:
        picked_a_first = "A" if judge_fn(a, b) == 0 else "B"
        picked_b_first = "B" if judge_fn(b, a) == 0 else "A"
        flips += picked_a_first != picked_b_first
    flip_rate = flips / len(pairs)
    return {
        "total_pairs": len(pairs),
        "flips": flips,
        "flip_rate": flip_rate,
        "significant": flip_rate > POSITION_BIAS_FLIP_THRESHOLD,
    }


def detect_verbosity_bias(records: list[dict]) -> dict:
    """Pearson correlation of output word count vs the judge's mean score (records missing a score are skipped)."""
    lengths, means = [], []
    for r in records:
        label = r.get("judge_label") or {}
        scores = [label.get(d) for d in REQUIRED_DIMENSIONS]
        if any(s is None for s in scores):
            continue
        lengths.append(len(r["output"].split()))
        means.append(sum(scores) / len(scores))
    if len(lengths) < 3 or len(set(lengths)) < 2 or len(set(means)) < 2:
        return {"pearson_r": 0.0, "p_value": 1.0, "significant": False, "n": len(lengths)}  # correlation undefined
    r, p = pearsonr(lengths, means)
    if math.isnan(r):
        return {"pearson_r": 0.0, "p_value": 1.0, "significant": False, "n": len(lengths)}
    return {
        "pearson_r": float(r),
        "p_value": float(p),
        "significant": bool(p < VERBOSITY_MAX_P and r > VERBOSITY_MIN_R),
        "n": len(lengths),
    }
