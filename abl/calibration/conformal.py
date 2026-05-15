"""Split conformal prediction sets and the rolling-window variant for time series.

Sources
-------
- Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic Learning in a Random World.
  Springer. https://link.springer.com/book/10.1007/b106715
- Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction
  and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.

Statement (split conformal, two-sided binary classification)
------------------------------------------------------------
Given a calibration set of n IID-from-the-data-distribution examples with nonconformity
scores s_1, ..., s_n, the prediction set at level (1 - alpha) contains every label whose
test-point nonconformity score is at most the ⌈(n+1)(1-alpha)⌉ / n quantile of the
calibration scores. Under exchangeability, marginal coverage ≥ 1 - alpha.

The caveat we shout about
-------------------------
Time series are NOT exchangeable. Regime changes destroy the marginal-coverage guarantee.
We expose a rolling-window variant that computes the quantile from a trailing window. It
does NOT recover the exchangeability assumption — it is a pragmatic compromise. The
scorecard reports the empirical coverage actually achieved, not the nominal level.
"""
from __future__ import annotations

import numpy as np


def split_conformal_quantile(cal_scores: np.ndarray, alpha: float = 0.1) -> float:
    """The split-conformal threshold: prediction set contains every label with score <= q.

    Under exchangeability, this guarantees P(true_score <= q) >= 1 - alpha.
    """
    cal = np.asarray(cal_scores, dtype=float)
    cal = cal[np.isfinite(cal)]
    n = cal.size
    if n < 1:
        raise ValueError("need at least one calibration score")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    # The ⌈(n+1)(1-alpha)⌉ / n quantile is the canonical choice.
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(n, k))
    sorted_scores = np.sort(cal)
    return float(sorted_scores[k - 1])


def rolling_split_conformal_coverage(
    scores: np.ndarray,
    *,
    alpha: float = 0.1,
    window: int = 100,
) -> dict[str, float]:
    """Run rolling split-conformal: at step t, calibrate on scores[t-window:t], predict t.

    Returns the empirical coverage actually achieved (fraction of test scores <= rolling q),
    plus auxiliary stats. Useful as a diagnostic for whether the exchangeability assumption
    is plausible for this series.
    """
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1:
        raise ValueError("scores must be 1-D")
    if window < 10:
        raise ValueError("window must be >= 10 for stable rolling-quantile estimates")
    if s.size <= window:
        raise ValueError(f"need more than `window` points; got {s.size} <= {window}")
    covered = 0
    total = 0
    qs: list[float] = []
    for t in range(window, s.size):
        cal = s[t - window:t]
        q = split_conformal_quantile(cal, alpha=alpha)
        qs.append(q)
        if s[t] <= q:
            covered += 1
        total += 1
    return {
        "alpha_nominal": alpha,
        "empirical_coverage": covered / total if total else float("nan"),
        "n_eval": int(total),
        "median_quantile": float(np.median(qs)) if qs else float("nan"),
    }
