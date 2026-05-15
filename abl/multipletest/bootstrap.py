"""BCa (bias-corrected and accelerated) bootstrap confidence intervals.

Source
------
Efron, B. (1987). Better Bootstrap Confidence Intervals. *Journal of the American
Statistical Association*, 82(397), 171-185.

Statement
---------
The BCa bootstrap improves on the naive percentile bootstrap by correcting for (i) bias
in the statistic and (ii) skewness in its sampling distribution. For a statistic θ̂(X)
on a sample X = (x_1, ..., x_n), draw B bootstrap resamples θ̂*_b, compute:

  - bias correction z_0 = Φ⁻¹(P(θ̂*_b < θ̂))
  - acceleration   a   ≈ Σ (θ̂_(.) - θ̂_(-i))³ / (6 · (Σ (θ̂_(.) - θ̂_(-i))²)^{3/2})
                         (jackknife)

The (1-α) BCa CI uses adjusted percentiles:

  α_lo = Φ(z_0 + (z_0 + z_{α/2}) / (1 - a(z_0 + z_{α/2})))
  α_hi = Φ(z_0 + (z_0 + z_{1-α/2}) / (1 - a(z_0 + z_{1-α/2})))

Returns the (α_lo, α_hi) percentiles of the bootstrap distribution.

When to use this
----------------
Whenever the statistic of interest is NOT well-approximated by a Gaussian — e.g. the
Sharpe ratio with fat-tailed returns, the Sortino ratio, the Calmar ratio, drawdown.
The IID-Gaussian or HAC SE is the right call when the asymptotic-normality assumption is
defensible; BCa is the right call when you'd rather not assume that.

Caveats
-------
- We use the IID bootstrap by default. For autocorrelated series, use the
  `stationary_block_indices` helper from `abl.multipletest.spa` and resample with it
  (an example is in the docstring of `bca_bootstrap_ci`).
- B = 2000 is a reasonable default for 95% CIs; smaller `B` makes the endpoints noisier.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BCaResult:
    statistic: float
    ci_low: float
    ci_high: float
    alpha: float
    n_bootstrap: int
    z_0: float
    acceleration: float


def bca_bootstrap_ci(
    data: np.ndarray,
    statistic_fn: Callable[[np.ndarray], float],
    *,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> BCaResult:
    """BCa bootstrap CI for `statistic_fn(data)`.

    Parameters
    ----------
    data : array-like, shape (n,) or (n, k)
        The sample.
    statistic_fn : callable
        Maps a resampled view of `data` to a scalar.
    n_bootstrap : int, default 2000
    alpha : float, default 0.05 -> 95% CI

    Returns
    -------
    BCaResult with `ci_low`, `ci_high`, plus the diagnostic (z_0, acceleration).
    """
    arr = np.asarray(data)
    n = arr.shape[0]
    if n < 4:
        raise ValueError("need >=4 observations for a meaningful BCa CI")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if rng is None:
        rng = np.random.default_rng(20260514)

    theta_hat = float(statistic_fn(arr))
    # Bootstrap resamples
    boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(statistic_fn(arr[idx]))
    # Bias correction
    p_less = float((boot < theta_hat).mean())
    p_less = min(max(p_less, 1.0 / (n_bootstrap + 1)), 1.0 - 1.0 / (n_bootstrap + 1))
    z_0 = float(norm.ppf(p_less))

    # Jackknife for acceleration
    jack = np.empty(n)
    for i in range(n):
        jack[i] = float(statistic_fn(np.delete(arr, i, axis=0)))
    jack_mean = jack.mean()
    num = float(np.sum((jack_mean - jack) ** 3))
    den = float(6.0 * (np.sum((jack_mean - jack) ** 2)) ** 1.5)
    accel = num / den if den > 0 else 0.0

    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1.0 - alpha / 2)

    def _adjust(z):
        denom = 1.0 - accel * (z_0 + z)
        if denom == 0:
            return float("nan")
        return float(norm.cdf(z_0 + (z_0 + z) / denom))

    a_lo = _adjust(z_lo)
    a_hi = _adjust(z_hi)
    # Clip to [0, 1] for numerical safety
    a_lo = min(max(a_lo, 1e-6), 1.0 - 1e-6)
    a_hi = min(max(a_hi, 1e-6), 1.0 - 1e-6)
    ci_low = float(np.quantile(boot, a_lo))
    ci_high = float(np.quantile(boot, a_hi))
    return BCaResult(
        statistic=theta_hat,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        z_0=z_0,
        acceleration=accel,
    )


def bca_sharpe_ci(
    returns: np.ndarray, *, n_bootstrap: int = 2000, alpha: float = 0.05,
    annualization: int = 252, rng: np.random.Generator | None = None,
) -> BCaResult:
    """BCa CI specifically for the annualized Sharpe ratio."""
    scale = float(np.sqrt(annualization))

    def _sr(x: np.ndarray) -> float:
        if x.size < 2:
            return float("nan")
        sd = float(x.std(ddof=1))
        if sd <= 0:
            return float("nan")
        return float(x.mean() / sd * scale)

    return bca_bootstrap_ci(returns, _sr, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng)
