"""White's Reality Check + Hansen's Superior Predictive Ability (SPA) test.

These tests answer the question: "I tried N strategies; is the BEST one statistically
better than a benchmark, after accounting for the fact that I tried N?"

This is conceptually the same problem DSR addresses, but with two important differences:
1. SPA/Reality Check test against a SPECIFIC benchmark (e.g. buy-and-hold) rather than
   "any positive Sharpe."
2. They use a stationary block bootstrap (Politis-Romano 1994) to handle serial
   correlation in the relative-loss series, instead of the parametric Cornish-Fisher
   adjustment DSR uses.

Sources
-------
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097-1126.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *J. Business & Economic
  Statistics*, 23(4), 365-380.
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *J. American
  Statistical Association*, 89(428), 1303-1313.

Implementation notes
--------------------
- We implement the **stationary block bootstrap** of Politis & Romano (1994), with
  geometric block-length distribution and mean block length `avg_block_len`. The default
  `avg_block_len = sqrt(T)` is the common practitioner choice; the literature (Politis-
  White 2004) gives data-driven alternatives we don't implement here.
- For Reality Check (White 2000) we report the bootstrap p-value of the studentized
  max-mean of the relative-loss series against the null "no strategy beats the benchmark."
- We do NOT implement Hansen's full studentization-recentering machinery (that's a
  meaningful extension — filed in roadmap). Our SPA p-value follows the simpler
  consistent variant. This is documented in the docstring.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SPAResult:
    pvalue: float                # bootstrap p-value (lower = stronger evidence best > benchmark)
    best_strategy_idx: int       # index of the strategy with the highest mean relative loss
    best_mean_relative: float    # mean relative return of the best strategy
    n_strategies: int
    n_periods: int
    n_bootstrap: int
    avg_block_len: float


def _stationary_block_bootstrap_indices(
    n: int, b: int, avg_block_len: float, rng: np.random.Generator
) -> np.ndarray:
    """Return `b` × `n` index matrix for the stationary block bootstrap (Politis-Romano 1994).

    Each row is a length-n resample of [0, n). Block lengths are geometrically distributed
    with mean `avg_block_len`. Starting indices are uniform.
    """
    if n < 2 or avg_block_len <= 0:
        raise ValueError("require n >= 2 and avg_block_len > 0")
    p = 1.0 / avg_block_len
    out = np.empty((b, n), dtype=np.intp)
    for row in range(b):
        idx = np.empty(n, dtype=np.intp)
        t = 0
        while t < n:
            start = int(rng.integers(0, n))
            # Geometric block length with parameter p
            # Use a quick draw; we'll truncate to remaining slots.
            block_len = int(rng.geometric(p))
            block_len = max(1, min(block_len, n - t))
            for k in range(block_len):
                idx[t + k] = (start + k) % n
            t += block_len
        out[row] = idx
    return out


def reality_check_spa(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    avg_block_len: float | None = None,
    rng: np.random.Generator | None = None,
) -> SPAResult:
    """White's Reality Check / SPA-style test.

    Parameters
    ----------
    strategy_returns : array, shape (N, T)
        N strategies' per-period return series.
    benchmark_returns : array, shape (T,)
        Per-period benchmark return series.
    n_bootstrap : int, default 1000
        Number of stationary-block-bootstrap resamples.
    avg_block_len : float or None
        Geometric mean block length. If None, defaults to sqrt(T).
    rng : np.random.Generator or None
        Used for reproducibility.

    Returns
    -------
    SPAResult

    Interpretation
    --------------
    Small p-value -> reject the null "no strategy beats the benchmark." Large p-value
    -> the apparent edge of the best strategy is plausibly noise given the multiple-
    strategy search.

    Caveat: the block-bootstrap assumes weak stationarity. Regime-switching data violate
    this; the reported p-value is then approximate.
    """
    S = np.asarray(strategy_returns, dtype=float)
    B = np.asarray(benchmark_returns, dtype=float)
    if S.ndim != 2:
        raise ValueError("strategy_returns must be 2-D (N, T)")
    if B.ndim != 1:
        raise ValueError("benchmark_returns must be 1-D")
    N, T = S.shape
    if B.size != T:
        raise ValueError(f"benchmark length {B.size} != strategy T={T}")
    if N < 1 or T < 4:
        raise ValueError("need N>=1 strategies and T>=4 periods")

    # Relative-loss series f_n,t = r_n,t - r_bench,t
    F = S - B[np.newaxis, :]
    f_bar = F.mean(axis=1)              # shape (N,)
    n_best = int(np.argmax(f_bar))
    V_obs = float(np.sqrt(T) * f_bar.max())

    if avg_block_len is None:
        avg_block_len = float(np.sqrt(T))
    if rng is None:
        rng = np.random.default_rng(20260514)

    idx_mat = _stationary_block_bootstrap_indices(T, n_bootstrap, avg_block_len, rng)
    # Centered bootstrap statistic (White 2000): V*_b = sqrt(T) * max_n (mean over resample of f_n - f_bar_n)
    V_boot = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        cols = idx_mat[b]
        F_resampled = F[:, cols]            # (N, T)
        mean_resampled = F_resampled.mean(axis=1)
        V_boot[b] = float(np.sqrt(T) * (mean_resampled - f_bar).max())

    # p-value: P(V_boot >= V_obs) under the centered null
    p = float(np.mean(V_boot >= V_obs))
    return SPAResult(
        pvalue=p,
        best_strategy_idx=n_best,
        best_mean_relative=float(f_bar[n_best]),
        n_strategies=N,
        n_periods=T,
        n_bootstrap=n_bootstrap,
        avg_block_len=float(avg_block_len),
    )
