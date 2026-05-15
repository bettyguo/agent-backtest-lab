"""Bonferroni-Holm and Romano-Wolf stepwise multiple-testing procedures.

These control the FAMILYWISE ERROR RATE (FWER) — P(at least one false rejection) — at
level alpha. They are STRICTER than FDR procedures (BH / BY) but in many regulatory and
academic contexts FWER is the right thing to control.

Procedures
----------

A. **Bonferroni-Holm** (Holm 1979). Stepdown variant of Bonferroni:
   sort p-values, find largest k with p_(k) > alpha / (m - k + 1); reject everything
   strictly smaller. Controls FWER under arbitrary dependence.

B. **Romano-Wolf** (Romano & Wolf 2005). Stepwise multiple testing built on the
   stationary block bootstrap; controls FWER and is generally more powerful than
   Bonferroni-Holm because it uses the joint distribution of the test statistics.

Sources
-------
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure.
  *Scandinavian Journal of Statistics*, 6(2), 65-70.
- Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data
  Snooping. *Econometrica*, 73(4), 1237-1282.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abl.multipletest.spa import _stationary_block_bootstrap_indices


@dataclass(frozen=True)
class StepwiseResult:
    pvals: np.ndarray
    rejected: np.ndarray
    n_rejected: int
    alpha: float
    method: str


def bonferroni_holm(pvals: np.ndarray, alpha: float = 0.05) -> StepwiseResult:
    """Holm (1979) step-down FWER control."""
    pvals = np.asarray(pvals, dtype=float)
    if pvals.ndim != 1:
        raise ValueError("pvals must be 1-D")
    if np.any((pvals < 0) | (pvals > 1) | ~np.isfinite(pvals)):
        raise ValueError("pvals must be in [0, 1] and finite")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    m = pvals.size
    if m == 0:
        return StepwiseResult(pvals=pvals, rejected=np.zeros(0, dtype=bool),
                              n_rejected=0, alpha=alpha, method="holm")

    order = np.argsort(pvals, kind="mergesort")
    p_sorted = pvals[order]
    thresholds = alpha / (m - np.arange(m))
    # Walk down: stop at the first p_(k) > threshold(k); reject everything strictly before.
    rejected_sorted = np.zeros(m, dtype=bool)
    for k in range(m):
        if p_sorted[k] <= thresholds[k]:
            rejected_sorted[k] = True
        else:
            break

    rejected = np.empty(m, dtype=bool)
    rejected[order] = rejected_sorted
    return StepwiseResult(
        pvals=pvals,
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        alpha=alpha,
        method="holm",
    )


def romano_wolf_stepwise(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    avg_block_len: float | None = None,
    rng: np.random.Generator | None = None,
) -> StepwiseResult:
    """Romano-Wolf (2005) stepwise multiple-testing for "which strategies beat the benchmark."

    Operates on the SAME loss-differential matrix used by Reality Check / SPA:
    `f_n,t = r_n,t − r_bench,t`. At each step we compute the max-statistic of remaining
    strategies, calibrate a one-sided critical value via the stationary block bootstrap,
    reject any strategies whose loss-differential mean clears it, and iterate on the
    remaining strategies. Stops when no further rejections are made.

    Controls FWER under weak stationarity; assumes the block bootstrap is a valid
    resampling scheme for the underlying time series.

    Returns
    -------
    StepwiseResult with `rejected[i]` indicating strategy i was rejected (i.e., beat
    the benchmark) at level alpha after the stepwise correction.
    """
    S = np.asarray(strategy_returns, dtype=float)
    B = np.asarray(benchmark_returns, dtype=float)
    if S.ndim != 2:
        raise ValueError("strategy_returns must be 2-D (N, T)")
    if B.ndim != 1:
        raise ValueError("benchmark_returns must be 1-D")
    N, T = S.shape
    if B.size != T:
        raise ValueError("benchmark length mismatch")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if rng is None:
        rng = np.random.default_rng(20260514)
    if avg_block_len is None:
        avg_block_len = float(np.sqrt(T))

    F = S - B[np.newaxis, :]
    f_bar = F.mean(axis=1)
    sqrt_T = float(np.sqrt(T))

    # Pre-compute bootstrap resample indices once
    idx_mat = _stationary_block_bootstrap_indices(T, n_bootstrap, avg_block_len, rng)

    active = np.ones(N, dtype=bool)
    rejected = np.zeros(N, dtype=bool)

    while True:
        if not active.any():
            break
        # Centered bootstrap statistic over the active set
        max_boot = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            cols = idx_mat[b]
            F_resampled = F[np.ix_(active, cols)]  # (n_active, T)
            mean_resampled = F_resampled.mean(axis=1) - f_bar[active]
            max_boot[b] = float(sqrt_T * mean_resampled.max())
        # One-sided (1-alpha) critical value
        crit = float(np.quantile(max_boot, 1.0 - alpha))
        # Observed studentized statistics for the active set
        v_obs = sqrt_T * f_bar[active]
        # Identify which of the active strategies clear the critical value
        v_full = np.full(N, -np.inf)
        v_full[active] = v_obs
        new_rej = (v_full > crit) & active
        if not new_rej.any():
            break
        rejected |= new_rej
        active &= ~new_rej

    return StepwiseResult(
        pvals=np.full(N, np.nan),  # Romano-Wolf returns rejections, not adjusted p-values
        rejected=rejected,
        n_rejected=int(rejected.sum()),
        alpha=alpha,
        method="romano_wolf",
    )
