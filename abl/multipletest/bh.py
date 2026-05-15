"""Benjamini-Hochberg FDR control.

Source
------
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical
and powerful approach to multiple testing. *Journal of the Royal Statistical Society,
Series B (Methodological)*, 57(1), 289-300.
https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x

Statement
---------
Given m p-values sorted as p_(1) ≤ ... ≤ p_(m) testing m null hypotheses,
let k be the largest index satisfying p_(k) ≤ (k/m) * alpha. Reject H_(1), ..., H_(k).

The procedure controls the False Discovery Rate (expected proportion of false rejections
among all rejections) at level alpha under the assumptions of independence or positive
regression dependence (PRDS). Under arbitrary dependence, use the BH-Yekutieli variant
which divides alpha by sum(1/k for k in 1..m).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BHResult:
    """Result of a Benjamini-Hochberg run."""

    pvals: np.ndarray  # original order
    rejected: np.ndarray  # bool array, same shape as pvals
    pvals_adjusted: np.ndarray  # BH-adjusted q-values (monotone)
    alpha: float
    n_rejected: int
    threshold: float | None  # the largest p_(k) that crossed; None if no rejections


def benjamini_hochberg(
    pvals: np.ndarray, alpha: float = 0.05, *, method: str = "bh"
) -> BHResult:
    """Apply BH or BH-Yekutieli FDR control at level alpha.

    Parameters
    ----------
    pvals : array-like, shape (m,)
        The m p-values to correct.
    alpha : float, default 0.05
        Target FDR level.
    method : {"bh", "by"}, default "bh"
        - "bh"  : Benjamini-Hochberg (1995). Controls FDR under independence and PRDS.
        - "by"  : Benjamini-Yekutieli (2001). Controls FDR under ARBITRARY dependence
                  by dividing the BH threshold by c(m) = Σ_{k=1}^m 1/k. Conservative
                  but assumption-free; the right choice when the p-values' dependence
                  structure is unknown (e.g. multiple correlated trading-strategy variants).

    Returns
    -------
    BHResult
        rejected : boolean array. rejected[i] is True if H_i is rejected at FDR alpha.
        pvals_adjusted : adjusted q-values, monotone-corrected (a q-value is never
                         lower than a smaller-ranked p-value's q-value).

    References
    ----------
    - Benjamini, Y., & Hochberg, Y. (1995). JRSS-B 57(1), 289-300.
    - Benjamini, Y., & Yekutieli, D. (2001). The Control of the False Discovery Rate in
      Multiple Testing under Dependency. *Annals of Statistics*, 29(4), 1165-1188.
    """
    pvals = np.asarray(pvals, dtype=float)
    if pvals.ndim != 1:
        raise ValueError("pvals must be 1-D")
    if np.any((pvals < 0) | (pvals > 1) | ~np.isfinite(pvals)):
        raise ValueError("pvals must be in [0, 1] and finite")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if method not in ("bh", "by"):
        raise ValueError(f"method must be 'bh' or 'by', got {method!r}")

    m = pvals.size
    if m == 0:
        return BHResult(
            pvals=pvals,
            rejected=np.zeros(0, dtype=bool),
            pvals_adjusted=np.zeros(0, dtype=float),
            alpha=alpha,
            n_rejected=0,
            threshold=None,
        )

    order = np.argsort(pvals, kind="mergesort")
    p_sorted = pvals[order]
    ranks = np.arange(1, m + 1)
    # BH-Yekutieli divides by the harmonic sum c(m) = sum 1/k
    c_m = float(np.sum(1.0 / ranks)) if method == "by" else 1.0
    thresholds = (ranks / (m * c_m)) * alpha

    # Largest k with p_(k) <= (k / (m * c_m)) * alpha
    below = p_sorted <= thresholds
    if below.any():
        k = int(np.max(np.where(below)[0])) + 1  # 1-indexed
        cutoff = p_sorted[k - 1]
    else:
        k = 0
        cutoff = None

    # BH/BY-adjusted q-values with monotone correction
    q_sorted = np.minimum.accumulate((p_sorted * m * c_m / ranks)[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    rejected_sorted = np.zeros(m, dtype=bool)
    if k > 0:
        rejected_sorted[:k] = True

    # Unsort back to the input order
    rejected = np.empty(m, dtype=bool)
    rejected[order] = rejected_sorted
    pvals_adjusted = np.empty(m, dtype=float)
    pvals_adjusted[order] = q_sorted

    return BHResult(
        pvals=pvals,
        rejected=rejected,
        pvals_adjusted=pvals_adjusted,
        alpha=alpha,
        n_rejected=int(rejected.sum()),
        threshold=float(cutoff) if cutoff is not None else None,
    )
