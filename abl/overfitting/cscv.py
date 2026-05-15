"""Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation.

Source
------
Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of
Backtest Overfitting. *Journal of Computational Finance*. SSRN:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

Companion (expository): Bailey, Borwein, López de Prado, Zhu (2014). Pseudo-Mathematics
and Financial Charlatanism. *Notices of the AMS*, 61(5), 458-471.

Statement (CSCV)
----------------
Given an N×T matrix M of N candidate strategies' returns over T periods:

1. Partition the T time-steps into S equal-sized contiguous blocks (S must be even).
2. For each of the C(S, S/2) ways of choosing S/2 blocks as "in-sample":
   a. Let J = chosen-block indices; let J' = its complement.
   b. Compute each strategy's IS Sharpe on M[:, J] and OOS Sharpe on M[:, J'].
   c. Let n* = the strategy with the highest IS Sharpe.
   d. Let ω = rank(OOS Sharpe of n*) / (N + 1).
   e. λ = log(ω / (1 - ω))  (logit of the OOS rank).
3. PBO = fraction of partitions in which λ <= 0 (i.e. the IS-best strategy has below-median
   OOS performance).

A PBO close to 0.5 indicates pure noise (the IS-best is no better OOS than random); a low
PBO indicates the IS-best is reliably good OOS; a high PBO indicates the IS-best is
reliably *bad* OOS — a strong sign of overfitting.

Assumptions
-----------
- Strategy return series are commensurable.
- S divides T cleanly enough that the equal-sized partition is reasonable.
- The Sharpe metric is monotone — higher is better.

Practical tuning
----------------
For T ≈ 252 and S = 16, C(16, 8) = 12870 partitions, manageable. For larger N×T or
finer partitions the cost grows quickly; we cap at 50k partitions and sample uniformly
if needed (documented).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from abl.config import DEFAULT_PBO_BLOCKS

_MAX_PARTITIONS = 50_000


def _block_sharpes(returns_matrix: np.ndarray, block_indices: list[np.ndarray]) -> np.ndarray:
    """Per-strategy Sharpe over a concatenation of the listed block-index arrays.

    `returns_matrix` is (N, T). Returns shape (N,).
    """
    if not block_indices:
        raise ValueError("block_indices must be non-empty")
    cols = np.concatenate(block_indices)
    sub = returns_matrix[:, cols]
    mean = sub.mean(axis=1)
    sd = sub.std(axis=1, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    return mean / sd


def _logit_of_rank(values: np.ndarray, target_idx: int) -> float:
    """Compute log(ω / (1 - ω)) where ω is target_idx's rank fraction in `values`."""
    n = values.size
    # Rank in [1, n] (1 = smallest); average-rank for ties.
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    # Tie-handling: average ranks for ties.
    sorted_vals = values[order]
    for i in range(n):
        ties = sorted_vals == sorted_vals[i]
        ranks[order[ties]] = ranks[order[ties]].mean()
    rank = ranks[target_idx]
    omega = rank / (n + 1.0)
    omega = float(np.clip(omega, 1e-12, 1 - 1e-12))
    return float(np.log(omega / (1.0 - omega)))


@dataclass(frozen=True)
class OverfittingFlag:
    code: str
    severity: str  # "info" | "warn" | "critical"
    pbo: float
    message: str


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    *,
    s_blocks: int = DEFAULT_PBO_BLOCKS,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute PBO via CSCV.

    Parameters
    ----------
    returns_matrix : array, shape (N, T)
        N strategies, T time periods.
    s_blocks : int
        Number of equal-sized blocks; must be even. Default 16.

    Returns
    -------
    dict
        {
            "pbo": float in [0, 1],
            "n_partitions_evaluated": int,
            "n_strategies": int,
            "n_periods": int,
            "n_blocks": int,
            "median_logit_oos_rank": float,
            "flag": OverfittingFlag | None,
        }
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (N strategies × T periods)")
    if s_blocks % 2 != 0:
        raise ValueError("s_blocks must be even")
    if s_blocks < 4:
        raise ValueError("s_blocks must be >= 4 for a non-degenerate CSCV")
    N, T = M.shape
    if N < 2:
        raise ValueError("need at least 2 strategies")
    if T < 2 * s_blocks:
        raise ValueError(f"T={T} must be >= 2 * s_blocks = {2 * s_blocks}")

    # Equal-sized contiguous blocks; drop the trailing fractional if T not divisible.
    block_size = T // s_blocks
    block_indices = [np.arange(b * block_size, (b + 1) * block_size) for b in range(s_blocks)]

    half = s_blocks // 2
    all_combos = list(combinations(range(s_blocks), half))

    if rng is None:
        rng = np.random.default_rng(20260514)
    if len(all_combos) > _MAX_PARTITIONS:
        idx = rng.choice(len(all_combos), size=_MAX_PARTITIONS, replace=False)
        combos_iter = (all_combos[i] for i in idx)
        n_eval = _MAX_PARTITIONS
    else:
        combos_iter = iter(all_combos)
        n_eval = len(all_combos)

    n_overfit = 0
    n_valid = 0
    logits: list[float] = []
    for combo in combos_iter:
        is_blocks = [block_indices[b] for b in combo]
        oos_blocks = [block_indices[b] for b in range(s_blocks) if b not in combo]
        is_sharpe = _block_sharpes(M, is_blocks)
        oos_sharpe = _block_sharpes(M, oos_blocks)
        # If a strategy is degenerate (NaN Sharpe in either side), skip it.
        valid = np.isfinite(is_sharpe) & np.isfinite(oos_sharpe)
        if valid.sum() < 2:
            continue
        is_sub = np.where(valid, is_sharpe, -np.inf)
        n_star = int(np.argmax(is_sub))
        oos_for_rank = np.where(valid, oos_sharpe, np.nan)
        # Replace NaNs with the smallest finite value to keep rank well-defined among valid strategies.
        if not np.isfinite(oos_for_rank[n_star]):
            continue
        finite_min = np.nanmin(oos_for_rank)
        oos_filled = np.where(np.isfinite(oos_for_rank), oos_for_rank, finite_min - 1e9)
        lam = _logit_of_rank(oos_filled, n_star)
        logits.append(lam)
        if lam <= 0:
            n_overfit += 1
        n_valid += 1

    if n_valid == 0:
        return {
            "pbo": float("nan"),
            "n_partitions_evaluated": 0,
            "n_strategies": N,
            "n_periods": T,
            "n_blocks": s_blocks,
            "median_logit_oos_rank": float("nan"),
            "flag": None,
        }
    pbo = n_overfit / n_valid

    flag: OverfittingFlag | None = None
    if pbo >= 0.7:
        flag = OverfittingFlag(
            code="HIGH_PBO",
            severity="critical",
            pbo=pbo,
            message=(
                f"Probability of Backtest Overfitting = {pbo:.3f}. The in-sample-best "
                "strategy reliably underperforms OOS. Treat 'best' results as overfit."
            ),
        )
    elif pbo >= 0.5:
        flag = OverfittingFlag(
            code="ELEVATED_PBO",
            severity="warn",
            pbo=pbo,
            message=(
                f"PBO = {pbo:.3f}. Elevated overfitting risk; the OOS rank of the IS-best "
                "is below median in a majority of partitions."
            ),
        )

    return {
        "pbo": pbo,
        "n_partitions_evaluated": n_eval,
        "n_strategies": N,
        "n_periods": T,
        "n_blocks": s_blocks,
        "median_logit_oos_rank": float(np.median(logits)),
        "flag": flag,
    }
