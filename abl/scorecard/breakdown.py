"""Per-ticker breakdown and cross-strategy correlation analysis.

Per-ticker breakdown answers: "is the strategy's edge concentrated in a few tickers,
or robust across the universe?" A real edge spreads; a fragile edge concentrates.

Cross-strategy correlation answers: "are the N candidate strategies actually
independent trials, or 30 versions of the same thing?" High pairwise correlation
inflates the effective trial count and is a setup for DSR mis-specification.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from abl.config import DEFAULT_ANNUALIZATION


@dataclass(frozen=True)
class TickerBreakdownRow:
    ticker: str
    n_decisions: int
    n_non_flat: int
    hit_rate: float           # P(call matches sign(ret_next)) over non-FLAT calls
    mean_ret_contribution: float  # mean ticker-level contribution to portfolio return
    sharpe_annualized: float  # per-ticker ann. Sharpe of its return contribution


def per_ticker_breakdown(
    decisions: pd.DataFrame,
    *,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> list[TickerBreakdownRow]:
    """Break down a BacktestResult.decisions DataFrame by ticker.

    The decisions table is built by the engine and has columns:
    `as_of, ticker, direction, confidence, position, ret_next`. We use
    `position * ret_next` as the per-(ticker, day) return contribution.

    Returns one row per ticker. Empty universes return an empty list.
    """
    if decisions is None or decisions.empty:
        return []
    out: list[TickerBreakdownRow] = []
    for ticker, grp in decisions.groupby("ticker", sort=True):
        n_decisions = int(len(grp))
        non_flat = grp[grp["direction"] != "FLAT"]
        n_non_flat = int(len(non_flat))
        if n_non_flat > 0:
            sign_call = np.where(non_flat["direction"] == "LONG", 1, -1)
            sign_real = np.where(non_flat["ret_next"].to_numpy() > 0, 1, -1)
            hit = float((sign_call == sign_real).mean())
        else:
            hit = float("nan")
        contrib = (grp["position"].astype(float) * grp["ret_next"].astype(float)).dropna()
        if contrib.size >= 2 and contrib.std(ddof=1) > 0:
            sr_obs = float(contrib.mean() / contrib.std(ddof=1))
            sr_ann = sr_obs * float(np.sqrt(annualization))
        else:
            sr_ann = float("nan")
        out.append(
            TickerBreakdownRow(
                ticker=str(ticker),
                n_decisions=n_decisions,
                n_non_flat=n_non_flat,
                hit_rate=hit,
                mean_ret_contribution=float(contrib.mean()) if contrib.size else float("nan"),
                sharpe_annualized=sr_ann,
            )
        )
    return out


def cross_strategy_correlation(
    returns_by_name: dict[str, pd.Series],
) -> pd.DataFrame:
    """Pearson correlation matrix of per-strategy net-return series.

    Aligns series on their common index; correlations are over the intersection.

    Returns a square DataFrame indexed by strategy name. Diagonal is 1.0 by construction.
    A high off-diagonal (say > 0.8) flags a redundant trial set — the effective number
    of independent strategies is far less than `N`, and using that `N` in DSR will
    UNDER-correct (because DSR assumes N IID trials).
    """
    if not returns_by_name:
        return pd.DataFrame()
    df = pd.DataFrame(returns_by_name).dropna(how="all")
    return df.corr(method="pearson")


def effective_n_trials(corr_matrix: pd.DataFrame) -> float:
    """Heuristic 'effective N' for use in DSR when strategies are correlated.

    Uses the eigenvalue-spread heuristic: effective N ≈ (Σ λ_i)² / Σ λ_i² where λ_i are
    the eigenvalues of the correlation matrix. This is the "participation ratio" of
    random-matrix theory; it equals N for IID strategies and collapses toward 1 for
    perfectly-correlated ones.

    This is a HEURISTIC, not a theorem; we expose it so users can see the magnitude of
    the redundancy and reach their own DSR-N choice.
    """
    if corr_matrix.empty:
        return 0.0
    M = corr_matrix.to_numpy(dtype=float)
    # Symmetrize defensively (numerical noise can break perfect symmetry)
    M = 0.5 * (M + M.T)
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.clip(eigvals, 1e-12, None)
    s1 = float(eigvals.sum())
    s2 = float((eigvals ** 2).sum())
    return (s1 * s1) / s2 if s2 > 0 else 0.0
