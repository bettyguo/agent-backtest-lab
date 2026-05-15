"""Probabilistic Sharpe Ratio (PSR).

Source
------
Bailey, D. H., & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
*Journal of Risk*, 15(2). SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643

Statement
---------
Given a return series of length T with observed Sharpe ratio ŜR, sample skewness γ̂₃,
and sample excess kurtosis γ̂₄ - 1 (where γ̂₄ is the standardized fourth moment),

    PSR(SR*) = Φ( ( (ŜR - SR*) · √(T - 1) ) / √( 1 - γ̂₃·ŜR + ((γ̂₄ - 1)/4)·ŜR² ) )

where Φ is the standard normal CDF and SR* is the benchmark Sharpe to test against
(usually 0).

The Cornish-Fisher-style denominator corrects for the well-known fact that the
standard error of the Sharpe estimator depends on the higher moments of the return
distribution; a high observed Sharpe in a skewed or fat-tailed series is less impressive
than the same number in a Gaussian one.

Notes
-----
The Sharpe inputs here are observed (raw) Sharpes, not annualized. If the user has
annualized them by √252, the standard error scaling already absorbs that — PSR is a
quantity of the observed ratio. To stay consistent with that convention we accept the
return series in its sampled cadence (typically daily) and compute the daily Sharpe
internally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from abl.config import DEFAULT_ANNUALIZATION


@dataclass(frozen=True)
class SharpeStats:
    """Summary statistics of a return series, sufficient for PSR / DSR."""

    sharpe_observed: float       # sample Sharpe ratio at the input cadence (e.g. daily)
    sharpe_annualized: float     # sharpe_observed * sqrt(annualization)
    skewness: float              # sample skewness γ̂₃
    excess_kurtosis: float       # γ̂₄ - 1
    n_obs: int


def sharpe_ratio(returns: np.ndarray, annualization: int = DEFAULT_ANNUALIZATION) -> SharpeStats:
    """Compute the sample Sharpe (at the input cadence and annualized) plus higher moments."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        raise ValueError(f"need at least 2 finite observations, got {r.size}")
    mean = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        raise ValueError("standard deviation is zero or negative")
    sr_obs = mean / sd
    sr_ann = sr_obs * np.sqrt(annualization)
    # Sample skewness γ̂₃ = E[(x-μ)^3] / σ^3, using ddof=1 normalization in σ
    centered = r - mean
    skew = float(np.mean(centered**3) / sd**3)
    # Sample kurtosis γ̂₄ = E[(x-μ)^4] / σ^4 (NOT excess); we return excess = γ̂₄ - 1
    # Note the PSR paper uses excess kurtosis = γ4 - 3 + 2 = γ4 - 1 (mistranscribed in some
    # secondary sources). Specifically: the denominator is √(1 - γ3*SR + ((γ4 - 1)/4)*SR^2),
    # where γ4 here is the standardized fourth moment (NOT excess kurtosis), so (γ4 - 1)
    # rather than (γ4 - 3). We follow the canonical formulation: γ4 standardized fourth
    # moment, term is (γ4 - 1)/4.
    kurt_standardized = float(np.mean(centered**4) / sd**4)
    excess = kurt_standardized - 1.0
    return SharpeStats(
        sharpe_observed=sr_obs,
        sharpe_annualized=sr_ann,
        skewness=skew,
        excess_kurtosis=excess,
        n_obs=int(r.size),
    )


def probabilistic_sharpe(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    *,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> float:
    """Probabilistic Sharpe Ratio: probability that the true (observed-cadence) Sharpe exceeds sr_benchmark.

    `sr_benchmark` is interpreted at the same cadence as the input return series (daily by
    default). If you have an annualized benchmark Sharpe S_a, pass `sr_benchmark = S_a / sqrt(252)`.
    """
    stats = sharpe_ratio(returns, annualization=annualization)
    return _psr_from_stats(stats.sharpe_observed, sr_benchmark, stats.skewness,
                           stats.excess_kurtosis, stats.n_obs)


def _psr_from_stats(sr_obs: float, sr_benchmark: float, skew: float, excess_kurt: float, n: int) -> float:
    """Core PSR formula. Exposed for DSR which shares the kernel."""
    if n < 2:
        raise ValueError("n must be >= 2")
    denom_inside = 1.0 - skew * sr_obs + (excess_kurt / 4.0) * (sr_obs ** 2)
    # The Cornish-Fisher correction can in principle go negative for extreme skew/kurt; clip.
    denom_inside = max(denom_inside, 1e-12)
    z = (sr_obs - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom_inside)
    return float(norm.cdf(z))
