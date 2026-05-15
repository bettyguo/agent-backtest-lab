"""Deflated Sharpe Ratio (DSR).

Source
------
Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting, and Non-Normality. *Journal of Portfolio Management*,
40(5), 94-107. SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
PDF: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

Statement
---------
DSR adjusts the PSR for the multiple-testing problem: if you tried N strategies and report
the best, the observed Sharpe is inflated by the maximum-over-N selection. DSR replaces
PSR's benchmark `SR_benchmark` with an `SR_0` derived from the expected-max-of-N under
the null:

    SR_0 = √V[ŜR] · ( (1 - γ) · Φ⁻¹(1 - 1/N) + γ · Φ⁻¹(1 - 1/(N·e)) )

where γ is the Euler-Mascheroni constant (~0.5772) and V[ŜR] is the cross-trial variance
of estimated Sharpes (the variability in measured Sharpes across the N trials).

DSR is then PSR(SR_0):

    DSR = Φ( ( (ŜR - SR_0) · √(T - 1) ) / √( 1 - γ̂₃·ŜR + ((γ̂₄ - 1)/4)·ŜR² ) )

Notes
-----
- `SR_0` and `ŜR` must be in the same cadence (typically observed-daily; both NOT annualized).
- For `N=1` and `V[ŜR] = 0`, `SR_0 = 0` and DSR reduces to `PSR(0)`.
- The Gumbel approximation to the expected maximum of N standard normals is what produces
  the closed form; for small N (N < ~10) it is less accurate but still monotone.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from abl.config import DEFAULT_ANNUALIZATION
from abl.multipletest.psr import _psr_from_stats, sharpe_ratio


def expected_max_sharpe_under_null(n_trials: int, var_trial_sharpe: float) -> float:
    """Expected maximum Sharpe over N IID trials with cross-trial variance V[ŜR].

    Uses the standard Gumbel approximation to E[max_N N(0,1)] from Bailey & López de Prado:

        E[max] ≈ (1 - γ) · Φ⁻¹(1 - 1/N) + γ · Φ⁻¹(1 - 1/(N·e))

    where γ is Euler-Mascheroni. Multiplied by √V[ŜR] to scale into the Sharpe metric.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if var_trial_sharpe < 0:
        raise ValueError("var_trial_sharpe must be >= 0")
    if n_trials == 1 or var_trial_sharpe == 0:
        return 0.0
    gamma = float(np.euler_gamma)  # 0.57721566...
    e = math.e
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(np.sqrt(var_trial_sharpe) * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe(
    returns: np.ndarray,
    *,
    n_trials: int,
    var_trial_sharpe: float,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> dict[str, float]:
    """Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    returns : array-like
        Per-period return series of the *selected* strategy (the one with the best
        observed Sharpe among the N candidates).
    n_trials : int
        The N — number of strategies / configurations / prompt variants tried. If the
        user reports best-of-50, n_trials=50. Lying about this defeats the correction.
    var_trial_sharpe : float
        V[ŜR] — variance of the *estimated* Sharpe ratios across the N trials, at the
        same cadence as `returns`. If only one strategy was tried (n_trials=1), pass 0.

    Returns
    -------
    dict
        {"dsr": float, "sr_0": float, "sharpe_observed": float, "sharpe_annualized": float}
    """
    stats = sharpe_ratio(returns, annualization=annualization)
    sr_0 = expected_max_sharpe_under_null(n_trials, var_trial_sharpe)
    dsr = _psr_from_stats(
        sr_obs=stats.sharpe_observed,
        sr_benchmark=sr_0,
        skew=stats.skewness,
        excess_kurt=stats.excess_kurtosis,
        n=stats.n_obs,
    )
    return {
        "dsr": dsr,
        "sr_0": sr_0,
        "sharpe_observed": stats.sharpe_observed,
        "sharpe_annualized": stats.sharpe_annualized,
        "n_trials": n_trials,
        "var_trial_sharpe": var_trial_sharpe,
    }
