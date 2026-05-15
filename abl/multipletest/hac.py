"""HAC / Newey-West standard error for the Sharpe ratio (Lo 2002).

Source
------
Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4), 36-52.

Statement
---------
For a return series with mean µ, variance σ², and the GMM-based asymptotic variance of
the Sharpe estimator under SERIAL CORRELATION, the heteroskedasticity-and-autocorrelation
consistent (HAC) standard error is

    SE_HAC(ŜR) = √( (1 + 0.5 · ŜR²) · η(q) / T )

where η(q) is a serial-correlation correction factor based on the autocorrelations
ρ_1, ..., ρ_q of the return series:

    η(q) = 1 + 2 · Σ_{k=1}^{q} (1 - k/(q+1)) · ρ_k

The Bartlett kernel weights (1 - k/(q+1)) are the standard Newey-West (1987) choice. For
q=0 this reduces to the IID-Gaussian formula `(1 + 0.5·ŜR²)/T` used elsewhere in the
package. For positively-autocorrelated series, η(q) > 1 and the HAC SE is LARGER than
the IID SE — that is, the IID CI is overconfident.

Practical choice of q
---------------------
A common rule of thumb is q ≈ ⌊4 · (T/100)^(2/9)⌋ (Newey-West 1994 automatic bandwidth).
For daily returns over a typical 1–5 year sample this gives q ≈ 5–10. The package
exposes `q` as a parameter, defaulting to that automatic choice.

References
----------
- Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4), 36-52.
- Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703-708.
- Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix
  Estimation. *Review of Economic Studies*, 61(4), 631-653.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from abl.config import DEFAULT_ANNUALIZATION


@dataclass(frozen=True)
class HACSharpeResult:
    sharpe_observed: float
    sharpe_annualized: float
    se_hac: float                # observed-cadence HAC SE of ŜR
    se_iid: float                # IID Gaussian SE of ŜR (for comparison)
    eta_q: float                 # serial-correlation correction factor
    q_lags: int                  # lag truncation actually used
    ci_low_obs: float            # observed-cadence HAC 95% CI lower
    ci_high_obs: float           # observed-cadence HAC 95% CI upper
    ci_low_ann: float            # annualized HAC 95% CI lower
    ci_high_ann: float           # annualized HAC 95% CI upper
    autocorr_lags: tuple[float, ...]  # estimated ρ_1, ..., ρ_q


def auto_lag_truncation(n: int) -> int:
    """Newey-West 1994 automatic bandwidth: q ≈ floor(4 * (T/100)^(2/9))."""
    if n < 10:
        return 0
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _sample_autocorrelations(r: np.ndarray, q: int) -> np.ndarray:
    """Sample autocorrelations ρ_1, ..., ρ_q of a 1-D series."""
    if q <= 0:
        return np.zeros(0)
    r = r - r.mean()
    denom = float(np.dot(r, r))
    if denom <= 0:
        return np.zeros(q)
    out = np.empty(q)
    for k in range(1, q + 1):
        if k >= r.size:
            out[k - 1] = 0.0
        else:
            out[k - 1] = float(np.dot(r[:-k], r[k:])) / denom
    return out


def newey_west_eta(autocorr_lags: np.ndarray) -> float:
    """η(q) = 1 + 2·Σ_k (1 − k/(q+1))·ρ_k under the Bartlett kernel."""
    q = autocorr_lags.size
    if q == 0:
        return 1.0
    k = np.arange(1, q + 1)
    weights = 1.0 - k / (q + 1.0)
    return float(1.0 + 2.0 * np.sum(weights * autocorr_lags))


def hac_sharpe_ci(
    returns: np.ndarray,
    *,
    q_lags: int | None = None,
    alpha: float = 0.05,
    annualization: int = DEFAULT_ANNUALIZATION,
) -> HACSharpeResult:
    """HAC / Newey-West-corrected CI for the Sharpe ratio (Lo 2002).

    Parameters
    ----------
    returns : array-like
        Per-period return series.
    q_lags : int or None
        Bartlett-kernel lag truncation. If None, uses `auto_lag_truncation(T)`.
    alpha : float
        CI level: returned CI is (1 - alpha) coverage. Default 0.05 -> 95% CI.
    annualization : int
        Sharpe annualization factor (252 for daily US-equity).

    Returns
    -------
    HACSharpeResult with both observed-cadence and annualized CIs, plus the
    IID-comparison SE so users can see the magnitude of the autocorrelation correction.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = r.size
    if T < 2:
        raise ValueError(f"need >=2 finite observations, got {T}")
    mean = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 0:
        raise ValueError("standard deviation is zero or negative")
    sr_obs = mean / sd
    sr_ann = sr_obs * np.sqrt(annualization)

    if q_lags is None:
        q_lags = auto_lag_truncation(T)
    q_lags = max(0, min(q_lags, T - 1))
    rhos = _sample_autocorrelations(r, q_lags)
    eta = newey_west_eta(rhos)
    eta_pos = max(eta, 1e-12)  # numerical floor for sqrt

    se_iid = float(np.sqrt((1.0 + 0.5 * sr_obs**2) / T))
    se_hac = float(np.sqrt((1.0 + 0.5 * sr_obs**2) * eta_pos / T))
    z = norm.ppf(1.0 - alpha / 2)
    lo_obs = sr_obs - z * se_hac
    hi_obs = sr_obs + z * se_hac
    scale = float(np.sqrt(annualization))
    return HACSharpeResult(
        sharpe_observed=sr_obs,
        sharpe_annualized=sr_ann,
        se_hac=se_hac,
        se_iid=se_iid,
        eta_q=float(eta),
        q_lags=int(q_lags),
        ci_low_obs=lo_obs,
        ci_high_obs=hi_obs,
        ci_low_ann=lo_obs * scale,
        ci_high_ann=hi_obs * scale,
        autocorr_lags=tuple(float(x) for x in rhos),
    )
