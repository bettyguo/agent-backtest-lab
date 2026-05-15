"""Multiple-hypothesis correction and selection-bias-aware Sharpe ratios.

The "we tried N configurations / N tickers / N prompts" problem corrupts any naive
significance claim. This module exposes:

- `benjamini_hochberg(pvals, alpha)` — controls the False Discovery Rate.
- `probabilistic_sharpe(returns, sr_benchmark)` — Bailey & López de Prado's PSR.
- `deflated_sharpe(returns, n_trials, var_trial_sharpe)` — DSR, the multiple-trial-corrected PSR.

Sources
-------
- Benjamini & Hochberg, "Controlling the False Discovery Rate," JRSS-B 57(1), 1995, pp. 289-300.
- Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier," J. Risk 15(2), Winter 2012/13.
- Bailey & López de Prado, "The Deflated Sharpe Ratio," J. Portfolio Management 40(5), 2014.
"""
from __future__ import annotations

from abl.multipletest.bh import benjamini_hochberg
from abl.multipletest.bootstrap import BCaResult, bca_bootstrap_ci, bca_sharpe_ci
from abl.multipletest.dsr import deflated_sharpe, expected_max_sharpe_under_null
from abl.multipletest.hac import HACSharpeResult, auto_lag_truncation, hac_sharpe_ci, newey_west_eta
from abl.multipletest.psr import probabilistic_sharpe, sharpe_ratio
from abl.multipletest.spa import SPAResult, reality_check_spa

__all__ = [
    "benjamini_hochberg",
    "probabilistic_sharpe",
    "deflated_sharpe",
    "expected_max_sharpe_under_null",
    "sharpe_ratio",
    "hac_sharpe_ci",
    "HACSharpeResult",
    "newey_west_eta",
    "auto_lag_truncation",
    "reality_check_spa",
    "SPAResult",
    "bca_bootstrap_ci",
    "bca_sharpe_ci",
    "BCaResult",
]
