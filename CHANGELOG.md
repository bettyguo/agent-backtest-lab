# Changelog

All notable changes to `agent-backtest-lab` are tracked here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-05-14

First public release. Statistical-audit harness for LLM trading-agent frameworks.

### Added

- **Point-in-time data firewall** (`abl.data.firewall`, `abl.data.pit_view`): refuses any data access with `requested_date > as_of`; reconstructs adjusted OHLCV from raw bars + corporate actions filtered by `ex_date <= as_of`; writes an audit log of every access.
- **Synthetic GBM fixture** (`abl.data.fixture_synth`): deterministic, seeded, ground-truth-known dataset used by every fixture test and the offline quickstart.
- **Walk-forward engine** (`abl.backtest.engine`): drives a `StrategyAdapter` day by day across a `Universe` and `Window`, returning a `BacktestResult` with the per-day audit trail.
- **Purged K-fold CV with embargo** (`abl.backtest.cv`): López de Prado-style splits for label-overlap-aware cross-validation.
- **Transaction-cost models** (`abl.costs`): constant-bps cost (default 5 bps each side) and an Almgren-style square-root impact term (the 3/5-vs-1/2 caveat documented in the docstring).
- **Leakage detectors** (`abl.leakage`): corporate-action-proximity clustering, audit-log violation parser. Tested against a deliberately-leaky fixture strategy which it MUST catch (CI gate).
- **Multiple-testing module** (`abl.multipletest`): Benjamini-Hochberg FDR, Probabilistic Sharpe Ratio (PSR), Deflated Sharpe Ratio (DSR).
- **Calibration module** (`abl.calibration`): reliability diagrams, Expected Calibration Error (ECE), split conformal prediction with a rolling-window option for non-exchangeable time series.
- **Overfitting detection** (`abl.overfitting.cscv`): Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation. Tested against a deliberately-overfit fixture which it MUST flag (CI gate).
- **Three adapters** (`abl.adapters`): `CallableAdapter` (generic), `PlainStrategyAdapter` (used by built-in baselines), `TradingAgentsAdapter` (optional dependency on `TauricResearch/TradingAgents`).
- **Three baselines** (`abl.baselines`): buy-and-hold, naive momentum, random. Every scorecard shows these alongside the evaluated adapter.
- **Honest scorecard** (`abl.scorecard`): Markdown + JSON report with net-of-cost returns, confidence intervals, multiple-testing corrections, baseline comparison, and explicit leakage / overfitting / universe flags. The not-advice disclaimer is appended to every report and not removable.
- **CLI** (`abl`): `evaluate`, `scorecard`, `leakage-check`, `baselines`, `methods`, `version` commands.

### Not added (deliberately)

- No broker integration. No live data feed. No paper-trading bridge. No "buy/sell" language. No alpha claims about any strategy.

### References

- López de Prado, *Advances in Financial Machine Learning*, Wiley 2018 (Ch. 7).
- Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest Overfitting," *J. Computational Finance*, 2017.
- Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality," *J. Portfolio Management* 40(5), 2014.
- Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier," *J. Risk* 15(2), Winter 2012/13.
- Benjamini & Hochberg, "Controlling the False Discovery Rate," *JRSS-B* 57(1), 1995, pp. 289-300.
- Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, Springer 2005.
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," arXiv:2107.07511.
- Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks," ICML 2017.
- Almgren & Chriss, "Optimal Execution of Portfolio Transactions," *J. Risk* 2000.
- Almgren, Thum, Hauptmann, Li, "Direct Estimation of Equity Market Impact," *Risk* 2005.
