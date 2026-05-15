# Changelog

All notable changes to `agent-backtest-lab` are tracked here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] — 2026-05-15

"Better CIs, more baselines." 134 fixture tests pass.

### Added

- **BCa bootstrap CIs** (`abl.multipletest.bootstrap.bca_bootstrap_ci`,
  `bca_sharpe_ci`). Bias-corrected and accelerated bootstrap per Efron (1987).
  The right CI when the statistic's sampling distribution is NOT well-approximated
  by a Gaussian — most useful for Sharpe with fat-tailed returns, Sortino, Calmar.
  Includes the jackknife acceleration estimate; degraded gracefully on degenerate
  input.
- **Three new baselines** in `abl.baselines`:
  - `equal_vol_adapter` — inverse-volatility weighted LONG. The "risk parity for
    one factor" minimum bar.
  - `naive_mean_reversion_adapter` — mirror of `naive_momentum`. Useful so a
    "this agent is just trend-following" hypothesis is falsifiable.
  - `dca_adapter` — dollar-cost averaging ramp. Captures the "I'd have just
    averaged in" retail counterfactual that benchmarks beginning-of-trough windows.

### Added (tests)

- `tests/test_bca_bootstrap.py` (6 tests) — BCa for the mean achieves near-95%
  empirical coverage on Gaussian samples across 200 reps; BCa Sharpe CI contains
  the point estimate; degenerate input does not crash; acceleration / z_0 are finite.
- `tests/test_more_baselines.py` (4 tests) — equal-vol and naive-mean-reversion
  run end-to-end on the bundled fixture; DCA is FLAT during the ramp then LONG;
  naive-mean-reversion and naive-momentum produce mirror-image directions.

## [0.4.0] — 2026-05-15

"More adapters, reward-hacking detection, HTML report." 124 fixture tests pass.

### Added

- **FinGPT adapter** (`abl.adapters.fingpt.FinGPTAdapter`). Wraps a user-supplied
  sentiment-in-[-1, +1] callable. Honest about what FinGPT emits: a sentiment, not a
  calibrated probability. `|score|` becomes the confidence — feeds cleanly into ECE
  and conformal analysis. The `fingpt` framework itself stays optional; users supply
  the callable.
- **FinRobot adapter** (`abl.adapters.finrobot.FinRobotAdapter`). Wraps a
  `(decision_text, confidence)` callable. Reuses the defensive TradingAgents
  direction parser; safely drops out-of-range confidences.
- **Reward-hacking detection** (`abl.leakage.reward_hacking.detect_reward_hacking`).
  Three single-strategy heuristics for window-overfitting (complementary to
  cross-strategy PBO):
  - `SHARPE_DROP_IS_OOS` — Sharpe collapses from the first ~70% of the window to the
    final ~30%. Critical when the drop is large and OOS Sharpe goes negative.
  - `DRAWDOWN_WIDENS_OOS` — max drawdown materially worse on holdout.
  - `CALIBRATION_DIVERGES_OOS` — ECE rises sharply OOS when the agent emits a confidence.
  Auto-included in every scorecard; surfaces in both Markdown and HTML reports.
- **HTML scorecard renderer** (`abl.scorecard.html_render.render_html`). Self-contained
  single-file HTML with inline CSS, color-coded banner, base64-embedded plots when
  available. Same not-advice disclaimer block at the top, attribution at the bottom.
  `abl evaluate` now writes `report.md`, `report.json`, AND `report.html` by default.

### Changed

- `Scorecard.reward_hacking_flags` is a new field. Backward-compatible addition.
- Banner logic in `render_markdown` and `render_html` now flips red on a critical
  reward-hacking flag, not just on leakage / critical overfitting.

### Added (tests)

- `tests/test_new_adapters.py` (8 tests) — FinGPT positive/negative/below-threshold,
  NaN handling, invalid threshold; FinRobot parses text, drops out-of-range confidence,
  handles None decisions.
- `tests/test_reward_hacking.py` (5 tests) — no flag on consistent series, critical
  flag on Sharpe collapse, drawdown-widening flag, too-short-series returns empty,
  calibration-divergence flag.
- `tests/test_html_render.py` (4 tests) — disclaimer + attribution present, baselines
  + per-ticker breakdown rendered, plots embed as base64 data URIs, no gross-return
  leakage in the HTML.

## [0.3.0] — 2026-05-15

"Real data, visuals, breakdown." 107 fixture tests pass.

### Added

- **yfinance loader** (`abl.data.yfinance_loader.load_yfinance`). Pulls raw OHLCV plus
  the dividends/splits actions table from Yahoo Finance — **and REFUSES to read
  `Adj Close`** because Yahoo's adjusted series is retroactively back-adjusted (the
  single most common silent leakage in retail backtests). The leakage firewall + PITView
  reconstruct the adjusted series at as_of using only known corporate actions. Optional
  dependency: `pip install "agent-backtest-lab[yfinance]"`. Tests are network-free —
  yfinance is mocked.
- **Matplotlib plotters** (`abl.plots`). Three renderers, headless (Agg backend), each
  stamped with the not-advice disclaimer at the bottom:
  - `plot_equity_curve(...)` — primary strategy vs baselines, net of cost.
  - `plot_drawdown(...)` — drawdown curve with max-DD trough annotated.
  - `plot_reliability_diagram(...)` — confidence bins vs empirical accuracy with the
    perfect-calibration diagonal.
- **Per-ticker breakdown** (`abl.scorecard.breakdown.per_ticker_breakdown`). Per-ticker
  hit rate, decision count, mean return contribution, and ticker-level annualized
  Sharpe. Auto-included in every scorecard so users see whether the edge is robust or
  concentrates in one ticker.
- **Cross-strategy correlation** (`cross_strategy_correlation` + `effective_n_trials`).
  Pearson correlation matrix of N strategies' return series, plus the participation-ratio
  heuristic for "effective N" — a sanity check before using `n_trials_reported = N` in DSR.
  High correlation → effective N << N → DSR under-corrects unless you pass the smaller value.

### Changed

- CLI `abl evaluate` now renders `equity.png` and `drawdown.png` alongside the Markdown
  and JSON reports by default. `--no-plots` disables.
- Scorecard Markdown now includes a "Per-ticker breakdown" section between the drawdown
  table and the baselines table.
- `Scorecard.ticker_breakdown` is a new field on the dataclass.

### Added (tests)

- `tests/test_yfinance_loader.py` (7 tests) — `Adj Close` IS REFUSED (load-bearing
  assertion), corporate-actions extraction handles dividends + splits + zero-filtering,
  clean ImportError when yfinance is missing, end-to-end with a stub.
- `tests/test_plots.py` (4 tests) — equity / drawdown / reliability PNGs are written.
- `tests/test_breakdown.py` (8 tests) — hit rates correct, FLAT-only tickers give NaN,
  identity correlation = 1.0, large-N independents → ~0, effective N = N on identity,
  effective N → 1 on all-ones, intermediate block correlation gives ~2.

## [0.2.0] — 2026-05-15

"More rigor." Five additions, each motivated by the Phase 6 hostile review or by user-feedback paths well-trodden in the academic literature. 87 fixture tests pass.

### Added

- **HAC / Newey-West Sharpe SE** (`abl.multipletest.hac.hac_sharpe_ci`). Lo (2002) variance estimator with Bartlett kernel and Newey-West (1987, 1994) automatic bandwidth `q ≈ ⌊4 · (T/100)^(2/9)⌋`. Resolves the Phase 6 review item A1 — the IID-Gaussian CI was overconfident on autocorrelated series. The scorecard now defaults to HAC and exposes the correction factor `η` and lag truncation `q` in the rendered Markdown. `sharpe_ci_method="iid"` preserves the previous behavior.
- **BH-Yekutieli FDR variant** (`abl.multipletest.bh.benjamini_hochberg(..., method="by")`). Benjamini & Yekutieli (2001). Divides the BH threshold by the harmonic sum `c(m) = Σ 1/k`; controls FDR under arbitrary dependence. The right choice when the dependence structure of the p-values is unknown — common in trading-strategy backtesting (correlated alpha signals).
- **Combinatorial Purged K-Fold** (`abl.backtest.cv.combinatorial_purged_kfold_splits`). López de Prado (2018), Chapter 12. Returns `C(n_groups, n_test_groups)` splits and exposes `n_recoverable_paths(...)` so users can size the number of independent OOS realizations they get.
- **Drawdown metrics** (`abl.scorecard.drawdown.drawdown_stats`). Max drawdown, longest underwater stretch (in trading days), and the Calmar ratio (annualized return / |max drawdown|). Computed from the net-of-cost equity curve and rendered as its own table in every scorecard.
- **Reality Check / SPA test** (`abl.multipletest.spa.reality_check_spa`). White (2000) Reality Check via the Politis-Romano (1994) stationary block bootstrap. Tests "does the best of N strategies beat a specified benchmark, accounting for the N-trial search?" Complements DSR, which tests against `any positive Sharpe`. Hansen (2005) studentization-recentering is not implemented — filed as an open extension.

### Changed

- `abl.scorecard.Scorecard` gained `sharpe_ci_method`, `sharpe_hac_eta`, `sharpe_hac_lags`, `max_drawdown`, `longest_underwater_days`, `calmar_ratio` fields. Backward-compatible additions; downstream consumers reading by attribute name continue to work.
- `build_scorecard(..., sharpe_ci_method="hac" | "iid")`. Default is `"hac"`.

### Added (tests)

- `tests/test_hac_sharpe.py` (7 tests) — η reduces to 1 with zero autocorr, widens CI on AR(1), reduces to IID at q=0, matches the Bartlett-weighted formula on hand-computed ρ.
- `tests/test_bh_yekutieli.py` (4 tests) — BY is at least as conservative as BH, textbook example, controls FDR under correlated p-values across 500 reps.
- `tests/test_combinatorial_cv.py` (5 tests) — split count = C(n, k), train/test disjoint, purge respected, recoverable-paths formula.
- `tests/test_drawdown.py` (5 tests) — no drawdown when monotone positive, hand-computed three-day case, trough index correct, Calmar finite when DD exists.
- `tests/test_spa.py` (3 tests) — high p-value on pure noise, low p-value on a clear winner, output-shape sanity.

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
