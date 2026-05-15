# PLANNING/01_design.md — Phase 1 (DESIGN)

**Project:** `agent-backtest-lab` (importable as `abl`)
**Author:** Betty Guo (Dongxin Guo / 郭东欣), University of Hong Kong, advised by Prof. Siu-Ming Yiu
**Status:** design frozen pre-implementation

This document defines the architecture, contracts, test plan, README wireframe, and CLI surface. Every concrete decision below carries forward to Phase 2 implementation.

---

## 1. Package layout (final)

```
agent-backtest-lab/
  README.md
  LICENSE                       # Apache-2.0
  CONTRIBUTING.md
  pyproject.toml
  CHANGELOG.md

  abl/
    __init__.py                 # version, public surface, attribution
    types.py                    # Decision, Universe, Window, BacktestResult, dataclasses
    config.py                   # global defaults (cost in bps, annualization, seed)

    adapters/
      __init__.py
      base.py                   # StrategyAdapter protocol
      callable_adapter.py       # CallableAdapter
      plain_strategy.py         # PlainStrategyAdapter (used by baselines)
      tradingagents.py          # TradingAgentsAdapter — optional dependency

    data/
      __init__.py
      pit_view.py               # PITView (the as-of restricted view)
      firewall.py               # the leakage firewall + audit log
      loaders.py                # bundled-fixture loader, yfinance loader
      fixture_synth.py          # synthetic GBM fixture generator
      corporate_actions.py      # split/dividend application by ex-date

    backtest/
      __init__.py
      engine.py                 # WalkForwardEngine, executes adapter day by day
      cv.py                     # purged k-fold CV with embargo
      pnl.py                    # PnL accumulation (net-of-cost)

    costs/
      __init__.py
      models.py                 # ConstantBpsCost, AlmgrenImpactCost

    leakage/
      __init__.py
      detector.py               # post-hoc detectors (impossible-call, corp-action proximity)
      audit.py                  # parses the firewall audit log

    multipletest/
      __init__.py
      bh.py                     # Benjamini-Hochberg FDR
      psr.py                    # Probabilistic Sharpe Ratio
      dsr.py                    # Deflated Sharpe Ratio

    calibration/
      __init__.py
      reliability.py            # reliability diagram + ECE
      conformal.py              # split conformal with rolling window

    overfitting/
      __init__.py
      cscv.py                   # CSCV / Probability of Backtest Overfitting

    scorecard/
      __init__.py
      report.py                 # Markdown + JSON report builder
      flags.py                  # standardized flag types (LEAKAGE, OVERFITTING, ...)

    baselines/
      __init__.py
      buy_and_hold.py
      naive_momentum.py
      random_baseline.py

    cli/
      __init__.py
      main.py                   # click-based `abl` CLI

  tests/
    __init__.py
    conftest.py
    test_data_firewall.py
    test_leakage_fixture.py     # the CI gate: catches the deliberately-leaky strategy
    test_walkforward.py
    test_costs.py
    test_bh_fdr.py
    test_dsr_psr.py
    test_cscv_pbo.py
    test_overfitting_fixture.py # the CI gate: flags the deliberately-overfit strategy
    test_reliability_ece.py
    test_conformal.py
    test_scorecard.py
    test_adapters.py
    test_cli.py
    fixtures/
      synth_clean.parquet
      synth_leaky_strategy.py
      synth_overfit_strategy.py

  examples/
    01_evaluate_plain_strategy.py
    02_evaluate_tradingagents.py  # documented, not CI-run

  docs/
    LAUNCH.md
    PROFILE_SNIPPET.md
    METHODS.md                  # one-page summary of every statistical method + source
    DISCLAIMERS.md              # full disclaimer text, single source of truth

  assets/
    RECORD_DEMO.md              # exact vhs/asciinema script
    scorecard_demo.gif          # produced by RECORD_DEMO.md (manual step)

  .github/
    workflows/
      ci.yml                    # lint + tests on 3.10, 3.11, 3.12
```

Every source file targets < 300 lines (hard ceiling 500). The reason is reviewability: this library's job is to be trusted, and trust scales with how easily a reader can verify each module in isolation.

---

## 2. Adapter contract

```python
# abl/types.py
from dataclasses import dataclass
from typing import Literal, Protocol
from datetime import date

Direction = Literal["LONG", "SHORT", "FLAT"]

@dataclass(frozen=True)
class Decision:
    direction: Direction
    confidence: float | None   # in [0,1] if provided; None if framework doesn't emit one
    raw: dict | None = None    # opaque framework-native output

class StrategyAdapter(Protocol):
    name: str
    def predict(self, ticker: str, as_of: date, pit_data: "PITView") -> Decision: ...
```

**Three concrete adapters:**

- `CallableAdapter(fn, name)` — wraps any `Callable[[str, date, PITView], Decision]`.
- `PlainStrategyAdapter(strategy)` — used by the three baselines; same interface, no special-casing.
- `TradingAgentsAdapter(ta_config=None, confidence_parser=None)` — optional dependency. Calls `TradingAgentsGraph(debug=False, config=cfg).propagate(ticker, as_of.isoformat())`, parses the returned `decision` into `Decision`. If `tradingagents` is not installed, raises `ImportError` with a one-line install hint at construction time, not at module import time.

**Confidence handling — three honest cases:**

1. Framework emits a probability → use it.
2. Framework emits a categorical only → `confidence = None`; the calibration module reports *"not evaluable — confidence not provided"*. We do not fabricate a confidence.
3. Framework emits text containing a confidence (TradingAgents-style) → `confidence_parser` callable extracts it; if extraction fails, fall back to case 2.

**Data access discipline.** The adapter receives a `pit_data: PITView`. The protocol is *advisory* — Python cannot prevent a framework from making its own network calls. We document this honestly. The firewall is therefore (a) a **hard guarantee** for adapter-mediated data, and (b) a **best-effort detector** via the post-hoc leakage module for framework-internal channels.

---

## 3. Leakage firewall contract

**The point-in-time guarantee:** when the engine calls `adapter.predict(ticker, as_of, pit_data)`, the `pit_data` object refuses to return any datum with `date > as_of` or any corporate action with `ex_date > as_of`.

```python
# abl/data/pit_view.py
class PITView:
    """An as-of-restricted view of the data store. Hard-refuses future data."""
    def __init__(self, store: DataStore, as_of: date, firewall: Firewall): ...
    def bars(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        # Returns adjusted OHLCV up to and including as_of, with adjustments
        # reconstructed using only corporate actions with ex_date <= as_of.
        ...
    def corporate_actions(self, ticker: str) -> pd.DataFrame: ...
    def as_of(self) -> date: ...
```

```python
# abl/data/firewall.py
class Firewall:
    """Mediates every PIT data access and writes an audit log entry per access."""
    def check(self, ticker: str, requested_date: date, as_of: date) -> None: ...
    def log(self, event: AccessEvent) -> None: ...
    def violations(self) -> list[AccessEvent]: ...
```

**Post-hoc leakage detectors** (run after the backtest, never during):

- **Impossible-call clustering near corporate actions.** Score: count of correct directional calls within `[-2, 0]` trading days of a known split/dividend ex-date, normalized to base rate. A score significantly above 1 is flagged.
- **Same-day reaction implausibility.** A directional call that perfectly predicts a same-day return distribution shift suggests the call was conditioned on close-of-day data when only open-of-day should have been visible (a common subtle bug).
- **Audit-log anomalies.** Any firewall violation events (should always be zero in clean runs).

The leakage module emits a `LeakageReport` (list of flags + severity + the supporting statistic) which becomes part of the scorecard. A non-empty report does NOT crash the run — it shows up as a red banner on the scorecard.

**Gate test:** `tests/test_leakage_fixture.py` ships a `synth_leaky_strategy.py` that *deliberately* reads the next-day close (via a synthetic data path that simulates a backdoor — used only in tests, never in product code). The test asserts the leakage detector flags it. If the test ever passes silently (i.e. the detector misses the leak) the build fails. This is the most important test in the repository.

---

## 4. Statistical-method contract

Every method exposes (a) a precise function signature, (b) a docstring with the formal statement and assumptions, (c) a citation to its source, and (d) a known-answer fixture test.

| Method | Function | Fixture test |
|---|---|---|
| Walk-forward + purged CV | `abl.backtest.cv.purged_kfold_splits(n, k, embargo, label_window)` | Construct a dataset with known label-overlap structure; assert no train index falls within `[test_start − label_window, test_end + embargo]`. |
| Constant-bps cost | `abl.costs.models.ConstantBpsCost(bps).apply(pnl_series, trades)` | Synthetic 1-trade-per-day series, hand-computed net P&L for 3 days; assert exact match. |
| Almgren impact | `abl.costs.models.AlmgrenImpactCost(eta, beta).apply(...)` | Plug `eta=1, beta=0.5, sigma=0.01, Q/ADV=0.04` → expected `Δp/p = 0.01 · 0.2 = 2e-3`; assert. |
| BH FDR | `abl.multipletest.bh.benjamini_hochberg(pvals, alpha)` | Monte Carlo: 1000 reps of 100 p-vals (20 from `H_1`, 80 uniform); assert empirical FDR ≤ α + 2σ. |
| Deflated Sharpe | `abl.multipletest.dsr.deflated_sharpe(returns, n_trials, var_trial_sharpe)` | Plug returns with `T=252`, `ŜR=2`, `n=100`, `V[SR]=1`, zero skew/kurtosis → reproduce DSR from the Bailey-López de Prado paper Table I (or the analytic closed-form for the simplified inputs). |
| Probabilistic Sharpe | `abl.multipletest.psr.probabilistic_sharpe(returns, sr_benchmark)` | Plug `ŜR=2, T=252, SR*=0, zero skew/kurtosis` → DSR-with-N=1 reduces to PSR; assert numerical equivalence with an independent direct formula. |
| CSCV / PBO | `abl.overfitting.cscv.probability_of_backtest_overfitting(returns_matrix, S)` | Two fixtures: (i) `N=20` IID-noise strategies → PBO ≈ 0.5; (ii) one strategy with persistent OOS edge dominating 19 noise strategies → PBO ≈ 0. |
| Reliability + ECE | `abl.calibration.reliability.expected_calibration_error(probs, labels, n_bins)` | Perfectly-calibrated synthetic (`probs ~ Uniform`, `labels ~ Bernoulli(probs)`) → ECE → 0 as `N → ∞`; assert ECE < 0.02 at N=20000. |
| Split conformal | `abl.calibration.conformal.split_conformal_set(scores_cal, score_test, alpha)` | Exchangeable synthetic → empirical coverage ≥ 1−α across 1000 reps. |
| Leakage detector | `abl.leakage.detector.detect(audit_log, predictions, returns, corporate_actions)` | The leaky-fixture strategy MUST be flagged; the clean-fixture strategy MUST NOT be flagged. |
| Overfitting detector | (same as CSCV) | The deliberately-overfit fixture (a 50-strategy hyperparameter scan over noise) MUST report PBO > 0.5. |

**Every fixture test is a CI gate.** Phase 3 ends only when all of these pass on 3.10 / 3.11 / 3.12.

---

## 5. Scorecard contract

The scorecard is the *honest report*. It is the only thing most users will read, so it carries the framing.

```python
@dataclass
class Scorecard:
    adapter_name: str
    universe: Universe
    window: Window
    cost_model: str
    annualization: int

    # Per-adapter results, AFTER costs, with CIs.
    net_return_total: float
    net_return_annualized: float
    sharpe_observed: float
    sharpe_ci_95: tuple[float, float]
    psr: float
    dsr: float            # if N>1 trials reported by user
    pbo: float | None     # if a strategy matrix is supplied

    # Baseline comparison — always present.
    baseline_results: dict[str, BaselineRow]   # buy_and_hold, naive_momentum, random

    # Calibration — only if confidences were emitted.
    ece: float | None
    conformal_coverage: float | None

    # Flags.
    leakage_flags: list[LeakageFlag]
    overfitting_flags: list[OverfittingFlag]
    universe_flags: list[UniverseFlag]   # survivorship, etc.

    # The honest disclaimer line — always present, never removable.
    disclaimer: str = DISCLAIMER_LINE
```

Rendered as:

- `report.md` — Markdown with a clear top banner: green if zero flags AND beats buy-and-hold net of cost AND DSR > 0; **amber** if flags but no leakage; **red** if any leakage flag. Tables for net return / Sharpe / PSR / DSR / PBO with CIs. Side-by-side baseline comparison. The disclaimer line at the bottom.
- `report.json` — machine-readable mirror; same fields.

**Hard rule:** the report **always** prints net-of-cost numbers. Gross-only numbers are not emitted by any public function. (Gross PnL is computed internally and can be inspected via `BacktestResult.internals`, but never rendered to the report.)

Every report contains, verbatim near the top:

> *This report is produced by `agent-backtest-lab`, an evaluation harness for trading-agent frameworks. It is not financial, investment, or trading advice. It does not execute trades. Backtest results are not predictive of live performance. Treat any positive result with skepticism proportional to the number of configurations tried.*

---

## 6. CLI contract

```
abl evaluate <adapter-spec> --universe <file> --window <start>..<end>
    [--cost-bps 5] [--impact none|almgren] [--annualization 252]
    [--n-trials 1] [--out scorecard/]
abl scorecard <run-dir>                # re-render an existing run
abl leakage-check <run-dir>            # rerun post-hoc leakage detectors
abl baselines --universe <file> --window <start>..<end>
abl version
abl methods                            # prints docs/METHODS.md
```

`<adapter-spec>` is either:
- `module.path:CallableName` (resolved to a `CallableAdapter`), or
- one of the built-in baselines (`buy_and_hold`, `naive_momentum`, `random`), or
- `tradingagents:default` (requires `tradingagents` installed; prints a clear error otherwise).

Every CLI command prints, on first line:

```
agent-backtest-lab — not financial advice, not a trading system. See `abl methods` for the math.
```

This is enforced in `abl/cli/main.py`. The banner is *not* removable via a flag — a `--quiet` flag exists, but the disclaimer is the *one* thing `--quiet` does not silence.

---

## 7. README wireframe

```
# agent-backtest-lab
> Before you trust an LLM trading agent, audit it. A rigorous evaluation harness — look-ahead-leak detection, transaction-cost modeling, multiple-testing correction, calibration, and reward-hacking detection — for trading-agent frameworks.

[badges: license=Apache-2.0 | CI passing | Python 3.10|3.11|3.12 | fixture tests passing | PyPI-ready]

> ⚠️  This is a research and evaluation tool. It is not financial, investment, or trading advice. It does not execute trades or connect to brokerages. It exists to help researchers and practitioners rigorously measure how trading-agent frameworks actually perform — including, and especially, when they perform badly. Backtest results are not predictive of live performance. You are responsible for any decisions you make.

## The problem
A trading agent shows +40% in a backtest. Was the agent good — or did it quietly see tomorrow's prices through retroactively-adjusted data, or get the best of 50 untracked prompt-tuning attempts, or earn returns no real trader could net of costs? This is the question `agent-backtest-lab` answers.

[ASCIINEMA GIF: `abl evaluate` on an overfit fixture; scorecard banner flips red; leakage flag prints.]

## 30-second quickstart
```bash
pip install agent-backtest-lab
abl evaluate buy_and_hold --universe abl:fixtures/spy_only.csv --window 2020-01-01..2024-12-31 --out out/
cat out/report.md
```

## What it does (and what it doesn't)
[capability table: Walk-forward / Leakage firewall / Costs / FDR / DSR / PSR / PBO / Calibration / Conformal / Honest scorecard]
[NOT-A-TRADING-SYSTEM table: no execution / no broker / no buy-or-sell recommendation / no alpha claims]

## Integrating TradingAgents
[exact ~15-line code example wrapping TradingAgentsGraph().propagate()]
[note: requires `tradingagents` and your own API keys; illustrative, not CI-tested.]

## Honest by design
- Net of cost, always.
- Confidence intervals on every Sharpe.
- Multiple-testing correction (BH-FDR, DSR) whenever you report best-of-N.
- Backtest-overfitting probability (PBO) for every parameter scan.
- Reliability diagram + ECE if your agent emits a confidence.
- Split conformal sets on directional calls; rolling window because returns are not exchangeable. We say so.
- Buy-and-hold and naive baselines printed alongside every result. No exceptions.
- Survivorship and PIT caveats flagged on every report.

## Method docs
See [docs/METHODS.md](docs/METHODS.md). Every method names its paper, its assumptions, and the fixture test that verifies it.

## Attribution
Built by Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong, advised by Prof. Siu-Ming Yiu. ORCID: 0009-0000-2388-1072. Apache-2.0.

## Companion to TradingAgents
This library is designed as a companion to LLM trading-agent frameworks — `TauricResearch/TradingAgents` first. Their job is to *generate* decisions; ours is to *audit* them. Cite respectfully; never as an attack.

## Star history
[shields.io / star-history.com embed]
```

---

## 8. Test plan

**Phase 2 gates (must pass before checkpoint):**
- Firewall refuses any future-dated access (unit).
- The deliberately-leaky fixture is caught by the leakage detector.
- Walk-forward engine produces deterministic, seed-stable results.
- Constant-bps cost matches hand-computed expected P&L.
- Three baselines run end-to-end on the bundled fixture.

**Phase 3 gates (must pass before checkpoint):**
- BH FDR controls empirical FDR ≤ α + 2σ across 1000 Monte Carlo reps.
- DSR reproduces a published example (or, in the simplified zero-skew zero-kurtosis case, matches an independent closed-form).
- PSR equals DSR-with-N=1 numerically.
- CSCV/PBO returns ≈0.5 on IID noise and ≈0 when one strategy persistently dominates.
- Split conformal returns ≥ 1−α empirical coverage on exchangeable synthetic.
- ECE → 0 on perfectly-calibrated synthetic.
- The deliberately-overfit fixture is flagged by PBO > 0.5.

**Phase 4 gates:**
- All three adapters round-trip a synthetic strategy through the engine.
- The CLI emits a valid scorecard JSON validated against a schema.
- The not-advice disclaimer appears in every emitted Markdown and on every CLI invocation.
- The `tradingagents` adapter raises a clean `ImportError` when the framework is not installed.

**Cross-cutting:**
- `ruff` lint clean.
- `pytest` green on Python 3.10 / 3.11 / 3.12 in CI.
- Every fixture test produces a deterministic numeric result that is asserted against a tolerance.
- No test depends on network access (all data is bundled or synthetic).

---

## 9. Disclaimer text — single source of truth

Stored in `docs/DISCLAIMERS.md`. Re-imported by `abl.config.DISCLAIMER_LINE` and `abl.config.DISCLAIMER_BLOCK`. Any new emitted artifact MUST include the line.

Block (top of README, top of `abl methods`, top of every report):

> This project is a research and evaluation tool. It is not financial, investment, or trading advice. It does not execute trades or connect to brokerages. It exists to help researchers and practitioners rigorously measure how trading-agent frameworks actually perform — including, and especially, when they perform badly. Backtest results are not predictive of live performance. You are responsible for any decisions you make.

Line (every CLI invocation and every scorecard footer):

> agent-backtest-lab is a research tool. Not financial advice. Not a trading system. Backtests don't predict the future.

---

## 10. Carry-forward to Phase 2

Implementation order:
1. `pyproject.toml`, `LICENSE`, `README.md` stub, `CHANGELOG.md`, `CONTRIBUTING.md`, CI workflow.
2. `abl/types.py`, `abl/config.py`, `docs/DISCLAIMERS.md`.
3. `abl/data/fixture_synth.py` (synthetic GBM with known params), bundled parquet.
4. `abl/data/pit_view.py`, `abl/data/firewall.py`, `abl/data/corporate_actions.py`, `abl/data/loaders.py`.
5. `tests/test_data_firewall.py`, `tests/test_leakage_fixture.py` (with `synth_leaky_strategy.py`).
6. `abl/costs/models.py`, `tests/test_costs.py`.
7. `abl/backtest/engine.py`, `abl/backtest/cv.py`, `abl/backtest/pnl.py`, `tests/test_walkforward.py`.
8. `abl/baselines/*`, `abl/adapters/plain_strategy.py`.

Phase 2 checkpoint requires: data firewall test + leaky-fixture test green, walk-forward + cost + baselines green, CI green.

Proceeding to Phase 2.
