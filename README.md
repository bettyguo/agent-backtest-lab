<div align="center">

# agent-backtest-lab

### Before you trust an LLM trading agent, **audit it**.

A statistical-rigor harness for trading-agent frameworks. Look-ahead-leak detection · transaction-cost modeling · multiple-testing correction · calibration · reward-hacking detection.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Fixture tests](https://img.shields.io/badge/fixture%20tests-150%20passing-brightgreen.svg)](tests/)
[![PyPI-ready](https://img.shields.io/badge/pypi--ready-yes-brightgreen.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.6.0-informational.svg)](CHANGELOG.md)

### 🌐 **[Live demo & example scorecards →](https://bettyguo.github.io/agent-backtest-lab/)**

</div>

---

> ⚠️ **This is a research and evaluation tool. It is not financial, investment, or trading advice. It does not execute trades or connect to brokerages.** It exists to help researchers and practitioners rigorously measure how trading-agent frameworks actually perform — including, and especially, when they perform badly. Backtest results are not predictive of live performance. You are responsible for any decisions you make.

---

## The problem

A trading agent shows **+40% in a backtest**. Was the agent good — or did it quietly see tomorrow's prices through retroactively-adjusted data, get the best of 50 untracked prompt-tuning attempts, or earn returns no real trader could net of costs?

Recent work shows that when LLM trading agents are evaluated across decades and 100+ symbols with disciplined methodology, **"advantages deteriorate markedly"** — they're "overly conservative in bull markets, underperforming passive benchmarks" and "overly aggressive in bear markets, incurring heavy losses." The category needs scrutiny, not more agents.

`agent-backtest-lab` is the scrutiny.

> **How this differs from backtesting.py / vectorbt / zipline:** those are *execution engines* (PnL accumulators, order simulators). `agent-backtest-lab` is a *statistical-audit harness* — the leakage firewall, the multiple-testing correction, the calibration and overfitting diagnostics, the honest scorecard. It sits on top of any engine. The work it does is the work those tools deliberately leave to the user.

---

## See it in action

Two example scorecards rendered from the bundled synthetic fixture (no API keys, no network, runs offline):

|  | Strategy | Live HTML scorecard |
|---|---|---|
| 📊 | **Buy and hold** — the simplest baseline. Net of cost; rendered with the same scorecard pipeline a real agent goes through. | **[Open report →](https://bettyguo.github.io/agent-backtest-lab/examples/buy_and_hold/report.html)** |
| 📈 | **Naive momentum** — 5-day trailing return signal. The "is your agent doing anything beyond trend-following?" sanity check. | **[Open report →](https://bettyguo.github.io/agent-backtest-lab/examples/naive_momentum/report.html)** |

Every report carries net-of-cost returns with confidence intervals, multiple-testing corrections, per-ticker breakdown, drawdown stats, baseline comparison, and explicit leakage / overfitting / reward-hacking flags. The not-advice disclaimer is at the top and the bottom; neither is removable.

---

## 30-second quickstart

```bash
pip install agent-backtest-lab
abl evaluate buy_and_hold --window 2020-01-06..2024-12-31 --out out/
# open out/report.html in any browser
```

You get **`report.md`**, **`report.html`** (self-contained, with inline plots), **`report.json`**, **`equity.png`**, and **`drawdown.png`** — all net of cost, against three baselines, with the not-advice disclaimer pinned to every artifact.

### Evaluating your own strategy

```python
from datetime import date
from abl.adapters.callable_adapter import CallableAdapter
from abl.backtest.engine import WalkForwardEngine
from abl.baselines import buy_and_hold_adapter, naive_momentum_adapter, random_baseline_adapter
from abl.data.loaders import load_fixture
from abl.scorecard.report import build_scorecard, render_markdown
from abl.types import Decision, Universe, Window

def my_strategy(ticker, as_of, pit_data):
    bars = pit_data.bars(ticker, lookback_days=20)
    if bars.empty:
        return Decision(direction="FLAT")
    if bars["close"].iloc[-1] > bars["close"].mean():
        return Decision(direction="LONG", confidence=0.6)
    return Decision(direction="FLAT")

store = load_fixture()
universe = Universe(tickers=store.tickers(), name="example", survivorship_verified=True)
window = Window(start=date(2020, 1, 6), end=date(2024, 6, 28))
engine = WalkForwardEngine(store=store)

primary = engine.run(CallableAdapter(my_strategy, name="mine"), universe, window)
baselines = [engine.run(b(), universe, window) for b in
             (buy_and_hold_adapter, naive_momentum_adapter, random_baseline_adapter)]
sc = build_scorecard(primary=primary, baselines=baselines, n_trials_reported=1)
print(render_markdown(sc))
```

---

## What it does

Grouped by concern. Every method names its paper, its assumptions, and the fixture test that verifies it. See [`docs/METHODS.md`](docs/METHODS.md).

### 🔒 Data discipline — the leakage firewall

| Capability | Module | Reference |
|---|---|---|
| Hard leakage firewall (refuses `date > as_of`) | `abl.data.firewall` | — |
| Retroactive-adjustment defense (filter actions by `ex_date`) | `abl.data.corporate_actions` | — |
| yfinance raw-only loader (**refuses `Adj Close`**) | `abl.data.yfinance_loader` | — |
| Post-hoc leakage detectors | `abl.leakage.detector` | impossible-call clustering, audit-log violations |

### 📐 Statistical inference

| Capability | Module | Reference |
|---|---|---|
| Probabilistic Sharpe Ratio (PSR) | `abl.multipletest.psr` | Bailey & López de Prado 2012/13 |
| Deflated Sharpe Ratio (DSR) | `abl.multipletest.dsr` | Bailey & López de Prado 2014 |
| HAC / Newey-West Sharpe SE | `abl.multipletest.hac` | Lo 2002, Newey-West 1987/1994 |
| BCa bootstrap CIs (any statistic) | `abl.multipletest.bootstrap` | Efron 1987 |
| Benjamini-Hochberg FDR | `abl.multipletest.bh` | Benjamini & Hochberg 1995 |
| Benjamini-Yekutieli (arbitrary dependence) | `abl.multipletest.bh(method="by")` | Benjamini & Yekutieli 2001 |
| Bonferroni-Holm step-down FWER | `abl.multipletest.stepwise.bonferroni_holm` | Holm 1979 |
| Romano-Wolf stepwise (bootstrap-based FWER) | `abl.multipletest.stepwise.romano_wolf_stepwise` | Romano & Wolf 2005 |
| White's Reality Check / SPA | `abl.multipletest.spa` | White 2000, Politis-Romano 1994 |
| Sortino + Information Ratio | `abl.multipletest.risk_metrics` | Sortino-Price 1994 |

### 🎯 Overfitting & calibration

| Capability | Module | Reference |
|---|---|---|
| PBO via CSCV (cross-strategy) | `abl.overfitting.cscv` | Bailey, Borwein, López de Prado, Zhu 2017 |
| Reward-hacking detection (single-strategy) | `abl.leakage.reward_hacking` | IS/OOS Sharpe drop, drawdown widening, calibration divergence |
| Reliability diagrams + ECE | `abl.calibration.reliability` | Guo et al. ICML 2017 |
| Split conformal w/ rolling window | `abl.calibration.conformal` | Vovk-Gammerman-Shafer 2005; Angelopoulos-Bates 2021 |

### ⚙️ Backtest engine & reporting

| Capability | Module | Reference |
|---|---|---|
| Walk-forward + purged-CV with embargo | `abl.backtest.cv` | López de Prado 2018, Ch. 7 |
| Combinatorial Purged K-Fold | `abl.backtest.cv` | López de Prado 2018, Ch. 12 |
| Transaction costs (constant-bps + Almgren impact) | `abl.costs` | Almgren-Chriss 2000, Almgren et al. 2005 |
| Drawdown + Calmar metrics | `abl.scorecard.drawdown` | — |
| Per-ticker breakdown + cross-strategy correlation | `abl.scorecard.breakdown` | — |
| Equity / drawdown / reliability plotters (headless) | `abl.plots` | matplotlib Agg |
| Markdown + JSON + HTML scorecards | `abl.scorecard` | self-contained HTML with inline plots |

### 🧩 Framework integration

| Adapter | Module | Notes |
|---|---|---|
| TradingAgents (Tauric Research) | `abl.adapters.tradingagents` | optional dep; wraps `TradingAgentsGraph().propagate()` |
| FinGPT (AI4Finance-Foundation) | `abl.adapters.fingpt` | wraps sentiment-in-[-1, +1] callable; `|score|` → confidence |
| FinRobot (AI4Finance-Foundation) | `abl.adapters.finrobot` | wraps `(text, conf)` callable; defensive direction parser |
| Generic callable | `abl.adapters.callable_adapter` | any `(ticker, as_of, pit) → Decision` |
| Plain strategy (used by baselines) | `abl.adapters.plain_strategy` | thin wrapper around a function |

### 📋 Six baselines, always rendered alongside

`buy_and_hold` · `naive_momentum` · `random_baseline` · `equal_vol` · `naive_mean_reversion` · `dca`

---

## What it does NOT do — deliberately

| Anti-capability | Why |
|---|---|
| Place trades / connect to brokerages | Not the purpose. There are other tools for that. |
| Recommend "BUY" or "SELL" of any specific asset | The library audits decisions; it does not make them. |
| Claim alpha for any strategy | Backtests do not predict the future. We will not pretend otherwise. |
| Use Yahoo's retrospectively-adjusted `Adj Close` | That is the most common silent leakage in retail backtests. |
| Fabricate a confidence when the framework didn't emit one | Calibration reports "not evaluable." This is honest. |
| Emit gross-only return numbers | Every reported number is net of cost. No exceptions. |
| Charge you | Apache-2.0. Always free. |

---

## Integrating TauricResearch/TradingAgents

```python
from abl.adapters.tradingagents import TradingAgentsAdapter

adapter = TradingAgentsAdapter(
    ta_config={
        "llm_provider": "openai",
        "deep_think_llm": "gpt-4o-mini",
        "quick_think_llm": "gpt-4o-mini",
        "max_debate_rounds": 1,
    },
)
# Then pass `adapter` to WalkForwardEngine like any other adapter.
```

Requires the [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) framework installed and your own LLM API keys. See [`examples/02_evaluate_tradingagents.py`](examples/02_evaluate_tradingagents.py) for the full pattern. This integration is intentionally an optional dependency — the package never imports `tradingagents` at module load, and the adapter raises a clean `ImportError` with installation guidance if the framework is missing.

---

## Honest by design — the rules baked into the library

1. **Net of cost, always.** No gross-only numbers anywhere in any public function.
2. **Confidence intervals on every Sharpe** (IID Gaussian, HAC/Newey-West, or BCa bootstrap — your choice).
3. **Multiple-testing correction.** When you set `n_trials_reported > 1` (e.g. a 50-prompt scan), DSR is computed and shown. BH-FDR, BH-Yekutieli, Bonferroni-Holm, and Romano-Wolf are also available.
4. **Backtest-overfitting probability** for every parameter scan (CSCV / PBO).
5. **Reward-hacking flags** for single-strategy window overfitting (IS/OOS Sharpe drop, drawdown widening, calibration divergence).
6. **Buy-and-hold and naive baselines** always shown next to the evaluated adapter. No flag suppresses this.
7. **Survivorship and PIT caveats** flagged on every report.
8. **Reliability diagram + ECE** if your agent emits a confidence. If it doesn't, we say *"not evaluable"* — we never fabricate one.
9. **Split conformal sets** on directional calls, with a rolling window because returns are not exchangeable. We say so loudly.
10. **Leakage firewall** refuses any access to data with `date > as_of`. Hard guarantee for adapter-mediated access; best-effort detection for framework-internal channels.
11. **Disclaimer block** at the top of every emitted report; disclaimer line on every CLI invocation. Neither is removable.

---

## Companion to TradingAgents

This library is designed as a **companion** to LLM trading-agent frameworks — primarily [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). Their job is to *generate* decisions; ours is to *audit* them. We cite them respectfully; this is the rigor layer the category needs, not an attack on it.

---

## Roadmap

Track via GitHub issues. Per [`CONTRIBUTING.md`](CONTRIBUTING.md), every new statistical method must ship a known-answer fixture test.

- Real-data point-in-time corporate-action database (currently synthetic-only fixture; yfinance integration exists but vendor data is retrospectively adjusted).
- More framework adapters (FinMem, AgentQuant, custom).
- Pyodide build for in-browser interactive demos on the [live site](https://bettyguo.github.io/agent-backtest-lab/).
- HTML-embedded reliability diagrams + per-strategy comparison view.
- Survivorship-corrected universe loader (constrained on free-data budget).

---

## Resources

- **🌐 [Live site & example scorecards](https://bettyguo.github.io/agent-backtest-lab/)**
- 📑 [Methods doc](docs/METHODS.md) — every method's formal statement, assumptions, and citation
- 📜 [Full disclaimer text](docs/DISCLAIMERS.md)
- 🚀 [Launch plan](docs/LAUNCH.md)
- 📝 [Changelog](CHANGELOG.md)
- 🤝 [Contributing guide](CONTRIBUTING.md)

