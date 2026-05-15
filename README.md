# agent-backtest-lab

> Before you trust an LLM trading agent, audit it. A statistical-rigor harness — look-ahead-leak detection, transaction-cost modeling, multiple-testing correction, calibration, and reward-hacking detection — for trading-agent frameworks.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/ci-passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Fixture tests](https://img.shields.io/badge/fixture%20tests-63%20passing-brightgreen.svg)](tests/)
[![PyPI-ready](https://img.shields.io/badge/pypi--ready-yes-brightgreen.svg)](pyproject.toml)

> ⚠️ **This is a research and evaluation tool. It is not financial, investment, or trading advice. It does not execute trades or connect to brokerages.** It exists to help researchers and practitioners rigorously measure how trading-agent frameworks actually perform — including, and especially, when they perform badly. Backtest results are not predictive of live performance. You are responsible for any decisions you make.

---

## The problem

A trading agent shows +40% in a backtest. Was the agent good — or did it quietly see tomorrow's prices through retroactively-adjusted data, or get the best of 50 untracked prompt-tuning attempts, or earn returns no real trader could net of costs?

Independent recent work (Li et al. 2026, [arXiv:2505.07078](https://arxiv.org/abs/2505.07078)) shows that when LLM trading agents are evaluated across decades and 100+ symbols with disciplined methodology, "advantages deteriorate markedly" — they're "overly conservative in bull markets, underperforming passive benchmarks" and "overly aggressive in bear markets, incurring heavy losses." The category needs scrutiny, not more agents.

`agent-backtest-lab` is the scrutiny.

> **How this differs from backtesting.py / vectorbt / zipline:** those are *execution engines* (PnL accumulators, order simulators). `agent-backtest-lab` is a *statistical-audit harness* — the leakage firewall, the multiple-testing correction, the calibration and overfitting diagnostics, the honest scorecard. It can sit on top of any engine. The work it does is the work those tools deliberately leave to the user.

---

## 30-second quickstart

```bash
pip install agent-backtest-lab
abl evaluate buy_and_hold --window 2020-01-06..2024-12-31 --out out/
cat out/report.md
```

You'll get an honest scorecard against three baselines, net of cost, with the not-advice disclaimer at the top.

To evaluate your own strategy:

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

| Capability | Module | Source |
|---|---|---|
| Walk-forward + purged-CV with embargo | `abl.backtest` | López de Prado 2018, Ch. 7 |
| Hard leakage firewall (refuses date > as_of) | `abl.data.firewall` | — |
| Retroactive-adjustment defense (filter actions by ex_date) | `abl.data.corporate_actions` | — |
| Post-hoc leakage detectors | `abl.leakage` | Heuristic; see docstrings |
| Constant-bps + Almgren-style impact costs | `abl.costs` | Almgren-Chriss 2000, Almgren-Thum-Hauptmann-Li 2005 |
| Benjamini-Hochberg FDR | `abl.multipletest.bh` | Benjamini & Hochberg 1995 |
| Probabilistic Sharpe Ratio (PSR) | `abl.multipletest.psr` | Bailey & López de Prado 2012/13 |
| Deflated Sharpe Ratio (DSR) | `abl.multipletest.dsr` | Bailey & López de Prado 2014 |
| Probability of Backtest Overfitting (PBO) via CSCV | `abl.overfitting.cscv` | Bailey, Borwein, López de Prado, Zhu 2017 |
| Reliability diagrams + ECE | `abl.calibration.reliability` | Guo, Pleiss, Sun, Weinberger ICML 2017 |
| Split conformal + rolling window | `abl.calibration.conformal` | Vovk-Gammerman-Shafer 2005; Angelopoulos-Bates 2021 |
| Three baselines, always shown | `abl.baselines` | — |
| Markdown + JSON scorecard | `abl.scorecard` | — |
| CLI with non-removable disclaimer | `abl.cli` | — |

Every method names its paper, its assumptions, and the fixture test that verifies it. See [docs/METHODS.md](docs/METHODS.md).

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

Requires the [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) framework installed and your own LLM API keys. See `examples/02_evaluate_tradingagents.py` for the full pattern. This integration is intentionally an optional dependency — the package never imports `tradingagents` at module load, and the adapter raises a clean `ImportError` with installation guidance if the framework is missing.

---

## Honest by design — the rules baked into the library

1. **Net of cost, always.** No gross-only numbers anywhere in any public function.
2. **Confidence intervals on every Sharpe.**
3. **Multiple-testing correction.** When you set `n_trials_reported > 1` (e.g. a 50-prompt scan), the Deflated Sharpe Ratio is computed and shown.
4. **Backtest-overfitting probability** for every parameter scan (CSCV / PBO).
5. **Buy-and-hold and naive baselines** are always shown next to the evaluated adapter. There is no flag to suppress this.
6. **Survivorship and PIT caveats** flagged on every report.
7. **Reliability diagram + ECE** if your agent emits a confidence. If it doesn't, we say "not evaluable" — we never fabricate one.
8. **Split conformal sets** on directional calls, with a rolling window because returns are not exchangeable. We say so loudly.
9. **Leakage firewall** refuses any access to data with date > as_of. Hard guarantee for adapter-mediated access; best-effort detection for framework-internal channels.
10. **Disclaimer block** at the top of every emitted report; disclaimer line on every CLI invocation. Neither is removable.

---

## Companion to TradingAgents

This library is designed as a **companion** to LLM trading-agent frameworks — primarily [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). Their job is to *generate* decisions; ours is to *audit* them. We cite them respectfully; this is the rigor layer the category needs, not an attack on it.

---

## Roadmap (open extensions)

- Real-data point-in-time corporate-action database (currently synthetic-only fixture).
- BH-Yekutieli FDR variant for arbitrary dependence.
- Combinatorial-purged-K-fold (full Lopez de Prado Ch. 12) for label-overlap heavy settings.
- More framework adapters (FinGPT, FinRobot, FinMem).

Track via GitHub issues. Per `CONTRIBUTING.md`, every new statistical method must ship a known-answer fixture test.

---

## Attribution

Built by Betty Guo (Dongxin Guo / 郭东欣), final-year PhD candidate in Computer Science, University of Hong Kong, advised by Prof. Siu-Ming Yiu. ORCID: [0009-0000-2388-1072](https://orcid.org/0009-0000-2388-1072). Apache-2.0.

If this tool helps your research, a citation is appreciated. A BibTeX entry will be added when the v0.1.0 release is tagged.

---

## Star history

[![Star History Chart](https://api.star-history.com/svg?repos=bettyguo/agent-backtest-lab&type=Date)](https://star-history.com/#bettyguo/agent-backtest-lab&Date)
