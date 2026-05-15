# PLANNING/00_think.md — Phase 0 (THINK)

**Project:** `agent-backtest-lab`
**Author:** Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong (advised by Prof. Siu-Ming Yiu)
**ORCID:** 0009-0000-2388-1072 · **GitHub:** `bettyguo` · **License:** Apache-2.0
**Date:** 2026-05-14

This document is the research-and-decision log produced before any code is written. Every citation has been web-verified via a research subagent (see CHECKPOINT 0 transcript). Items I could not verify exactly are tagged `[verify]`; nothing is invented.

---

## 1. Target study — what `TauricResearch/TradingAgents` actually is, and the gap

`TauricResearch/TradingAgents` (https://github.com/TauricResearch/TradingAgents) is the dominant LLM-trading-agent framework: roughly **75.5k stars**, Apache-2.0, built as a LangGraph multi-agent system whose public entry point is:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
_, decision = ta.propagate("NVDA", "2026-01-15")  # 2-tuple; decision shape undocumented [verify]
```

A separate CLI (`python -m cli.main`) exists. Configuration includes `llm_provider`, `deep_think_llm`, `quick_think_llm`, `max_debate_rounds`. A reflection/memory loop writes to `~/.tradingagents/memory/trading_memory.md` and, on subsequent runs, fetches realised return (raw and alpha vs SPY) to feed a one-paragraph reflection.

**What the README does NOT contain — verified by reading it:**

- No section discussing **transaction costs, slippage, or market impact**.
- No section discussing **multiple-testing correction** for prompt/config search.
- No section discussing **look-ahead leakage** or **point-in-time data integrity**.
- No **calibrated confidence** in the returned `decision`; it is a recommendation, not a probability.
- No **buy-and-hold / passive benchmark comparison** in any reported result. The SPY reference appears only as feedback to the LLM's memory, not as a reported result.
- **No `backtest/` or `evaluation/` directory** at the top level of the repo. (Confirmed against the repo's directory listing.)

This is exactly the empty niche. `TradingAgents` is an **agent system**, not an evaluated strategy. The framework provides a decision-at-a-date callable; everything that turns repeated decisions into an honest performance estimate is absent. We do not build a competing agent — we build the audit layer the entire category lacks.

**Scan of three other prominent LLM-trading-agent repos** confirms the gap is category-wide:

- **FinGPT** (AI4Finance-Foundation/FinGPT, ~20.1k stars, MIT): evaluation is **NLP** (weighted F1 on FPB / FiQA-SA / TFNS / NWGI). The README does not provide a trading-strategy backtest harness with PnL, costs, or risk-adjusted metrics. The README itself states "Nothing herein is financial advice."
- **FinRobot** (AI4Finance-Foundation/FinRobot, ~7.0k, Apache-2.0): no evaluation/backtest framework documented in the README. A `finrobot/functional/quantitative.py` exists `[verify contents]`.
- **FinMem** (pipiku915/FinMem-LLM-StockTrading, ~789–893 stars `[verify exact]`, MIT, paper arXiv:2311.13743): has explicit train/test modes for memory population; the paper reports cumulative return; the README does not describe a reusable statistical-audit framework.

**Strongest empirical motivation for this library** is independent: Li et al., "Can LLM-based Financial Investing Strategies Outperform the Market in Long Run?" (arXiv:2505.07078) introduces **FINSABER**, extends evaluation across two decades and 100+ symbols, and reports that under more rigorous testing LLM trading-agent advantages "deteriorate markedly" — overly conservative in bull markets, overly aggressive in bear markets, underperforming passive benchmarks. This is exactly the empirical pattern an honest evaluation harness should make routinely visible. Cite this in the README — it is the third-party evidence that the rigor gap is real, not just our framing.

---

## 2. The rigor gap — methodological failures we will make impossible to ignore

Each item below is a specific, recurring failure mode in LLM-trading-agent research. Each becomes a capability of `agent-backtest-lab`.

| Failure | What goes wrong | Capability we ship |
|---|---|---|
| **Look-ahead leakage** | Agent's prompt or feature set includes data not yet known on the as-of date (the most common subtle case: vendor "Adjusted Close" series back-adjusted for future splits/dividends). | Point-in-time **data firewall** that filters every input by as-of date, plus post-hoc leakage detectors. |
| **Survivorship bias** | Evaluating on the current S&P 500 constituents, ignoring delistings. | We do not bundle survivorship-corrected data (out of scope on free budget) but we **document the bias** in every scorecard and flag when the universe used cannot be verified survivorship-free. |
| **Gross vs net returns** | Reported P&L ignores commissions, half-spread, and impact. A 5 bps round-trip cost destroys a "winning" 3 bps/day signal. | **Transaction-cost + slippage models** (constant-bps default + Almgren-style square-root impact, with the 3/5 exponent caveat from Almgren-Thum-Hauptmann-Li 2005 documented). |
| **Multiple testing without correction** | "I tried 50 prompt variants and 20 tickers and report the best." With no correction, the best Sharpe is overwhelmingly likely to be noise. | **FDR control** (Benjamini-Hochberg 1995) **plus Deflated Sharpe Ratio (DSR)** and **Probabilistic Sharpe Ratio (PSR)** from Bailey & López de Prado. |
| **Backtest overfitting** | Hyperparameter / prompt search optimizes the same in-sample period repeatedly. | **Probability of Backtest Overfitting (PBO)** via **Combinatorially Symmetric Cross-Validation (CSCV)** from Bailey, Borwein, López de Prado, Zhu. |
| **Uncalibrated confidence** | Agent says "85% confident BUY" — we never check whether 85%-confident calls are right 85% of the time. | **Reliability diagrams + Expected Calibration Error (ECE)** (Guo et al. 2017) **and split conformal prediction sets** on directional calls (Vovk, Gammerman, Shafer 2005; modern intro Angelopoulos & Bates arXiv:2107.07511). |
| **Cherry-picked tickers / periods** | "It works on NVDA from 2023–2024." | The scorecard requires a declared universe and window; we report **per-ticker and pooled** results, and the PBO/DSR corrections fold the size of the search into the null. |
| **No buy-and-hold comparison** | Reporting raw returns without showing what an index would have done over the same window. | **Mandatory baseline set** (buy-and-hold, naive momentum, random); every scorecard renders the agent vs all baselines with CIs after costs. |

The deliverable is not a paper-quality calculation of each metric in isolation — many libraries can compute a Sharpe ratio. The deliverable is that **all of these checks run together, with no escape hatch**, and the report makes any failure unmistakable.

---

## 3. Method inventory — precise statements, assumptions, sources

Every method below has a known-answer fixture test (Phase 2/3) before it's allowed into the public API.

### 3.1 Walk-forward / Purged K-Fold CV with embargo
- **Statement:** When splitting a financial time series into train/test folds for CV, (a) **purge** any training observation whose label-construction window overlaps any test observation, and (b) **embargo** a percentage of observations immediately after each test fold so that label leakage via serial correlation is neutralised.
- **Assumptions:** Labels are constructed from forward windows of known length; serial correlation decays within the embargo window.
- **Source:** López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch. 7.
- **Implementation reference:** `skfolio.model_selection.CombinatorialPurgedCV` and `mlfinlab.cross_validation.combinatorial` are the canonical references in the open-source ecosystem; we implement a minimal, self-contained walk-forward + purged variant rather than depending on `mlfinlab` (now partially closed-source).

### 3.2 Probability of Backtest Overfitting (PBO) via CSCV
- **Statement:** Given `N` candidate strategies' return series across `T` time-steps, partition `T` into `S` equal blocks, take every `(S/2)`-sized combination of blocks as in-sample and the complement as out-of-sample, compute the **rank** of the in-sample-best strategy's OOS performance, transform to a logit, and estimate PBO as the fraction of combinations in which the in-sample-best strategy is below median OOS.
- **Assumptions:** Strategy return series are commensurable; `S` is even; performance metric is monotone.
- **Source:** Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest Overfitting," *J. Computational Finance*, 2017 — SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253. Companion expository piece: "Pseudo-Mathematics and Financial Charlatanism…" *AMS Notices*, May 2014.

### 3.3 Deflated Sharpe Ratio (DSR)
- **Statement:** DSR adjusts an observed Sharpe ratio for (i) the number of trials `N`, (ii) skewness `γ̂₃`, (iii) excess kurtosis `γ̂₄ − 1`, and (iv) sample size `T`. The reported quantity is the probability that the true Sharpe exceeds zero given the multi-trial null.
- **Working formula** (transcribed from secondary sources, to be **re-verified against the original PDF before any user-facing doc cites it** — current placeholder):

  `DSR = Φ( ( (ŜR − SR₀) · √(T − 1) ) / √( 1 − γ̂₃·ŜR + ((γ̂₄ − 1)/4)·ŜR² ) )`

  with `SR₀ = √V[ŜR] · ((1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N e)))`, `γ` the Euler–Mascheroni constant, `V[ŜR]` the cross-trial Sharpe variance.
- **Assumptions:** IID-ish returns within each trial (the skew/kurtosis correction softens the IID-Gaussian assumption); `N` is the number of trials over which the maximum was taken.
- **Source:** Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality," *J. Portfolio Management* 40(5), 2014. SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551. PDF https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- **Decision:** read the PDF before writing the implementation; pin the formula in our docs to that PDF; keep the `[verify against PDF]` tag in code comments until then.

### 3.4 Probabilistic Sharpe Ratio (PSR)
- **Statement:** Same Cornish-Fisher-style kernel as DSR but with a user-chosen benchmark `SR*` instead of the multiple-testing-adjusted `SR₀`. Reports `P(true SR > SR*)`.
- **Source:** Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier," *J. Risk* 15(2), Winter 2012/13. SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643.

### 3.5 Benjamini-Hochberg FDR
- **Statement:** Sort p-values `p_(1) ≤ ... ≤ p_(m)`. Find the largest `k` with `p_(k) ≤ (k/m)·α`. Reject `H_(1),...,H_(k)`. Controls the **false discovery rate** at level `α` under independence (and PRDS).
- **Source:** Benjamini & Hochberg, *JRSS-B* 57(1), 1995, pp. 289-300. https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x
- **Test fixture:** generate `m` p-values, a known subset from `H_1` (small p) and the rest from `Uniform(0,1)`. Verify our BH controls FDR at α across 1000 simulations.

### 3.6 Conformal prediction sets (directional)
- **Statement (split conformal, classification):** Hold out a calibration set, score each calibration example, take the `⌈(n+1)(1−α)⌉/n` quantile of nonconformity scores, and at test time return all labels whose score is below that quantile. Marginal coverage `≥ 1−α` under exchangeability.
- **Assumption:** Exchangeability of calibration + test points. **Time series violate this**, so we additionally support a **rolling-window** variant where the calibration set is the trailing `n` examples (still imperfect under regime shift; we document this honestly).
- **Sources:** Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, Springer 2005 (2nd ed. 2022). Modern tutorial: Angelopoulos & Bates, arXiv:2107.07511.

### 3.7 Reliability diagrams / Expected Calibration Error (ECE)
- **Statement:** Bin predictions by predicted probability; in each bin compute the mean predicted probability and the empirical accuracy; ECE is the bin-population-weighted average of `|mean_pred − empirical_acc|`.
- **Sources:** ECE per Guo et al., "On Calibration of Modern Neural Networks," ICML 2017 (arXiv:1706.04599). Reliability-diagram lineage: Murphy, "A New Vector Partition of the Probability Score," *J. Applied Meteorology* 1973.

### 3.8 Transaction-cost models
- **Default (always on):** constant per-trade cost `c_bps` applied to notional (default 5 bps each side, configurable).
- **Optional impact term:** `Δp/p ≈ σ · η · (Q / ADV)^β`, where `σ` is daily volatility, `Q/ADV` is participation, `η` and `β` are model parameters. The **canonical Almgren-Thum-Hauptmann-Li (2005)** estimate is `β = 3/5` for temporary impact (NOT 1/2 — important to flag). We expose `β` as a parameter, default to `0.5` (square-root) for accessibility, and document the 3/5 finding.
- **Sources:** Almgren & Chriss, "Optimal Execution of Portfolio Transactions," *J. Risk* 2000; Almgren, Thum, Hauptmann, Li, "Direct Estimation of Equity Market Impact," *Risk* 2005.

---

## 4. Adapter design sketch

The interface a framework must satisfy to be evaluated is intentionally minimal: produce a **directional call** (one of `LONG / SHORT / FLAT`) and an optional **confidence** in `[0, 1]` for a `(ticker, as_of_date)` pair, given a point-in-time-restricted bundle of inputs.

```python
class StrategyAdapter(Protocol):
    name: str
    def predict(self, ticker: str, as_of: date, pit_data: PITView) -> Decision: ...

@dataclass(frozen=True)
class Decision:
    direction: Literal["LONG", "SHORT", "FLAT"]
    confidence: float | None    # None means the framework does not emit one
    raw: dict | None            # opaque framework-native output, kept for forensics
```

Three concrete adapters:

1. **`TradingAgentsAdapter`** (reference): wraps `TradingAgentsGraph().propagate(ticker, as_of.isoformat())`, parses the second return element into `Decision`. The exact parsing path is `[verify]` against `tradingagents/graph/trading_graph.py` source — current README does not document the schema. **Important:** `TradingAgents` does not natively expose a calibrated probability, so `confidence = None` is the default; if the agent's text decision contains a numeric confidence we extract it, otherwise we leave it null and the calibration module reports "not evaluable, confidence not provided." This is what *honest* looks like.
2. **`CallableAdapter`** (generic): wraps any `Callable[[ticker, as_of, pit_data], Decision]`. The escape hatch for users with their own framework.
3. **`PlainStrategyAdapter`**: for non-LLM baselines (buy-and-hold, naive momentum, random) — a thin wrapper so baselines flow through the same evaluation pipeline as agents. No special-casing.

**Critical:** the `pit_data` argument is **always** produced by our leakage firewall, never directly by the framework. The adapter does not get to fetch its own data behind our back. (This is enforceable in practice for adapters that ask for data through us; we cannot prevent `TradingAgents` from making its own LLM-mediated tool calls. We document this honestly — the leakage firewall is a **guarantee for adapter-mediated inputs** and a **best-effort detector** for framework-internal data access. The post-hoc leakage detectors exist precisely for the latter.)

`TradingAgents` is an **optional** dependency. The package never imports it at module load; the adapter degrades to a clean ImportError-with-instructions if a user tries to use it without it installed. CI does not exercise the `TradingAgents` adapter (it would require API keys); it is illustrated in `examples/` only.

---

## 5. Data discipline

**The point-in-time problem in one paragraph.** A typical free data vendor (yfinance, stooq) gives you a *retrospectively adjusted* time series: today's "Adjusted Close" for 2020-03-15 reflects every split and dividend that happened *after* 2020-03-15. A strategy backtested on this series can — without anyone noticing — earn returns by trading patterns that only became visible after a future stock split was applied. **This is the single most common silent leakage in retail backtests.**

**Our defense:** the data layer stores the **unadjusted raw OHLCV** plus a `corporate_actions` table keyed by ex-date. At as-of date `t`, the `PITView` reconstructs the adjusted series using **only** corporate actions with `ex-date ≤ t`. The leakage firewall additionally:

- refuses to serve any bar with date `> as_of`;
- refuses to serve a corporate action with `ex-date > as_of`;
- logs every data access with the `(as_of, ticker, requested_date)` triple, so the post-hoc detector can run on the audit log;
- bundles a small fixture dataset (≤10 tickers, ~5 years of daily bars) with the package so quickstart, CI, and the leakage-detection test all run offline with no API keys.

**Sources (free, with caveats):**
- `yfinance` — Apache-2.0; the unadjusted close is accessible; treat `Adj Close` as retroactively adjusted (do not feed to the agent).
- `stooq` — bulk CSV; close is already split-adjusted (same leakage concern); license terms not clearly stated `[verify before redistribution]`.
- `alphavantage` — 25 reqs/day free tier; corroboration only.

Survivorship-corrected delisting data is *not* freely available on a research budget. We do not pretend to fix this. Every scorecard prints a warning when the user-declared universe cannot be verified survivorship-free.

---

## 6. Risk log

| Risk | Mitigation |
|---|---|
| **Reputational adjacency to "LLM trading" hype.** | The not-advice / no-execution / no-alpha-claims disclaimer block is high in the README, and "honest by design" is the framing on every emitted report. The library exists to *make users more skeptical*, not to help them deploy an agent. |
| **A subtle statistical bug ships and produces *false* honesty.** A tool that quietly under-corrects multiple testing is worse than no tool. | Every statistical method has a **known-answer fixture test** that is a CI gate. The leakage detector is tested against a *deliberately-leaky* fixture it must catch; the overfitting detector against a deliberately-overfit fixture; the BH corrector against simulated FDR. Phase 6 is a hostile review pass dedicated to finding these bugs. |
| **`TradingAgents` interface drifts.** | The adapter is a thin wrapper; the generic `CallableAdapter` is the load-bearing path. We pin against a known `TradingAgents` commit in the example, and the package never imports it at load time. |
| **Maintenance burden.** | No source file > 500 lines; minimal dependencies (numpy, pandas, scipy, matplotlib, click, optionally yfinance); CI runs fully on the bundled fixture so external-data outages don't break builds. |
| **Compliance / "this is investment advice" misreading.** | Disclaimer block; no `buy`/`sell` language in *our* outputs (we say "directional call" or "predicted direction"); the CLI top-line help text repeats the not-advice disclaimer. |
| **License clash with bundled data.** | Bundled fixture data is either (a) synthetic (GBM with known parameters, properties documented), or (b) sourced from a clearly free/redistributable source with attribution. **Default: synthetic.** This avoids vendor-license trouble entirely and has a side benefit — for fixture tests we *know* the true Sharpe / true directional accuracy and can verify our metrics. |
| **Hostile-reviewer attack: "isn't this just backtesting.py with extra steps?"** | Direct contrast in README: backtesting.py / vectorbt / zipline are *execution engines*; we are a *statistical-audit harness* that can sit on top of any of them. Our value is the multiple-testing, calibration, PBO, and leakage layers, not the PnL accumulator. |
| **Subagent / web-search facts I couldn't verify.** | Marked `[verify]` throughout; CI does not depend on any unverified claim; before any user-facing doc cites a paper formula we re-read the source PDF. |

---

## 7. Open questions (proceeding by best judgment per the autonomous-execution directive)

1. **Bundled fixture data: synthetic GBM vs free real data.** Decision: **synthetic** (GBM with known drift, vol, true direction-conditional-probability) for the primary fixture, because it lets every statistical test verify against a *known* truth. A second tiny real-data fixture (≤3 tickers, 1 year, from a clearly-free source) for the demo screencast.
2. **Conformal prediction under time-series non-exchangeability.** Decision: ship **split conformal with a rolling calibration window** as the default; document loudly that exchangeability is violated and coverage may degrade under regime shift; recommend the user combine it with the calibration diagnostics rather than trusting the conformal set blindly.
3. **Sharpe annualization convention.** Decision: `√252` for daily, `√52` for weekly, explicit parameter; the scorecard prints the chosen factor. No silent assumptions.
4. **Default transaction-cost level.** Decision: **5 bps each side** as the default for daily-bar US equity strategies, plus an optional Almgren-style impact term off by default. This is conservative-but-realistic for retail; the scorecard prints the assumption.
5. **Universe + window declaration.** Decision: a `Universe` object is a *required* argument to the top-level `evaluate()` call; we refuse to run if the user does not declare it explicitly. Cherry-picking is harder when the universe is a first-class API object.
6. **What we will NOT build:** a broker integration, a paper-trading bridge, a live-data feed, any `buy`/`sell` recommendation in the CLI output, a "strategy marketplace," any benchmark against a real customer's portfolio. If a feature can only be built by crossing the not-advice line, we don't build it — log it as out-of-scope here.

---

## Decision summary (carried forward to Phase 1)

- Library is called `agent-backtest-lab`, importable as `abl`, CLI command `abl`.
- Apache-2.0; Python 3.10+; numpy/pandas/scipy/matplotlib/click required, yfinance optional, `tradingagents` optional.
- Bundled-synthetic fixture is the primary CI dataset; a tiny real-data fixture for the demo.
- Adapter contract: `predict(ticker, as_of, pit_data) -> Decision(direction, confidence, raw)`.
- Three adapters: `TradingAgents` (optional dep), generic callable, plain strategy (for baselines).
- Statistical methods: walk-forward + purged CV (López de Prado), BH FDR (Benjamini-Hochberg 1995), DSR + PSR (Bailey & López de Prado), CSCV/PBO (Bailey/Borwein/López de Prado/Zhu), reliability + ECE (Guo et al. 2017), split conformal with rolling window (Vovk/Gammerman/Shafer 2005; Angelopoulos & Bates 2021).
- Transaction-cost default: 5 bps each side; optional Almgren impact with documented 3/5-vs-1/2 caveat.
- The empty niche is confirmed: no existing open-source library combines (rigorous backtest + leakage + multiple-testing + calibration + PBO/DSR + LLM-agent integration). `mlfinlab` covers some statistical pieces but is partly closed-source and not LLM-agent-focused; `backtesting.py` / `vectorbt` / `zipline` are execution engines, not audit harnesses. We position as a **companion layer** to both — not a competitor to either.

Proceeding to Phase 1.
