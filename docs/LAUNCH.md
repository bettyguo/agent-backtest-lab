# Launch playbook

This file holds the launch material for `agent-backtest-lab`. Drafted by the author; revise before posting. Every public-facing line follows the not-advice rule: this is a research tool, not a trading system.

---

## Show HN

**Title** (≤ 80 chars):

```
Show HN: agent-backtest-lab – audit your LLM trading agent before you trust it
```

**Body:**

> The LLM-trading-agent category is hot. TradingAgents is at ~75k stars; clones are
> proliferating. Almost none of them check for the failure modes that actually destroy
> alpha in published quant research: look-ahead leakage via retrospectively-adjusted
> data, gross-vs-net returns, multiple-testing without correction, miscalibrated
> confidence, backtest overfitting from prompt search. Independent recent work
> (arXiv:2505.07078) shows that under disciplined evaluation, LLM-trading-agent
> advantages "deteriorate markedly" — they underperform passive benchmarks in bull
> markets and incur heavy losses in bear markets.
>
> `agent-backtest-lab` is a Python library and CLI that audits these frameworks.
> It ships a hard leakage firewall (refuses any access with date > as_of), defends
> against the retroactive-adjustment failure mode, applies transaction costs by
> default, implements Benjamini-Hochberg FDR, the Probabilistic and Deflated Sharpe
> Ratios (Bailey & López de Prado), the Probability of Backtest Overfitting via
> CSCV, reliability diagrams + ECE, and split conformal sets — and assembles them
> into a single Markdown scorecard that prints buy-and-hold and naive baselines next
> to whatever you're evaluating. Every method has a known-answer fixture test in CI;
> the leaky-fixture test is a gate (the build fails if the firewall ever lets the
> leak through).
>
> This is a companion to TradingAgents, not an attack on it. Their job is to
> generate decisions; ours is to audit them.
>
> Importantly: **this is not a trading system**. It doesn't execute trades, doesn't
> connect to brokerages, doesn't recommend buying or selling anything. It exists to
> help researchers and practitioners measure how trading-agent frameworks actually
> perform, especially when they perform badly.
>
> Repo: https://github.com/bettyguo/agent-backtest-lab
> Methods reference: https://github.com/bettyguo/agent-backtest-lab/blob/main/docs/METHODS.md

---

## X (Twitter) thread

1/ Your LLM trading agent shows +40% in a backtest.

Was it good — or did it secretly see tomorrow's prices through retroactively-adjusted data, pick the best of 50 untracked prompts, or earn returns no real trader could net of costs?

`agent-backtest-lab` tells you which.

2/ TradingAgents (~75k stars) and its many clones are agent *systems*. They don't ship transaction-cost models, leakage firewalls, multiple-testing correction, calibration analysis, or backtest-overfitting probability. The category needs an audit layer.

3/ Concretely. Yahoo's `Adj Close` is RETROACTIVELY adjusted for every future split and dividend. A strategy backtest on that series can earn returns by trading patterns only visible after a future split. This is the most common silent leakage in retail backtests. We never use that series.

4/ Tried 50 prompts and reported the best? The Deflated Sharpe Ratio (Bailey & López de Prado 2014) corrects for that. The Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation (BBLZ 2017) tells you whether the IS-best is reliably OOS-bad. Both run with one flag.

5/ Hard leakage firewall: every PIT data access goes through `Firewall.check()`, which refuses any datum with date > as_of and writes an audit log. Tested against a deliberately-leaky fixture strategy it MUST catch. Gate test. Build fails if it ever doesn't.

6/ Honest by design: every scorecard is net of cost, has CIs, shows DSR after multiple-testing correction, prints buy-and-hold alongside, flags leakage and overfitting in red. If your agent doesn't emit a confidence, calibration says "not evaluable" — we don't fabricate one.

7/ This is not a trading system. It doesn't execute trades, connect to brokerages, or recommend buying anything. It exists to help researchers and practitioners measure trading-agent frameworks honestly — especially when they perform badly.

8/ Apache-2.0. 63 fixture tests pass. Companion to TradingAgents — their job is to generate decisions; ours is to audit them.

https://github.com/bettyguo/agent-backtest-lab

---

## r/algotrading post

**Title:** `agent-backtest-lab — a statistical-audit harness for LLM trading agents (Apache-2.0)`

**Body:**

I built a Python library that audits LLM trading-agent frameworks (TradingAgents and similar) for the methodological failures common to the whole category: look-ahead leakage via retroactively-adjusted data, gross-vs-net returns, multiple-testing without correction, uncalibrated confidence, backtest overfitting from prompt search, cherry-picked universes.

It ships:

- A hard point-in-time leakage firewall (refuses any access with date > as_of; reconstructs adjusted OHLCV using only corporate actions known at as_of)
- Transaction-cost models (constant bps + Almgren-style impact)
- Benjamini-Hochberg FDR, Probabilistic + Deflated Sharpe Ratios (Bailey & López de Prado), Probability of Backtest Overfitting via CSCV
- Reliability diagrams + ECE, split conformal sets on directional calls (rolling-window variant for non-exchangeable time series — caveat documented)
- Three mandatory baselines (buy-and-hold, naive momentum, random) always shown alongside
- Markdown + JSON scorecard with explicit leakage / overfitting / survivorship flags

Every method has a known-answer fixture test in CI. The leaky-fixture test is a gate.

This is explicitly NOT a trading system. It doesn't execute trades, doesn't connect to brokerages, doesn't recommend buying or selling. It exists to help you measure honestly. https://github.com/bettyguo/agent-backtest-lab

---

## r/MachineLearning post

**Title:** [P] agent-backtest-lab: statistical-rigor harness for LLM trading agents

**Body:** (similar to r/algotrading but emphasize: BH-FDR, conformal prediction with the non-exchangeability caveat, CSCV, ECE; explicitly position as a *measurement* tool not a *deployment* tool; cite Bailey/Borwein/López de Prado/Zhu, Benjamini/Hochberg, Vovk/Gammerman/Shafer, Angelopoulos/Bates, Guo/Pleiss/Sun/Weinberger.)

---

## r/quant post

**Title:** Companion library to LLM-trading-agent frameworks: leakage firewall, DSR, PBO, conformal calibration

**Body:** (terse; quants know the references. Lead with the methods table.)

---

## Newsletter outreach

**AlphaSignal / Latent Space / Import AI:**

> Subject: Tool for measuring LLM trading agents honestly
>
> Hi —
>
> I built and just open-sourced `agent-backtest-lab`, a statistical-audit harness for
> LLM trading agents. It plugs into frameworks like TauricResearch/TradingAgents
> (~75k stars) and runs the rigor checks that the category mostly skips:
> point-in-time leakage firewall, transaction-cost-net returns, Benjamini-Hochberg
> FDR, Deflated Sharpe Ratio, Probability of Backtest Overfitting via CSCV,
> reliability diagrams, split conformal sets. Every method has a known-answer
> fixture test.
>
> Independent recent work (arXiv:2505.07078) shows LLM-trading-agent advantages
> "deteriorate markedly" under more rigorous evaluation — this library is the
> evaluation that surfaces that, with no escape hatches.
>
> This is explicitly NOT a trading system. It audits, it doesn't execute or recommend.
>
> Apache-2.0. https://github.com/bettyguo/agent-backtest-lab. Happy to write more if
> a longer piece would suit your audience.
>
> — Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong

---

## Launch sequence (recommended)

1. **D-7 to D-1:** Record the demo GIF (`assets/RECORD_DEMO.md`); finalize PyPI publish; tag v0.1.0; smoke-test `pip install` in a clean environment.
2. **D-0 (Tue 09:00 PT / 17:00 BST):** Show HN post goes live; X thread fires 60 seconds later.
3. **D-0 (afternoon):** r/algotrading and r/MachineLearning posts.
4. **D-1:** r/quant post (less time-sensitive; quant communities react more slowly).
5. **D-3:** newsletter outreach emails.
6. **D-7:** retrospective on what landed; pin best feedback as GitHub issues.

**Things NOT to do during launch:**

- Do NOT claim alpha for any strategy.
- Do NOT respond to "should I trade this?" questions with anything except "this is a research tool; not financial advice."
- Do NOT remove the disclaimer block from any artifact, ever.
- Do NOT pick fights with TradingAgents or any LLM-trading-agent maintainer. Companion, not competitor.
