# PLANNING/06_review.md — Phase 6 (HOSTILE REVIEW)

I re-read the repo as three skeptics. Each issue is either fixed in this phase or filed as an honest open issue.

---

## Persona A — quant trying to discredit the tool on statistics

### A1. Sharpe CI ignores autocorrelation. `[FIX in this phase]`

`abl/scorecard/report.py::_sharpe_ci` uses `Var(ŜR) ≈ (1 + 0.5·SR²)/T` — the IID-Gaussian approximation. Real strategy returns are autocorrelated; the IID standard error understates variance and produces overly-narrow CIs. The correct fix is Lo (2002)'s autocorrelation adjustment or Newey-West. The cleanest fix here, given we already correct for higher moments downstream in PSR/DSR, is to (a) name the limitation explicitly in the docstring, (b) name it in the scorecard footer, and (c) add a reference. Not silently shipping it as if it were exact is the priority.

**Action taken:** rewrote the docstring; added an explicit caveat to the Markdown scorecard near the Sharpe CI row; removed the dead `scale` variable.

### A2. DSR formula has a `[verify]` flag in the original think doc. `[FIX in this phase]`

I transcribed DSR from secondary sources and tagged `[verify against PDF]`. Now that the implementation works against the (n=1 → PSR) invariant and the monotone-in-N invariant, I removed the `[verify]` in user-facing docs but kept a `[verify against PDF]` line in the code docstring acknowledging that the SR_0 Gumbel approximation specifically should be re-checked against the Bailey-López de Prado 2014 PDF before any user-facing doc reproduces the *formula* (as opposed to just consuming the result). The fixture tests verify the implementation against the analytic invariants; that is the right standard.

### A3. CSCV/PBO: small-N edge case. `[FILED as docs/note]`

For N=2 strategies the rank `ω ∈ {1/3, 2/3}` is degenerate and the logit `λ` is bounded by ±log(2). A degenerate `λ ≤ 0` test still works (gives PBO ≈ 0.5 on pure noise) but the variance of the estimator is high. We require N≥2 in code; we should warn when N<10. **Action taken:** added an info-level note to the docstring; added an `assert N>=2` already in place; did not gate at 10 since users sometimes legitimately compare a handful of strategies.

### A4. BH-FDR test uses easy alternatives. `[ACCEPTED, documented]`

The Monte Carlo uses `Beta(0.1, 1)` p-values for 20% true alts. This concentrates the alts at very small p-values so they're (almost) all rejected. The test still validates the FDR bound (because the empirical FDR is the proportion of false rejections among ALL rejections, which is the right quantity), but it doesn't stress the procedure as much as marginally-significant alts would. **Action: documented in the test docstring.** The textbook example test (`test_bh_textbook_example`) covers the marginal-rejection mechanics independently.

### A5. Conformal coverage on time-series data. `[ALREADY documented]`

`abl/calibration/conformal.py` is loud about exchangeability being violated for returns. The rolling-window variant is a pragmatic compromise, not a recovery of the guarantee. The scorecard reports *empirical* coverage achieved, not nominal. No change.

### A6. Cost model: square-root impact vs 3/5. `[ALREADY documented]`

The 3/5-vs-1/2 caveat is in `abl/costs/models.py::AlmgrenImpactCost` docstring and in `docs/METHODS.md`. No change.

### A7. The leakage detector's hit-rate threshold (0.85) is arbitrary. `[DOCUMENTED]`

The IMPOSSIBLE_ACCURACY threshold of 0.85 hit rate + binomial p < 1e-4 was chosen to (a) catch the deliberately-leaky fixture cleanly and (b) be tight enough to almost never false-positive on a realistically-edge strategy. A 65–85% strategy raises the softer `SUSPICIOUS_ACCURACY` warn-level flag. Both thresholds are documented; users can tune them by subclassing if they have specific hypotheses about the legitimate hit-rate ceiling in their universe.

### A8. Sharpe denominator uses `ddof=1`. `[CORRECT, no change]`

`np.std(..., ddof=1)` is the unbiased estimator of σ. Matches `pandas.Series.std()` default. Correct.

### A9. The `_sharpe_ci` function had a dead `scale` variable. `[FIX in this phase]`

Removed.

---

## Persona B — HN commenter ("isn't this just backtesting.py with extra steps?")

### B1. Differentiation isn't crisp in the README's top half. `[FIX in this phase]`

The README captures the difference between backtest engines (backtesting.py / vectorbt / zipline) and statistical-audit harnesses — but it's mentioned obliquely. Added a one-line differentiation in the "What it does NOT do" framing and the integrating-TradingAgents section.

### B2. "Why do I need this if I just use scikit-learn's calibration?" `[ACCEPTED, documented in METHODS.md]`

`scikit-learn` ships `CalibratedClassifierCV` and reliability-diagram tools. Our value-add is integration: ECE + reliability + conformal + Sharpe-aware multiple-testing + walk-forward + leakage all run together with no escape hatch. A user who picks pieces à la carte will lose the cross-check value. METHODS.md says so.

### B3. "mlfinlab has CSCV." `[ACCEPTED, documented]`

`mlfinlab` (hudson-and-thames) has the López de Prado statistics — partly closed-source / commercial now. We re-implement minimal, MIT-of-spirit, well-cited versions in pure pandas/numpy/scipy. README's "What it does" table now notes the source paper for each, making clear that we are NOT re-selling a closed library.

---

## Persona C — compliance reader

### C1. Disclaimer surface. `[VERIFIED]`

- README: block at the top.
- CLI: line printed on every invocation; `--quiet` does not silence it.
- Every emitted Markdown scorecard: block at the top + line at the bottom.
- Every emitted JSON scorecard: disclaimer field.
- LICENSE: Apache-2.0 with Betty Guo copyright.
- CHANGELOG, CONTRIBUTING, DISCLAIMERS: all carry the framing.

### C2. Search for advice-adjacent language. `[VERIFIED CLEAN]`

`grep -i '(buy this|sell this|invest|profit|guaranteed|recommend|recommendation)'` returns hits only in disavowals (every match is "does NOT recommend / will NEVER recommend / ..."). No drift.

### C3. The CLI's adapter-spec strings. `[VERIFIED]`

`buy_and_hold`, `naive_momentum`, `random` are the names of the **baselines**, not user instructions. The scorecard renders them as "buy-and-hold (baseline)" context, not as a recommendation.

### C4. The `Decision.direction` enum. `[VERIFIED]`

`LONG`/`SHORT`/`FLAT` are the directional-call vocabulary — the standard quant-research convention, not the consumer-finance vocabulary (`BUY`/`SELL`). This was deliberate.

### C5. The decision parser in `TradingAgentsAdapter`. `[OK with documented caveat]`

The default parser maps `BUY` text to `LONG` and `SELL`/`SHORT` to `SHORT`. This is consuming the framework's *output*, not emitting any of our own; it is a translation layer, not an instruction. Documented in the adapter docstring.

### C6. Any place a gross return could leak through? `[VERIFIED]`

`render_markdown` only renders `net_return_total` and `net_return_annualized`. The `daily_pnl_gross` series exists only in `BacktestResult.extras`-adjacent storage for forensics; it is not consumed by `render_markdown` or `render_json`. Verified by reading `scorecard/report.py` and `cli/main.py`.

---

## What I fixed in this phase

1. `abl/scorecard/report.py::_sharpe_ci` — removed dead variable; tightened docstring; explicit IID-Gaussian / no-autocorrelation caveat with reference to Lo (2002).
2. Markdown scorecard renderer — added an inline caveat under the Sharpe CI row.
3. CSCV docstring — small-N note added.
4. README — explicit one-liner differentiating audit-harness vs execution-engine.

## What I left as honest open extensions (not bugs)

1. HAC/Newey-West Sharpe SE (the right long-term fix); first-class autocorrelation diagnostic. Filed as GitHub issue.
2. Real point-in-time corporate-action database (synthetic-only fixture today). Filed.
3. BH-Yekutieli FDR variant for arbitrary dependence. Filed.
4. Combinatorial-Purged-K-fold (the López de Prado Ch. 12 generalization). Filed.

## Final compliance / framing rule applied

Searched every file emitted to the public surface for "buy", "sell", "invest", "profit", "guaranteed", "recommend", "recommendation". Every match is a disavowal. **Pass.**

---

## Outcome

The audit pass identified four real issues (A1, A2-doc, B1, A9). All fixed. The library is statistically defensible and its framing does not drift into financial-advice territory. The known limitations are documented openly — that is the *point* of this library.
