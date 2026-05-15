# Statistical methods reference

This file is the source of truth for every statistical method `agent-backtest-lab` implements. Each entry lists the formal statement, the assumptions, the citation, and a pointer to the fixture test that verifies the implementation against a known-answer.

If you spot a transcription error against the cited source, please open an issue. We mark anything we could not verify exactly against the original PDF with `[verify]`; that flag is honest, not embarrassing.

---

## 0. Sharpe ratio standard error — HAC / Newey-West (Lo 2002)

**Statement.** Under serial correlation, the IID-Gaussian SE understates Var(ŜR). The HAC estimator is

```
SE_HAC(ŜR) = √( (1 + 0.5·ŜR²) · η(q) / T )
η(q) = 1 + 2 · Σ_{k=1}^{q} (1 − k/(q+1)) · ρ_k
```

where ρ_k are sample autocorrelations and the Bartlett kernel weights (1 − k/(q+1)) are the standard Newey-West choice. For q=0 this reduces to the IID formula. For positively-autocorrelated series, η(q) > 1 — the IID CI was overconfident.

**Assumptions.** Weak stationarity. Bartlett kernel; lag truncation q can be set automatically per Newey-West (1994) `q ≈ ⌊4·(T/100)^(2/9)⌋`.

**Source.** Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4), 36-52. · Newey, W. K., & West, K. D. (1987). *Econometrica*, 55(3), 703-708. · Newey, W. K., & West, K. D. (1994). *Review of Economic Studies*, 61(4), 631-653.

**Implementation.** `abl.multipletest.hac.hac_sharpe_ci`. **Tests:** `tests/test_hac_sharpe.py` — reduces to IID at q=0, widens CI under AR(1) positive autocorrelation, matches Bartlett-weighted formula on hand-computed ρ.

---

## 1. Walk-forward / Purged K-fold CV with embargo

**Statement.** When labels are constructed from forward windows of length `label_window`, a naive K-fold split leaks information into training whenever a training observation's label window overlaps any test observation. Fix: (a) **purge** training observations whose label window `[i, i + label_window]` overlaps the test fold `[test_start, test_end + label_window]`; (b) **embargo** an additional `embargo_pct * n` observations immediately after each test fold to neutralize serial correlation the purge alone cannot remove.

**Assumptions.** Labels constructed from a forward window of known constant length; serial correlation decays within the embargo.

**Source.** López de Prado, M. (2018). *Advances in Financial Machine Learning*, Wiley, Chapter 7. https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

**Implementation.** `abl.backtest.cv.purged_kfold_splits`. **Test:** `tests/test_walkforward.py::test_purged_kfold_purges_label_window`.

### 1b. Combinatorial Purged K-Fold (López de Prado Ch. 12)

**Statement.** Partition the chronological index into `n_groups` equal-sized contiguous groups; for each of the `C(n_groups, n_test_groups)` ways of choosing groups as test, return a split with the corresponding test_idx and a train_idx that excludes the purge+embargo window around each test group. Number of recoverable OOS paths: `C(n_groups − 1, n_test_groups − 1)`.

**Implementation.** `abl.backtest.cv.combinatorial_purged_kfold_splits`, `n_recoverable_paths`. **Tests:** `tests/test_combinatorial_cv.py` — split count = C(n,k), purge respected, recoverable-paths formula.

---

## 2. Benjamini-Hochberg FDR control

**Statement.** Given m p-values sorted as p_(1) ≤ ... ≤ p_(m), let k = max{i : p_(i) ≤ (i/m)·α}. Reject H_(1), ..., H_(k). Controls the **False Discovery Rate** (expected proportion of false rejections among all rejections) at level α under independence and Positive Regression Dependence on a Subset (PRDS).

**Assumptions.** Independence or PRDS. Under arbitrary dependence use the BH-Yekutieli variant (multiply α by Σ 1/k for k=1..m).

**Source.** Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B (Methodological)*, 57(1), 289-300. https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x

**Implementation.** `abl.multipletest.bh.benjamini_hochberg`. **Tests:** `tests/test_bh_fdr.py` — textbook example, Monte Carlo verification that empirical FDR ≤ α under independence across 1000 reps.

### 2b. Benjamini-Yekutieli (FDR under arbitrary dependence)

**Statement.** Same procedure as BH but divide the threshold by `c(m) = Σ_{k=1}^m 1/k`. Controls FDR under arbitrary dependence — strictly more conservative, but assumption-free.

**Source.** Benjamini, Y., & Yekutieli, D. (2001). The Control of the False Discovery Rate in Multiple Testing under Dependency. *Annals of Statistics*, 29(4), 1165-1188.

**Implementation.** `abl.multipletest.bh.benjamini_hochberg(..., method="by")`. **Tests:** `tests/test_bh_yekutieli.py` — strictly fewer rejections than BH on the textbook example; controls FDR under correlated p-values across 500 reps.

### 2c. Reality Check / Superior Predictive Ability (SPA) test

**Statement.** Tests "does the best of N strategies beat a specified benchmark, accounting for the N-trial search?" Uses the Politis-Romano stationary block bootstrap on the relative-loss series (`r_strategy − r_benchmark`).

**Source.** White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097-1126. · Hansen, P. R. (2005). A Test for Superior Predictive Ability. *J. Business & Economic Statistics*, 23(4), 365-380. · Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *J. American Statistical Association*, 89(428), 1303-1313.

**Honest scope.** We implement the centered White (2000) variant. Hansen's full studentization-recentering machinery is filed as an open extension — `abl.multipletest.spa` says so in its docstring.

**Implementation.** `abl.multipletest.spa.reality_check_spa`. **Tests:** `tests/test_spa.py` — high p-value on pure noise; low p-value on a clear winner.

---

## 3. Probabilistic Sharpe Ratio (PSR)

**Statement.** Given a return series of length T with observed Sharpe `ŜR`, sample skewness `γ̂₃`, and standardized fourth moment `γ̂₄`,

```
PSR(SR*) = Φ( (ŜR − SR*) · √(T − 1) / √(1 − γ̂₃·ŜR + ((γ̂₄ − 1)/4)·ŜR²) )
```

The Cornish-Fisher-style denominator corrects the standard error of `ŜR` for higher moments of the return distribution.

**Assumptions.** Returns are IID-ish (the higher-moment correction softens the IID-Gaussian assumption but does not eliminate it).

**Source.** Bailey, D. H., & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*, 15(2). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643

**Implementation.** `abl.multipletest.psr.probabilistic_sharpe`. **Test:** `tests/test_dsr_psr.py::test_psr_zero_skew_zero_kurt_closed_form`.

---

## 4. Deflated Sharpe Ratio (DSR)

**Statement.** DSR adjusts PSR for the multiple-testing problem (we tried `N` strategies and reported the best). Replace PSR's benchmark `SR*` with the expected maximum-of-N under the null:

```
SR_0 = √V[ŜR] · ( (1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e)) )
```

(γ is the Euler-Mascheroni constant ≈ 0.5772; V[ŜR] is the cross-trial variance of estimated Sharpes.) Then DSR := PSR(SR_0).

**Assumptions.** N independent trials; Gumbel approximation to E[max of N standard normals] (degrades for small N but remains monotone).

**Source.** Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. *J. Portfolio Management*, 40(5), 94-107. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 · PDF: https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

**Implementation.** `abl.multipletest.dsr.deflated_sharpe`. **Tests:** `tests/test_dsr_psr.py::test_dsr_reduces_to_psr_when_one_trial`, `test_dsr_lower_than_psr_when_many_trials`, `test_expected_max_monotone_in_n`.

---

## 5. Probability of Backtest Overfitting (PBO) via CSCV

**Statement.** Given N candidate strategies' returns over T periods: (1) partition T into S equal-size blocks (S even); (2) for each of the C(S, S/2) choices of S/2 blocks as in-sample (IS), compute each strategy's IS and OOS Sharpe, identify the IS-best `n*`, compute the logit `λ` of `n*`'s OOS rank; (3) **PBO** = fraction of partitions with λ ≤ 0 — i.e., the IS-best is below median OOS.

**Assumptions.** Strategies' return series are commensurable; the Sharpe metric is monotone. We cap evaluation at 50k partitions and sample uniformly otherwise.

**Source.** Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

**Implementation.** `abl.overfitting.cscv.probability_of_backtest_overfitting`. **Tests:** `tests/test_cscv_pbo.py` — PBO ≈ 0.5 on IID noise; ≤ 0.1 when one strategy persistently dominates; ≥ 0.5 with critical flag on the anti-correlated overfit fixture.

---

## 6. Reliability diagrams + Expected Calibration Error (ECE)

**Statement.** Partition [0,1] into M equal-width bins. For each populated bin b: acc(b) = empirical positive rate; conf(b) = mean predicted probability. **ECE** = Σ_b (|b|/N) · |acc(b) − conf(b)|.

**Assumptions.** Predictions are scalar probabilities in [0,1]; labels are 0/1.

**Source.** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*, PMLR 70:1321-1330. https://arxiv.org/abs/1706.04599 · Reliability-diagram lineage: Murphy, A. H. (1973). A New Vector Partition of the Probability Score. *J. Applied Meteorology*, 12, 595-600.

**Implementation.** `abl.calibration.reliability.expected_calibration_error`. **Test:** `tests/test_reliability_ece.py::test_perfectly_calibrated_synthetic` — ECE < 0.02 at N=50000 on `probs ~ Uniform, labels ~ Bernoulli(probs)`.

---

## 7. Split conformal prediction sets (rolling-window for time series)

**Statement.** Given calibration set of size n with nonconformity scores s_1,...,s_n, the prediction set at level (1 − α) contains every test label whose score is ≤ the ⌈(n+1)(1−α)⌉/n quantile of the calibration scores. Under **exchangeability**, marginal coverage ≥ 1 − α.

**Assumption — and our honest caveat.** Time series are NOT exchangeable. Regime change destroys the guarantee. We ship a rolling-window variant that calibrates on the trailing n observations; it is a pragmatic compromise, NOT a recovery of the exchangeability assumption. The scorecard reports the empirical coverage actually achieved, not the nominal level.

**Sources.** Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*, Springer. https://link.springer.com/book/10.1007/b106715 · Modern tutorial: Angelopoulos, A. N., & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511. https://arxiv.org/abs/2107.07511

**Implementation.** `abl.calibration.conformal.split_conformal_quantile`, `rolling_split_conformal_coverage`. **Test:** `tests/test_conformal.py::test_split_conformal_empirical_coverage_iid` — empirical coverage ≥ 1 − α on 500 reps of IID standard normals.

---

## 8. Transaction-cost models

### 8a. Constant per-trade bps
Constant cost `c_bps` applied to turnover. The library default is **5 bps each side** for daily-bar US-equity strategies (conservative-but-realistic for retail). `abl.costs.models.ConstantBpsCost`.

### 8b. Almgren-style temporary impact
`Δp/p ≈ σ · η · (Q/ADV)^β`. Parameters `η` (impact coefficient) and `β` (exponent) exposed.

**The 3/5-vs-1/2 caveat.** The "square-root law of impact" (β = 1/2) is the practitioner default. The Almgren-Thum-Hauptmann-Li (2005) empirical study on Citigroup US-equity trades found temporary impact closer to **β = 3/5** (NOT 1/2). We default to 1/2 for accessibility and let you set 0.6 — documented in the docstring.

**Sources.** Almgren, R., & Chriss, N. (2000). Optimal Execution of Portfolio Transactions. *J. Risk*. https://www.smallake.kr/wp-content/uploads/2016/03/optliq.pdf · Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). Direct Estimation of Equity Market Impact. *Risk*, July 2005. https://www.cis.upenn.edu/~mkearns/finread/costestim.pdf

**Implementation.** `abl.costs.models.ConstantBpsCost`, `AlmgrenImpactCost`, `CompositeCost`. **Tests:** `tests/test_costs.py`.

---

## 8c. Drawdown metrics

**Statement.** From the cumulative net-of-cost equity curve `E_t = Π_{s≤t}(1+r_s)`, compute the running peak `M_t = max_{s≤t} E_s`, drawdown `DD_t = E_t/M_t − 1 ∈ [−1, 0]`, max drawdown `min_t DD_t`, longest underwater stretch (longest contiguous run of `DD_t < 0`), and the Calmar ratio (annualized return / |max drawdown|).

**Notes.** These are path-statistics, not distribution-statistics. They describe what *did* happen, not what *could* happen. The scorecard reports them next to Sharpe so users see both risk-adjusted return and realized rope-to-drawdown.

**Implementation.** `abl.scorecard.drawdown.drawdown_stats`. **Tests:** `tests/test_drawdown.py` — zero drawdown when monotone positive, hand-computed three-day case (−50% trough), trough index correct, Calmar finite when DD exists.

---

## 9. Leakage detection

Two flavors:

- **Hard guarantee.** `abl.data.firewall.Firewall` raises `FirewallViolation` on any PIT access with `requested_date > as_of`. The PITView reconstructs adjusted OHLCV using only corporate actions with `ex_date ≤ as_of`, defeating the most common retail leakage (Yahoo's retroactively-adjusted `Adj Close`).
- **Best-effort post-hoc detection.** `abl.leakage.detector.detect_leakage` flags impossibly-high directional accuracy and clustered correct calls near corporate-action ex-dates.

**Gate tests.** `tests/test_leakage_fixture.py::test_leaky_strategy_is_caught` — the deliberately-leaky strategy MUST trigger `IMPOSSIBLE_ACCURACY` with severity `critical`. `test_clean_strategy_is_not_flagged` — the naive-momentum baseline MUST NOT trigger any flag. Both are CI gates.

---

## What we deliberately do NOT do

- We do NOT use Yahoo's `Adj Close` series. The retroactive adjustment is the most common silent leakage in retail backtests.
- We do NOT bundle survivorship-corrected data — none is freely available on a research budget. Every scorecard prints `SURVIVORSHIP_UNVERIFIED` when the universe is not explicitly marked verified.
- We do NOT fabricate a confidence value when the framework didn't emit one. Calibration reports "not evaluable."
- We do NOT report gross-only numbers. Every Markdown report is net of cost.
- We do NOT execute trades, recommend buying or selling, connect to brokerages, or claim alpha for any strategy.

---

## Attribution

Methods catalog maintained by Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong, advised by Prof. Siu-Ming Yiu. ORCID: 0009-0000-2388-1072. Apache-2.0.
