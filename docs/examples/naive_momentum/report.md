# Scorecard — naive_momentum_20

> This project is a research and evaluation tool. It is not financial,
> investment, or trading advice. It does not execute trades or connect to
> brokerages. It exists to help researchers and practitioners rigorously
> measure how trading-agent frameworks actually perform — including, and
> especially, when they perform badly. Backtest results are not predictive
> of live performance. You are responsible for any decisions you make.

🟢 No critical flags raised. *Note: the absence of a flag is not a guarantee.*

- **Universe:** bundled_synthetic (3 tickers)
- **Window:** 2020-01-06 → 2024-12-31 (1259 trading days)
- **Cost model:** constant_bps_5
- **Annualization:** 252
- **Generated:** 2026-05-15T09:25:25.239102+00:00

## Net-of-cost performance

| Metric | Value |
|---|---|
| Net total return | +13.0526% |
| Net annualized return | +2.4860% |
| Sharpe (annualized) | +0.333 |
| Sharpe 95% CI (annualized)¹ — HAC (HAC, q=7, η=0.97) | [-0.530, +1.196] |
| PSR (vs SR=0) | 0.7719 |
| DSR | not computed (n_trials_reported=1, var_trial_sharpe=0) |
| Sortino (annualized, MAR=0) | +0.490 |
| Information Ratio (vs buy-and-hold, ann.) | -0.854 |

## Drawdown

| Metric | Value |
|---|---|
| Max drawdown | -12.5898% |
| Longest underwater stretch | 398 trading days |
| Calmar ratio (ann. ret / |max DD|) | +0.197 |

## Per-ticker breakdown

_A real edge spreads across the universe; a fragile edge concentrates in a single ticker. Hit-rate is over non-FLAT calls only._

| Ticker | Decisions | Non-FLAT | Hit rate | Mean contribution | Sharpe (ann.) |
|---|---|---|---|---|---|
| SYN-A | 1259 | 640 | 0.500 | -0.0000% | -0.002 |
| SYN-B | 1259 | 676 | 0.496 | +0.0084% | +0.411 |
| SYN-C | 1259 | 690 | 0.491 | +0.0078% | +0.403 |

## Baselines (same window, same cost model)

| Baseline | Net total return | Sharpe (ann.) |
|---|---|---|
| buy_and_hold | +55.6254% | +0.829 |
| naive_momentum_20 | +13.0526% | +0.333 |
| random | -51.1864% | -1.425 |

## Calibration

- Not evaluable: the adapter did not emit a confidence. This is honest — we do not fabricate a probability where the framework provided none.

## Flags

- No flags raised.

---

¹ The Sharpe 95% CI uses the HAC / Newey-West SE (Lo 2002 with Bartlett kernel, q=7 lags, autocorrelation correction factor η=0.97). η > 1 indicates the IID CI would have been overconfident; η < 1 the opposite. Use `build_scorecard(..., sharpe_ci_method='iid')` for the IID-Gaussian variant.

_agent-backtest-lab is a research tool. Not financial advice. Not a trading system. Backtests don't predict the future._

_Built by Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong, advised by Prof. Siu-Ming Yiu. ORCID: 0009-0000-2388-1072. Apache-2.0._
