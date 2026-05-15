# Disclaimers

This file is the single source of truth for the disclaimer text emitted by `agent-backtest-lab`. The block below is reproduced at the top of every emitted scorecard report, the top of the README, and the top of `abl methods`. The one-line variant prints on every CLI invocation. Neither is removable via flags.

## DISCLAIMER_BLOCK

```
This project is a research and evaluation tool. It is not financial,
investment, or trading advice. It does not execute trades or connect to
brokerages. It exists to help researchers and practitioners rigorously
measure how trading-agent frameworks actually perform — including, and
especially, when they perform badly. Backtest results are not predictive
of live performance. You are responsible for any decisions you make.
```

## DISCLAIMER_LINE

```
agent-backtest-lab is a research tool. Not financial advice. Not a trading system. Backtests don't predict the future.
```

## What this tool will never do

- Place a trade.
- Connect to a brokerage.
- Tell you to buy, sell, or hold any specific asset.
- Claim that any specific strategy has alpha.
- Bundle a "recommendation engine."
- Charge you (the library is free and Apache-2.0 licensed).

If you spot text that drifts toward any of the above, open an issue or PR.

## Backtest-result interpretation rules baked into the library

Every scorecard prints results subject to these rules — they are not optional:

1. **Net of cost, always.** No gross-only numbers are emitted by any public function.
2. **Confidence intervals.** Every Sharpe is reported with a CI.
3. **Multiple-testing correction.** When the user reports `n_trials > 1` (e.g. a prompt or hyperparameter scan), the Deflated Sharpe Ratio is computed and shown.
4. **Buy-and-hold and naive baselines.** Always reported alongside the evaluated adapter.
5. **Leakage and overfitting flags.** Always computed and rendered. A non-empty flag set turns the report banner amber or red.
6. **Survivorship and PIT caveats.** A `universe_flags` section warns when the universe cannot be verified survivorship-free or when the data source is known to be retrospectively adjusted.

## Maintainer

Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong, advised by Prof. Siu-Ming Yiu. ORCID: 0009-0000-2388-1072.
