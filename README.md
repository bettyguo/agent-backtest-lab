# agent-backtest-lab

> Before you trust an LLM trading agent, audit it. A statistical-rigor harness — look-ahead-leak detection, transaction-cost modeling, multiple-testing correction, calibration, and reward-hacking detection — for trading-agent frameworks.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/ci-passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Fixture tests](https://img.shields.io/badge/fixture%20tests-passing-brightgreen.svg)](tests/)
[![PyPI-ready](https://img.shields.io/badge/pypi--ready-yes-brightgreen.svg)](pyproject.toml)

> ⚠️ **This is a research and evaluation tool. It is not financial, investment, or trading advice. It does not execute trades or connect to brokerages.** It exists to help researchers and practitioners rigorously measure how trading-agent frameworks actually perform — including, and especially, when they perform badly. Backtest results are not predictive of live performance. You are responsible for any decisions you make.

(Status: in active development. This file will be expanded in Phase 5 with the screencast GIF, full integration example, and launch material.)

## The problem

A trading agent shows +40% in a backtest. Was the agent good — or did it quietly see tomorrow's prices through retroactively-adjusted data, or get the best of 50 untracked prompt-tuning attempts, or earn returns no real trader could net of costs? This is the question `agent-backtest-lab` answers.

## What it does

- **Walk-forward backtesting** with strict point-in-time data discipline.
- **Leakage firewall** that hard-refuses any access to data after `as_of`.
- **Post-hoc leakage detection** (impossible-accuracy and corporate-action-proximity).
- **Transaction-cost models** — constant-bps default; Almgren-style square-root impact optional.
- **Multiple-testing correction** — Benjamini-Hochberg FDR; Probabilistic and Deflated Sharpe.
- **Backtest-overfitting probability** via Combinatorially Symmetric Cross-Validation.
- **Calibration analysis** — reliability diagrams, ECE, and split conformal sets on directional calls.
- **Honest scorecard** — net of cost, with CIs, after multiple-testing correction, against buy-and-hold and naive baselines, with explicit leakage and overfitting flags.

## What it does NOT do — deliberately

- Place trades. Connect to brokerages. Recommend buying or selling anything. Claim alpha for any strategy.
- Touch live data. The whole library is offline-first; the bundled synthetic fixture covers quickstart and CI.

## Attribution

Built by Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong, advised by Prof. Siu-Ming Yiu. ORCID: 0009-0000-2388-1072. Apache-2.0.

## Companion to TradingAgents

This library is designed as a companion to LLM trading-agent frameworks — primarily [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents). Their job is to *generate* decisions; ours is to *audit* them.
