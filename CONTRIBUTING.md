# Contributing

Thanks for considering a contribution. `agent-backtest-lab` is a **statistical-audit harness for trading-agent frameworks**. Its credibility depends on every statistical method shipping correct. The contribution rules below exist to protect that.

## Ground rules

1. **No financial advice, anywhere.** No `buy`/`sell` language in code, docs, or commit messages. The library audits decisions; it does not make them. Pull requests that drift toward "use this to trade" will be sent back.
2. **No execution paths.** The library never connects to a broker, never places an order, never produces a recommendation. Don't add one.
3. **Every new statistical method ships a known-answer fixture test.** A method without a fixture test is a vector for false confidence — exactly what this library exists to destroy. The fixture must compute a result for which the right answer is known (analytically, from a published example, or by Monte Carlo with explicit tolerances) and assert against it.
4. **Cite your sources.** Every new method gets its docstring updated with a precise citation: author, year, title, venue, URL. Mark anything you couldn't verify as `[verify]` — that flag is honest, not embarrassing.
5. **No file > 500 lines.** Reviewability is a feature. If you need more, split the module.
6. **The disclaimer block, the leakage-fixture test, and the overfitting-fixture test are CI gates.** They are not removable.

## Setup

```bash
git clone https://github.com/bettyguo/agent-backtest-lab
cd agent-backtest-lab
pip install -e ".[dev]"
pytest
```

## Adding a statistical method — checklist

- [ ] Implementation in the right submodule (`abl/multipletest/`, `abl/calibration/`, ...).
- [ ] Docstring: formal statement, assumptions, citation (with URL).
- [ ] Known-answer fixture test in `tests/`.
- [ ] Method appears in `docs/METHODS.md` with the same citation.
- [ ] If the method has any place in the user-facing scorecard, the scorecard fields and the report renderer are updated.
- [ ] `ruff` clean, `pytest` green on 3.10 / 3.11 / 3.12.

## Adding a framework adapter

- [ ] Subclass / satisfy the `StrategyAdapter` protocol in `abl/adapters/`.
- [ ] The framework must be an **optional** dependency. Importing the adapter without the framework installed must raise a clear `ImportError` at construction time, not at module load.
- [ ] An `examples/` entry showing the integration. Examples that require API keys or paid services are illustrative, not CI-tested.
- [ ] A test (with the framework mocked) for the adapter's translation of framework output into `Decision`.

## Bug reports

Open an issue. Include: Python version, `pip freeze`, a minimal reproducer, and the scorecard output (if any). For suspected statistical-correctness bugs, include the inputs and the expected vs. observed result.

## Code style

- `ruff` for lint and format.
- `from __future__ import annotations` at the top of every module.
- Type hints on every public function. We don't (yet) enforce `mypy --strict`, but submissions that type-check cleanly under `mypy` are appreciated.
- No emojis in source. No marketing language in docstrings.

## License

By contributing you agree your work is licensed under Apache-2.0.

## Attribution

Maintainer: Betty Guo (Dongxin Guo / 郭东欣), PhD candidate, University of Hong Kong, advised by Prof. Siu-Ming Yiu. ORCID: 0009-0000-2388-1072.
