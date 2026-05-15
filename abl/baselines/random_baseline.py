"""Random baseline: independent uniform LONG/FLAT/SHORT call per day, seeded.

Useful as a sanity check — if an agent does not statistically beat random after costs,
something is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from abl.adapters.plain_strategy import PlainStrategyAdapter
from abl.config import DEFAULT_SEED
from abl.types import Decision, Direction, PITViewProtocol

_DIRECTIONS: tuple[Direction, ...] = ("LONG", "FLAT", "SHORT")


@dataclass
class _Random:
    seed: int = DEFAULT_SEED

    def __call__(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        # Per-call deterministic seeding so the same (ticker, as_of) always yields the same call.
        h = hash((self.seed, ticker, as_of.isoformat())) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        idx = int(rng.integers(0, 3))
        return Decision(direction=_DIRECTIONS[idx], confidence=None, raw={"baseline": "random"})


def random_baseline_adapter(seed: int = DEFAULT_SEED) -> PlainStrategyAdapter:
    return PlainStrategyAdapter(fn=_Random(seed=seed), name="random")
