"""Adapter for AI4Finance-Foundation/FinGPT.

FinGPT is primarily an NLP-fine-tuned family of LLMs for finance. It does not ship a
single "decide a direction on (ticker, date)" callable the way TradingAgents does; users
typically run sentiment / forecasting heads via the `fingpt` package or HF transformers.

This adapter wraps a USER-PROVIDED callable that returns a sentiment score in [-1, +1].
We map sentiment to direction with a configurable threshold:

  score >  threshold   -> LONG
  score < -threshold   -> SHORT
  otherwise            -> FLAT

We pass `|score|` as the confidence in [0, 1] (clipped). This keeps the adapter honest
about what FinGPT actually emits — a sentiment, not a calibrated probability — while
plugging cleanly into our calibration analysis.

The `fingpt` framework itself is intentionally an OPTIONAL dependency. The adapter
construction works WITHOUT it installed because the user supplies the callable. If a
caller wants to lazy-import FinGPT internals, they do so in their callable.

References
----------
- AI4Finance-Foundation/FinGPT — https://github.com/AI4Finance-Foundation/FinGPT
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from abl.types import Decision, PITViewProtocol


@dataclass
class FinGPTAdapter:
    """Wrap a user-supplied (ticker, as_of, pit_data) -> sentiment_in_[-1,+1] callable.

    Parameters
    ----------
    sentiment_fn : callable
        Returns a float in [-1, +1] for each (ticker, as_of). The caller is responsible
        for routing the inputs through FinGPT-the-framework or any other sentiment model.
    threshold : float, default 0.2
        Magnitude below which the call is FLAT. 0.2 is conservative; tune to your model.
    name : str, default "fingpt_sentiment"
    """

    sentiment_fn: Callable[[str, date, PITViewProtocol], float]
    threshold: float = 0.2
    name: str = "fingpt_sentiment"

    def __post_init__(self) -> None:
        if not (0.0 <= self.threshold < 1.0):
            raise ValueError(f"threshold must be in [0, 1), got {self.threshold}")

    def predict(self, ticker: str, as_of: date, pit_data: PITViewProtocol) -> Decision:
        score = float(self.sentiment_fn(ticker, as_of, pit_data))
        if score != score:  # NaN
            return Decision(direction="FLAT", confidence=None, raw={"sentiment": score})
        # Clip the absolute value into [0, 1] as the confidence.
        conf = max(0.0, min(1.0, abs(score)))
        if score > self.threshold:
            direction = "LONG"
        elif score < -self.threshold:
            direction = "SHORT"
        else:
            direction = "FLAT"
        return Decision(direction=direction, confidence=conf, raw={"sentiment": score})
