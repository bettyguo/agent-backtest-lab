"""Tests for FinGPTAdapter and FinRobotAdapter."""
from __future__ import annotations

from datetime import date

import pytest

from abl.adapters.fingpt import FinGPTAdapter
from abl.adapters.finrobot import FinRobotAdapter


def _fake_pit_view():
    class _PIT:
        def bars(self, *a, **k):
            import pandas as pd
            return pd.DataFrame()
        def corporate_actions(self, *a, **k):
            import pandas as pd
            return pd.DataFrame()
        def as_of(self):
            return date(2024, 1, 1)
    return _PIT()


def test_fingpt_positive_score_long():
    adapter = FinGPTAdapter(sentiment_fn=lambda t, d, p: 0.6, threshold=0.2)
    dec = adapter.predict("AAPL", date(2024, 1, 1), _fake_pit_view())
    assert dec.direction == "LONG"
    assert dec.confidence == pytest.approx(0.6)


def test_fingpt_negative_score_short():
    adapter = FinGPTAdapter(sentiment_fn=lambda t, d, p: -0.8, threshold=0.2)
    dec = adapter.predict("AAPL", date(2024, 1, 1), _fake_pit_view())
    assert dec.direction == "SHORT"
    assert dec.confidence == pytest.approx(0.8)


def test_fingpt_below_threshold_flat():
    adapter = FinGPTAdapter(sentiment_fn=lambda t, d, p: 0.05, threshold=0.2)
    dec = adapter.predict("AAPL", date(2024, 1, 1), _fake_pit_view())
    assert dec.direction == "FLAT"


def test_fingpt_nan_score_flat():
    import math
    adapter = FinGPTAdapter(sentiment_fn=lambda t, d, p: math.nan)
    dec = adapter.predict("AAPL", date(2024, 1, 1), _fake_pit_view())
    assert dec.direction == "FLAT"


def test_fingpt_invalid_threshold():
    with pytest.raises(ValueError):
        FinGPTAdapter(sentiment_fn=lambda *a: 0.0, threshold=1.5)


def test_finrobot_parses_text_decision():
    adapter = FinRobotAdapter(decision_fn=lambda t, d, p: ("Final answer: BUY", 0.7))
    dec = adapter.predict("AAPL", date(2024, 1, 1), _fake_pit_view())
    assert dec.direction == "LONG"
    assert dec.confidence == pytest.approx(0.7)


def test_finrobot_handles_invalid_confidence():
    adapter = FinRobotAdapter(decision_fn=lambda t, d, p: ("SELL", 1.5))
    dec = adapter.predict("AAPL", date(2024, 1, 1), _fake_pit_view())
    assert dec.direction == "SHORT"
    # Out-of-range confidence is dropped, never coerced.
    assert dec.confidence is None


def test_finrobot_handles_none_decision():
    adapter = FinRobotAdapter(decision_fn=lambda t, d, p: (None, None))
    dec = adapter.predict("AAPL", date(2024, 1, 1), _fake_pit_view())
    assert dec.direction == "FLAT"
    assert dec.confidence is None
