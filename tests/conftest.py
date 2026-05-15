"""Shared pytest fixtures."""
from __future__ import annotations

from datetime import date

import pytest

from abl.data.fixture_synth import generate_synthetic_fixture
from abl.data.loaders import DataStore
from abl.types import Universe, Window


@pytest.fixture
def synth_store() -> DataStore:
    fix = generate_synthetic_fixture(seed=20260514)
    return DataStore.from_synthetic(fix)


@pytest.fixture
def synth_universe() -> Universe:
    return Universe(tickers=("SYN-A", "SYN-B", "SYN-C"), name="synthetic_3", survivorship_verified=True)


@pytest.fixture
def synth_window() -> Window:
    # ~1 trading year inside the default 1260-day fixture
    return Window(start=date(2020, 1, 6), end=date(2022, 1, 5))
