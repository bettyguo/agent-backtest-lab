"""Tests for combinatorial purged K-fold CV (López de Prado Ch. 12)."""
from __future__ import annotations

from math import comb

import numpy as np
import pytest

from abl.backtest.cv import (
    combinatorial_purged_kfold_splits,
    n_recoverable_paths,
)


def test_combinatorial_split_count():
    splits = combinatorial_purged_kfold_splits(
        n=500, n_groups=6, n_test_groups=2, label_window=2, embargo_pct=0.01
    )
    assert len(splits) == comb(6, 2) == 15


def test_n_recoverable_paths():
    # For n_groups=6, n_test_groups=2 the recoverable paths = C(5,1) = 5
    assert n_recoverable_paths(6, 2) == 5
    assert n_recoverable_paths(10, 2) == 9
    assert n_recoverable_paths(8, 3) == comb(7, 2)


def test_combinatorial_train_test_disjoint():
    splits = combinatorial_purged_kfold_splits(
        n=400, n_groups=8, n_test_groups=3, label_window=3, embargo_pct=0.02
    )
    for s in splits:
        assert set(s.train_idx).isdisjoint(set(s.test_idx))


def test_combinatorial_purge_respected():
    n = 600
    label = 5
    embargo_pct = 0.02
    embargo = int(round(embargo_pct * n))
    splits = combinatorial_purged_kfold_splits(
        n=n, n_groups=6, n_test_groups=2, label_window=label, embargo_pct=embargo_pct
    )
    bounds = np.linspace(0, n, 7, dtype=int)
    for s in splits:
        for g in s.test_groups:
            g_lo, g_hi = bounds[g], bounds[g + 1]
            purge_lo = max(0, g_lo - label)
            purge_hi = min(n, g_hi + label + embargo)
            # No training index can lie inside the purge window
            mask = (s.train_idx >= purge_lo) & (s.train_idx < purge_hi)
            assert not mask.any(), f"train idx inside purge for group {g}"


def test_combinatorial_invalid_inputs():
    with pytest.raises(ValueError):
        combinatorial_purged_kfold_splits(n=100, n_groups=1, n_test_groups=1)
    with pytest.raises(ValueError):
        combinatorial_purged_kfold_splits(n=100, n_groups=5, n_test_groups=5)
    with pytest.raises(ValueError):
        combinatorial_purged_kfold_splits(n=100, n_groups=5, n_test_groups=0)
