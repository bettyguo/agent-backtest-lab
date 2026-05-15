"""Tests for Bonferroni-Holm and Romano-Wolf stepwise procedures."""
from __future__ import annotations

import numpy as np
import pytest

from abl.multipletest.stepwise import bonferroni_holm, romano_wolf_stepwise


def test_holm_textbook_example():
    """For m=4 p-values [0.001, 0.012, 0.024, 0.060] and alpha=0.05:
    thresholds α/(m-k+1) = [0.0125, 0.01667, 0.025, 0.05].
    p_(1)=0.001 <= 0.0125 -> reject. p_(2)=0.012 <= 0.01667 -> reject.
    p_(3)=0.024 <= 0.025 -> reject. p_(4)=0.060 > 0.05 -> stop.
    So 3 rejections."""
    pvals = np.array([0.001, 0.012, 0.024, 0.060])
    res = bonferroni_holm(pvals, alpha=0.05)
    assert res.n_rejected == 3
    assert res.rejected[0] and res.rejected[1] and res.rejected[2]
    assert not res.rejected[3]


def test_holm_no_rejections_when_all_large():
    pvals = np.array([0.5, 0.6, 0.7, 0.8])
    res = bonferroni_holm(pvals, alpha=0.05)
    assert res.n_rejected == 0


def test_holm_at_most_as_aggressive_as_bh():
    """Holm controls FWER; BH controls FDR. Holm rejects at most as many."""
    from abl.multipletest.bh import benjamini_hochberg
    rng = np.random.default_rng(0)
    for _ in range(20):
        pvals = np.concatenate([
            rng.beta(0.1, 1.0, size=20),
            rng.uniform(0, 1, size=80),
        ])
        holm = bonferroni_holm(pvals, alpha=0.05)
        bh = benjamini_hochberg(pvals, alpha=0.05)
        assert holm.n_rejected <= bh.n_rejected


def test_holm_invalid_inputs():
    with pytest.raises(ValueError):
        bonferroni_holm(np.array([-0.1, 0.5]))
    with pytest.raises(ValueError):
        bonferroni_holm(np.array([0.1, 0.5]), alpha=1.5)


def test_romano_wolf_no_rejection_on_noise():
    rng = np.random.default_rng(20260514)
    N, T = 10, 500
    S = rng.normal(0.0, 0.01, size=(N, T))
    B = rng.normal(0.0, 0.01, size=T)
    res = romano_wolf_stepwise(S, B, alpha=0.05, n_bootstrap=300, rng=rng)
    # Under the null, expected number of rejections at FWER 0.05 is small
    assert res.n_rejected <= 2, f"got {res.n_rejected} false rejections on pure noise"


def test_romano_wolf_rejects_clear_winner():
    rng = np.random.default_rng(20260514)
    N, T = 5, 600
    S = rng.normal(0.0, 0.01, size=(N, T))
    B = rng.normal(0.0, 0.01, size=T)
    # Strategy 0 has a real edge
    S[0] += 0.005
    res = romano_wolf_stepwise(S, B, alpha=0.05, n_bootstrap=400, rng=rng)
    assert res.rejected[0]


def test_romano_wolf_invalid_shapes():
    with pytest.raises(ValueError):
        romano_wolf_stepwise(np.zeros(10), np.zeros(10))  # 1-D strategies
    with pytest.raises(ValueError):
        romano_wolf_stepwise(np.zeros((3, 10)), np.zeros(9))  # length mismatch
