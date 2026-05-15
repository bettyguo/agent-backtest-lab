"""Tests for the Reality Check / SPA test."""
from __future__ import annotations

import numpy as np

from abl.multipletest.spa import reality_check_spa


def test_spa_does_not_reject_noise():
    """N IID-noise strategies vs IID-noise benchmark. The 'best' is noise, p-value should be high."""
    rng = np.random.default_rng(20260514)
    N, T = 10, 500
    S = rng.normal(0.0, 0.01, size=(N, T))
    B = rng.normal(0.0, 0.01, size=T)
    res = reality_check_spa(S, B, n_bootstrap=500, rng=rng)
    assert res.pvalue >= 0.10, f"expected high p-value, got {res.pvalue:.3f}"


def test_spa_rejects_clear_winner():
    """One strategy beats benchmark substantially -> p-value should be low."""
    rng = np.random.default_rng(20260514)
    N, T = 5, 500
    S = rng.normal(0.0, 0.01, size=(N, T))
    B = rng.normal(0.0, 0.01, size=T)
    S[0] += 0.005  # ~0.5%/day excess return, robust signal
    res = reality_check_spa(S, B, n_bootstrap=500, rng=rng)
    assert res.pvalue <= 0.05, f"expected low p-value, got {res.pvalue:.3f}"
    assert res.best_strategy_idx == 0


def test_spa_result_shape():
    rng = np.random.default_rng(0)
    S = rng.normal(0.0, 0.01, size=(3, 200))
    B = rng.normal(0.0, 0.01, size=200)
    res = reality_check_spa(S, B, n_bootstrap=200, rng=rng)
    assert 0.0 <= res.pvalue <= 1.0
    assert res.n_strategies == 3
    assert res.n_periods == 200
    assert res.avg_block_len > 0
