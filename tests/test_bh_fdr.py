"""Tests for Benjamini-Hochberg FDR control."""
from __future__ import annotations

import numpy as np
import pytest

from abl.multipletest.bh import benjamini_hochberg


def test_bh_no_rejections_when_all_large():
    pvals = np.array([0.4, 0.5, 0.6, 0.7])
    res = benjamini_hochberg(pvals, alpha=0.05)
    assert res.n_rejected == 0
    assert not res.rejected.any()


def test_bh_rejects_all_when_all_tiny():
    pvals = np.array([1e-6, 1e-7, 1e-8, 1e-9])
    res = benjamini_hochberg(pvals, alpha=0.05)
    assert res.n_rejected == 4
    assert res.rejected.all()


def test_bh_textbook_example():
    """Standard textbook example: pvals [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205].

    With alpha=0.05 and m=8, the thresholds (k/m)*alpha are:
    [0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.04375, 0.05].
    Largest k with p_(k) <= (k/m)*alpha: p_(1)=0.001 <= 0.00625 ✓, p_(2)=0.008 <= 0.0125 ✓,
    p_(3)=0.039 vs 0.01875 ✗, p_(4)=0.041 vs 0.025 ✗, p_(5)=0.042 vs 0.03125 ✗.
    So k=2 — reject the two smallest p-values.
    """
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    res = benjamini_hochberg(pvals, alpha=0.05)
    assert res.n_rejected == 2
    assert res.rejected[0]
    assert res.rejected[1]
    assert not res.rejected[2:].any()


def test_bh_monotone_adjusted_pvals():
    """BH q-values must be monotone non-decreasing in the original p-value rank."""
    rng = np.random.default_rng(42)
    pvals = rng.uniform(0, 1, size=200)
    res = benjamini_hochberg(pvals, alpha=0.1)
    order = np.argsort(pvals)
    qs_sorted = res.pvals_adjusted[order]
    assert (np.diff(qs_sorted) >= -1e-12).all()


def test_bh_controls_fdr_under_independence():
    """Monte Carlo check: with 20% true alternatives, BH at α=0.05 controls empirical FDR.

    1000 reps × 200 hypotheses each. We construct pvals: 40 from a beta(0.1, 1) distribution
    (concentrated near 0 — true H1 with strong signal) and 160 uniform (true H0). BH at α=0.05
    should achieve empirical FDR ≤ 0.05 + a small Monte Carlo slack.
    """
    rng = np.random.default_rng(20260514)
    n_rep = 1000
    m = 200
    n_alt = 40
    fdrs: list[float] = []
    for _ in range(n_rep):
        p_alt = rng.beta(0.1, 1.0, size=n_alt)
        p_null = rng.uniform(0, 1, size=m - n_alt)
        pvals = np.concatenate([p_alt, p_null])
        is_null = np.concatenate([np.zeros(n_alt, dtype=bool), np.ones(m - n_alt, dtype=bool)])
        res = benjamini_hochberg(pvals, alpha=0.05)
        if res.n_rejected == 0:
            fdrs.append(0.0)
        else:
            false_disc = (res.rejected & is_null).sum()
            fdrs.append(false_disc / res.n_rejected)
    emp_fdr = float(np.mean(fdrs))
    # Standard error of the mean of a [0,1] variable across 1000 reps is at most 0.5/√1000 ≈ 0.016.
    # Allow 2σ slack.
    assert emp_fdr <= 0.05 + 0.04, f"empirical FDR = {emp_fdr:.4f} exceeds 0.05 + slack"


def test_bh_invalid_inputs():
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([-0.1, 0.5]), alpha=0.05)
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, 1.1]), alpha=0.05)
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, 0.5]), alpha=1.5)
