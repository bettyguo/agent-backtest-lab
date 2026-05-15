"""Tests for the BY (Benjamini-Yekutieli) FDR variant."""
from __future__ import annotations

import numpy as np
import pytest

from abl.multipletest.bh import benjamini_hochberg


def test_by_more_conservative_than_bh():
    """BY divides the threshold by c(m) = sum 1/k -> fewer rejections."""
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    bh = benjamini_hochberg(pvals, alpha=0.05, method="bh")
    by = benjamini_hochberg(pvals, alpha=0.05, method="by")
    assert by.n_rejected <= bh.n_rejected


def test_by_textbook_example():
    """For 8 p-values, c(8) = 1+1/2+1/3+1/4+1/5+1/6+1/7+1/8 ≈ 2.7179.
    BY threshold at k=2: (2 / (8 * 2.7179)) * 0.05 ≈ 0.0046. p_(2)=0.008 > 0.0046, so reject 0.
    BY threshold at k=1: (1 / (8 * 2.7179)) * 0.05 ≈ 0.0023. p_(1)=0.001 < 0.0023, so reject 1.
    """
    pvals = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205])
    res = benjamini_hochberg(pvals, alpha=0.05, method="by")
    assert res.n_rejected == 1
    assert res.rejected[0]
    assert not res.rejected[1:].any()


def test_by_invalid_method():
    with pytest.raises(ValueError):
        benjamini_hochberg(np.array([0.1, 0.5]), alpha=0.05, method="bonferroni")


def test_by_controls_fdr_under_dependence():
    """Under positively-correlated p-values, BY still controls FDR at alpha while BH may
    inflate. We construct correlated p-values via a single shared shock and check that
    BY's empirical FDR is bounded."""
    rng = np.random.default_rng(20260514)
    n_rep = 500
    m = 100
    n_alt = 20
    bh_fdrs: list[float] = []
    by_fdrs: list[float] = []
    for _ in range(n_rep):
        # Common shock + idiosyncratic; map to p-values via normal CDF
        shock = rng.standard_normal()
        idio = rng.standard_normal(m)
        z = shock + idio
        from scipy.stats import norm

        p = 1.0 - norm.cdf(z)  # right-tailed
        # First n_alt are alts: push their z to be positive on average
        p[:n_alt] = 1.0 - norm.cdf(z[:n_alt] + 2.5)  # shifted alternatives
        is_null = np.concatenate([np.zeros(n_alt, dtype=bool), np.ones(m - n_alt, dtype=bool)])
        bh = benjamini_hochberg(p, alpha=0.05, method="bh")
        by = benjamini_hochberg(p, alpha=0.05, method="by")
        bh_fdrs.append(((bh.rejected & is_null).sum() / max(1, bh.n_rejected)) if bh.n_rejected else 0.0)
        by_fdrs.append(((by.rejected & is_null).sum() / max(1, by.n_rejected)) if by.n_rejected else 0.0)
    # BY must control FDR <= alpha (within Monte Carlo slack) even here
    emp_by = float(np.mean(by_fdrs))
    assert emp_by <= 0.05 + 0.04, f"BY empirical FDR = {emp_by:.4f} exceeds 0.05 + slack"
