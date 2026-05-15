"""Tests for the CSCV-based Probability of Backtest Overfitting (PBO)."""
from __future__ import annotations

import numpy as np

from abl.overfitting.cscv import probability_of_backtest_overfitting


def test_pbo_pure_noise_near_half():
    """N=20 IID-noise strategies × T=512 periods.

    Under the null (no strategy has any edge), the IS-best is OOS-rank-random. PBO ≈ 0.5.
    """
    rng = np.random.default_rng(20260514)
    N, T = 20, 512
    returns = rng.standard_normal((N, T)) * 0.01
    out = probability_of_backtest_overfitting(returns, s_blocks=16, rng=rng)
    assert 0.35 <= out["pbo"] <= 0.65, f"expected ~0.5, got {out['pbo']:.3f}"


def test_pbo_low_when_one_strategy_dominates():
    """One strategy with a real positive mean dominates noise -> PBO should be low (close to 0).

    The dominant strategy will be the IS-best in nearly every partition AND OOS-rank-top, so
    its OOS logit-rank > 0 → not flagged as overfit.
    """
    rng = np.random.default_rng(20260514)
    N, T = 20, 512
    returns = rng.standard_normal((N, T)) * 0.01
    returns[0] += 0.005  # add a real edge to strategy 0
    out = probability_of_backtest_overfitting(returns, s_blocks=16, rng=rng)
    assert out["pbo"] <= 0.1, f"expected low PBO, got {out['pbo']:.3f}"


def test_pbo_high_when_winners_are_overfit():
    """A 'parameter scan' where each strategy is tuned to a specific in-sample block.

    Build N strategies each of which spikes positively in one specific block and is noisy
    elsewhere. The IS-best in any partition is the one whose spike-block was in-sample —
    but OOS that strategy reverts to noise. PBO should be high (well above 0.5).
    """
    rng = np.random.default_rng(20260514)
    N, T = 16, 512
    s_blocks = 16
    block_size = T // s_blocks
    returns = rng.standard_normal((N, T)) * 0.01
    # Each strategy i has a positive bump in block i only
    for i in range(N):
        lo = i * block_size
        hi = lo + block_size
        returns[i, lo:hi] += 0.05
    out = probability_of_backtest_overfitting(returns, s_blocks=s_blocks, rng=rng)
    assert out["pbo"] >= 0.5, f"expected high PBO, got {out['pbo']:.3f}"
    # And the overfitting flag should fire.
    assert out["flag"] is not None


def test_pbo_overfit_fixture_flagged_critical():
    """The deliberately-overfit fixture MUST be flagged. This is the CI gate.

    Construction: 50 strategies, each one is "good" in 8 randomly-chosen blocks and "bad"
    in the other 8. Across the C(16, 8) = 12870 partitions, the in-sample-best strategy is
    reliably one whose "good 8" coincides with the IS choice — and therefore its "bad 8"
    coincides with the OOS choice. This gives the anti-correlated IS/OOS Sharpe pattern that
    is the textbook overfitting signal. PBO should be well above 0.5.
    """
    rng = np.random.default_rng(20260514)
    N, T = 50, 384
    s_blocks = 16
    block_size = T // s_blocks  # 24

    # Each row gets its own random partition of blocks into "good" and "bad".
    returns = rng.standard_normal((N, T)) * 0.001  # tiny noise
    half = s_blocks // 2
    for i in range(N):
        perm = rng.permutation(s_blocks)
        good_blocks = perm[:half]
        bad_blocks = perm[half:]
        for b in good_blocks:
            returns[i, b * block_size:(b + 1) * block_size] += 0.05
        for b in bad_blocks:
            returns[i, b * block_size:(b + 1) * block_size] -= 0.05

    out = probability_of_backtest_overfitting(returns, s_blocks=s_blocks, rng=rng)
    # The PBO must be >= 0.5 and a flag must fire — this is the gate.
    assert out["pbo"] >= 0.5, f"overfit fixture: PBO {out['pbo']:.3f} below 0.5 threshold"
    assert out["flag"] is not None, f"overfit fixture not flagged; pbo={out['pbo']:.3f}"
