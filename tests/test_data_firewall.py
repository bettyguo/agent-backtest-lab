"""Tests for the leakage firewall — the hard guarantee at the core of the library."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from abl.data.firewall import Firewall, FirewallViolation
from abl.data.pit_view import PITView


def test_firewall_blocks_future_bar(synth_store):
    fw = Firewall(strict=True)
    pit = PITView(synth_store, as_of=date(2020, 6, 1), firewall=fw)
    # Asking for bars is fine — PITView itself clips to as_of.
    bars = pit.bars("SYN-A", lookback_days=10)
    assert (bars.index <= pd.Timestamp(date(2020, 6, 1))).all()
    assert len(fw.events) == 1
    assert fw.events[0].allowed is True


def test_firewall_direct_check_blocks(synth_store):
    fw = Firewall(strict=True)
    with pytest.raises(FirewallViolation):
        fw.check(
            as_of=date(2020, 6, 1),
            ticker="SYN-A",
            kind="bars",
            requested_min_date=date(2020, 5, 1),
            requested_max_date=date(2020, 6, 30),  # past as_of -> violation
        )
    assert len(fw.violations()) == 1


def test_firewall_nonstrict_logs_but_does_not_raise(synth_store):
    fw = Firewall(strict=False)
    fw.check(
        as_of=date(2020, 6, 1),
        ticker="SYN-A",
        kind="bars",
        requested_min_date=date(2020, 5, 1),
        requested_max_date=date(2020, 6, 30),
    )
    assert len(fw.violations()) == 1


def test_pitview_corp_actions_filtered(synth_store):
    # Inject a future corporate action and confirm PITView refuses to surface it.
    synth_store.add_corporate_action("SYN-A", ex_date=date(2020, 9, 1), kind="split", ratio=2.0)
    fw = Firewall(strict=True)
    pit = PITView(synth_store, as_of=date(2020, 6, 1), firewall=fw)
    actions = pit.corporate_actions("SYN-A")
    assert actions.empty  # the future action is filtered out


def test_pitview_corp_actions_after_ex_date_visible(synth_store):
    synth_store.add_corporate_action("SYN-A", ex_date=date(2020, 4, 1), kind="split", ratio=2.0)
    fw = Firewall(strict=True)
    pit = PITView(synth_store, as_of=date(2020, 6, 1), firewall=fw)
    actions = pit.corporate_actions("SYN-A")
    assert len(actions) == 1
    assert actions.iloc[0]["kind"] == "split"


def test_pitview_bars_apply_only_known_actions(synth_store):
    # Add a 2-for-1 split after the PIT cutoff; PITView should NOT apply it.
    synth_store.add_corporate_action("SYN-A", ex_date=date(2020, 9, 1), kind="split", ratio=2.0)
    fw = Firewall(strict=True)
    pit_before = PITView(synth_store, as_of=date(2020, 6, 1), firewall=fw)
    bars_before = pit_before.bars("SYN-A", lookback_days=5)
    raw = synth_store.raw_bars("SYN-A").loc[bars_before.index]
    # No split known yet: adjusted should equal raw close.
    assert bars_before["close"].iloc[-1] == pytest.approx(raw["close"].iloc[-1])

    fw2 = Firewall(strict=True)
    pit_after = PITView(synth_store, as_of=date(2020, 9, 15), firewall=fw2)
    bars_after = pit_after.bars("SYN-A", lookback_days=5)
    # Now the split IS known; bars before ex_date should be half of raw close.
    # Pick a date strictly before ex_date and check.
    pre_ex = bars_after.index[bars_after.index < pd.Timestamp(date(2020, 9, 1))]
    if len(pre_ex) > 0:
        raw_pre = synth_store.raw_bars("SYN-A").loc[pre_ex]
        assert bars_after.loc[pre_ex[-1], "close"] == pytest.approx(raw_pre.loc[pre_ex[-1], "close"] / 2.0)
