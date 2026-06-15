"""Tests for walk-forward evaluation and the no-look-ahead property in OOS."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import BacktestConfig
from walk_forward import run_walk_forward


def _synth_panel(n_assets=10, n_days=900, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-01", periods=n_days)
    rets = rng.normal(0.0003, 0.011, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_walk_forward_produces_folds():
    prices = _synth_panel()

    def equal_weight(p):
        n = p.shape[1]
        return pd.DataFrame(1.0 / n, index=p.index, columns=p.columns)

    m = run_walk_forward(prices, equal_weight, train_months=24, test_months=1)
    assert m["n_folds"] > 0
    assert m["n_days"] > 0
    assert "sharpe" in m


def test_walk_forward_oos_excludes_train_days():
    """OOS day count must be far smaller than the full sample: only test windows
    are counted, not the warm-up train history.
    """
    prices = _synth_panel(n_days=1200)

    def equal_weight(p):
        n = p.shape[1]
        return pd.DataFrame(1.0 / n, index=p.index, columns=p.columns)

    m = run_walk_forward(prices, equal_weight, train_months=24, test_months=1)
    # OOS days should be roughly (total - 24 months warmup). Definitely less than
    # the full series and greater than zero.
    assert 0 < m["n_days"] < len(prices)


def test_walk_forward_lag_holds_inside_folds():
    """The t->t+1 lag must hold inside each OOS fold: a weight that turns on the
    day an asset spikes earns the spike one day later, not the same day.

    We plant a single +20% spike on a known business day inside the OOS region
    of a flat panel. An oracle weight that enters ON the spike day captures zero
    spike under a correct lag; a realistic weight that enters the day before
    captures it. We compare the two through the full walk-forward path.
    """
    n_days = 900
    idx = pd.bdate_range("2016-01-01", periods=n_days)
    # Flat prices except a single +20% jump deep in the OOS region. Anchor the
    # spike by DATE so the test is robust to the walk-forward sub-frame slicing
    # (strategies receive a windowed frame, not the full panel).
    spike_pos = 800
    spike_date = idx[spike_pos]
    day_before = idx[spike_pos - 1]
    px = np.ones(n_days) * 100.0
    for i in range(1, n_days):
        px[i] = px[i - 1] * (1.20 if i == spike_pos else 1.0)
    prices = pd.DataFrame({"A0": px, "A1": [100.0] * n_days}, index=idx)
    cfg = BacktestConfig(tc_bps=0.0, slippage_bps=0.0)

    def realistic(p):
        w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        if day_before in w.index:
            w.loc[day_before, "A0"] = 1.0  # enter day before spike
        return w

    def oracle_same_day(p):
        w = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        if spike_date in w.index:
            w.loc[spike_date, "A0"] = 1.0  # enter on the spike day
        return w

    m_real = run_walk_forward(prices, realistic, cfg=cfg)
    m_oracle = run_walk_forward(prices, oracle_same_day, cfg=cfg)
    # Realistic strategy's OOS stream contains the +20% day; oracle's does not.
    assert m_real["ann_return"] > m_oracle["ann_return"], (
        "same-day-entry oracle matched the realistic entry; the t->t+1 lag "
        "did not hold inside the walk-forward folds"
    )


def test_shift_minus_one_leaks_through_walk_forward():
    """Companion to the backtester documenting test: a .shift(-1) strategy is a
    construction-level look-ahead bug the harness cannot catch, so it leaks even
    through walk-forward. Asserting the leak keeps the failure mode visible.
    """
    prices = _synth_panel(seed=11)

    def peeking(p):
        fwd = p.pct_change(fill_method=None).shift(-1)
        w = (fwd > 0).astype(float)
        gross = w.sum(axis=1).replace(0, 1)
        return w.div(gross, axis=0).fillna(0.0)

    cfg = BacktestConfig(tc_bps=0.0, slippage_bps=0.0)
    peek = run_walk_forward(prices, peeking, cfg=cfg)
    assert peek["sharpe"] > 3.0, "expected the .shift(-1) construction bug to leak"


def test_regime_gap_reduces_or_keeps_fold_count():
    prices = _synth_panel(n_days=1500)

    def equal_weight(p):
        n = p.shape[1]
        return pd.DataFrame(1.0 / n, index=p.index, columns=p.columns)

    base = run_walk_forward(prices, equal_weight, train_test_gap_months=0)
    gapped = run_walk_forward(prices, equal_weight, train_test_gap_months=3)
    # A 3-month gap drops the first part of each 1-month test window, so OOS day
    # count must not exceed the no-gap run.
    assert gapped["n_days"] <= base["n_days"]
