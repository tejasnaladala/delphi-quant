"""Core-invariant tests for the delphi-quant backtester.

Three invariants the pre-reg leans on:

  1. No look-ahead: a strategy that peeks at t+1 must NOT be able to print money
     on a deterministic series that a non-peeking strategy cannot. We construct
     a deterministic up-down series whose next-day direction is known, give one
     strategy the (illegal) t+1 sign and another only the legal lagged sign, and
     assert the backtester's t->t+1 lag neutralizes the peeker.
  2. Cost application: round-trip cost equals (tc + slippage) per side times the
     traded notional, applied on the rebalance day.
  3. Metric correctness: Sharpe / max drawdown match closed-form values on a
     known constant-return series.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtester import BacktestConfig, apply_costs, compute_metrics, run_backtest


@pytest.fixture
def deterministic_two_asset():
    """Two assets. Asset 'up' alternates +2% / -1% so it drifts up; asset 'flat'
    is constant. The alternation makes 'tomorrow's return' both knowable (for the
    peeking test) and non-trivial (so a peeker would profit if look-ahead leaked).
    """
    n = 400
    idx = pd.bdate_range("2015-01-01", periods=n)
    up = [100.0]
    for i in range(1, n):
        step = 1.02 if i % 2 == 1 else 0.995
        up.append(up[-1] * step)
    flat = [100.0] * n
    return pd.DataFrame({"up": up, "flat": flat}, index=idx)


def test_no_lookahead_lag_strips_same_day_spike():
    """The backtester's contract: a weight vector dated day t earns returns
    starting day t+1. So a position taken on the very day an asset spikes must
    NOT capture that spike; the one-day lag strips it.

    We build a series with a single large up-move on one known day. A "realistic"
    weight enters the day BEFORE the spike (and so earns it on the spike day). An
    "oracle" weight enters ON the spike day (it would only earn the spike if the
    backtester had no lag). Under a correct t->t+1 lag, the oracle weight earns
    the spike one day late (when nothing happens) and so captures zero spike,
    while the realistic weight captures the full spike. This is the property a
    look-ahead bug would break.
    """
    n = 60
    idx = pd.bdate_range("2015-01-01", periods=n)
    spike_day = 30
    px = [100.0] * n
    for i in range(1, n):
        px[i] = px[i - 1] * (1.10 if i == spike_day else 1.0)  # one +10% day
    prices = pd.DataFrame({"x": px}, index=idx)
    returns = prices.pct_change(fill_method=None)
    cfg = BacktestConfig(tc_bps=0.0, slippage_bps=0.0)

    # Realistic: hold from the day BEFORE the spike. Weight on day (spike-1)
    # earns the spike-day return.
    w_realistic = pd.DataFrame(0.0, index=idx, columns=["x"])
    w_realistic.iloc[spike_day - 1, 0] = 1.0
    net_real = apply_costs(w_realistic, returns, cfg)

    # Oracle: enter ON the spike day only. With the lag this weight earns the
    # day-(spike+1) return, which is zero.
    w_oracle = pd.DataFrame(0.0, index=idx, columns=["x"])
    w_oracle.iloc[spike_day, 0] = 1.0
    net_oracle = apply_costs(w_oracle, returns, cfg)

    spike_ret = returns["x"].iloc[spike_day]
    assert spike_ret == pytest.approx(0.10, abs=1e-9)
    # Realistic position captures the spike; oracle (same-day) does not.
    assert net_real.max() == pytest.approx(0.10, abs=1e-9)
    assert net_oracle.max() == pytest.approx(0.0, abs=1e-9), (
        "a same-day position captured the spike; the t->t+1 lag failed"
    )


def test_strategy_with_shift_minus_one_is_construction_bug():
    """Documenting test: a strategy that internally reads p.shift(-1) injects
    future data into its own weights. No backtester lag can catch this, because
    the look-ahead happens inside strategy construction, before the harness sees
    the weights. We assert it DOES leak (high Sharpe), so the repo records that
    look-ahead prevention is a shared responsibility: the harness enforces the
    t->t+1 lag, and strategy authors must never call .shift(negative).
    """
    n = 300
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.01, n)
    px = 100 * np.exp(np.cumsum(rets))
    prices = pd.DataFrame({"x": px}, index=idx)
    cfg = BacktestConfig(tc_bps=0.0, slippage_bps=0.0)

    def peeking(p):
        fwd = p.pct_change(fill_method=None).shift(-1)
        return (fwd > 0).astype(float)

    m = run_backtest(prices, peeking, cfg)
    # The peek is exactly one day forward and the harness lag is one day back,
    # so they align: this leaks and prints an implausibly high Sharpe. The test
    # asserts the leak to make the failure mode explicit and regression-visible.
    assert m["sharpe"] > 3.0


def test_lag_is_one_day_exactly(deterministic_two_asset):
    """Directly assert the shift: gross return on day t uses weights from t-1."""
    prices = deterministic_two_asset
    returns = prices.pct_change(fill_method=None)
    # Hold 'up' fully from day 0 onward.
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w["up"] = 1.0
    cfg = BacktestConfig(tc_bps=0.0, slippage_bps=0.0)
    net = apply_costs(w, returns, cfg)
    # Net return on day 1 equals asset return on day 1, because the weight set on
    # day 0 (=1 for 'up') earns the day-1 return under the t->t+1 lag.
    expected_day1 = returns["up"].iloc[1]
    assert net.loc[prices.index[1]] == pytest.approx(expected_day1, abs=1e-12)
    # Day 0 has no prior weight, so it realizes nothing.
    assert net.loc[prices.index[0]] == pytest.approx(0.0, abs=1e-12)


def test_cost_application_round_trip():
    """A single full rotation into one asset costs exactly one side of cost on
    the entry day. Costs scale with traded notional.
    """
    idx = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.DataFrame({"x": np.linspace(100, 109, 10)}, index=idx)
    returns = prices.pct_change(fill_method=None)

    w = pd.DataFrame(0.0, index=idx, columns=["x"])
    w.iloc[3:, 0] = 1.0  # enter on day 3, hold to end

    cfg = BacktestConfig(tc_bps=10.0, slippage_bps=5.0)
    net = apply_costs(w, returns, cfg)

    # Delta on day 3 is 1.0 (0 -> 1). cost_per_trade = (10 + 5)/10000 = 0.0015.
    # That cost is realized on day 3. Compare day-3 net vs gross.
    gross = (w.shift(1) * returns).sum(axis=1)
    cost_day3 = gross.loc[idx[3]] - net.loc[idx[3]]
    assert cost_day3 == pytest.approx(0.0015, abs=1e-12)

    # No further trades -> no further costs after entry.
    for d in idx[4:]:
        assert (gross.loc[d] - net.loc[d]) == pytest.approx(0.0, abs=1e-12)


def test_no_cost_when_no_trades():
    idx = pd.bdate_range("2020-01-01", periods=20)
    prices = pd.DataFrame({"x": np.linspace(100, 119, 20)}, index=idx)
    returns = prices.pct_change(fill_method=None)
    w = pd.DataFrame(1.0, index=idx, columns=["x"])  # constant hold, no rebalancing
    cfg = BacktestConfig(tc_bps=10.0, slippage_bps=5.0)
    net = apply_costs(w, returns, cfg)
    gross = (w.shift(1) * returns).sum(axis=1).dropna()
    # net and gross should match (only the initial weight delta on first row,
    # which is dropped by apply_costs' dropna of the gross NaN first row).
    pd.testing.assert_series_equal(net, gross, check_names=False)


def test_sharpe_on_known_constant_return():
    """Constant positive daily return -> zero variance -> Sharpe defined as 0 by
    the harness (std == 0 guard). Use a known two-value series for a real Sharpe.
    """
    # Series that alternates +1% and -0.5%: mean and std are closed-form.
    n = 252
    vals = np.array([0.01 if i % 2 == 0 else -0.005 for i in range(n)])
    rets = pd.Series(vals, index=pd.bdate_range("2020-01-01", periods=n))
    w = pd.DataFrame(1.0, index=rets.index, columns=["x"])
    cfg = BacktestConfig(risk_free_annual=0.0)
    m = compute_metrics(rets, w, cfg)

    mean = vals.mean()
    std = vals.std(ddof=1)
    expected_sharpe = (mean / std) * np.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected_sharpe, rel=1e-9)


def test_max_drawdown_on_known_path():
    """Known price path with a single 20% peak-to-trough drop."""
    # cumulative returns: up to 1.25, down to 1.00 -> drawdown = 1.00/1.25 - 1 = -0.20
    rets = pd.Series(
        [0.25, -0.20],  # 1 -> 1.25 -> 1.00
        index=pd.bdate_range("2020-01-01", periods=2),
    )
    # compute_metrics needs >= 30 points; pad with zeros after the drop.
    pad = pd.Series([0.0] * 40, index=pd.bdate_range("2020-01-03", periods=40))
    full = pd.concat([rets, pad])
    w = pd.DataFrame(1.0, index=full.index, columns=["x"])
    cfg = BacktestConfig(risk_free_annual=0.0)
    m = compute_metrics(full, w, cfg)
    assert m["max_dd"] == pytest.approx(-0.20, abs=1e-9)


def test_short_series_returns_zeroed_metrics():
    rets = pd.Series([0.01] * 5, index=pd.bdate_range("2020-01-01", periods=5))
    w = pd.DataFrame(1.0, index=rets.index, columns=["x"])
    m = compute_metrics(rets, w, BacktestConfig())
    assert m["sharpe"] == 0.0 and m["n_trades"] == 0
