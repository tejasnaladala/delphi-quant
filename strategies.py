"""Pre-registered baseline strategies for delphi-quant v0.1.

All strategies in this file were committed to git BEFORE running any backtests.
See PRE_REGISTRATION.md for the locked hypotheses and the success criteria.

Each strategy returns a (T x N) weights DataFrame aligned to the input prices.
Weights at row t represent the desired end-of-day-t position. The backtester
enforces a 1-day lag before the position is taken (no look-ahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight buy-and-hold across the universe. Sanity-check baseline."""
    n = prices.shape[1]
    weights = pd.DataFrame(1.0 / n, index=prices.index, columns=prices.columns)
    weights = weights.where(prices.notna(), 0.0)
    return weights


def time_series_momentum(prices: pd.DataFrame, lookback: int = 252, top_n: int = 20) -> pd.DataFrame:
    """Cross-sectional momentum: hold the top_n assets by 12-month return.

    Standard academic spec: skip the most recent month to avoid 1-month reversal
    contamination (Jegadeesh & Titman 1993, Asness 1994). Long-only. Equal-weight.
    Rebalanced monthly.
    """
    skip = 21  # 1 month
    momentum = prices.shift(skip) / prices.shift(skip + lookback) - 1
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Rebalance every 21 trading days (monthly cadence, guaranteed alignment)
    rebalance_dates = prices.index[skip + lookback::21]
    for date in rebalance_dates:
        mom_t = momentum.loc[date].dropna()
        if len(mom_t) < top_n:
            continue
        top = mom_t.nlargest(top_n).index
        weights.loc[date:, :] = 0.0
        weights.loc[date:, top] = 1.0 / top_n
    return weights


def cross_sectional_mean_reversion(prices: pd.DataFrame, lookback: int = 5, bottom_n: int = 20) -> pd.DataFrame:
    """Cross-sectional short-term reversal: hold bottom_n assets by 5-day return.

    Standard academic spec: weekly losers tend to bounce (Lo & MacKinlay 1990,
    Jegadeesh 1990). Long-only. Equal-weight. Rebalanced weekly.
    """
    reversal_signal = prices.pct_change(lookback, fill_method=None)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Rebalance every 5 trading days (weekly cadence, guaranteed alignment)
    rebalance_dates = prices.index[lookback::5]
    for date in rebalance_dates:
        sig_t = reversal_signal.loc[date].dropna()
        if len(sig_t) < bottom_n:
            continue
        bottom = sig_t.nsmallest(bottom_n).index
        weights.loc[date:, :] = 0.0
        weights.loc[date:, bottom] = 1.0 / bottom_n
    return weights


# Registry for the autonomous iteration loop
STRATEGIES = {
    "buy_and_hold": buy_and_hold,
    "time_series_momentum": time_series_momentum,
    "cross_sectional_mean_reversion": cross_sectional_mean_reversion,
}
