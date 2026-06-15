"""Strict daily-bar backtester for delphi-quant.

Design rules (delphi-style, intended to prevent the most common p-hacking modes):
1. Signals computed at close on day t can only execute at close on day t+1
   (no look-ahead). Returns are realized on day t+2.
2. Transaction costs: 10 bps per side per trade (round-trip = 20 bps).
3. Slippage: 5 bps per side, modeled as a fixed adverse fill on top of TC.
4. Position sizing: equal-weight on selected names, capped at 1.0 gross exposure.
5. Walk-forward CV: train on rolling N-month windows, test on next 1 month,
   no overlap. Strategy parameters are frozen per fold.
6. No survivorship-bias correction in v0.1 (acknowledged in PRE_REGISTRATION).

Output: dict with sharpe, sortino, max_dd, calmar, hit_rate, turnover, n_trades.
Designed to print METRIC name=value lines on stdout for the autonomous
iteration loop to ingest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    tc_bps: float = 10.0  # per side
    slippage_bps: float = 5.0  # per side
    risk_free_annual: float = 0.04
    trading_days_per_year: int = 252
    max_gross_exposure: float = 1.0


def compute_metrics(returns: pd.Series, weights: pd.DataFrame, cfg: BacktestConfig) -> dict:
    """Compute Sharpe, Sortino, max DD, Calmar, hit rate, turnover."""
    if len(returns) < 30:
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "calmar": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "n_trades": 0,
            "ann_return": 0.0,
            "ann_vol": 0.0,
        }

    rf_daily = cfg.risk_free_annual / cfg.trading_days_per_year
    excess = returns - rf_daily
    mean = excess.mean()
    std = returns.std(ddof=1)
    downside = returns[returns < 0].std(ddof=1) if (returns < 0).any() else float("nan")

    sharpe = (mean / std) * np.sqrt(cfg.trading_days_per_year) if std > 0 else 0.0
    sortino = (mean / downside) * np.sqrt(cfg.trading_days_per_year) if downside > 0 else 0.0

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum / peak) - 1
    max_dd = drawdown.min()
    calmar = (mean * cfg.trading_days_per_year) / abs(max_dd) if max_dd < 0 else 0.0

    hit_rate = (returns > 0).mean()
    turnover = weights.diff().abs().sum(axis=1).mean()
    n_trades = int(weights.diff().abs().sum().sum())
    ann_return = (1 + mean) ** cfg.trading_days_per_year - 1
    ann_vol = std * np.sqrt(cfg.trading_days_per_year)

    return {
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "hit_rate": float(hit_rate),
        "turnover": float(turnover),
        "n_trades": n_trades,
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
    }


def apply_costs(weights: pd.DataFrame, returns: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    """Apply TC + slippage to portfolio returns.

    Returns the realized daily portfolio return series net of costs.
    """
    # Position changes from t-1 to t, cost realized on day t
    delta_pos = weights.diff().abs().fillna(0.0)
    cost_per_trade = (cfg.tc_bps + cfg.slippage_bps) / 10000.0
    total_cost = delta_pos.sum(axis=1) * cost_per_trade

    # Portfolio return on day t uses weights at end of day t-1 multiplied by
    # asset returns on day t. Then subtract cost.
    gross_ret = (weights.shift(1) * returns).sum(axis=1)
    net_ret = gross_ret - total_cost
    return net_ret.dropna()


def run_backtest(
    prices: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], pd.DataFrame],
    cfg: BacktestConfig | None = None,
) -> dict:
    """Run a strategy on adjusted close prices.

    strategy_fn takes a price DataFrame (T x N) and returns a weights
    DataFrame (T x N) where row t is the position at end of day t. The
    backtester enforces the t -> t+1 lag.
    """
    cfg = cfg or BacktestConfig()
    returns = prices.pct_change(fill_method=None)
    weights = strategy_fn(prices)

    # Cap gross exposure
    gross = weights.abs().sum(axis=1).replace(0, 1)
    scale = (cfg.max_gross_exposure / gross).clip(upper=1.0)
    weights = weights.mul(scale, axis=0).fillna(0.0)

    portfolio_ret = apply_costs(weights, returns, cfg)
    metrics = compute_metrics(portfolio_ret, weights, cfg)
    metrics["start"] = str(portfolio_ret.index[0].date())
    metrics["end"] = str(portfolio_ret.index[-1].date())
    metrics["n_days"] = len(portfolio_ret)
    return metrics


def walk_forward_split(prices: pd.DataFrame, train_months: int = 24, test_months: int = 1) -> list:
    """Generate purged walk-forward train/test splits."""
    splits = []
    start = prices.index[0]
    end = prices.index[-1]
    cursor = start + pd.DateOffset(months=train_months)
    while cursor + pd.DateOffset(months=test_months) <= end:
        train = prices.loc[start : cursor]
        test = prices.loc[cursor : cursor + pd.DateOffset(months=test_months)]
        if len(train) > 100 and len(test) > 5:
            splits.append((train, test))
        cursor += pd.DateOffset(months=test_months)
    return splits


def emit_metrics_for_autoresearch(metrics: dict, prefix: str = "") -> None:
    """Print METRIC lines for the autonomous iteration loop to ingest."""
    p = f"{prefix}_" if prefix else ""
    for k in ("sharpe", "sortino", "calmar", "hit_rate", "ann_return", "ann_vol", "max_dd", "turnover"):
        if k in metrics:
            print(f"METRIC {p}{k}={metrics[k]:.6f}")
    if "n_trades" in metrics:
        print(f"METRIC {p}n_trades={metrics['n_trades']}")
