"""Capital-deployment gates for delphi-quant.

The pre-reg (section 4) requires any candidate to additionally hold up under
three robustness re-runs before it can be called a deployment candidate:

  1. 5x transaction-cost sensitivity (50 bps per side instead of 10).
  2. Re-run with the universe restricted to the most-liquid S&P 100 subset
     (liquidity sanity check).
  3. Re-run with a 3-month gap between train and test (regime-stability check).

Each gate is a walk-forward OOS re-run with one parameter changed. A gate
passes if the OOS Sharpe stays above a survival floor. The floor is expressed
relative to the strategy's own pre-reg failure threshold so the gates do not
silently invent a new bar: a strategy that the pre-reg already calls "alive"
should not flip to "dead" under a reasonable stress, and a gate that pushes it
below its own dead-line is a genuine rejection.

These gates double as the AI-safety-eval-gating pattern: a capability that only
clears under benign conditions and collapses under a cost/distribution shift is
exactly the thing a gate should reject before deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from backtester import BacktestConfig
from walk_forward import StrategyFn, run_walk_forward

# Liquidity subset: the 50 most-liquid mega-caps from the v0.1 universe. Used as
# the S&P-100-style restriction. Members must also exist in the loaded frame.
SP100_LIQUID_SUBSET = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "BRK-B", "JPM", "V", "UNH", "XOM", "WMT", "JNJ", "MA", "PG", "AVGO",
    "HD", "CVX", "ABBV", "MRK", "LLY", "PEP", "KO", "BAC", "PFE", "COST",
    "TMO", "CSCO", "MCD", "ACN", "ADBE", "ABT", "LIN", "CRM", "DHR", "AMD",
    "DIS", "WFC", "TXN", "NFLX", "VZ", "PM", "CMCSA", "RTX", "INTC", "NEE",
    "BMY", "QCOM",
]


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    realized_sharpe: float
    threshold: float
    detail: str


def _restrict_universe(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    cols = [t for t in tickers if t in prices.columns]
    return prices[cols].dropna(axis=1, how="all")


def tc_sensitivity_gate(
    prices: pd.DataFrame,
    strategy_fn: StrategyFn,
    survival_floor: float,
    base_cfg: BacktestConfig | None = None,
    multiplier: float = 5.0,
) -> GateResult:
    """Re-run walk-forward at 5x TC (50 bps/side) and check the survival floor."""
    base_cfg = base_cfg or BacktestConfig()
    stressed = replace(
        base_cfg,
        tc_bps=base_cfg.tc_bps * multiplier,
        slippage_bps=base_cfg.slippage_bps * multiplier,
    )
    m = run_walk_forward(prices, strategy_fn, cfg=stressed)
    sharpe = m["sharpe"]
    return GateResult(
        name="tc_5x_sensitivity",
        passed=sharpe >= survival_floor,
        realized_sharpe=float(sharpe),
        threshold=float(survival_floor),
        detail=f"OOS Sharpe at {stressed.tc_bps:.0f} bps/side TC + {stressed.slippage_bps:.0f} bps slippage",
    )


def liquidity_gate(
    prices: pd.DataFrame,
    strategy_fn: StrategyFn,
    survival_floor: float,
    cfg: BacktestConfig | None = None,
) -> GateResult:
    """Re-run walk-forward on the S&P-100-style liquid subset."""
    cfg = cfg or BacktestConfig()
    sub = _restrict_universe(prices, SP100_LIQUID_SUBSET)
    m = run_walk_forward(sub, strategy_fn, cfg=cfg)
    sharpe = m["sharpe"]
    return GateResult(
        name="sp100_liquidity",
        passed=sharpe >= survival_floor,
        realized_sharpe=float(sharpe),
        threshold=float(survival_floor),
        detail=f"OOS Sharpe on {sub.shape[1]}-name liquid subset",
    )


def regime_gap_gate(
    prices: pd.DataFrame,
    strategy_fn: StrategyFn,
    survival_floor: float,
    cfg: BacktestConfig | None = None,
    gap_months: int = 3,
) -> GateResult:
    """Re-run walk-forward with a 3-month gap between train and test."""
    cfg = cfg or BacktestConfig()
    m = run_walk_forward(prices, strategy_fn, cfg=cfg, train_test_gap_months=gap_months)
    sharpe = m["sharpe"]
    return GateResult(
        name="regime_gap_3mo",
        passed=sharpe >= survival_floor,
        realized_sharpe=float(sharpe),
        threshold=float(survival_floor),
        detail=f"OOS Sharpe with {gap_months}-month train/test gap ({m['n_folds']} folds)",
    )


def run_all_gates(
    prices: pd.DataFrame,
    strategy_fn: StrategyFn,
    survival_floor: float,
    cfg: BacktestConfig | None = None,
) -> list[GateResult]:
    """Run the three pre-registered deployment gates in order."""
    cfg = cfg or BacktestConfig()
    return [
        tc_sensitivity_gate(prices, strategy_fn, survival_floor, base_cfg=cfg),
        liquidity_gate(prices, strategy_fn, survival_floor, cfg=cfg),
        regime_gap_gate(prices, strategy_fn, survival_floor, cfg=cfg),
    ]
