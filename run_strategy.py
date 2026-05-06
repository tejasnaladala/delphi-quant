"""Run a single strategy end-to-end and emit METRIC lines for /autoresearch.

Usage:
    python run_strategy.py --strategy buy_and_hold
    python run_strategy.py --strategy time_series_momentum --param lookback=126 --param top_n=30
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtester import BacktestConfig, emit_metrics_for_autoresearch, run_backtest, walk_forward_split
from strategies import STRATEGIES

DATA_PATH = Path(__file__).parent / "data" / "sp500_daily.parquet"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_prices() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print("ERROR: data not fetched yet. Run: python fetch_data.py", file=sys.stderr)
        sys.exit(1)
    df = pd.read_parquet(DATA_PATH)
    # yfinance returns multi-index columns (ticker, field). Pull adj close only.
    if isinstance(df.columns, pd.MultiIndex):
        # Try common field names
        for field in ("Close", "Adj Close"):
            try:
                close = df.xs(field, axis=1, level=1)
                return close.dropna(how="all")
            except KeyError:
                continue
    return df


def parse_params(param_args: list[str]) -> dict:
    params = {}
    for p in param_args or []:
        k, v = p.split("=", 1)
        try:
            params[k] = int(v)
        except ValueError:
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    return params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES.keys()))
    ap.add_argument("--param", action="append", default=[], help="key=value")
    ap.add_argument("--walk-forward", action="store_true")
    ap.add_argument("--train-months", type=int, default=24)
    ap.add_argument("--test-months", type=int, default=1)
    args = ap.parse_args()

    print(f"loading data from {DATA_PATH}...", file=sys.stderr)
    prices = load_prices()
    print(f"  shape: {prices.shape}, date range: {prices.index[0].date()} to {prices.index[-1].date()}", file=sys.stderr)

    strategy = STRATEGIES[args.strategy]
    params = parse_params(args.param)
    print(f"running strategy: {args.strategy}, params: {params}", file=sys.stderr)

    cfg = BacktestConfig()
    fn = lambda p: strategy(p, **params) if params else strategy(p)

    if args.walk_forward:
        splits = walk_forward_split(prices, train_months=args.train_months, test_months=args.test_months)
        print(f"walk-forward: {len(splits)} folds", file=sys.stderr)
        all_metrics = []
        for i, (train, test) in enumerate(splits):
            metrics = run_backtest(test, fn, cfg)
            metrics["fold"] = i
            all_metrics.append(metrics)
        # Aggregate over folds
        agg = {}
        for k in ("sharpe", "sortino", "calmar", "max_dd", "ann_return"):
            vals = [m[k] for m in all_metrics if not (isinstance(m[k], float) and (m[k] != m[k]))]
            if vals:
                agg[k] = sum(vals) / len(vals)
                agg[f"{k}_std"] = pd.Series(vals).std()
        emit_metrics_for_autoresearch(agg, prefix="oos")
        result = {"strategy": args.strategy, "params": params, "walk_forward": True, "folds": all_metrics, "agg": agg}
    else:
        metrics = run_backtest(prices, fn, cfg)
        emit_metrics_for_autoresearch(metrics)
        result = {"strategy": args.strategy, "params": params, "walk_forward": False, "metrics": metrics}

    # Save result
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"{args.strategy}_{ts}.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"saved: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
