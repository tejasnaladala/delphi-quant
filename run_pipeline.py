"""Run the full delphi-quant verification pipeline over the baseline family.

This is the v0.1 report driver. It evaluates every pre-registered strategy
walk-forward OOS, applies the per-strategy pre-reg checks, runs Holm-Bonferroni
across the family, runs the deployment gates for any multiple-comparison
survivor, writes the structured rejection log, and emits the markdown report.

Usage:
    python run_pipeline.py                      # cached data, write report + log
    python run_pipeline.py --report-out REPORT.md
    python run_pipeline.py --synthetic          # no data feed; logic check only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import BacktestConfig
from pipeline import evaluate_family
from report import generate_report

RESULTS_DIR = Path(__file__).parent / "results"


def _synthetic_prices(n_assets: int = 30, n_days: int = 2000, seed: int = 7) -> pd.DataFrame:
    """Deterministic geometric-random-walk panel for a logic-only run."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    drift = rng.normal(0.0003, 0.0002, n_assets)
    rets = rng.normal(drift, 0.012, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=idx, columns=cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-out", default=str(Path(__file__).parent / "REPORT_v0.1.md"))
    ap.add_argument("--log-out", default=str(RESULTS_DIR / "rejection_log.jsonl"))
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic data (no feed)")
    args = ap.parse_args()

    pending = False
    if args.synthetic:
        prices = _synthetic_prices()
        pending = True
        print("running on SYNTHETIC data (logic check only)", file=sys.stderr)
    else:
        from run_strategy import load_prices

        try:
            prices = load_prices()
        except SystemExit:
            print("price feed unreachable; falling back to synthetic data", file=sys.stderr)
            prices = _synthetic_prices()
            pending = True

    print(f"universe: {prices.shape[1]} names, {prices.shape[0]} days", file=sys.stderr)

    cfg = BacktestConfig()
    log, context = evaluate_family(prices, cfg=cfg)

    log_path = log.write_jsonl(args.log_out)
    print(f"rejection log written: {log_path}", file=sys.stderr)

    md = generate_report(log, context, numbers_pending_live_run=pending)
    Path(args.report_out).write_text(md, encoding="utf-8")
    print(f"report written: {args.report_out}", file=sys.stderr)

    # Console summary
    for row in log.summary_rows():
        print(
            f"  {row['label']:34s} sharpe={row['oos_sharpe']:.3f} "
            f"verdict={row['verdict']}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
