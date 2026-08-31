"""Run the full delphi-quant verification pipeline over the baseline family.

This is the v0.1 report driver. It evaluates every declared baseline strategy
walk-forward OOS, applies the per-strategy protocol checks, runs Holm-Bonferroni
across the family, runs the deployment gates for any multiple-comparison
survivor, writes the structured rejection log, and emits the markdown report.

Usage:
    python run_pipeline.py                      # cached real data; fail closed if absent
    python run_pipeline.py --report-out REPORT.md
    python run_pipeline.py --synthetic          # explicit logic check, separate outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import BacktestConfig
from pipeline import evaluate_family
from provenance import dataset_manifest, synthetic_manifest, write_manifest
from report import generate_report
from run_strategy import DATA_PATH, load_prices

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


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.synthetic:
        report_default = RESULTS_DIR / "SYNTHETIC_REPORT.md"
        log_default = RESULTS_DIR / "synthetic_rejection_log.jsonl"
        manifest_default = RESULTS_DIR / "synthetic_input_manifest.json"
    else:
        report_default = Path(__file__).parent / "REPORT_v0.1.md"
        log_default = RESULTS_DIR / "rejection_log.jsonl"
        manifest_default = RESULTS_DIR / "input_manifest.json"
    return (
        Path(args.report_out) if args.report_out else report_default,
        Path(args.log_out) if args.log_out else log_default,
        Path(args.manifest_out) if args.manifest_out else manifest_default,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-out")
    ap.add_argument("--log-out")
    ap.add_argument("--manifest-out")
    ap.add_argument("--data-path", type=Path, default=DATA_PATH)
    ap.add_argument(
        "--data-source-note",
        default="Unspecified local snapshot; publication rights are not attested by this code.",
    )
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic data (no feed)")
    args = ap.parse_args(argv)
    report_out, log_out, manifest_out = _output_paths(args)

    if args.synthetic:
        prices = _synthetic_prices()
        evidence_kind = "synthetic"
        manifest = synthetic_manifest(prices, n_assets=30, n_days=2000, seed=7)
        print("running on explicit SYNTHETIC data (logic check only)", file=sys.stderr)
    else:
        try:
            prices = load_prices(args.data_path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print(
                "No report was generated. Use --synthetic explicitly for a logic-only run.",
                file=sys.stderr,
            )
            return 2
        evidence_kind = "real"
        manifest = dataset_manifest(args.data_path, prices, args.data_source_note)

    print(f"universe: {prices.shape[1]} names, {prices.shape[0]} days", file=sys.stderr)

    cfg = BacktestConfig()
    log, context = evaluate_family(prices, cfg=cfg)

    log_path = log.write_jsonl(log_out)
    print(f"rejection log written: {log_path}", file=sys.stderr)

    manifest_path = write_manifest(manifest, manifest_out)
    print(f"input manifest written: {manifest_path}", file=sys.stderr)

    md = generate_report(
        log,
        context,
        evidence_kind=evidence_kind,
        provenance=manifest,
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(md, encoding="utf-8")
    print(f"report written: {report_out}", file=sys.stderr)

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
