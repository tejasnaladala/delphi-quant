# delphi-quant

A small verification harness for systematic equity research. It keeps the
strategy family, thresholds, walk-forward evaluation, multiple-comparison
correction, deployment gates, and rejection log in one inspectable path.

## Evidence status

This repository is a **retrospective research artifact**, not a preregistered
study and not a deployable trading system.

- The first verifiable Git commit contains both the protocol and its
  implementation. History therefore cannot establish that the hypotheses were
  locked before analysis.
- The parquet snapshot behind the historical June 2026 report is not committed,
  hashed, or otherwise available here. The report is retained as an explicitly
  unverified historical artifact, not as a reproducible result.
- A normal pipeline run now fails closed when the requested real-data snapshot
  is absent. Synthetic data is used only with an explicit `--synthetic` flag and
  writes to separate, unmistakably labeled outputs.
- No result in this repository is investment advice or evidence of a strategy
  suitable for capital deployment.

## What is implemented

1. Rolling walk-forward evaluation with 24-month warm-up windows and one-month
   scored windows.
2. A declared three-strategy baseline family with fixed decision thresholds.
3. Two-sided Sharpe significance tests and Holm-Bonferroni family-wise error
   control.
4. Transaction-cost, liquid-universe, and regime-gap stress gates.
5. JSONL rejection records containing every strategy, check, realized value,
   threshold, and verdict.
6. Input manifests that record the exact real-data SHA-256, panel shape, and
   observation range without publishing a private local path.

The harness is useful as code even when every strategy fails. Its job is to make
rejection legible, not to manufacture a candidate.

## Reproduce the code path

```bash
python -m venv .venv
.venv/Scripts/pip install yfinance pandas numpy scipy pyarrow pytest
.venv/Scripts/python -m pytest -q
```

The test suite is network-free and uses deterministic synthetic fixtures.

### Explicit synthetic logic check

```bash
.venv/Scripts/python run_pipeline.py --synthetic
```

This writes:

- `results/SYNTHETIC_REPORT.md`
- `results/synthetic_rejection_log.jsonl`
- `results/synthetic_input_manifest.json`

Those metrics test control flow only. They cannot support a market or strategy
claim and do not overwrite `REPORT_v0.1.md`.

### Real-data computation

```bash
.venv/Scripts/python fetch_data.py
.venv/Scripts/python run_pipeline.py \
  --data-path data/sp500_daily.parquet \
  --data-source-note "yfinance pull for local research; verify redistribution rights"
```

If the parquet is missing or unreadable, the command exits with status 2 and
generates nothing. A successful run writes `results/input_manifest.json`; cite
that hash with any result. The manifest improves byte-level reproducibility but
does not establish data licensing, point-in-time constituent correctness, or
economic validity.

Single-strategy runs follow the same fail-closed rule:

```bash
.venv/Scripts/python run_strategy.py \
  --strategy time_series_momentum \
  --walk-forward \
  --data-path data/sp500_daily.parquet
```

## Repository map

| Path | Purpose |
|---|---|
| `PRE_REGISTRATION.md` | Historical filename for the retrospective declared protocol; its evidence caveat is part of the document. |
| `DEVIATION_LOG.md` | Chronological changes and the 2026 audit correction. |
| `backtester.py` | Daily-bar accounting, lag, costs, and metrics. |
| `walk_forward.py` | Fold construction and out-of-sample aggregation. |
| `strategies.py` | Fixed baseline strategy definitions. |
| `stats.py` | Sharpe p-values and Holm-Bonferroni correction. |
| `gates.py` | Cost, liquid-universe, and regime-gap stress checks. |
| `pipeline.py` | Family-level evaluation and verdict path. |
| `rejection_log.py` | Structured, append-only-within-run records. |
| `provenance.py` | Real and synthetic input manifests. |
| `REPORT_v0.1.md` | Unverified historical report retained for auditability. |

## Known limitations

- Current-constituent data is survivorship-biased.
- The repository has no immutable licensed market-data snapshot.
- Daily bars do not model intraday execution or market impact.
- There is no factor risk model, covariance shrinkage, or regime model.
- The current p-value calculation is a simplified research diagnostic, not a
  complete treatment of autocorrelation, non-normality, or data snooping.
- No paper-trading or prospective holdout evidence is present.

MIT licensed. See `LICENSE`.
