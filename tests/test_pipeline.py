"""End-to-end pipeline tests on synthetic data (no network)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import evaluate_family
from rejection_log import CheckRecord, RejectionLog, StrategyRecord
from report import generate_report


def _synth_panel(n_assets=30, n_days=2200, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2014-01-01", periods=n_days)
    drift = rng.normal(0.0003, 0.0002, n_assets)
    rets = rng.normal(drift, 0.012, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"A{i:02d}" for i in range(n_assets)])


def test_pipeline_runs_and_records_every_strategy():
    prices = _synth_panel()
    log, context = evaluate_family(prices)
    labels = {r.label for r in log.records}
    # All three baselines must appear, including any that failed.
    assert labels == {"buy_and_hold", "time_series_momentum", "cross_sectional_mean_reversion"}
    # Holm table covers the whole family.
    assert len(context["holm"]) == 3


def test_pipeline_records_failed_checks():
    prices = _synth_panel()
    log, _ = evaluate_family(prices)
    for rec in log.records:
        # Every record has at least the pre-reg checks + raw p + Holm check.
        assert len(rec.checks) >= 3
        # Verdict is from the known vocabulary.
        assert rec.verdict in {
            "PASS", "FAIL", "ALIVE", "REJECTED_MC", "CANDIDATE", "GATED",
        }


def test_rejection_log_jsonl_roundtrip(tmp_path):
    log = RejectionLog()
    log.add(
        StrategyRecord(
            label="demo",
            strategy="demo",
            params={"lookback": 60},
            checks=(
                CheckRecord("oos_sharpe_above_target", 0.7, 0.5, ">", False, "below target"),
            ),
            verdict="FAIL",
            reason="below target",
            oos_sharpe=0.5,
            raw_p_value=0.4,
        )
    )
    p = log.write_jsonl(tmp_path / "log.jsonl")
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["label"] == "demo"
    assert obj["checks"][0]["passed"] is False
    assert obj["verdict"] == "FAIL"


def test_first_failed_check():
    log = RejectionLog()
    log.add(
        StrategyRecord(
            label="s",
            strategy="s",
            params={},
            checks=(
                CheckRecord("a", 1.0, 1.0, ">=", True),
                CheckRecord("b", 1.0, 0.0, ">=", False),
                CheckRecord("c", 1.0, 0.0, ">=", False),
            ),
            verdict="FAIL",
            reason="b failed",
            oos_sharpe=0.0,
        )
    )
    failed = log.first_failed_check("s")
    assert failed is not None and failed.check == "b"


def test_report_generation_includes_all_strategies_and_verdict():
    prices = _synth_panel()
    log, context = evaluate_family(prices)
    md = generate_report(log, context)
    assert "Search trajectory" in md
    assert "Holm-Bonferroni correction" in md
    assert "Verdict" in md
    for rec in log.records:
        assert rec.label in md


def test_report_no_candidate_message_when_none_qualify():
    prices = _synth_panel(seed=123)  # random panel: no real alpha expected
    log, context = evaluate_family(prices)
    md = generate_report(log, context)
    candidates = [r for r in log.records if r.verdict == "CANDIDATE"]
    if not candidates:
        assert "No strategy cleared every stage" in md
