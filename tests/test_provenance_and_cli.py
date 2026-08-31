from __future__ import annotations

import json

import pandas as pd
import pytest

from provenance import dataset_manifest
from run_pipeline import _output_paths
from run_pipeline import main as pipeline_main
from run_strategy import load_prices


def test_missing_real_data_fails_closed(tmp_path, capsys):
    report = tmp_path / "report.md"
    log = tmp_path / "log.jsonl"
    manifest = tmp_path / "manifest.json"

    result = pipeline_main(
        [
            "--data-path",
            str(tmp_path / "missing.parquet"),
            "--report-out",
            str(report),
            "--log-out",
            str(log),
            "--manifest-out",
            str(manifest),
        ]
    )

    assert result == 2
    assert "No report was generated" in capsys.readouterr().err
    assert not report.exists()
    assert not log.exists()
    assert not manifest.exists()


def test_load_prices_raises_instead_of_exiting(tmp_path):
    with pytest.raises(FileNotFoundError, match="price snapshot not found"):
        load_prices(tmp_path / "missing.parquet")


def test_synthetic_defaults_cannot_overwrite_canonical_report():
    args = type(
        "Args",
        (),
        {"synthetic": True, "report_out": None, "log_out": None, "manifest_out": None},
    )()
    report, log, manifest = _output_paths(args)

    assert report.name == "SYNTHETIC_REPORT.md"
    assert log.name == "synthetic_rejection_log.jsonl"
    assert manifest.name == "synthetic_input_manifest.json"


def test_real_manifest_fingerprints_exact_input(tmp_path):
    path = tmp_path / "prices.parquet"
    prices = pd.DataFrame(
        {"A": [100.0, 101.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
    )
    prices.to_parquet(path)

    manifest = dataset_manifest(path, prices, "test fixture")

    assert manifest["evidence_kind"] == "real"
    assert manifest["file_name"] == "prices.parquet"
    assert len(manifest["sha256"]) == 64
    assert manifest["rows"] == 2
    assert manifest["assets"] == 1
    assert json.dumps(manifest)
