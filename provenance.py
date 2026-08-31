"""Small, non-secret input manifests for reproducible Delphi runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _panel_shape(prices: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(prices.shape[0]),
        "assets": int(prices.shape[1]),
        "first_observation": str(prices.index.min()),
        "last_observation": str(prices.index.max()),
    }


def dataset_manifest(
    data_path: str | Path,
    prices: pd.DataFrame,
    source_note: str,
) -> dict[str, object]:
    """Describe the exact local bytes used without publishing a private path."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "schema_version": 1,
        "evidence_kind": "real",
        "file_name": path.name,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "source_note": source_note,
        **_panel_shape(prices),
    }


def synthetic_manifest(
    prices: pd.DataFrame,
    *,
    n_assets: int,
    n_days: int,
    seed: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "evidence_kind": "synthetic",
        "generator": "deterministic_geometric_random_walk",
        "n_assets": n_assets,
        "n_days": n_days,
        "seed": seed,
        **_panel_shape(prices),
    }


def write_manifest(manifest: dict[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
