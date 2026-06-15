"""Structured rejection logging for delphi-quant.

Every strategy that is evaluated produces a record. A record names each check
the strategy faced, the threshold it had to clear, the metric it actually
realized, and whether it passed. Rejections are first-class: the log is the
audit trail that lets a reader reconstruct why a strategy was dropped without
re-running anything.

This mirrors the structured-rejection-logging discipline used in evaluation
harnesses generally: an eval gate that silently drops a candidate is
unauditable; one that records (check, threshold, realized, verdict) for every
candidate is. The same record shape applies whether the "candidate" is a
trading strategy or a model capability being gated for safety.

The log is append-only within a run and serialized to JSON Lines so it can be
diffed and committed. No secrets ever enter this layer.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CheckRecord:
    """One pass/fail check applied to a strategy."""

    check: str  # human-readable check name
    threshold: float  # the bar the metric had to clear
    realized: float  # the metric the strategy actually produced
    comparison: str  # ">=", "<=", "in_range", etc.
    passed: bool
    note: str = ""


@dataclass(frozen=True)
class StrategyRecord:
    """Full evaluation record for one strategy: every check + final verdict."""

    label: str
    strategy: str
    params: dict
    checks: tuple[CheckRecord, ...]
    verdict: str  # "PASS", "FAIL", "CANDIDATE", "REJECTED_MC", etc.
    reason: str  # short human-readable reason for the verdict
    oos_sharpe: float
    raw_p_value: float | None = None
    holm_adjusted_alpha: float | None = None
    holm_significant: bool | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [asdict(c) for c in self.checks]
        return d


class RejectionLog:
    """Append-only collector for strategy evaluation records."""

    def __init__(self) -> None:
        self._records: list[StrategyRecord] = []

    def add(self, record: StrategyRecord) -> None:
        self._records.append(record)

    @property
    def records(self) -> list[StrategyRecord]:
        return list(self._records)

    def first_failed_check(self, label: str) -> CheckRecord | None:
        """Return the first failed check for a strategy, or None if all passed."""
        for rec in self._records:
            if rec.label == label:
                for c in rec.checks:
                    if not c.passed:
                        return c
        return None

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec.to_dict(), default=str) + "\n")
        return path

    def summary_rows(self) -> list[dict]:
        """Compact per-strategy rows for a report table."""
        rows = []
        for rec in self._records:
            failed = [c for c in rec.checks if not c.passed]
            rows.append(
                {
                    "label": rec.label,
                    "oos_sharpe": rec.oos_sharpe,
                    "verdict": rec.verdict,
                    "reason": rec.reason,
                    "failed_checks": [c.check for c in failed],
                    "raw_p_value": rec.raw_p_value,
                    "holm_significant": rec.holm_significant,
                }
            )
        return rows
