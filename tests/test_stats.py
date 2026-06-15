"""Tests for the statistics module: Sharpe p-value + Holm-Bonferroni."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stats import holm_bonferroni, sharpe_pvalue


def test_sharpe_pvalue_zero_sharpe_is_one():
    assert sharpe_pvalue(0.0, 1000) == pytest.approx(1.0, abs=1e-9)


def test_sharpe_pvalue_monotonic_in_sharpe():
    p_low = sharpe_pvalue(0.5, 2520)
    p_high = sharpe_pvalue(1.5, 2520)
    assert p_high < p_low  # higher Sharpe -> smaller p-value


def test_sharpe_pvalue_monotonic_in_length():
    # Same Sharpe, more observations -> more significant (smaller p).
    p_short = sharpe_pvalue(0.8, 252)
    p_long = sharpe_pvalue(0.8, 2520)
    assert p_long < p_short


def test_sharpe_pvalue_known_value():
    # t = SR * sqrt(T/P). SR=1.0, T=252, P=252 -> t=1.0 -> two-sided p ~ 0.3173.
    p = sharpe_pvalue(1.0, 252, trading_days_per_year=252)
    assert p == pytest.approx(0.31731, abs=1e-4)


def test_holm_empty():
    assert holm_bonferroni({}) == []


def test_holm_single_hypothesis_equals_raw():
    # With m=1, adjusted alpha == family alpha.
    res = holm_bonferroni({"a": 0.04}, family_alpha=0.05)
    assert len(res) == 1
    assert res[0].adjusted_alpha == pytest.approx(0.05)
    assert res[0].reject_null is True


def test_holm_step_down_ordering():
    # Three hypotheses. m=3, alpha=0.05.
    # sorted p: 0.01, 0.02, 0.04
    # rank1 threshold 0.05/3 = 0.01667 -> 0.01 <= 0.01667 reject
    # rank2 threshold 0.05/2 = 0.025   -> 0.02 <= 0.025   reject
    # rank3 threshold 0.05/1 = 0.05    -> 0.04 <= 0.05    reject
    pvals = {"x": 0.04, "y": 0.01, "z": 0.02}
    res = holm_bonferroni(pvals, family_alpha=0.05)
    by_label = {r.label: r for r in res}
    assert by_label["y"].rank == 1 and by_label["y"].reject_null
    assert by_label["z"].rank == 2 and by_label["z"].reject_null
    assert by_label["x"].rank == 3 and by_label["x"].reject_null


def test_holm_step_down_stops_after_first_failure():
    # sorted p: 0.001, 0.03, 0.04 ; m=3
    # rank1 0.001 <= 0.01667 reject
    # rank2 0.03  <= 0.025 ? NO -> stop. rank3 cannot reject even though
    # 0.04 <= 0.05 would pass plain Bonferroni at that step.
    pvals = {"a": 0.001, "b": 0.03, "c": 0.04}
    res = holm_bonferroni(pvals, family_alpha=0.05)
    by_label = {r.label: r for r in res}
    assert by_label["a"].reject_null is True
    assert by_label["b"].reject_null is False
    assert by_label["c"].reject_null is False  # step-down monotonicity


def test_holm_more_conservative_than_uncorrected():
    # Two hypotheses, both p < 0.05 (uncorrected "significant"). Holm rank-1
    # threshold is 0.05/2 = 0.025. The smallest p (0.03) exceeds it, so NOTHING
    # is rejected even though both p-values clear the naive 0.05 bar. This is the
    # multiple-comparison artifact the pre-reg exists to catch.
    pvals = {"a": 0.03, "b": 0.04}
    res = holm_bonferroni(pvals, family_alpha=0.05)
    by_label = {r.label: r for r in res}
    assert by_label["a"].p_value < 0.05 and by_label["b"].p_value < 0.05
    assert by_label["a"].reject_null is False  # 0.03 > 0.025 adjusted alpha
    assert by_label["b"].reject_null is False  # step-down: blocked too
