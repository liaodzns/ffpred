"""Tests for the paired comparisons, including the seed-averaged one that counts training noise."""

import numpy as np
import pandas as pd
import pytest

from ffml.models import evaluate

ACTUAL = pd.Series([10.0, 10.0, 10.0, 10.0])


def test_paired_comparison_matches_hand_computed_values() -> None:
    """Errors 1, 1, 0, 0 against 0, 0, 1, 2 differ by -1, -1, 1, 2.

    Mean 0.25, standard deviation 1.5, standard error 0.75, so t is one third.
    """
    comparison = evaluate.paired_error_comparison(
        ACTUAL,
        pd.Series([9.0, 11.0, 10.0, 10.0]),
        pd.Series([10.0, 10.0, 9.0, 12.0]),
    )

    assert comparison["delta_mae"] == pytest.approx(0.25)
    assert comparison["se"] == pytest.approx(0.75)
    assert comparison["t"] == pytest.approx(1.0 / 3.0)
    assert comparison["rows"] == 4


def two_seed_reference() -> list[pd.Series]:
    """Reference predictions under two seeds.

    Takes nothing. Returns them. Errors are 1, 1, 0, 0 and then 0, 0, 0, 2, so
    both seeds score MAE 0.5 and the per-row seed-averaged errors are 0.5, 0.5,
    0, 1.
    """
    return [pd.Series([9.0, 11.0, 10.0, 10.0]), pd.Series([10.0, 10.0, 10.0, 12.0])]


def two_seed_candidate() -> list[pd.Series]:
    """Candidate predictions under the same two seeds.

    Takes nothing. Returns them. Errors are all 0 and then 0, 0, 1, 0, so the
    seeds score MAE 0 and 0.25 and the per-row averaged errors are 0, 0, 0.5, 0.
    """
    return [pd.Series([10.0, 10.0, 10.0, 10.0]), pd.Series([10.0, 10.0, 9.0, 10.0])]


def test_seed_averaged_comparison_matches_hand_computed_values() -> None:
    """Both standard error terms and their combination are computed correctly.

    Per-seed differences are -0.5 and -0.25: mean -0.375, SD 0.25 / sqrt 2, so
    the seed term is 0.125. Per-row averaged differences are -0.5, -0.5, 0.5,
    -1: SD sqrt(1.1875 / 3), so the row term is that over 2. The combined error
    is the root of the two squared.
    """
    comparison = evaluate.seed_averaged_comparison(ACTUAL, two_seed_reference(), two_seed_candidate())

    se_row = float(np.sqrt(1.1875 / 3.0) / 2.0)
    se = float(np.sqrt(se_row**2 + 0.125**2))

    assert comparison["seeds"] == 2
    assert comparison["rows"] == 4
    assert comparison["seed_differences"] == pytest.approx([-0.5, -0.25])
    assert comparison["delta_mae"] == pytest.approx(-0.375)
    assert comparison["se_seed"] == pytest.approx(0.125)
    assert comparison["se_row"] == pytest.approx(se_row)
    assert comparison["se"] == pytest.approx(se)
    assert comparison["t"] == pytest.approx(-0.375 / se)
    assert comparison["row_only_t"] == pytest.approx(-0.375 / se_row)

    assert comparison["mae_a_mean"] == pytest.approx(0.5)
    assert comparison["mae_a_range"] == pytest.approx(0.0)
    assert comparison["mae_b_mean"] == pytest.approx(0.125)
    assert comparison["mae_b_sd"] == pytest.approx(0.25 / np.sqrt(2.0))
    assert comparison["mae_b_range"] == pytest.approx(0.25)
    assert comparison["seed_range"] == pytest.approx(0.25)


def test_a_deterministic_reference_contributes_no_seed_spread() -> None:
    """The baseline, repeated once per seed, has no spread, so the seed term is the model's alone."""
    baseline = pd.Series([9.0, 11.0, 10.0, 10.0])
    candidate = two_seed_candidate()
    comparison = evaluate.seed_averaged_comparison(ACTUAL, [baseline, baseline], candidate)

    model_mae_sd = 0.25 / np.sqrt(2.0)
    assert comparison["mae_a_sd"] == 0.0
    assert comparison["mae_a_range"] == 0.0
    assert comparison["se_seed"] == pytest.approx(model_mae_sd / np.sqrt(2.0))
    assert comparison["seed_range"] == pytest.approx(comparison["mae_b_range"])


def test_unequal_seed_counts_are_refused() -> None:
    """Pairing by seed needs both configurations trained under the same seeds."""
    with pytest.raises(ValueError, match="Seed counts differ"):
        evaluate.seed_averaged_comparison(ACTUAL, two_seed_reference(), two_seed_candidate()[:1])


def test_a_single_seed_is_refused() -> None:
    """One seed says nothing about training randomness, which is the point of the comparison."""
    with pytest.raises(ValueError, match="At least two seeds"):
        evaluate.seed_averaged_comparison(ACTUAL, two_seed_reference()[:1], two_seed_candidate()[:1])


def test_an_effect_inside_the_seed_range_is_unresolved_however_large_its_t() -> None:
    """The stage nine trap: WR opponent strength at 0.035 against a 0.036 range, t = -3."""
    verdict = evaluate.seed_averaged_verdict({"delta_mae": -0.035, "t": -3.0, "seed_range": 0.036}, 2.0)

    assert verdict["verdict"] == evaluate.UNRESOLVED
    assert "seed range" in verdict["reason"]
    assert "|t|" not in verdict["reason"]


def test_an_effect_at_exactly_the_seed_range_is_unresolved() -> None:
    """The effect must exceed the range, not merely reach it."""
    verdict = evaluate.seed_averaged_verdict({"delta_mae": -0.036, "t": -5.0, "seed_range": 0.036}, 2.0)
    assert verdict["verdict"] == evaluate.UNRESOLVED


def test_an_effect_beyond_the_range_but_within_two_errors_is_unresolved() -> None:
    """Clearing seed noise does not rescue an effect the rows cannot separate."""
    verdict = evaluate.seed_averaged_verdict({"delta_mae": -0.05, "t": -1.5, "seed_range": 0.036}, 2.0)

    assert verdict["verdict"] == evaluate.UNRESOLVED
    assert "|t|" in verdict["reason"]
    assert "seed range" not in verdict["reason"]


def test_clearing_both_bars_decides_in_the_direction_of_the_effect() -> None:
    """Negative deltas help the candidate and positive ones hurt it."""
    helps = evaluate.seed_averaged_verdict({"delta_mae": -0.05, "t": -4.0, "seed_range": 0.036}, 2.0)
    hurts = evaluate.seed_averaged_verdict({"delta_mae": 0.05, "t": 4.0, "seed_range": 0.036}, 2.0)

    assert helps["verdict"] == evaluate.HELPS
    assert hurts["verdict"] == evaluate.HURTS
