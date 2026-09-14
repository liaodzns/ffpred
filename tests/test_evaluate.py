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


def test_the_comparison_counts_the_seeds_on_each_side() -> None:
    """Both hand-made seeds favour the candidate, by 0.5 and 0.25."""
    comparison = evaluate.seed_averaged_comparison(ACTUAL, two_seed_reference(), two_seed_candidate())

    assert comparison["favourable_seeds"] == 2
    assert comparison["unfavourable_seeds"] == 0


def verdict_input(delta_mae: float, t_statistic: float, favourable: int, unfavourable: int) -> dict:
    """Build the parts of a ten-seed comparison the verdict reads.

    Takes the effect, its t, and how many seeds fell on each side. Returns the
    comparison. The seed range is set wider than every effect used here, so a
    verdict that still read it would show up.
    """
    return {
        "seeds": 10,
        "delta_mae": delta_mae,
        "t": t_statistic,
        "favourable_seeds": favourable,
        "unfavourable_seeds": unfavourable,
        "seed_range": 0.2,
    }


def test_a_small_effect_measured_reliably_is_adopted_inside_the_seed_range() -> None:
    """Stage ten's WR opponent strength: -0.0224 at t = -3.48, ten of ten seeds, inside its range.

    The dropped range bar left this unresolved however many seeds were run.
    """
    verdict = evaluate.seed_averaged_verdict(verdict_input(-0.0224, -3.48, 10, 0), 2.0, 8)
    assert verdict["verdict"] == evaluate.HELPS


def test_eight_agreeing_seeds_of_ten_is_enough_and_seven_is_not() -> None:
    """The consistency bar is inclusive at its minimum."""
    enough = evaluate.seed_averaged_verdict(verdict_input(-0.05, -4.0, 8, 2), 2.0, 8)
    too_few = evaluate.seed_averaged_verdict(verdict_input(-0.05, -4.0, 7, 3), 2.0, 8)

    assert enough["verdict"] == evaluate.HELPS
    assert too_few["verdict"] == evaluate.UNRESOLVED
    assert "7 of 10 seeds favourable, needs 8" in too_few["reason"]
    assert "|t|" not in too_few["reason"]


def test_consistent_seeds_do_not_rescue_an_imprecise_effect() -> None:
    """Every seed agreeing is not enough when the rows cannot separate the two."""
    verdict = evaluate.seed_averaged_verdict(verdict_input(-0.05, -1.5, 10, 0), 2.0, 8)

    assert verdict["verdict"] == evaluate.UNRESOLVED
    assert "|t|" in verdict["reason"]
    assert "needs" not in verdict["reason"]


def test_te_injuries_fails_both_bars() -> None:
    """Stage ten's TE injuries: -0.0089 at t = -1.37, favourable on seven seeds of ten."""
    verdict = evaluate.seed_averaged_verdict(verdict_input(-0.0089, -1.37, 7, 3), 2.0, 8)

    assert verdict["verdict"] == evaluate.UNRESOLVED
    assert "|t| 1.37 is not above 2.0" in verdict["reason"]
    assert "7 of 10 seeds favourable, needs 8" in verdict["reason"]


def test_a_harmful_effect_needs_the_same_consistency_on_the_other_side() -> None:
    """A positive effect counts the seeds it hurt on, not the ones it helped on."""
    hurts = evaluate.seed_averaged_verdict(verdict_input(0.05, 4.0, 1, 9), 2.0, 8)
    mixed = evaluate.seed_averaged_verdict(verdict_input(0.05, 4.0, 9, 1), 2.0, 8)

    assert hurts["verdict"] == evaluate.HURTS
    assert mixed["verdict"] == evaluate.UNRESOLVED
    assert "1 of 10 seeds unfavourable" in mixed["reason"]


def test_requiring_more_agreeing_seeds_than_were_run_is_refused() -> None:
    """Eleven of ten could never be met, so it is a configuration error."""
    with pytest.raises(ValueError, match="only 10 were run"):
        evaluate.seed_averaged_verdict(verdict_input(-0.05, -4.0, 10, 0), 2.0, 11)


def test_rmse_comparison_matches_hand_computed_values() -> None:
    """Reference RMSEs are root 0.5 and 1.0 over the two seeds; the candidate's are 0.0 and 0.5.

    The row term linearises RMSE: each row contributes its seed-averaged squared
    error over twice its model's seed-averaged RMSE. The reference's averaged
    squared errors are 0.5, 0.5, 0, 2 and the candidate's 0, 0, 0.5, 0.
    """
    comparison = evaluate.seed_averaged_rmse_comparison(ACTUAL, two_seed_reference(), two_seed_candidate())

    rmse_reference = [float(np.sqrt(0.5)), 1.0]
    rmse_candidate = [0.0, 0.5]
    first_difference = rmse_candidate[0] - rmse_reference[0]
    second_difference = rmse_candidate[1] - rmse_reference[1]
    contributions = np.array([0.0, 0.0, 0.5, 0.0]) / (2.0 * 0.25) - np.array([0.5, 0.5, 0.0, 2.0]) / (
        2.0 * float(np.mean(rmse_reference))
    )

    assert comparison["metric_a_by_seed"] == pytest.approx(rmse_reference)
    assert comparison["metric_b_by_seed"] == pytest.approx(rmse_candidate)
    assert comparison["delta"] == pytest.approx((first_difference + second_difference) / 2.0)
    assert comparison["se_unit"] == pytest.approx(float(np.std(contributions, ddof=1)) / 2.0)
    assert comparison["se_seed"] == pytest.approx(abs(first_difference - second_difference) / 2.0)
    assert comparison["favourable_seeds"] == 2
    assert comparison["higher_is_better"] is False


def rank_metadata(weeks: list[int]) -> pd.DataFrame:
    """Build metadata placing every row in one position and season, in the given weeks.

    Takes each row's week. Returns the frame.
    """
    positions = []
    seasons = []
    for _ in weeks:
        positions.append("WR")
        seasons.append(2024)
    return pd.DataFrame({"position": positions, "season": seasons, "week": weeks})


def test_spearman_comparison_averages_groups_and_drops_undefined_ones() -> None:
    """Two weeks of three players each, and a one-player week with no correlation.

    Reference: seed one ranks week 1 perfectly and week 2 backwards (mean 0),
    seed two scores 0.5 and 1 (mean 0.75). Candidate: 1 and 1 (mean 1), then 1
    and 0.5 (mean 0.75). Seed differences are 1 and 0, so the effect is 0.5 on
    one favourable seed. Averaged over seeds, the week differences are 0.25 and
    0.75, so the group term is 0.25; the seed term is 0.5.
    """
    weeks = [1, 1, 1, 2, 2, 2, 3]
    actual = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0])
    reference = [
        pd.Series([1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 5.0]),
        pd.Series([1.0, 3.0, 2.0, 1.0, 2.0, 3.0, 5.0]),
    ]
    candidate = [
        pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 5.0]),
        pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 3.0, 5.0]),
    ]

    comparison = evaluate.seed_averaged_spearman_comparison(actual, reference, candidate, rank_metadata(weeks))

    assert comparison["groups_dropped"] == 1
    assert comparison["units"] == 2
    assert comparison["metric_a_by_seed"] == pytest.approx([0.0, 0.75])
    assert comparison["metric_b_by_seed"] == pytest.approx([1.0, 0.75])
    assert comparison["delta"] == pytest.approx(0.5)
    assert comparison["se_unit"] == pytest.approx(0.25)
    assert comparison["se_seed"] == pytest.approx(0.5)
    assert comparison["favourable_seeds"] == 1
    assert comparison["unfavourable_seeds"] == 0


def test_a_higher_is_better_metric_is_judged_in_its_own_direction() -> None:
    """For rank correlation a consistent rise helps and a consistent fall hurts."""
    rises = {"seeds": 10, "delta": 0.05, "t": 4.0, "favourable_seeds": 10, "unfavourable_seeds": 0,
             "higher_is_better": True}
    falls = {"seeds": 10, "delta": -0.05, "t": -4.0, "favourable_seeds": 0, "unfavourable_seeds": 10,
             "higher_is_better": True}

    assert evaluate.seed_averaged_verdict(rises, 2.0, 8)["verdict"] == evaluate.HELPS
    assert evaluate.seed_averaged_verdict(falls, 2.0, 8)["verdict"] == evaluate.HURTS


def test_group_bias_catches_a_predictor_that_over_projects_low_and_under_projects_high() -> None:
    """Projections at half the spread of the actual scores, around the same mean.

    Group 1 is projected 10 and scores 5, a bias of +5; group 2 is projected
    20 and scores 25, a bias of -5. With two identical seeds the seed term is
    zero, and the row term is the SD of 7.5 and 2.5 over root 2, which is 2.5.
    """
    actual = np.array([0.0, 10.0, 20.0, 30.0])
    projection = np.array([7.5, 12.5, 17.5, 22.5])
    rows = evaluate.projection_group_bias(actual, [projection, projection], np.array([1, 1, 2, 2]), 2)

    assert rows[0]["rows"] == 2
    assert rows[0]["mean_projected"] == pytest.approx(10.0)
    assert rows[0]["mean_actual"] == pytest.approx(5.0)
    assert rows[0]["bias"] == pytest.approx(5.0)
    assert rows[0]["se_row"] == pytest.approx(2.5)
    assert rows[0]["t"] == pytest.approx(2.0)
    assert rows[0]["seeds_same_sign"] == 2
    assert rows[1]["bias"] == pytest.approx(-5.0)


def test_the_calibration_slope_is_two_for_half_spread_and_one_when_calibrated() -> None:
    """Actual scores spread twice as wide as these projections, so their slope is 2."""
    actual = np.array([0.0, 10.0, 20.0, 30.0])
    compressed = np.array([7.5, 12.5, 17.5, 22.5])

    slope = evaluate.calibration_slope(actual, [compressed, compressed])
    assert slope["slope"] == pytest.approx(2.0)
    assert slope["seeds_above_one"] == 2
    assert slope["overall_bias"] == pytest.approx(0.0)

    calibrated = evaluate.calibration_slope(actual, [actual, actual])
    assert calibrated["slope"] == pytest.approx(1.0)
    assert calibrated["seeds_above_one"] == 0


def test_compression_is_called_only_when_both_ends_and_the_slope_agree() -> None:
    """A clear low end and slope are not enough if the high end is not under-projected."""
    low = {"bias": 1.0, "t": 5.0, "seeds_same_sign": 10}
    high = {"bias": -1.0, "t": -5.0, "seeds_same_sign": 10}
    steep = {"slope": 1.2, "t_against_one": 4.0}

    assert evaluate.compression_verdict([low, high], steep, 2.0, 8)["present"] is True

    flat_high = {"bias": -0.1, "t": -1.0, "seeds_same_sign": 10}
    verdict = evaluate.compression_verdict([low, flat_high], steep, 2.0, 8)
    assert verdict["present"] is False
    assert "highest group" in verdict["reason"]
    assert "lowest group" not in verdict["reason"]
