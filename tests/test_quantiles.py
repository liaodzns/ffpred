"""Tests for the quantile models' configuration, calibration statistics, and adoption rule."""

import numpy as np
import pytest

from ffml.models import quantiles

ADOPTION = {
    "max_overall_gap": 0.05,
    "max_top_group_gap": 0.08,
    "min_calibrated_seeds": 8,
    "projection_groups": 5,
}


def test_quantile_config_changes_only_the_objective_and_leaves_the_original_alone() -> None:
    """Features, folds, parameters, and seed stay the mean model's; only the loss changes."""
    config = {
        "project": {"random_seed": 42},
        "model": {
            "lightgbm": {
                "shared": {"objective": "regression", "metric": "mae", "num_leaves": 31},
                "by_position": {"QB": {"num_leaves": 15}},
            }
        },
    }
    variant = quantiles.quantile_config(config, 0.9)

    assert variant["model"]["lightgbm"]["shared"] == {
        "objective": "quantile",
        "metric": "quantile",
        "alpha": 0.9,
        "num_leaves": 31,
    }
    assert variant["model"]["lightgbm"]["by_position"] == config["model"]["lightgbm"]["by_position"]
    assert variant["project"] == config["project"]

    assert config["model"]["lightgbm"]["shared"]["objective"] == "regression"
    assert "alpha" not in config["model"]["lightgbm"]["shared"]


def test_level_suffixes_name_the_percentile() -> None:
    """Each level gets its own file, so no quantile model can overwrite a mean model."""
    assert quantiles.level_suffix(0.1) == "_q10"
    assert quantiles.level_suffix(0.5) == "_q50"
    assert quantiles.level_suffix(0.9) == "_q90"


def test_coverage_counts_rows_strictly_below_the_prediction() -> None:
    """1 < 2 and 3 < 5 count; 2 < 2 and 4 < 3 do not, so half the rows are below."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([2.0, 2.0, 5.0, 3.0])
    assert quantiles.coverage(actual, predicted) == 0.5


def test_projection_groups_are_equal_sized_with_the_highest_projections_on_top() -> None:
    """Ten rows in five groups: two each, the two highest projections in group five."""
    projection = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 10.0])
    groups = quantiles.projection_groups(projection, 5)

    for group in range(1, 6):
        assert int(np.sum(groups == group)) == 2
    assert groups[9] == 5 and groups[2] == 5
    assert groups[1] == 1 and groups[5] == 1

    # Ties are split by row order, so the result never depends on sorting luck.
    tied = quantiles.projection_groups(np.array([1.0, 1.0, 1.0, 1.0]), 2)
    assert tied.tolist() == [1, 1, 2, 2]


def test_level_calibration_matches_hand_computed_coverage() -> None:
    """One seed predicts above every score and one below, so coverage averages to the level.

    Seed A: every actual is below its prediction, coverage 1.0. Seed B: none
    is, coverage 0.0. At level 0.5 the mean gap is zero, yet neither seed is
    within any tolerance, which is exactly what the per-seed count is for.
    """
    actual = np.arange(1.0, 11.0)
    groups = quantiles.projection_groups(actual, 5)
    calibration = quantiles.level_calibration(actual, [actual + 1.0, actual - 1.0], 0.5, groups, ADOPTION)

    overall = calibration["overall"]
    assert overall["coverage"] == pytest.approx(0.5)
    assert overall["gap"] == pytest.approx(0.0)
    assert overall["gap_range"] == pytest.approx(1.0)
    assert overall["seeds_within"] == 0

    assert calibration["top_group"] == 5
    assert calibration["group_rows"][5] == 2
    assert calibration["groups"][5]["coverage"] == pytest.approx(0.5)


def calibration_from_gaps(overall_gaps: list[float], top_gaps: list[float]) -> dict:
    """Build a 0.9 level's calibration directly from per-seed gaps.

    Takes the overall gap and the top group gap for each seed. Returns the
    calibration in the shape level_verdict reads.
    """
    level = 0.9
    overall_coverages = []
    for gap in overall_gaps:
        overall_coverages.append(level + gap)
    top_coverages = []
    for gap in top_gaps:
        top_coverages.append(level + gap)

    return {
        "level": level,
        "overall": quantiles.summarise_coverage(overall_coverages, level, ADOPTION["max_overall_gap"]),
        "groups": {5: quantiles.summarise_coverage(top_coverages, level, ADOPTION["max_top_group_gap"])},
        "top_group": 5,
    }


def repeated(value: float, count: int) -> list[float]:
    """Repeat one value.

    Takes the value and how many times. Returns the list.
    """
    values = []
    for _ in range(count):
        values.append(value)
    return values


def test_a_level_within_every_bar_passes() -> None:
    """Small gaps overall and in the top group, on every seed."""
    verdict = quantiles.level_verdict(calibration_from_gaps(repeated(0.01, 10), repeated(0.03, 10)), ADOPTION)
    assert verdict["passed"] is True


def test_a_gap_of_exactly_the_tolerance_is_within_it() -> None:
    """0.95 - 0.90 is not exactly 0.05 in floating point, and must still pass."""
    verdict = quantiles.level_verdict(calibration_from_gaps(repeated(0.05, 10), repeated(0.08, 10)), ADOPTION)
    assert verdict["passed"] is True


def test_an_overall_gap_of_six_points_fails() -> None:
    """Coverage 0.96 at the 0.9 level is beyond the 0.05 overall tolerance."""
    verdict = quantiles.level_verdict(calibration_from_gaps(repeated(0.06, 10), repeated(0.03, 10)), ADOPTION)

    assert verdict["passed"] is False
    assert "overall gap +0.060 is beyond 0.05" in verdict["reason"]


def test_a_level_calibrated_overall_but_not_for_starters_fails() -> None:
    """The top group missing by 0.09 fails even though every row together is calibrated."""
    verdict = quantiles.level_verdict(calibration_from_gaps(repeated(0.01, 10), repeated(0.09, 10)), ADOPTION)

    assert verdict["passed"] is False
    assert "top group gap +0.090 is beyond 0.08" in verdict["reason"]
    assert "overall" not in verdict["reason"]


def test_a_good_average_carried_by_too_few_seeds_fails() -> None:
    """Seven seeds on target and three off: the average is fine but the count is not."""
    overall_gaps = repeated(0.0, 7)
    overall_gaps.append(0.06)
    overall_gaps.append(0.06)
    overall_gaps.append(-0.06)
    verdict = quantiles.level_verdict(calibration_from_gaps(overall_gaps, repeated(0.0, 10)), ADOPTION)

    assert verdict["passed"] is False
    assert "overall within 0.05 on 7 of 10 seeds, needs 8" in verdict["reason"]


def test_floor_and_ceiling_are_judged_independently() -> None:
    """A failing floor does not take a passing ceiling down with it."""
    floor = quantiles.level_verdict(calibration_from_gaps(repeated(0.01, 10), repeated(-0.12, 10)), ADOPTION)
    ceiling = quantiles.level_verdict(calibration_from_gaps(repeated(0.01, 10), repeated(0.02, 10)), ADOPTION)

    assert floor["passed"] is False
    assert ceiling["passed"] is True


def test_interval_width_matches_hand_computed_values() -> None:
    """Widths 2 and 4, then 2 and 2: seed means 3 and 2, averaging 2.5."""
    floors = [np.array([1.0, 2.0]), np.array([2.0, 2.0])]
    ceilings = [np.array([3.0, 6.0]), np.array([4.0, 4.0])]
    width = quantiles.interval_width(floors, ceilings, np.array([1, 2]), 2)

    assert width["mean"] == pytest.approx(2.5)
    assert width["sd"] == pytest.approx(np.sqrt(0.5))
    assert width["groups"][1] == pytest.approx(2.0)
    assert width["groups"][2] == pytest.approx(3.0)


def test_crossing_and_containment_rates() -> None:
    """One crossed row of two in the first seed and none in the second average to a quarter."""
    lower = [np.array([1.0, 5.0]), np.array([1.0, 1.0])]
    upper = [np.array([2.0, 4.0]), np.array([2.0, 2.0])]
    assert quantiles.crossing_rate(lower, upper) == pytest.approx(0.25)

    points = [np.array([2.0, 5.0])]
    floors = [np.array([1.0, 1.0])]
    ceilings = [np.array([3.0, 4.0])]
    assert quantiles.containment_rate(points, floors, ceilings) == pytest.approx(0.5)
