"""Quantile models for floor and ceiling projections, and whether they can be believed.

A mean model says what a player is expected to score. A quantile model at 0.1
says the score he falls below one game in ten, a floor, and at 0.9 the score he
stays under nine games in ten, a ceiling. They are trained with LightGBM's
quantile objective on exactly the features, folds, parameters, and seeds the
mean models use, and they replace nothing.

A floor or ceiling is only useful if it is calibrated. A 0.9 ceiling that
players beat a third of the time will still be believed, and will mislead. So
every level is judged, per position, by how often actual scores fall below it
on the walk-forward folds, averaged over the noise floor seeds: over all scored
rows, and among the highest-projected fifth of players, who are the ones a
start/sit decision is actually about. A level that fails ships nothing.

This module computes no fold and trains nothing itself. The seeds are run by
tune.seed_sweep and the folds by train.score_walk_forward, as for every other
comparison in the project.
"""

import copy
from typing import Any

import numpy as np

QUANTILE_OBJECTIVE = "quantile"

FLOOR = "floor"
CEILING = "ceiling"
BOUNDS = [FLOOR, CEILING]

# The level whose model is a second point estimate, compared with the mean model.
MEDIAN_LEVEL = 0.5

# Coverage is a fraction of rows, so a gap such as 0.95 - 0.90 can land a hair
# above 0.05 in floating point. This keeps "within 0.05" meaning within.
TOLERANCE_EPSILON = 1e-9


def quantile_config(config: dict[str, Any], level: float) -> dict[str, Any]:
    """Copy the config with the models switched to one quantile level.

    Takes the parsed config and the level. Returns a deep copy in which only
    the objective, its alpha, and the early stopping metric have changed.

    Everything else, the features, folds, per-position parameters, round
    limits, and seed, is the mean model's. Early stopping then watches the
    pinball loss at this level rather than mean absolute error.
    """
    variant = copy.deepcopy(config)
    shared = variant["model"]["lightgbm"]["shared"]
    shared["objective"] = QUANTILE_OBJECTIVE
    shared["alpha"] = level
    shared["metric"] = QUANTILE_OBJECTIVE
    return variant


def level_suffix(level: float) -> str:
    """Build the model file name suffix for a quantile level.

    Takes the level. Returns, for example, _q10 for 0.1 and _q90 for 0.9.
    """
    return f"_q{int(round(level * 100)):02d}"


def bound_levels(config: dict[str, Any]) -> dict[str, float]:
    """Map each output bound to its quantile level.

    Takes the parsed config. Returns the floor and ceiling levels by name.
    """
    quantile_settings = config["model"]["quantiles"]
    return {FLOOR: quantile_settings["floor_level"], CEILING: quantile_settings["ceiling_level"]}


def coverage(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Find the fraction of rows whose actual score fell below the prediction.

    Takes aligned actual scores and quantile predictions. Returns the fraction,
    or NaN for no rows. A 0.9 quantile is calibrated when this is near 0.9.
    """
    if len(actual) == 0:
        return float("nan")
    return float(np.mean(actual < predicted))


def projection_groups(projection: np.ndarray, group_count: int) -> np.ndarray:
    """Split rows into equal groups by projection, the highest in the last group.

    Takes each row's point projection and the number of groups. Returns each
    row's group, from 1 for the lowest projections to group_count for the
    highest. Group sizes differ by at most one row, and tied projections are
    ordered by row position so the split is deterministic.
    """
    row_count = len(projection)
    order = np.argsort(projection, kind="stable")

    groups = np.zeros(row_count, dtype=int)
    for rank in range(row_count):
        groups[order[rank]] = rank * group_count // row_count + 1
    return groups


def summarise_coverage(coverages: list[float], level: float, tolerance: float) -> dict[str, Any]:
    """Summarise one level's per-seed coverage against its nominal value.

    Takes the coverage from each seed, the level, and the gap tolerated.
    Returns the seed-averaged coverage and gap, the gap's spread across seeds,
    and how many seeds were within the tolerance.
    """
    gaps = []
    seeds_within = 0
    for seed_coverage in coverages:
        gap = seed_coverage - level
        gaps.append(gap)
        if abs(gap) <= tolerance + TOLERANCE_EPSILON:
            seeds_within += 1

    return {
        "coverage": float(np.mean(coverages)),
        "gap": float(np.mean(gaps)),
        "gap_sd": float(np.std(gaps, ddof=1)),
        "gap_range": float(np.max(gaps) - np.min(gaps)),
        "seeds_within": seeds_within,
        "seeds": len(coverages),
        "tolerance": tolerance,
    }


def level_calibration(
    actual: np.ndarray,
    predictions_by_seed: list[np.ndarray],
    level: float,
    groups: np.ndarray,
    adoption: dict[str, Any],
) -> dict[str, Any]:
    """Measure one quantile level's calibration overall and in every projection group.

    Takes the actual scores, one array of predictions per seed, the level, each
    row's projection group, and the adoption settings. Returns the overall
    summary, a summary and row count per group, and which group is the top.

    Every group is reported, not only the top one. Coverage drifting steadily
    across groups says the interval scales wrongly with projection size, while
    one bad top group on few rows is more likely noise.
    """
    group_count = adoption["projection_groups"]
    overall_coverages = []
    group_coverages: dict[int, list[float]] = {}
    for group in range(1, group_count + 1):
        group_coverages[group] = []

    for predictions in predictions_by_seed:
        overall_coverages.append(coverage(actual, predictions))
        for group in range(1, group_count + 1):
            in_group = groups == group
            group_coverages[group].append(coverage(actual[in_group], predictions[in_group]))

    by_group = {}
    group_rows = {}
    for group in range(1, group_count + 1):
        by_group[group] = summarise_coverage(group_coverages[group], level, adoption["max_top_group_gap"])
        group_rows[group] = int(np.sum(groups == group))

    return {
        "level": level,
        "overall": summarise_coverage(overall_coverages, level, adoption["max_overall_gap"]),
        "groups": by_group,
        "group_rows": group_rows,
        "top_group": group_count,
    }


def _bar_failures(name: str, summary: dict[str, Any], min_seeds: int) -> list[str]:
    """List how one coverage summary misses its bar, if it does.

    Takes a name for the rows, the coverage summary, and the seeds required.
    Returns the failures, empty when the bar is met.
    """
    failures = []
    if abs(summary["gap"]) > summary["tolerance"] + TOLERANCE_EPSILON:
        failures.append(f"{name} gap {summary['gap']:+.3f} is beyond {summary['tolerance']}")
    if summary["seeds_within"] < min_seeds:
        failures.append(
            f"{name} within {summary['tolerance']} on {summary['seeds_within']} of "
            f"{summary['seeds']} seeds, needs {min_seeds}"
        )
    return failures


def level_verdict(calibration: dict[str, Any], adoption: dict[str, Any]) -> dict[str, Any]:
    """Apply the pre-registered adoption rule to one quantile level.

    Takes the level's calibration and the adoption settings. Returns whether
    it passed and why.

    Four conditions, all required: the seed-averaged gap and the per-seed count
    within tolerance, first over all rows at the overall gap, then in the top
    projection group at its looser gap.
    """
    min_seeds = adoption["min_calibrated_seeds"]
    overall = calibration["overall"]
    top = calibration["groups"][calibration["top_group"]]

    failures = _bar_failures("overall", overall, min_seeds)
    for failure in _bar_failures("top group", top, min_seeds):
        failures.append(failure)

    if len(failures) > 0:
        return {"passed": False, "reason": "; ".join(failures)}
    return {
        "passed": True,
        "reason": (
            f"overall gap {overall['gap']:+.3f}, within {overall['tolerance']} on "
            f"{overall['seeds_within']} seeds; top group gap {top['gap']:+.3f}, within "
            f"{top['tolerance']} on {top['seeds_within']} seeds"
        ),
    }


def interval_width(
    floors_by_seed: list[np.ndarray],
    ceilings_by_seed: list[np.ndarray],
    groups: np.ndarray,
    group_count: int,
) -> dict[str, Any]:
    """Measure the mean distance between floor and ceiling, overall and per group.

    Takes the floor and ceiling predictions per seed, each row's projection
    group, and the number of groups. Returns the seed-averaged width, its
    spread across seeds, and the seed-averaged width in each group.
    """
    overall_widths = []
    group_widths: dict[int, list[float]] = {}
    for group in range(1, group_count + 1):
        group_widths[group] = []

    for floor_values, ceiling_values in zip(floors_by_seed, ceilings_by_seed, strict=True):
        widths = ceiling_values - floor_values
        overall_widths.append(float(np.mean(widths)))
        for group in range(1, group_count + 1):
            group_widths[group].append(float(np.mean(widths[groups == group])))

    by_group = {}
    for group in range(1, group_count + 1):
        by_group[group] = float(np.mean(group_widths[group]))

    return {
        "mean": float(np.mean(overall_widths)),
        "sd": float(np.std(overall_widths, ddof=1)),
        "groups": by_group,
    }


def crossing_rate(lower_by_seed: list[np.ndarray], upper_by_seed: list[np.ndarray]) -> float:
    """Find how often a lower quantile model predicts above a higher one.

    Takes the lower and upper level's predictions per seed. Returns the
    seed-averaged fraction of rows where they cross.

    Each level is its own model, so nothing forces a floor below its ceiling.
    """
    rates = []
    for lower, upper in zip(lower_by_seed, upper_by_seed, strict=True):
        rates.append(float(np.mean(lower > upper)))
    return float(np.mean(rates))


def containment_rate(
    points_by_seed: list[np.ndarray],
    floors_by_seed: list[np.ndarray],
    ceilings_by_seed: list[np.ndarray],
) -> float:
    """Find how often the shipped point projection sits between floor and ceiling.

    Takes the point projection per seed, repeated for a deterministic baseline,
    and the floor and ceiling per seed. Returns the seed-averaged fraction of
    rows inside the interval, bounds included.
    """
    rates = []
    for point, floor_values, ceiling_values in zip(
        points_by_seed, floors_by_seed, ceilings_by_seed, strict=True
    ):
        inside = (point >= floor_values) & (point <= ceiling_values)
        rates.append(float(np.mean(inside)))
    return float(np.mean(rates))


def crossing_rates(predictions_by_level: dict[float, list[np.ndarray]]) -> dict[str, float]:
    """Find how often neighbouring levels cross, and how often the outermost pair does.

    Takes each level's predictions per seed. Returns the seed-averaged crossing
    rate under a name such as q10_above_q50.
    """
    ordered = sorted(predictions_by_level)
    rates = {}
    for index in range(len(ordered) - 1):
        lower = ordered[index]
        upper = ordered[index + 1]
        name = f"{level_suffix(lower)[1:]}_above_{level_suffix(upper)[1:]}"
        rates[name] = crossing_rate(predictions_by_level[lower], predictions_by_level[upper])

    if len(ordered) > 2:
        lowest = ordered[0]
        highest = ordered[-1]
        name = f"{level_suffix(lowest)[1:]}_above_{level_suffix(highest)[1:]}"
        rates[name] = crossing_rate(predictions_by_level[lowest], predictions_by_level[highest])
    return rates


def calibrate_position(
    actual: np.ndarray,
    points_by_seed: list[np.ndarray],
    predictions_by_level: dict[float, list[np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calibrate every quantile level for one position and judge each against the rule.

    Takes the actual scores, the position's shipped point projection per seed,
    each level's predictions per seed, and the parsed config. Returns each
    level's calibration and verdict, the floor to ceiling width, the crossing
    rates, how often the point projection sits inside the interval, and the
    row count of each projection group.

    Groups are fixed once, from the seed-averaged point projection, so every
    seed's coverage in a group describes the same players.
    """
    adoption = config["model"]["quantiles"]["adoption"]
    group_count = adoption["projection_groups"]
    groups = projection_groups(np.mean(points_by_seed, axis=0), group_count)

    levels = {}
    for level in predictions_by_level:
        calibration = level_calibration(actual, predictions_by_level[level], level, groups, adoption)
        calibration["verdict"] = level_verdict(calibration, adoption)
        levels[level] = calibration

    bounds = bound_levels(config)
    floors = predictions_by_level[bounds[FLOOR]]
    ceilings = predictions_by_level[bounds[CEILING]]
    group_rows = {}
    for group in range(1, group_count + 1):
        group_rows[group] = int(np.sum(groups == group))

    return {
        "levels": levels,
        "width": interval_width(floors, ceilings, groups, group_count),
        "crossing": crossing_rates(predictions_by_level),
        "containment": containment_rate(points_by_seed, floors, ceilings),
        "group_rows": group_rows,
    }
