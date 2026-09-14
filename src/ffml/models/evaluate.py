"""Metrics for scoring predictions against actual fantasy points.

Every model, the naive baseline included, is scored through this module so
that the numbers stay comparable across stages.

Mean absolute error and root mean squared error say how far the projections
land from the truth. The rank correlation matters most in practice, because
the real decision a manager makes is which of two players to start, not what
either one scores exactly.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logger = logging.getLogger(__name__)

# Rank correlation is computed inside each of these groups and then averaged.
# Comparing a quarterback against a tight end, or week 1 against week 12, would
# measure nothing a manager acts on.
#
# Season belongs in the key even though a single season report does not need
# it. Once walk-forward folds are pooled, grouping on week alone would merge
# week 5 of one season with week 5 of another into one ranking, which is not a
# choice any manager ever faced.
SPEARMAN_GROUP_COLUMNS = ["position", "season", "week"]

# Metric keys in the order they are printed, with their display labels.
METRIC_LABELS = [
    ("mae", "MAE (fantasy points)"),
    ("rmse", "RMSE (fantasy points)"),
    ("spearman", "Spearman rho, within position and week"),
]


def compute_spearman_within_position_week(
    y_true: pd.Series, y_pred: pd.Series, metadata: pd.DataFrame
) -> tuple[float, int]:
    """Average the rank correlation computed inside each position and week.

    Takes the actual values, the predictions, and a frame carrying the position
    and week of each row. Returns the mean correlation and the number of groups
    that were skipped.

    A group holding fewer than two players is skipped, because a correlation
    needs at least two points to exist. A group where every value ties is
    skipped too, since scipy reports that as undefined rather than as zero.
    """
    grouped = metadata.groupby(SPEARMAN_GROUP_COLUMNS, observed=True)

    correlations = []
    skipped_groups = 0
    for group_key in grouped.groups:
        row_index = grouped.groups[group_key]
        if len(row_index) < 2:
            skipped_groups += 1
            continue

        correlation = spearmanr(y_true.loc[row_index], y_pred.loc[row_index]).statistic
        if pd.isna(correlation):
            skipped_groups += 1
            continue
        correlations.append(correlation)

    if skipped_groups > 0:
        logger.info(
            "Rank correlation: skipped %s of %s position-week groups as undefined",
            skipped_groups,
            len(grouped.groups),
        )

    if len(correlations) == 0:
        return float("nan"), skipped_groups
    return float(np.mean(correlations)), skipped_groups


def compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, metadata: pd.DataFrame, metrics: list[str]
) -> dict[str, Any]:
    """Compute the named metrics for one set of predictions.

    Takes the actual values, the predictions, the metadata frame, and the list
    of metric names from evaluation.metrics in the config. Returns a dictionary
    of metric name to value. Raises ValueError for an unknown metric name.
    """
    computed: dict[str, Any] = {}
    for metric_name in metrics:
        if metric_name == "mae":
            computed["mae"] = float(mean_absolute_error(y_true, y_pred))
        elif metric_name == "rmse":
            computed["rmse"] = float(root_mean_squared_error(y_true, y_pred))
        elif metric_name == "spearman_within_position_week":
            correlation, skipped_groups = compute_spearman_within_position_week(
                y_true, y_pred, metadata
            )
            computed["spearman"] = correlation
            computed["groups_skipped"] = skipped_groups
        else:
            raise ValueError(
                f"Unknown metric '{metric_name}' in evaluation.metrics. Supported metrics "
                "are mae, rmse, and spearman_within_position_week."
            )
    return computed


def _breakdown_table(
    y_true: pd.Series,
    y_pred: pd.Series,
    metadata: pd.DataFrame,
    metrics: list[str],
    breakdown_column: str,
) -> pd.DataFrame:
    """Compute the metrics separately for each value of one column.

    Takes the actual values, the predictions, the metadata frame, the metric
    names, and the column to break out by. Returns a frame with one row per
    group.
    """
    grouped = metadata.groupby(breakdown_column, observed=True)

    table_rows = []
    for group_value in grouped.groups:
        row_index = grouped.groups[group_value]
        group_metrics = compute_metrics(
            y_true.loc[row_index], y_pred.loc[row_index], metadata.loc[row_index], metrics
        )
        group_metrics[breakdown_column] = group_value
        group_metrics["rows"] = len(row_index)
        table_rows.append(group_metrics)

    table = pd.DataFrame(table_rows)

    ordered_columns = [breakdown_column, "rows"]
    for metric_key, _ in METRIC_LABELS:
        if metric_key in table.columns:
            ordered_columns.append(metric_key)
    return table[ordered_columns].sort_values(breakdown_column).reset_index(drop=True)


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    metadata: pd.DataFrame,
    metrics: list[str],
    breakdowns: list[str],
) -> dict[str, Any]:
    """Score predictions overall and broken out by the requested columns.

    Takes the actual values, the predictions, a metadata frame carrying at
    least position, week, and season, the metric names from the config, and the
    breakdown columns from the config. Returns a dictionary holding the row
    counts, the overall metrics, and one table per breakdown.

    Rows with no prediction are dropped and counted rather than scored. The
    baseline cannot project a player without enough history, and counting those
    rows as errors would punish it for declining to guess.
    """
    usable_rows = y_true.notna() & y_pred.notna()
    rows_dropped = int((~usable_rows).sum())
    if rows_dropped > 0:
        logger.info(
            "Evaluating %s rows; dropped %s with no prediction available",
            int(usable_rows.sum()),
            rows_dropped,
        )

    y_true = y_true[usable_rows]
    y_pred = y_pred[usable_rows]
    metadata = metadata[usable_rows]

    results: dict[str, Any] = {
        "rows_evaluated": len(y_true),
        "rows_dropped": rows_dropped,
        "overall": compute_metrics(y_true, y_pred, metadata, metrics),
    }

    breakdown_tables = {}
    for breakdown_column in breakdowns:
        breakdown_tables[breakdown_column] = _breakdown_table(
            y_true, y_pred, metadata, metrics, breakdown_column
        )
    results["breakdowns"] = breakdown_tables

    return results


def run_evaluation(
    y_true: pd.Series,
    y_pred: pd.Series,
    metadata: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Score predictions using the metric and breakdown lists in the config.

    Takes the actual values, the predictions, the metadata frame, and the
    parsed config. Returns the results dictionary from evaluate_predictions.
    """
    return evaluate_predictions(
        y_true,
        y_pred,
        metadata,
        config["evaluation"]["metrics"],
        config["evaluation"]["breakdowns"],
    )


def print_evaluation_report(results: dict[str, Any], title: str) -> None:
    """Print the evaluation results as readable tables.

    Takes the results dictionary and a title describing what was scored.
    Returns nothing.
    """
    print("")
    print(title)
    print("=" * 72)
    print(f"  rows evaluated   {results['rows_evaluated']:,}")
    print(f"  rows dropped     {results['rows_dropped']:,} (no projection available)")
    print("")

    print("Overall")
    print("-" * 72)
    overall = results["overall"]
    for metric_key, metric_label in METRIC_LABELS:
        if metric_key in overall:
            print(f"  {metric_label:<42} {overall[metric_key]:>8.4f}")
    if "groups_skipped" in overall:
        print(f"  {'position-week groups skipped':<42} {overall['groups_skipped']:>8}")
    print("")

    for breakdown_column in results["breakdowns"]:
        print(f"By {breakdown_column}")
        print("-" * 72)
        table = results["breakdowns"][breakdown_column]
        print(table.round(4).to_string(index=False))
        print("")


def paired_error_comparison(
    y_true: pd.Series, y_pred_a: pd.Series, y_pred_b: pd.Series
) -> dict[str, Any]:
    """Test whether two sets of predictions really differ in accuracy.

    Takes the actual values and two aligned sets of predictions. Returns the
    difference in mean absolute error (b minus a), its standard error, the t
    statistic, and how many rows were compared.

    The comparison is paired, row by row. Both models face the same weeks and
    the same players, so almost all of the variance in fantasy scoring is
    common to them and cancels. An unpaired comparison of two MAEs would be
    swamped by how noisy the sport is and would call every real difference
    undetectable.

    A t much beyond two is a difference worth believing. Anything smaller means
    the data cannot separate the two models, which is a finding rather than a
    failure to measure.
    """
    usable_rows = y_true.notna() & y_pred_a.notna() & y_pred_b.notna()
    actual = y_true[usable_rows]

    errors_a = (actual - y_pred_a[usable_rows]).abs()
    errors_b = (actual - y_pred_b[usable_rows]).abs()
    differences = errors_b.to_numpy() - errors_a.to_numpy()

    delta_mae, standard_error, t_statistic = _mean_difference_and_se(differences)
    return {
        "delta_mae": delta_mae,
        "se": standard_error,
        "t": t_statistic,
        "rows": len(differences),
    }


def _mean_difference_and_se(differences: np.ndarray) -> tuple[float, float, float]:
    """Summarise paired differences as their mean, standard error, and t statistic.

    Takes the paired differences. Returns the mean, its standard error, and the
    t statistic. All three are NaN with fewer than two differences, and t is NaN
    when the standard error is zero.
    """
    count = len(differences)
    if count < 2:
        return float("nan"), float("nan"), float("nan")

    mean_difference = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(count))

    t_statistic = float("nan")
    if standard_error > 0:
        t_statistic = mean_difference / standard_error
    return mean_difference, standard_error, t_statistic


# Verdicts of the seed-averaged rule, read as what the candidate does to the reference.
HELPS = "helps"
HURTS = "hurts"
UNRESOLVED = "unresolved"


def _check_seed_predictions(
    y_true: pd.Series, predictions_a: list[pd.Series], predictions_b: list[pd.Series]
) -> None:
    """Refuse per-seed predictions that cannot be paired by seed and by row.

    Takes the actual values and the two lists of per-seed predictions. Returns
    nothing. Raises ValueError if the seed counts differ, fewer than two seeds
    were run, or any set of predictions has a different length from the actuals.
    """
    if len(predictions_a) != len(predictions_b):
        raise ValueError(
            f"Seed counts differ: {len(predictions_a)} against {len(predictions_b)}. "
            "Both configurations must be trained under the same seeds."
        )
    if len(predictions_a) < 2:
        raise ValueError(
            "At least two seeds are needed, because a single seed gives no estimate of "
            "how far training randomness moves the result."
        )

    all_predictions = predictions_a + predictions_b
    for predictions in all_predictions:
        if len(predictions) != len(y_true):
            raise ValueError(
                f"A set of predictions has {len(predictions)} rows against {len(y_true)} actuals."
            )


def _per_seed_absolute_errors(
    actual: np.ndarray, predictions: list[pd.Series], usable_rows: np.ndarray
) -> list[np.ndarray]:
    """Compute each seed's absolute errors on the usable rows.

    Takes the usable actual values, the per-seed predictions, and the usable row
    mask. Returns one error array per seed, in seed order.
    """
    errors_by_seed = []
    for seed_predictions in predictions:
        projected = seed_predictions.to_numpy(dtype=float)[usable_rows]
        errors_by_seed.append(np.abs(actual - projected))
    return errors_by_seed


def _seed_spread(maes: list[float]) -> dict[str, float]:
    """Summarise one configuration's MAE across seeds.

    Takes the per-seed MAEs. Returns the mean, the standard deviation, and the
    best-to-worst range. A deterministic model scores the same under every
    seed, so both its spread figures are zero.
    """
    return {
        "mean": float(np.mean(maes)),
        "sd": float(np.std(maes, ddof=1)),
        "range": float(np.max(maes) - np.min(maes)),
    }


def seed_averaged_comparison(
    y_true: pd.Series, predictions_a: list[pd.Series], predictions_b: list[pd.Series]
) -> dict[str, Any]:
    """Compare two configurations trained under the same seeds, counting seed noise.

    Takes the actual values and, for each configuration, one set of predictions
    per seed, both lists in the same seed order and every set on the same rows.
    A deterministic reference such as the naive baseline passes the same
    predictions once per seed. Returns each side's mean, spread, and range, the
    per-seed MAE differences, the effect (b minus a), its row and seed standard
    errors, the combined standard error, t, and the row-only t for contrast.

    paired_error_comparison counts which games happened to be scored but not
    the randomness of training, so two seeds of one model can "differ" at
    t = 3. Here the effect is the mean of the per-seed MAE differences, and its
    standard error adds two independent sources:

    - the seed term, the spread of those differences over the square root of
      the seed count, which is the training randomness left after averaging.
      Pairing by seed is valid whatever the correlation between the two sides
      and gains from it, since bagging draws depend only on the seed.
    - the row term, the paired standard error of the seed-averaged per-row
      errors, which is the uncertainty about which games were played. More
      seeds cannot shrink it.

    The small independent part of per-row seed noise sits in both terms, so the
    combined error is slightly conservative.
    """
    _check_seed_predictions(y_true, predictions_a, predictions_b)

    usable_rows: np.ndarray = y_true.notna().to_numpy()
    all_predictions = predictions_a + predictions_b
    for predictions in all_predictions:
        usable_rows = usable_rows & predictions.notna().to_numpy()
    rows_dropped = int((~usable_rows).sum())
    if rows_dropped > 0:
        logger.info("Seed-averaged comparison: dropped %s rows with a missing value", rows_dropped)
    if int(usable_rows.sum()) < 2:
        raise ValueError("Fewer than two rows have every prediction, so nothing can be compared.")

    actual = y_true.to_numpy(dtype=float)[usable_rows]
    errors_a = _per_seed_absolute_errors(actual, predictions_a, usable_rows)
    errors_b = _per_seed_absolute_errors(actual, predictions_b, usable_rows)

    maes_a = []
    maes_b = []
    seed_differences = []
    for seed_index in range(len(errors_a)):
        mae_a = float(np.mean(errors_a[seed_index]))
        mae_b = float(np.mean(errors_b[seed_index]))
        maes_a.append(mae_a)
        maes_b.append(mae_b)
        seed_differences.append(mae_b - mae_a)

    return _combine_seed_and_row_terms(errors_a, errors_b, maes_a, maes_b, seed_differences)


def _combine_seed_and_row_terms(
    errors_a: list[np.ndarray],
    errors_b: list[np.ndarray],
    maes_a: list[float],
    maes_b: list[float],
    seed_differences: list[float],
) -> dict[str, Any]:
    """Build the seed-averaged comparison from per-seed errors and MAEs.

    Takes each side's per-seed error arrays and MAEs, and the per-seed MAE
    differences. Returns the comparison dictionary described in
    seed_averaged_comparison. Raises ValueError if the two ways of computing
    the effect disagree, which would mean the rows were not paired.
    """
    delta_mae, se_seed, _ = _mean_difference_and_se(np.array(seed_differences))

    row_differences = np.mean(errors_b, axis=0) - np.mean(errors_a, axis=0)
    row_delta, se_row, row_only_t = _mean_difference_and_se(row_differences)

    # Averaging over seeds then rows equals averaging over rows then seeds. A
    # mismatch can only come from rows that do not line up.
    if not np.isclose(delta_mae, row_delta, rtol=0.0, atol=1e-9):
        raise ValueError(f"Effect computed by seed ({delta_mae}) and by row ({row_delta}) disagree.")

    standard_error = float(np.sqrt(se_row**2 + se_seed**2))
    t_statistic = float("nan")
    if standard_error > 0:
        t_statistic = delta_mae / standard_error

    spread_a = _seed_spread(maes_a)
    spread_b = _seed_spread(maes_b)
    favourable_seeds, unfavourable_seeds = _count_seed_directions(seed_differences)
    return {
        "seeds": len(maes_a),
        "rows": len(row_differences),
        "mae_a_mean": spread_a["mean"],
        "mae_a_sd": spread_a["sd"],
        "mae_a_range": spread_a["range"],
        "mae_b_mean": spread_b["mean"],
        "mae_b_sd": spread_b["sd"],
        "mae_b_range": spread_b["range"],
        "seed_range": max(spread_a["range"], spread_b["range"]),
        "seed_differences": seed_differences,
        "favourable_seeds": favourable_seeds,
        "unfavourable_seeds": unfavourable_seeds,
        "delta_mae": delta_mae,
        "se_row": se_row,
        "se_seed": se_seed,
        "se": standard_error,
        "t": t_statistic,
        "row_only_t": row_only_t,
    }


def _count_seed_directions(seed_differences: list[float]) -> tuple[int, int]:
    """Count the seeds on which the candidate scored better, and worse.

    Takes the per-seed MAE differences, candidate minus reference. Returns how
    many are below zero and how many above. A seed with no difference counts as
    neither.
    """
    favourable = 0
    unfavourable = 0
    for difference in seed_differences:
        if difference < 0:
            favourable += 1
        elif difference > 0:
            unfavourable += 1
    return favourable, unfavourable


def seed_averaged_verdict(
    comparison: dict[str, Any], decision_t: float, min_favourable_seeds: int
) -> dict[str, str]:
    """Apply the pre-registered adoption rule to a seed-averaged comparison.

    Takes the comparison from seed_averaged_comparison, the decision threshold
    in standard errors, and how many seeds must agree with the direction of the
    effect. Returns the verdict, helps, hurts, or unresolved, read as what the
    candidate does, and the reason. Raises ValueError if more seeds must agree
    than were run.

    Both bars are required. The t bar says the averaged effect is known
    precisely enough, counting both seed and row noise. The consistency bar
    says the effect is not the work of a few lucky seeds.

    Stage ten's rule also required the effect to exceed the seed range. That
    bar was dropped: the range describes how much a single run varies, not how
    precisely an average of many runs is known, and it never shrinks as seeds
    are added, so a small effect measured reliably could never pass it. The
    range is still reported, but it decides nothing.
    """
    seed_count = comparison["seeds"]
    if min_favourable_seeds > seed_count:
        raise ValueError(f"{min_favourable_seeds} seeds must agree but only {seed_count} were run.")

    delta_mae = comparison["delta_mae"]
    t_statistic = comparison["t"]
    clears_t = bool(pd.notna(t_statistic) and abs(t_statistic) > decision_t)

    # A harmful effect is held to the same consistency, counted on the other side.
    agreeing_seeds = comparison["favourable_seeds"]
    direction = "favourable"
    decided = HELPS
    if pd.notna(delta_mae) and delta_mae > 0:
        agreeing_seeds = comparison["unfavourable_seeds"]
        direction = "unfavourable"
        decided = HURTS
    consistent = agreeing_seeds >= min_favourable_seeds
    consistency = f"{agreeing_seeds} of {seed_count} seeds {direction}, needs {min_favourable_seeds}"

    if clears_t and consistent:
        return {"verdict": decided, "reason": f"|t| {abs(t_statistic):.2f} > {decision_t}; {consistency}"}

    failures = []
    if not clears_t:
        failures.append(f"|t| {abs(t_statistic):.2f} is not above {decision_t}")
    if not consistent:
        failures.append(consistency)
    return {"verdict": UNRESOLVED, "reason": "; ".join(failures)}


def two_sided_p_value(t_statistic: float) -> float:
    """Convert a t statistic into a two-sided p value.

    Takes the t statistic. Returns the p value, NaN if t is undefined.

    The normal distribution stands in for Student's t. The smallest paired
    comparison in this project covers over a thousand rows, where the two are
    indistinguishable.
    """
    if pd.isna(t_statistic):
        return float("nan")
    return float(2.0 * norm.sf(abs(t_statistic)))


def bonferroni_t_threshold(family_alpha: float, test_count: int) -> float:
    """Find how large |t| must be to survive a Bonferroni correction.

    Takes the family-wise error rate and the number of tests. Returns the
    absolute t statistic a single test must exceed.
    """
    return float(norm.isf(family_alpha / (2.0 * test_count)))


def _holm_verdicts(p_values: pd.Series, family_alpha: float) -> list[bool]:
    """Apply Holm's step-down procedure to a set of p values.

    Takes the p values, indexed 0 to n-1, and the family-wise error rate.
    Returns whether each test survives, in the same order.

    The smallest p value is held to the Bonferroni bar, the next to a bar one
    test less strict, and so on. The first test that fails stops the procedure,
    and every test with a larger p value fails with it. It controls the same
    error rate as Bonferroni while rejecting at least as often.
    """
    test_count = len(p_values)
    ordered_labels = p_values.fillna(1.0).sort_values(kind="mergesort").index.tolist()

    verdicts = []
    for _ in range(test_count):
        verdicts.append(False)

    for rank in range(test_count):
        label = ordered_labels[rank]
        p_value = p_values.loc[label]
        if pd.isna(p_value) or p_value > family_alpha / (test_count - rank):
            break
        verdicts[label] = True

    return verdicts


def correct_for_multiple_tests(
    tests: pd.DataFrame, family_alpha: float, decision_t: float
) -> pd.DataFrame:
    """Add raw p values and multiple-comparison verdicts to a table of tests.

    Takes a frame with one row per test carrying a t column, the family-wise
    error rate, and the decision threshold in standard errors. Returns a copy
    with p_value, clears_decision_t, survives_bonferroni, and survives_holm.

    Running many tests guarantees that some clear two standard errors by chance
    alone. The corrections say which results would still stand if every test
    in the family were held to account together, which is the honest bar for
    believing any one of them.
    """
    corrected = tests.reset_index(drop=True).copy()
    test_count = len(corrected)

    p_values = []
    clears = []
    for t_statistic in corrected["t"]:
        p_values.append(two_sided_p_value(t_statistic))
        clears.append(bool(pd.notna(t_statistic) and abs(t_statistic) > decision_t))
    corrected["p_value"] = p_values
    corrected["clears_decision_t"] = clears

    bonferroni = []
    for p_value in corrected["p_value"]:
        bonferroni.append(bool(pd.notna(p_value) and p_value < family_alpha / test_count))
    corrected["survives_bonferroni"] = bonferroni
    corrected["survives_holm"] = _holm_verdicts(corrected["p_value"], family_alpha)

    return corrected


def _check_same_rows(named_results: list[tuple[str, dict[str, Any]]]) -> None:
    """Warn if the scored sets do not cover the same number of rows.

    Takes the named results. Returns nothing.

    Two models scored on different rows cannot be compared, and the difference
    is easy to introduce by accident: the baseline declines to project a player
    without enough history, so any change to that rule silently moves one side
    of the comparison and not the other.
    """
    row_counts = []
    for _, results in named_results:
        row_counts.append(results["rows_evaluated"])

    for row_count in row_counts:
        if row_count != row_counts[0]:
            logger.warning(
                "The comparison is not like for like: rows evaluated differ across "
                "the scored sets (%s). Both must be scored on the same rows.",
                row_counts,
            )
            return


def print_comparison_report(
    named_results: list[tuple[str, dict[str, Any]]], title: str
) -> None:
    """Print two or more scored predictions side by side.

    Takes an ordered list of (name, results) pairs and a title describing what
    was scored. Returns nothing.

    The last entry is treated as the reference to beat, normally the naive
    baseline, so the delta column reads as first minus last.
    """
    _check_same_rows(named_results)

    print("")
    print(title)
    print("=" * 72)
    print(f"  rows evaluated   {named_results[0][1]['rows_evaluated']:,}")
    print(f"  rows dropped     {named_results[0][1]['rows_dropped']:,} (no projection available)")
    print("")

    header = f"{'Metric':<38}"
    for name, _ in named_results:
        header = header + f"{name:>11}"
    header = header + f"{'delta':>11}"
    print(header)
    print("-" * len(header))

    first_metrics = named_results[0][1]["overall"]
    last_metrics = named_results[-1][1]["overall"]

    for metric_key, metric_label in METRIC_LABELS:
        if metric_key not in first_metrics:
            continue
        line = f"{metric_label:<38}"
        for _, results in named_results:
            line = line + f"{results['overall'][metric_key]:>11.4f}"
        delta = first_metrics[metric_key] - last_metrics[metric_key]
        line = line + f"{delta:>+11.4f}"
        print(line)

    print("")
    print(f"  delta is {named_results[0][0]} minus {named_results[-1][0]}.")
    print("  Lower is better for MAE and RMSE; higher is better for Spearman rho.")
    print("")
