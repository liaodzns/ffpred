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

    row_count = len(differences)
    if row_count < 2:
        return {"delta_mae": float("nan"), "se": float("nan"), "t": float("nan"), "rows": row_count}

    delta_mae = float(np.mean(differences))
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(row_count))

    t_statistic = float("nan")
    if standard_error > 0:
        t_statistic = delta_mae / standard_error

    return {
        "delta_mae": delta_mae,
        "se": standard_error,
        "t": t_statistic,
        "rows": row_count,
    }


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
