"""Train one model per position and score each against its naive baseline.

Five modes. --baseline-only scores the naive last-N-game average and trains
nothing. --feature-study runs the pre-registered test inventory: every model
against its baseline, and each listed feature group flipped once against its
position's shipped configuration, with a multiple-comparisons report across the
whole family. --deploy trains the live models on every completed week, saved
apart from the evaluation models and never scored. --noise-floor retrains each
walk-forward model under several seeds to measure how far seed luck alone moves
it, which gates whether tuning is worth running. Otherwise every position's
walk-forward model is trained, scored, and saved.

Every mode retrains once per walk-forward test season and pools the scored
rows, rather than betting a verdict on a single holdout year.

This script stays thin. It reads the config, calls into ffml, and prints a
summary. All of the real work lives in src/ffml/models/ and src/ffml/features/.
"""

import argparse
import copy
import logging
import sys
from typing import Any

import pandas as pd

from ffml.config import ConfigError, load_config
from ffml.features import build_dataset
from ffml.features.rolling import rolling_feature_name
from ffml.models import baseline, evaluate, train, tune
from ffml.utils.io import read_parquet, resolve_path

# Columns the evaluation groups by when breaking results out.
METADATA_COLUMNS = ["position", "week", "season"]

# Columns identifying a scored row, used to prove two runs scored the same rows.
ROW_KEY_COLUMNS = ["player_id", "season", "week"]

# Columns the projections are attached to.
BASELINE_COLUMN = "baseline_projection"
MODEL_COLUMN = "model_projection"

# Largest difference tolerated between the baseline and the equivalent rolling
# feature. They are computed by the same code, so they should agree exactly.
BASELINE_MATCH_TOLERANCE = 1e-9

# Label of the model-against-baseline test in the study table.
BASELINE_TEST = "model vs baseline"

# How many features the importance report shows.
TOP_FEATURES = 15

# The pre-registered test inventory for stage seven, fixed before anything ran.
# Each group is flipped once against its position's shipped configuration:
# removed if it is on, added if it is off. WR's other groups were settled in
# stage six, so only the new injury group is tested for receivers.
STUDY_GROUPS_BY_POSITION = {
    "QB": ["game_context", "opponent_strength", "opportunity", "player_attributes", "injuries"],
    "RB": ["game_context", "opponent_strength", "opportunity", "player_attributes", "injuries"],
    "WR": ["injuries"],
    "TE": ["game_context", "opponent_strength", "opportunity", "player_attributes", "injuries"],
}

# The pre-registered seed-averaged re-tests for stage ten, fixed before anything ran.
# Only the decisions whose single-seed effect fell inside, or barely outside, their
# position's stage nine seed range are re-tested; the rest of the study stands. Each
# group is on in the shipped set and is tested as a single addition to that set.
SEED_AVERAGED_RETESTS = {
    "QB": ["game_context", "opportunity"],
    "WR": ["opponent_strength"],
    "TE": ["injuries"],
}

# How each seed-averaged verdict reads, for a model against its baseline and for a group.
HEADLINE_VERDICTS = {
    evaluate.HELPS: "model beats baseline",
    evaluate.HURTS: "BASELINE BEATS MODEL",
    evaluate.UNRESOLVED: "not distinguishable from baseline",
}
RETEST_VERDICTS = {
    evaluate.HELPS: "group helps: keep on",
    evaluate.HURTS: "group hurts: turn off",
    evaluate.UNRESOLVED: "unresolved: leave toggle",
}


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Takes nothing. Returns the parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train per-position models and compare them against the naive baseline."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to config/config.yaml at the project root.",
    )
    parser.add_argument(
        "--position",
        default=None,
        help="Train one position only. Defaults to every position in data.positions.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run the naive last-N-game average and print its evaluation, training nothing.",
    )
    parser.add_argument(
        "--feature-study",
        action="store_true",
        help="Run the pre-registered feature group tests with multiple-comparison correction.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help=(
            "Train the live deployment models on every completed week. They are saved "
            "separately and never touch the walk-forward evaluation models."
        ),
    )
    parser.add_argument(
        "--noise-floor",
        action="store_true",
        help=(
            "Retrain each walk-forward model under every seed in model.tuning and report "
            "how far seed luck alone moves it. Saves nothing."
        ),
    )
    parser.add_argument(
        "--seed-averaged",
        action="store_true",
        help=(
            "Restate each model against its baseline and re-test the marginal group decisions, "
            "averaged over every seed in model.tuning. Saves nothing."
        ),
    )
    return parser.parse_args()


def configure_logging(config: dict) -> None:
    """Set up log output using the level named in the config.

    Takes the parsed config. Returns nothing.
    """
    logging.basicConfig(
        level=config["logging"]["level"].upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )


def positions_to_run(config: dict[str, Any], requested: str | None) -> list[str]:
    """Decide which positions to train.

    Takes the parsed config and the position asked for, or None for all.
    Returns the positions. Raises ValueError for a position not in the config.
    """
    positions = config["data"]["positions"]
    if requested is None:
        return list(positions)
    if requested not in positions:
        raise ValueError(f"Unknown position '{requested}'. data.positions lists {positions}.")
    return [requested]


def load_player_weeks_with_baseline(config: dict[str, Any]) -> pd.DataFrame:
    """Read the player-week table and attach the naive baseline projection.

    Takes the parsed config. Returns the player-week frame with the baseline
    column added.

    The baseline is computed on the unfiltered table rather than on a feature
    matrix, which has already dropped each player's first games and the warmup
    season. Computing it there would leave a player's first surviving game with
    no history behind it and understate the baseline.
    """
    processed_directory = resolve_path(config["data"]["paths"]["processed"])
    player_weeks = read_parquet(processed_directory / "player_weeks.parquet")
    player_weeks[BASELINE_COLUMN] = baseline.run_baseline(player_weeks, config)
    return player_weeks


def attach_baseline(features: pd.DataFrame, player_weeks: pd.DataFrame) -> pd.DataFrame:
    """Join the baseline projection onto a feature matrix.

    Takes the feature matrix and the player-week frame carrying the baseline.
    Returns the matrix with the baseline attached. Raises ValueError if the join
    changed the row count.
    """
    projections = player_weeks[ROW_KEY_COLUMNS + [BASELINE_COLUMN]]

    row_count_before = len(features)
    merged = features.merge(projections, on=ROW_KEY_COLUMNS, how="left")
    if len(merged) != row_count_before:
        raise ValueError(
            f"Attaching the baseline changed the row count from {row_count_before} to "
            f"{len(merged)}. The player-week key is not unique."
        )
    return merged


def check_baseline_matches_feature(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    """Check the baseline agrees with the equivalent rolling feature.

    Takes the frame carrying both and the parsed config. Returns nothing.
    Raises ValueError if they disagree.

    The baseline and the rolling mean of the target over the same window come
    from two separately invoked code paths that must agree by construction, so
    a disagreement means the feature pipeline has drifted from the number the
    model is being asked to beat.
    """
    window = config["model"]["baseline"]["window"]
    feature_name = rolling_feature_name(config["scoring"]["target_column"], window)
    if feature_name not in frame.columns:
        return

    disagreeing_nulls = int((frame[BASELINE_COLUMN].isna() != frame[feature_name].isna()).sum())
    largest_difference = float((frame[BASELINE_COLUMN] - frame[feature_name]).abs().max())

    if disagreeing_nulls > 0 or largest_difference > BASELINE_MATCH_TOLERANCE:
        raise ValueError(
            f"The baseline and {feature_name} disagree: {disagreeing_nulls} rows differ on "
            f"whether a value exists, and the largest difference is {largest_difference}."
        )


def score_folds(
    features: pd.DataFrame, position: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Train one model per fold and gather every scored row.

    Takes the feature matrix, the position, and the parsed config. Returns the
    pooled scored rows and the per-fold training results.

    The fold loop lives in train.score_walk_forward, which the noise floor
    measurement shares, so both report numbers from the same code.
    """
    return train.score_walk_forward(features, position, config)


def score_predictions(
    scored: pd.DataFrame, projection_column: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Score one projection column over a set of rows.

    Takes the scored rows, which projection to read, and the parsed config.
    Returns the evaluation results dictionary.
    """
    return evaluate.run_evaluation(
        scored[config["scoring"]["target_column"]],
        scored[projection_column],
        scored[METADATA_COLUMNS],
        config,
    )


def run_baseline_evaluation(config: dict) -> None:
    """Project the baseline and score it on the validation season.

    Takes the parsed config. Returns nothing.
    """
    player_weeks = load_player_weeks_with_baseline(config)

    validation_season = config["validation"]["validation_season"]
    validation_rows = player_weeks[player_weeks["season"] == validation_season]

    results = score_predictions(validation_rows, BASELINE_COLUMN, config)
    window = config["model"]["baseline"]["window"]
    evaluate.print_evaluation_report(
        results,
        f"Baseline: mean of last {window} games, validation season {validation_season}",
    )


def _metrics_row(label: str, results: dict[str, Any]) -> dict[str, Any]:
    """Flatten one evaluation result into a table row.

    Takes a label and the results dictionary. Returns the row.
    """
    overall = results["overall"]
    return {
        "split": label,
        "rows": results["rows_evaluated"],
        "mae": overall["mae"],
        "rmse": overall["rmse"],
        "spearman": overall["spearman"],
    }


def print_fold_report(scored: pd.DataFrame, config: dict[str, Any], position: str) -> None:
    """Print model and baseline metrics per fold and pooled, with the paired test.

    Takes the pooled scored rows, the parsed config, and the position. Returns
    nothing.

    Per-fold rows sit beside the pooled figure because a model strong in one
    season and weak in the next is a different proposition from one that is
    consistently mediocre, and pooling alone hides which.
    """
    table_rows = []
    for fold_name in sorted(scored["fold"].unique().tolist()):
        fold_rows = scored[scored["fold"] == fold_name]
        model_results = score_predictions(fold_rows, MODEL_COLUMN, config)
        baseline_results = score_predictions(fold_rows, BASELINE_COLUMN, config)
        table_rows.append(_metrics_row(f"fold {fold_name} model", model_results))
        table_rows.append(_metrics_row(f"fold {fold_name} baseline", baseline_results))

    pooled_model = score_predictions(scored, MODEL_COLUMN, config)
    pooled_baseline = score_predictions(scored, BASELINE_COLUMN, config)
    table_rows.append(_metrics_row("POOLED model", pooled_model))
    table_rows.append(_metrics_row("POOLED baseline", pooled_baseline))

    print(f"{position}: walk-forward results, per fold and pooled")
    print("-" * 72)
    print(pd.DataFrame(table_rows).round(4).to_string(index=False))

    comparison = evaluate.paired_error_comparison(
        scored[config["scoring"]["target_column"]], scored[MODEL_COLUMN], scored[BASELINE_COLUMN]
    )
    print(
        f"  paired, baseline minus model: delta MAE {comparison['delta_mae']:+.4f}  "
        f"SE {comparison['se']:.4f}  t {comparison['t']:+.2f}  over {comparison['rows']:,} rows"
    )
    print("  (positive means the model is better)")
    print("")


def print_row_survival(results: list[dict[str, Any]], position: str) -> None:
    """Print the split sizes for every fold.

    Takes the per-fold training results and the position. Returns nothing.
    """
    table_rows = []
    for result in results:
        table_rows.append(
            {
                "fold": result["fold"]["name"],
                "train": len(result["train_rows"]),
                "early_stop": len(result["early_stopping_rows"]),
                "scored": len(result["validation_rows"]),
                "features": len(result["feature_names"]),
                "best_iteration": result["best_iteration"],
            }
        )

    print("")
    print(f"{position}: split sizes by fold")
    print("=" * 72)
    print(pd.DataFrame(table_rows).to_string(index=False))
    print("")


def print_importance(result: dict[str, Any], position: str) -> None:
    """Print the most important features by gain, with the concentration figures.

    Takes the final fold's training result and the position. Returns nothing.

    The largest share and its ratio to the next are printed because a single
    feature holding most of the gain is the usual signature of leakage. Whether
    a concentration actually is leakage takes a hand check, not a threshold.
    """
    table = train.gain_importance_table(result, TOP_FEATURES)

    print(f"{position}: top {TOP_FEATURES} features by gain (final fold)")
    print("-" * 72)
    print(table.round({"gain": 1, "gain_share": 4}).to_string(index=False))

    if len(table) >= 2:
        top_share = float(table["gain_share"].iloc[0])
        next_share = float(table["gain_share"].iloc[1])
        print(
            f"  largest share {top_share:.1%} ({table['feature'].iloc[0]}), next {next_share:.1%}, "
            f"ratio {top_share / next_share:.1f}x"
        )
    print("")


def load_position_features(
    config: dict[str, Any], position: str, player_weeks: pd.DataFrame
) -> pd.DataFrame:
    """Read one position's saved matrix and attach the baseline to it.

    Takes the parsed config, the position, and the player-week frame carrying
    the baseline. Returns the matrix ready to train on.
    """
    features = read_parquet(build_dataset.feature_file_path(config, position))
    features = attach_baseline(features, player_weeks)
    check_baseline_matches_feature(features, config)
    return features


def run_training(
    config: dict[str, Any], positions: list[str], player_weeks: pd.DataFrame
) -> None:
    """Train, score, report, and save every requested position's model.

    Takes the parsed config, the positions, and the player-week frame carrying
    the baseline. Returns nothing.

    A position whose model does not beat its baseline is reported as it is. It
    is a legitimate outcome, most likely for quarterbacks and tight ends, whose
    samples are far smaller than the receivers'.
    """
    for position in positions:
        features = load_position_features(config, position, player_weeks)
        scored, results = score_folds(features, position, config)

        print_row_survival(results, position)
        print_fold_report(scored, config, position)
        print_importance(results[-1], position)

        model_path, metadata_path = train.save_model(results[-1], config)
        print(f"  saved {model_path.name} and {metadata_path.name}")
        print("")


def config_with_group(
    config: dict[str, Any], position: str, group_name: str, enabled: bool
) -> dict[str, Any]:
    """Copy the config with one group set for one position.

    Takes the parsed config, the position, the group, and whether to enable it.
    Returns a deep copy, so one study variant cannot alter the next one's config.
    """
    variant = copy.deepcopy(config)
    by_position = variant["features"]["groups"].get("by_position") or {}
    overrides = by_position.get(position) or {}
    overrides[group_name] = enabled
    by_position[position] = overrides
    variant["features"]["groups"]["by_position"] = by_position
    return variant


def build_position_matrix(
    player_weeks: pd.DataFrame, config: dict[str, Any], position: str
) -> pd.DataFrame:
    """Build one position's matrix in memory and attach the baseline.

    Takes the player-week frame carrying the baseline, the parsed config, and
    the position. Returns the matrix ready to train on.
    """
    features, _ = build_dataset.build_feature_matrix(player_weeks, config, position)
    features = attach_baseline(features, player_weeks)
    check_baseline_matches_feature(features, config)
    return features


def check_same_scored_rows(
    reference: pd.DataFrame, candidate: pd.DataFrame, description: str
) -> None:
    """Fail unless two scored frames hold the same rows in the same order.

    Takes the two frames and a description for the error. Returns nothing.

    A paired comparison subtracts errors row by row. If a group changed which
    rows exist, or their order, the subtraction would pair different players
    and produce a confident number that means nothing.
    """
    same_length = len(reference) == len(candidate)
    same_rows = same_length
    for column_name in ROW_KEY_COLUMNS:
        if not same_rows:
            break
        same_rows = reference[column_name].to_numpy().tolist() == (
            candidate[column_name].to_numpy().tolist()
        )

    if not same_rows:
        raise ValueError(
            f"{description}: the two runs scored different rows, so they cannot be paired."
        )


def paired_test_row(
    position: str,
    test_name: str,
    direction: str,
    reference: tuple[pd.DataFrame, str],
    candidate: tuple[pd.DataFrame, str],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run one paired test and describe it as a row of the study table.

    Takes the position, the test's name and direction, the reference and
    candidate as (scored frame, projection column) pairs, and the parsed
    config. Returns the row.

    delta_mae is candidate minus reference, so a negative value favours the
    candidate: the model over the baseline, or the flipped group over shipped.
    """
    reference_frame, reference_column = reference
    candidate_frame, candidate_column = candidate
    target_column = config["scoring"]["target_column"]

    comparison = evaluate.paired_error_comparison(
        candidate_frame[target_column],
        reference_frame[reference_column],
        candidate_frame[candidate_column],
    )
    return {
        "position": position,
        "test": test_name,
        "direction": direction,
        "rows": comparison["rows"],
        "reference_mae": score_predictions(reference_frame, reference_column, config)["overall"]["mae"],
        "candidate_mae": score_predictions(candidate_frame, candidate_column, config)["overall"]["mae"],
        "delta_mae": comparison["delta_mae"],
        "se": comparison["se"],
        "t": comparison["t"],
    }


def study_position(
    config: dict[str, Any], position: str, player_weeks: pd.DataFrame
) -> list[dict[str, Any]]:
    """Run every pre-registered test for one position.

    Takes the parsed config, the position, and the player-week frame carrying
    the baseline. Returns one table row per test.

    Every group test is against the position's shipped configuration, never
    against another step in a sequence, so no result depends on the order the
    groups happened to be tried in.
    """
    shipped_features = build_position_matrix(player_weeks, config, position)
    shipped, _ = score_folds(shipped_features, position, config)
    tests = [
        paired_test_row(
            position, BASELINE_TEST, "", (shipped, BASELINE_COLUMN), (shipped, MODEL_COLUMN), config
        )
    ]

    groups = build_dataset.resolve_groups(config, position)
    for group_name in STUDY_GROUPS_BY_POSITION.get(position, []):
        currently_on = groups.get(group_name, False)
        direction = "add"
        if currently_on:
            direction = "remove"

        variant_config = config_with_group(config, position, group_name, not currently_on)
        variant_features = build_position_matrix(player_weeks, variant_config, position)
        variant, _ = score_folds(variant_features, position, variant_config)
        check_same_scored_rows(shipped, variant, f"{position} {group_name}")

        tests.append(
            paired_test_row(
                position, group_name, direction, (shipped, MODEL_COLUMN), (variant, MODEL_COLUMN), config
            )
        )
    return tests


def verdict_for(row: pd.Series) -> str:
    """Describe what the pre-registered rule concludes from one test.

    Takes one row of the corrected study table. Returns the verdict text.

    A result that clears the decision threshold but fails the Bonferroni
    correction is labelled marginal: with this many tests, about one such
    result is expected by chance alone.
    """
    favourable = bool(row["delta_mae"] < 0)
    decisive = bool(row["clears_decision_t"])

    if row["test"] == BASELINE_TEST:
        verdict = "not distinguishable from baseline"
        if decisive and favourable:
            verdict = "model beats baseline"
        if decisive and not favourable:
            verdict = "BASELINE BEATS MODEL"
    else:
        verdict = "inconclusive: keep default"
        if decisive and not favourable:
            verdict = "flipping hurts: keep default"
        if decisive and favourable:
            verdict = "turn on"
            if row["direction"] == "remove":
                verdict = "turn off"

    if decisive and not bool(row["survives_bonferroni"]):
        verdict = verdict + " (marginal: fails Bonferroni)"
    return verdict


def print_study_table(tests: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Print every test with its corrections, verdicts, and the family totals.

    Takes the test rows and the parsed config. Returns nothing.
    """
    evaluation_config = config["evaluation"]
    family_alpha = evaluation_config["family_alpha"]
    decision_t = evaluation_config["decision_t"]

    corrected = evaluate.correct_for_multiple_tests(pd.DataFrame(tests), family_alpha, decision_t)
    verdicts = []
    for _, row in corrected.iterrows():
        verdicts.append(verdict_for(row))
    corrected["verdict"] = verdicts

    test_count = len(corrected)
    expected_by_chance = test_count * evaluate.two_sided_p_value(decision_t)

    print("")
    print("Pre-registered feature study: every test in the family")
    print("=" * 100)
    print(corrected.round(4).to_string(index=False))
    print("")
    print(f"  tests run in this stage          {test_count}")
    print(f"  decision rule                    favourable and |t| > {decision_t}")
    print(f"  expected to clear |t| > {decision_t} by chance  {expected_by_chance:.2f}")
    print(
        f"  Bonferroni bar at alpha {family_alpha}      |t| > "
        f"{evaluate.bonferroni_t_threshold(family_alpha, test_count):.2f}"
    )
    print("  delta_mae is candidate minus reference; negative favours the candidate.")
    print(
        "  These are single-seed results and their standard errors ignore seed noise. "
        "Change a toggle only on --seed-averaged."
    )
    print("")


def run_feature_study(
    config: dict[str, Any], positions: list[str], player_weeks: pd.DataFrame
) -> None:
    """Run the pre-registered test inventory for every requested position.

    Takes the parsed config, the positions, and the player-week frame carrying
    the baseline. Returns nothing.

    The config is not changed by this function. Any toggle the rule says to
    change is applied by hand afterwards, with its evidence written beside it.
    """
    tests = []
    for position in positions:
        position_tests = study_position(config, position, player_weeks)
        for test_row in position_tests:
            tests.append(test_row)

    print_study_table(tests, config)


def main() -> int:
    """Run the training stage.

    Takes nothing. Returns a process exit code, 0 on success.
    """
    arguments = parse_arguments()

    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print("Config error: " + str(error), file=sys.stderr)
        return 1

    configure_logging(config)

    if arguments.baseline_only:
        run_baseline_evaluation(config)
        return 0

    try:
        positions = positions_to_run(config, arguments.position)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    if arguments.deploy:
        run_deployment(config, positions)
        return 0

    if arguments.noise_floor:
        run_noise_floor(config, positions)
        return 0

    player_weeks = load_player_weeks_with_baseline(config)
    if arguments.feature_study:
        run_feature_study(config, positions, player_weeks)
        return 0

    if arguments.seed_averaged:
        run_seed_averaged(config, positions, player_weeks)
        return 0

    run_training(config, positions, player_weeks)
    return 0


def run_deployment(config: dict[str, Any], positions: list[str]) -> None:
    """Train, save, and summarise the live deployment model for each position.

    Takes the parsed config and the positions. Returns nothing.

    No accuracy figure is printed, deliberately. These models train and early
    stop on the sealed test season, so any score read off them would be an
    evaluation on data the project has promised not to evaluate on yet.
    """
    raw_directory = resolve_path(config["data"]["paths"]["raw"])
    processed_directory = resolve_path(config["data"]["paths"]["processed"])
    schedules = read_parquet(raw_directory / "schedules.parquet")
    player_weeks = read_parquet(processed_directory / "player_weeks.parquet")
    completed = train.completed_weeks(schedules, player_weeks, config)

    table_rows = []
    for position in positions:
        features = read_parquet(build_dataset.feature_file_path(config, position))
        result = train.train_deployment_model(features, position, config, completed)
        model_path, _ = train.save_deployment_model(result, config)
        stop_weeks = result["early_stopping_weeks"]
        table_rows.append(
            {
                "position": position,
                "features": len(result["feature_names"]),
                "train_rows": len(result["train_rows"]),
                "stop_rows": len(result["stop_rows"]),
                "refit_rows": len(result["refit_rows"]),
                "rounds": result["rounds"],
                "stop_weeks": f"{stop_weeks[0]} to {stop_weeks[-1]}",
                "saved": str(model_path.relative_to(resolve_path(config["data"]["paths"]["models"]))),
            }
        )

    print("")
    print("Deployment models (live projection only; no accuracy is computed from these)")
    print("=" * 100)
    print(f"  completed weeks used: {len(completed)}, {completed[0]} through {completed[-1]}")
    print(f"  disabled for deployment: {config['predict']['deployment']['disabled_groups']}")
    print(pd.DataFrame(table_rows).to_string(index=False))
    print("")


def _noise_floor_row(
    summary: dict[str, Any], baseline_mae: float, verdict: dict[str, str]
) -> dict[str, Any]:
    """Flatten one position's noise floor summary into a table row.

    Takes the summary, the baseline MAE on the same rows, and the gate verdict.
    Returns the row.
    """
    return {
        "position": summary["position"],
        "rows": summary["scored_rows"],
        "mae_mean": round(summary["mae_mean"], 4),
        "mae_sd": round(summary["mae_sd"], 4),
        "mae_best": round(summary["mae_best"], 4),
        "mae_worst": round(summary["mae_worst"], 4),
        "mae_range": round(summary["mae_range"], 4),
        "rmse_sd": round(summary["rmse_sd"], 4),
        "spearman_sd": round(summary["spearman_sd"], 4),
        "worst_vs_best_t": round(summary["worst_vs_best_t"], 2),
        "shipped_seed_rank": f"{summary['reference_rank']} of {len(summary['seeds'])}",
        "baseline_mae": round(baseline_mae, 4),
        "mean_gain_vs_baseline": round(baseline_mae - summary["mae_mean"], 4),
        "verdict": verdict["verdict"],
    }


def run_noise_floor(config: dict[str, Any], positions: list[str]) -> None:
    """Measure and print how far each position's model moves when only the seed changes.

    Takes the parsed config and the positions. Returns nothing.

    Nothing is saved and no model is replaced. Each walk-forward model is
    retrained under every seed in model.tuning.noise_floor_seeds, and the spread
    decides, through the gate, whether tuning that position could be measured.
    """
    tuning_config = config["model"]["tuning"]
    seeds = tuning_config["noise_floor_seeds"]
    target_column = config["scoring"]["target_column"]
    shipped_seed = config["project"]["random_seed"]
    player_weeks = load_player_weeks_with_baseline(config)

    table_rows = []
    reasons = []
    per_seed_lines = []
    for position in positions:
        features = load_position_features(config, position, player_weeks)
        records = tune.seed_sweep(features, position, config, seeds)
        summary = tune.summarize_noise_floor(records, target_column, shipped_seed)
        summary["position"] = position
        verdict = tune.gate_verdict(summary, tuning_config["effect_band"])

        baseline_results = score_predictions(records[0]["scored"], BASELINE_COLUMN, config)
        table_rows.append(_noise_floor_row(summary, baseline_results["overall"]["mae"], verdict))
        reasons.append(f"  {position}: {verdict['verdict'].upper()}: {verdict['reason']}")

        seed_maes = []
        for seed, mae in zip(summary["seeds"], summary["maes"], strict=True):
            seed_maes.append(f"{seed}:{mae:.4f}")
        per_seed_lines.append(f"  {position}: " + ", ".join(seed_maes))

    print("")
    print(f"Seed noise floor: {len(seeds)} seeds, pooled walk-forward MAE, nothing saved")
    print("=" * 100)
    print(pd.DataFrame(table_rows).to_string(index=False))
    print("")
    print(f"Gate (effect band {tuning_config['effect_band']}):")
    for reason in reasons:
        print(reason)
    print("")
    print("Per-seed pooled MAE:")
    for line in per_seed_lines:
        print(line)
    print("")


def position_sweep(
    config: dict[str, Any], position: str, player_weeks: pd.DataFrame, seeds: list[int]
) -> list[dict[str, Any]]:
    """Build one configuration's matrix and train it under every seed.

    Takes the parsed config, the position, the player-week frame carrying the
    baseline, and the seeds. Returns the per-seed records from tune.seed_sweep.

    The matrix is built in memory from the config, as the feature study does,
    so a variant never needs a saved file of its own.
    """
    features = build_position_matrix(player_weeks, config, position)
    return tune.seed_sweep(features, position, config, seeds)


def sweep_projections(
    records: list[dict[str, Any]], column: str, reference: pd.DataFrame, description: str
) -> list[pd.Series]:
    """Collect one projection column from every seed, checking each scored the same rows.

    Takes the per-seed records, the projection column, the scored frame every
    record must match row for row, and a description for the error. Returns
    the per-seed projections in seed order.
    """
    projections = []
    for record in records:
        check_same_scored_rows(reference, record["scored"], f"{description}, seed {record['seed']}")
        projections.append(record["scored"][column])
    return projections


def seed_averaged_headline(
    config: dict[str, Any], position: str, shipped_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare the shipped model, averaged over seeds, against its baseline.

    Takes the parsed config, the position, and the shipped configuration's
    per-seed records. Returns the seed-averaged comparison, baseline as the
    reference and the model as the candidate.
    """
    target_column = config["scoring"]["target_column"]
    shipped_scored = shipped_records[0]["scored"]
    model_projections = sweep_projections(
        shipped_records, MODEL_COLUMN, shipped_scored, f"{position} shipped"
    )

    # The baseline has no seed, so it is passed once per seed and adds no spread.
    baseline_projections = []
    for _ in shipped_records:
        baseline_projections.append(shipped_scored[BASELINE_COLUMN])

    return evaluate.seed_averaged_comparison(
        shipped_scored[target_column], baseline_projections, model_projections
    )


def seed_averaged_retest(
    config: dict[str, Any],
    position: str,
    group_name: str,
    player_weeks: pd.DataFrame,
    shipped_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Re-test one group as a single addition to its position's shipped set.

    Takes the parsed config, the position, the group, the player-week frame
    carrying the baseline, and the shipped configuration's per-seed records.
    Returns the seed-averaged comparison with the group off as the reference
    and the shipped set, group on, as the candidate. Raises ValueError if the
    group is not on in the shipped set.
    """
    if not build_dataset.resolve_groups(config, position).get(group_name, False):
        raise ValueError(
            f"{position} {group_name} is off in the shipped set, but the re-test adds it to "
            "a set without it and compares against the shipped set."
        )

    seeds = []
    for record in shipped_records:
        seeds.append(record["seed"])

    without_config = config_with_group(config, position, group_name, False)
    without_records = position_sweep(without_config, position, player_weeks, seeds)

    shipped_scored = shipped_records[0]["scored"]
    description = f"{position} {group_name}"
    without_projections = sweep_projections(
        without_records, MODEL_COLUMN, shipped_scored, description + " off"
    )
    with_projections = sweep_projections(
        shipped_records, MODEL_COLUMN, shipped_scored, description + " on"
    )
    return evaluate.seed_averaged_comparison(
        shipped_scored[config["scoring"]["target_column"]], without_projections, with_projections
    )


def seed_averaged_row(
    position: str,
    test_name: str,
    comparison: dict[str, Any],
    verdict: dict[str, str],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Flatten one seed-averaged comparison into a table row.

    Takes the position, the test name, the comparison, its verdict, and the
    text each verdict reads as for this kind of test. Returns the row.
    """
    return {
        "position": position,
        "test": test_name,
        "rows": comparison["rows"],
        "reference_mae": comparison["mae_a_mean"],
        "reference_sd": comparison["mae_a_sd"],
        "candidate_mae": comparison["mae_b_mean"],
        "candidate_sd": comparison["mae_b_sd"],
        "seed_range": comparison["seed_range"],
        "delta_mae": comparison["delta_mae"],
        "se_row": comparison["se_row"],
        "se_seed": comparison["se_seed"],
        "se": comparison["se"],
        "t": comparison["t"],
        "row_only_t": comparison["row_only_t"],
        "verdict": labels[verdict["verdict"]],
        "reason": verdict["reason"],
    }


def per_seed_line(label: str, seeds: list[int], values: list[float]) -> str:
    """Format one value per seed on a single line.

    Takes a label, the seeds, and the values in the same order. Returns the line.
    """
    parts = []
    for seed, value in zip(seeds, values, strict=True):
        parts.append(f"{seed}:{value:+.4f}")
    return f"  {label}: " + ", ".join(parts)


def print_seed_averaged_table(title: str, table: pd.DataFrame, notes: list[str]) -> None:
    """Print one seed-averaged table, its verdict reasons, and notes.

    Takes a title, the table with a reason column, and note lines. Returns nothing.
    """
    print("")
    print(title)
    print("=" * 100)
    print(table.drop(columns=["reason"]).round(4).to_string(index=False))
    print("")
    for _, row in table.iterrows():
        print(f"  {row['position']} {row['test']}: {row['verdict']}: {row['reason']}")
    for note in notes:
        print(note)
    print("")


def print_seed_averaged_report(
    headline_rows: list[dict[str, Any]],
    retest_rows: list[dict[str, Any]],
    detail_lines: list[str],
    config: dict[str, Any],
) -> None:
    """Print the seed-averaged headline, the re-tests with corrections, and per-seed detail.

    Takes the headline rows, the re-test rows, the per-seed lines, and the
    parsed config. Returns nothing.
    """
    evaluation_config = config["evaluation"]
    decision_t = evaluation_config["decision_t"]
    family_alpha = evaluation_config["family_alpha"]
    seed_count = len(config["model"]["tuning"]["noise_floor_seeds"])
    rule = (
        f"  rule: favourable, |t| > {decision_t} on the seed-aware SE, and |delta| above the "
        "seed range. Both bars."
    )

    print_seed_averaged_table(
        f"Seed-averaged headline: {seed_count} seeds, pooled walk-forward rows, nothing saved",
        pd.DataFrame(headline_rows),
        ["  reference is the baseline, candidate the model; delta is model minus baseline.", rule],
    )

    if len(retest_rows) > 0:
        corrected = evaluate.correct_for_multiple_tests(
            pd.DataFrame(retest_rows), family_alpha, decision_t
        )
        bonferroni_bar = evaluate.bonferroni_t_threshold(family_alpha, len(corrected))
        print_seed_averaged_table(
            f"Seed-averaged re-tests: {len(corrected)} pre-registered tests, nothing saved",
            corrected.drop(columns=["clears_decision_t"]),
            [
                "  reference is the shipped set without the group, candidate the shipped set; "
                "delta is with minus without.",
                rule,
                f"  corrections are reported, not used to decide: Bonferroni bar |t| > {bonferroni_bar:.2f}",
            ],
        )

    print("Per-seed detail: shipped MAE, then each re-test's MAE difference (with minus without)")
    for line in detail_lines:
        print(line)
    print("")


def run_seed_averaged(
    config: dict[str, Any], positions: list[str], player_weeks: pd.DataFrame
) -> None:
    """Restate every model against its baseline and re-test the marginal groups, over seeds.

    Takes the parsed config, the positions, and the player-week frame carrying
    the baseline. Returns nothing.

    Nothing is saved and the config is not changed. Each shipped configuration
    is trained once under every seed in model.tuning.noise_floor_seeds, and
    that one sweep serves both its headline and its re-tests.
    """
    seeds = config["model"]["tuning"]["noise_floor_seeds"]
    decision_t = config["evaluation"]["decision_t"]

    headline_rows = []
    retest_rows = []
    detail_lines = []
    for position in positions:
        shipped_records = position_sweep(config, position, player_weeks, seeds)
        shipped_maes = []
        for record in shipped_records:
            shipped_maes.append(record["mae"])
        detail_lines.append(per_seed_line(f"{position} shipped MAE", seeds, shipped_maes))

        headline = seed_averaged_headline(config, position, shipped_records)
        verdict = evaluate.seed_averaged_verdict(headline, decision_t)
        headline_rows.append(
            seed_averaged_row(position, BASELINE_TEST, headline, verdict, HEADLINE_VERDICTS)
        )

        for group_name in SEED_AVERAGED_RETESTS.get(position, []):
            comparison = seed_averaged_retest(config, position, group_name, player_weeks, shipped_records)
            verdict = evaluate.seed_averaged_verdict(comparison, decision_t)
            retest_rows.append(
                seed_averaged_row(position, "add " + group_name, comparison, verdict, RETEST_VERDICTS)
            )
            detail_lines.append(
                per_seed_line(f"{position} {group_name} delta", seeds, comparison["seed_differences"])
            )

    print_seed_averaged_report(headline_rows, retest_rows, detail_lines, config)


if __name__ == "__main__":
    sys.exit(main())
