"""Train one model per position and score each against its naive baseline.

Five modes. --baseline-only scores the naive last-N-game average and trains
nothing. --feature-study runs the pre-registered test inventory: every model
against its baseline, and each listed feature group flipped once against its
position's shipped configuration, with a multiple-comparisons report across the
whole family. --deploy trains the live models on every completed week, saved
apart from the evaluation models and never scored. --noise-floor retrains each
walk-forward model under several seeds to measure how far seed luck alone moves
it, which gates whether tuning is worth running. --seed-averaged restates each
model against its baseline and re-tests the marginal groups over those seeds.
--quantiles measures the floor and ceiling models' calibration over the same
seeds and saves their walk-forward models; --deploy-quantiles trains the live
quantile models for the bounds that passed. --point-diagnostics compares the
median model with the mean model on MAE, RMSE, and within-week rank
correlation, and tests the point projections for compression; it saves nothing.
--final-evaluation walk-forward reproduces the stated figures with the final
evaluation's code, and --final-evaluation sealed scores the sealed test season,
which can be spent only once. Otherwise every position's walk-forward model is
trained, scored, and saved.

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
from ffml.models import baseline, evaluate, predict, quantiles, train, tune
from ffml.utils.io import read_parquet, resolve_path, use_utf8_output

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
# group is tested as a single addition: the shipped set with it against the shipped
# set without it, whichever of the two currently ships.
SEED_AVERAGED_RETESTS = {
    "QB": ["game_context", "opportunity"],
    "WR": ["opponent_strength"],
    "TE": ["injuries"],
}

# How each seed-averaged verdict reads, for a model against its baseline and for a group.
# A group is on only if it is adopted; one that fails the rule either way is off.
HEADLINE_VERDICTS = {
    evaluate.HELPS: "model beats baseline",
    evaluate.HURTS: "BASELINE BEATS MODEL",
    evaluate.UNRESOLVED: "not distinguishable from baseline",
}
RETEST_VERDICTS = {
    evaluate.HELPS: "adopted: on",
    evaluate.HURTS: "rejected: off",
    evaluate.UNRESOLVED: "not adopted: off",
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
    parser.add_argument(
        "--quantiles",
        action="store_true",
        help=(
            "Measure the quantile models' calibration and compare the median with the mean model, "
            "over every seed in model.tuning. Saves the seed-42 walk-forward quantile models."
        ),
    )
    parser.add_argument(
        "--deploy-quantiles",
        action="store_true",
        help="Train the live quantile models for the bounds predict.intervals ships. Mean models are untouched.",
    )
    parser.add_argument(
        "--point-diagnostics",
        action="store_true",
        help=(
            "Compare the median quantile model with the mean model on MAE, RMSE, and within-week Spearman, "
            "and test the point projections for compression, over every seed in model.tuning. Saves nothing."
        ),
    )
    parser.add_argument(
        "--final-evaluation",
        choices=FINAL_EVALUATION_SCOPES,
        default=None,
        help=(
            "Evaluate the shipped system over every seed in model.tuning. walk-forward scores 2023 and 2024 "
            "and must reproduce the stated figures; sealed scores validation.test_season, once. Saves nothing."
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
    use_utf8_output()
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

    if arguments.deploy_quantiles:
        run_quantile_deployment(config, positions)
        return 0

    player_weeks = load_player_weeks_with_baseline(config)
    if arguments.feature_study:
        run_feature_study(config, positions, player_weeks)
        return 0

    if arguments.seed_averaged:
        run_seed_averaged(config, positions, player_weeks)
        return 0

    if arguments.quantiles:
        run_quantiles(config, positions, player_weeks)
        return 0

    if arguments.point_diagnostics:
        run_point_diagnostics(config, positions, player_weeks)
        return 0

    if arguments.final_evaluation is not None:
        run_final_evaluation(config, positions, player_weeks, arguments.final_evaluation)
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
    and the group on as the candidate, whichever of the two currently ships.

    The shipped sweep serves the side that matches the config, so only the
    other side is trained here. A group that was turned off after failing the
    rule can therefore be re-tested exactly as it was before.
    """
    seeds = []
    for record in shipped_records:
        seeds.append(record["seed"])

    shipped_has_group = bool(build_dataset.resolve_groups(config, position).get(group_name, False))
    flipped_config = config_with_group(config, position, group_name, not shipped_has_group)
    flipped_records = position_sweep(flipped_config, position, player_weeks, seeds)

    with_records = shipped_records
    without_records = flipped_records
    if not shipped_has_group:
        with_records = flipped_records
        without_records = shipped_records

    shipped_scored = shipped_records[0]["scored"]
    description = f"{position} {group_name}"
    without_projections = sweep_projections(
        without_records, MODEL_COLUMN, shipped_scored, description + " off"
    )
    with_projections = sweep_projections(
        with_records, MODEL_COLUMN, shipped_scored, description + " on"
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
        "favourable_seeds": comparison["favourable_seeds"],
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
    min_favourable_seeds = evaluation_config["min_favourable_seeds"]
    rule = (
        f"  rule: favourable, |t| > {decision_t} on the seed-aware SE, and favourable on at least "
        f"{min_favourable_seeds} of {seed_count} seeds. Both bars; seed_range is reported, not used."
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
                "  reference is the shipped set without the group, candidate the shipped set with "
                "it; delta is with minus without. A group is on only if adopted.",
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
    min_favourable_seeds = config["evaluation"]["min_favourable_seeds"]

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
        verdict = evaluate.seed_averaged_verdict(headline, decision_t, min_favourable_seeds)
        headline_rows.append(
            seed_averaged_row(position, BASELINE_TEST, headline, verdict, HEADLINE_VERDICTS)
        )

        for group_name in SEED_AVERAGED_RETESTS.get(position, []):
            comparison = seed_averaged_retest(config, position, group_name, player_weeks, shipped_records)
            verdict = evaluate.seed_averaged_verdict(comparison, decision_t, min_favourable_seeds)
            retest_rows.append(
                seed_averaged_row(position, "add " + group_name, comparison, verdict, RETEST_VERDICTS)
            )
            detail_lines.append(
                per_seed_line(f"{position} {group_name} delta", seeds, comparison["seed_differences"])
            )

    print_seed_averaged_report(headline_rows, retest_rows, detail_lines, config)


# How the median quantile model reads against a point estimate. Reported only: no
# projector changes on this comparison without the user deciding it.
MEDIAN_VERDICTS = {
    evaluate.HELPS: "median better",
    evaluate.HURTS: "mean better",
    evaluate.UNRESOLVED: "not distinguishable",
}
MEDIAN_VS_BASELINE_VERDICTS = {
    evaluate.HELPS: "median better",
    evaluate.HURTS: "baseline better",
    evaluate.UNRESOLVED: "not distinguishable",
}


def series_arrays(projections: list[pd.Series]) -> list[Any]:
    """Convert per-seed projections to plain float arrays.

    Takes the projections, one Series per seed. Returns one array per seed.
    """
    arrays = []
    for projection in projections:
        arrays.append(projection.to_numpy(dtype=float))
    return arrays


def quantile_sweeps(
    config: dict[str, Any], position: str, features: pd.DataFrame, seeds: list[int]
) -> tuple[list[dict[str, Any]], dict[float, list[dict[str, Any]]]]:
    """Train the mean model and every quantile level under every seed.

    Takes the parsed config, the position, its feature matrix, and the seeds.
    Returns the mean model's per-seed records and each level's.

    Both run through tune.seed_sweep and train.score_walk_forward, so they
    share folds, guards, and seeds; only the objective differs.
    """
    mean_records = tune.seed_sweep(features, position, config, seeds)
    level_records = {}
    for level in config["model"]["quantiles"]["levels"]:
        level_config = quantiles.quantile_config(config, level)
        level_records[level] = tune.seed_sweep(features, position, level_config, seeds)
    return mean_records, level_records


def point_projections(
    config: dict[str, Any], position: str, reference: pd.DataFrame, mean_projections: list[pd.Series]
) -> tuple[list[Any], str]:
    """Pick the point projection a position ships, per seed.

    Takes the parsed config, the position, the scored rows, and the mean
    model's projections per seed. Returns the point projection per seed and
    its name. A baseline position repeats the baseline once per seed.
    """
    if predict.projector_for(config, position) == predict.PROJECTOR_BASELINE:
        points = []
        for _ in mean_projections:
            points.append(reference[BASELINE_COLUMN].to_numpy(dtype=float))
        return points, "baseline"
    return series_arrays(mean_projections), "model mean"


def median_comparison_rows(
    config: dict[str, Any],
    position: str,
    reference: pd.DataFrame,
    mean_projections: list[pd.Series],
    median_projections: list[pd.Series],
) -> list[dict[str, Any]]:
    """Compare the median quantile model with the mean model, and with the baseline where it ships.

    Takes the parsed config, the position, the scored rows, and the mean and
    median projections per seed. Returns one seed-averaged table row per
    comparison. Nothing is switched on the result.
    """
    target = reference[config["scoring"]["target_column"]]
    decision_t = config["evaluation"]["decision_t"]
    min_favourable_seeds = config["evaluation"]["min_favourable_seeds"]

    comparison = evaluate.seed_averaged_comparison(target, mean_projections, median_projections)
    verdict = evaluate.seed_averaged_verdict(comparison, decision_t, min_favourable_seeds)
    rows = [seed_averaged_row(position, "median vs mean", comparison, verdict, MEDIAN_VERDICTS)]

    if predict.projector_for(config, position) == predict.PROJECTOR_BASELINE:
        baseline_projections = []
        for _ in median_projections:
            baseline_projections.append(reference[BASELINE_COLUMN])
        comparison = evaluate.seed_averaged_comparison(target, baseline_projections, median_projections)
        verdict = evaluate.seed_averaged_verdict(comparison, decision_t, min_favourable_seeds)
        rows.append(
            seed_averaged_row(position, "median vs baseline", comparison, verdict, MEDIAN_VS_BASELINE_VERDICTS)
        )
    return rows


def calibration_rows(position: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten each level's calibration and verdict into a table row.

    Takes the position and its calibration summary. Returns one row per level.
    """
    rows = []
    for level in summary["levels"]:
        calibration = summary["levels"][level]
        overall = calibration["overall"]
        top_group = calibration["top_group"]
        top = calibration["groups"][top_group]
        verdict = "FAIL"
        if calibration["verdict"]["passed"]:
            verdict = "pass"
        rows.append(
            {
                "position": position,
                "test": f"level {level}",
                "coverage": overall["coverage"],
                "gap": overall["gap"],
                "gap_sd": overall["gap_sd"],
                "gap_range": overall["gap_range"],
                "seeds_within": overall["seeds_within"],
                "top_rows": calibration["group_rows"][top_group],
                "top_coverage": top["coverage"],
                "top_gap": top["gap"],
                "top_gap_sd": top["gap_sd"],
                "top_seeds_within": top["seeds_within"],
                "verdict": verdict,
                "reason": calibration["verdict"]["reason"],
            }
        )
    return rows


def group_coverage_rows(position: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Lay out each level's seed-averaged coverage in every projection group.

    Takes the position and its calibration summary. Returns one row per level.
    """
    rows = []
    for level in summary["levels"]:
        calibration = summary["levels"][level]
        row: dict[str, Any] = {"position": position, "level": level}
        for group in calibration["groups"]:
            row[f"group_{group}"] = calibration["groups"][group]["coverage"]
        rows.append(row)
    return rows


def width_and_diagnostic_rows(
    position: str, summary: dict[str, Any], point_label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Describe the interval's width, its crossings, and where the point projection sits.

    Takes the position, its calibration summary, and the name of its point
    projection. Returns the width row and the diagnostic row.
    """
    width = summary["width"]
    width_row: dict[str, Any] = {"position": position, "width": width["mean"], "width_sd": width["sd"]}
    counts = []
    for group in width["groups"]:
        width_row[f"group_{group}"] = width["groups"][group]
        counts.append(str(summary["group_rows"][group]))
    width_row["rows_per_group"] = "/".join(counts)

    diagnostic_row: dict[str, Any] = {"position": position, "point": point_label}
    for name in summary["crossing"]:
        diagnostic_row[name] = summary["crossing"][name]
    diagnostic_row["point_inside_interval"] = summary["containment"]
    return width_row, diagnostic_row


def shipping_line(position: str, summary: dict[str, Any], config: dict[str, Any]) -> str:
    """State which of a position's bounds the calibration rule lets ship.

    Takes the position, its calibration summary, and the parsed config.
    Returns the line.
    """
    levels = quantiles.bound_levels(config)
    parts = []
    for bound in quantiles.BOUNDS:
        state = "does NOT ship"
        if summary["levels"][levels[bound]]["verdict"]["passed"]:
            state = "ships"
        parts.append(f"{bound} {state}")
    return f"  {position}: " + ", ".join(parts)


def quantile_position_report(
    config: dict[str, Any], position: str, features: pd.DataFrame, seeds: list[int]
) -> dict[str, list[Any]]:
    """Run one position's quantile sweeps and build every report row for it.

    Takes the parsed config, the position, its feature matrix, and the seeds.
    Returns the rows for each report table, keyed by table.
    """
    mean_records, level_records = quantile_sweeps(config, position, features, seeds)
    reference = mean_records[0]["scored"]
    mean_projections = sweep_projections(mean_records, MODEL_COLUMN, reference, f"{position} mean")

    projections_by_level = {}
    arrays_by_level = {}
    for level in level_records:
        projections_by_level[level] = sweep_projections(
            level_records[level], MODEL_COLUMN, reference, f"{position} level {level}"
        )
        arrays_by_level[level] = series_arrays(projections_by_level[level])

    points, point_label = point_projections(config, position, reference, mean_projections)
    actual = reference[config["scoring"]["target_column"]].to_numpy(dtype=float)
    summary = quantiles.calibrate_position(actual, points, arrays_by_level, config)
    width_row, diagnostic_row = width_and_diagnostic_rows(position, summary, point_label)

    median_projections = projections_by_level[quantiles.MEDIAN_LEVEL]
    return {
        "calibration": calibration_rows(position, summary),
        "groups": group_coverage_rows(position, summary),
        "widths": [width_row],
        "diagnostics": [diagnostic_row],
        "median": median_comparison_rows(config, position, reference, mean_projections, median_projections),
        "shipping": [shipping_line(position, summary, config)],
    }


def save_walk_forward_quantiles(config: dict[str, Any], position: str, features: pd.DataFrame) -> list[str]:
    """Train and save one position's walk-forward quantile models at the project seed.

    Takes the parsed config, the position, and its feature matrix. Returns the
    saved file names.

    These are the walk-forward models, so they, not the deployment models, are
    the ones that may be scored on the sealed season once it is unsealed.
    Nothing here scores it.
    """
    saved = []
    for level in config["model"]["quantiles"]["levels"]:
        _, results = score_folds(features, position, quantiles.quantile_config(config, level))
        model_path, _ = train.save_model(results[-1], config, quantiles.level_suffix(level))
        saved.append(model_path.name)
    return saved


def print_plain_table(title: str, rows: list[dict[str, Any]]) -> None:
    """Print a titled table.

    Takes the title and the rows. Returns nothing.
    """
    print("")
    print(title)
    print("=" * 100)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def print_quantile_report(report: dict[str, list[Any]], config: dict[str, Any]) -> None:
    """Print the calibration, group, width, diagnostic, median, and shipping tables.

    Takes the collected report rows and the parsed config. Returns nothing.
    """
    adoption = config["model"]["quantiles"]["adoption"]
    seed_count = len(config["model"]["tuning"]["noise_floor_seeds"])
    rule = (
        f"  rule, per level: coverage within {adoption['max_overall_gap']} over all rows and within "
        f"{adoption['max_top_group_gap']} in the top of {adoption['projection_groups']} projection "
        f"groups, each seed-averaged and in at least {adoption['min_calibrated_seeds']} of {seed_count} seeds."
    )

    print_seed_averaged_table(
        f"Quantile calibration: {seed_count} seeds, pooled walk-forward rows; 2025 is never scored",
        pd.DataFrame(report["calibration"]),
        ["  coverage is the share of actual scores below the prediction; gap is coverage minus level.", rule],
    )
    print_plain_table("Coverage by projection group, seed-averaged (group 1 lowest projections)", report["groups"])
    print_plain_table("Floor to ceiling width in fantasy points, seed-averaged, by projection group", report["widths"])
    print_plain_table("Crossing between levels, and the point projection inside the interval", report["diagnostics"])
    print_seed_averaged_table(
        "Median against each position's point estimate: reported only, no projector changes",
        pd.DataFrame(report["median"]),
        ["  reference is the point estimate, candidate the median; delta is median minus reference."],
    )

    print("Bounds the pre-registered rule lets ship (record them in predict.intervals)")
    for line in report["shipping"]:
        print(line)
    print("")


def run_quantiles(config: dict[str, Any], positions: list[str], player_weeks: pd.DataFrame) -> None:
    """Measure every position's quantile calibration over seeds and save the walk-forward quantile models.

    Takes the parsed config, the positions, and the player-week frame carrying
    the baseline. Returns nothing.

    Every figure comes from the pooled walk-forward folds; the sealed season is
    never scored. The config is not changed here: the shipping verdicts are
    printed, and recorded in predict.intervals by hand with their evidence.
    """
    if not config["model"]["quantiles"]["enabled"]:
        print("model.quantiles.enabled is false, so no quantile model is trained.")
        return

    seeds = config["model"]["tuning"]["noise_floor_seeds"]
    report: dict[str, list[Any]] = {
        "calibration": [],
        "groups": [],
        "widths": [],
        "diagnostics": [],
        "median": [],
        "shipping": [],
    }
    saved = []
    for position in positions:
        features = load_position_features(config, position, player_weeks)
        position_report = quantile_position_report(config, position, features, seeds)
        for table_name in report:
            for item in position_report[table_name]:
                report[table_name].append(item)
        for name in save_walk_forward_quantiles(config, position, features):
            saved.append(name)

    print_quantile_report(report, config)
    print(f"Saved seed-{config['project']['random_seed']} walk-forward quantile models: {', '.join(saved)}")
    print("")


def run_quantile_deployment(config: dict[str, Any], positions: list[str]) -> None:
    """Train and save the live quantile models for the bounds predict.intervals ships.

    Takes the parsed config and the positions. Returns nothing.

    Only shipped bounds are trained, and the mean deployment models are neither
    retrained nor rewritten. No accuracy is printed, for the same reason as
    run_deployment: these models train on the sealed season.
    """
    raw_directory = resolve_path(config["data"]["paths"]["raw"])
    processed_directory = resolve_path(config["data"]["paths"]["processed"])
    schedules = read_parquet(raw_directory / "schedules.parquet")
    player_weeks = read_parquet(processed_directory / "player_weeks.parquet")
    completed = train.completed_weeks(schedules, player_weeks, config)
    levels = quantiles.bound_levels(config)

    table_rows = []
    for position in positions:
        bounds = predict.shipped_bounds(config, position)
        if len(bounds) == 0:
            continue
        features = read_parquet(build_dataset.feature_file_path(config, position))
        for bound in bounds:
            suffix = quantiles.level_suffix(levels[bound])
            level_config = quantiles.quantile_config(config, levels[bound])
            result = train.train_deployment_model(features, position, level_config, completed)
            model_path, _ = train.save_deployment_model(result, config, suffix)
            table_rows.append(
                {
                    "position": position,
                    "bound": bound,
                    "level": levels[bound],
                    "rounds": result["rounds"],
                    "refit_rows": len(result["refit_rows"]),
                    "saved": model_path.name,
                }
            )

    print("")
    print("Quantile deployment models (live projection only; no accuracy is computed from these)")
    print("=" * 100)
    print(f"  completed weeks used: {len(completed)}, {completed[0]} through {completed[-1]}")
    if len(table_rows) == 0:
        print("  predict.intervals ships no bound for these positions, so nothing was trained.")
    else:
        print(pd.DataFrame(table_rows).to_string(index=False))
    print("")



# The metrics the median is compared with the mean on. MAE favours a median by
# construction, so the decision also needs RMSE and the within-week rank
# correlation start/sit decisions actually use.
POINT_DIAGNOSTIC_METRICS = ["mae", "rmse", "spearman"]


def point_diagnostic_sweeps(
    config: dict[str, Any], position: str, features: pd.DataFrame, seeds: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Train the mean model and the median quantile model under every seed.

    Takes the parsed config, the position, its feature matrix, and the seeds.
    Returns both models' per-seed records.
    """
    mean_records = tune.seed_sweep(features, position, config, seeds)
    median_config = quantiles.quantile_config(config, quantiles.MEDIAN_LEVEL)
    median_records = tune.seed_sweep(features, position, median_config, seeds)
    return mean_records, median_records


def metric_comparison(
    metric: str,
    target: pd.Series,
    reference_projections: list[pd.Series],
    candidate_projections: list[pd.Series],
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    """Run the seed-averaged comparison for one metric.

    Takes the metric name, the actual scores, the reference and candidate
    projections per seed, and the metadata. Returns the comparison.
    """
    if metric == "rmse":
        return evaluate.seed_averaged_rmse_comparison(target, reference_projections, candidate_projections)
    if metric == "spearman":
        return evaluate.seed_averaged_spearman_comparison(
            target, reference_projections, candidate_projections, metadata
        )
    return evaluate.seed_averaged_comparison(target, reference_projections, candidate_projections)


def check_recorded_metric(
    comparison: dict[str, Any],
    metric: str,
    reference_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
) -> None:
    """Fail unless each seed's RMSE or rank correlation matches what run_evaluation recorded.

    Takes the comparison, the metric, and both models' per-seed records.
    Returns nothing. Raises ValueError on any difference.

    The rank correlation check is skipped when a group had to be dropped,
    because run_evaluation drops undefined groups seed by seed instead.
    """
    if metric == "mae":
        return
    if metric == "spearman" and comparison["groups_dropped"] > 0:
        return

    pairs = [(comparison["metric_a_by_seed"], reference_records), (comparison["metric_b_by_seed"], candidate_records)]
    for values, records in pairs:
        for value, record in zip(values, records, strict=True):
            if abs(value - record[metric]) > BASELINE_MATCH_TOLERANCE:
                raise ValueError(
                    f"Seed {record['seed']}: {metric} {value} differs from the recorded {record[metric]}."
                )


def metric_table_row(
    position: str,
    test_name: str,
    metric: str,
    comparison: dict[str, Any],
    verdict: dict[str, str],
    labels: dict[str, str],
) -> dict[str, Any]:
    """Flatten one metric comparison into a table row.

    Takes the position, the test name, the metric, the comparison, its
    verdict, and the verdict labels. Returns the row.

    The MAE comparison names its fields after MAE; the others use generic names.
    """
    prefix = "metric"
    effect_key = "delta"
    unit_key = "se_unit"
    units = comparison.get("units")
    if metric == "mae":
        prefix = "mae"
        effect_key = "delta_mae"
        unit_key = "se_row"
        units = comparison["rows"]

    return {
        "position": position,
        "test": f"{test_name} ({metric})",
        "units": units,
        "reference": comparison[f"{prefix}_a_mean"],
        "candidate": comparison[f"{prefix}_b_mean"],
        "difference": comparison[effect_key],
        "se_unit": comparison[unit_key],
        "se_seed": comparison["se_seed"],
        "se": comparison["se"],
        "t": comparison["t"],
        "favourable_seeds": comparison["favourable_seeds"],
        "groups_dropped": comparison.get("groups_dropped", 0),
        "verdict": labels[verdict["verdict"]],
        "reason": verdict["reason"],
    }


def median_metric_rows(
    config: dict[str, Any],
    position: str,
    reference: pd.DataFrame,
    mean_records: list[dict[str, Any]],
    median_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare the median with the mean model on every metric, and with the baseline where it ships.

    Takes the parsed config, the position, the scored rows, and both models'
    per-seed records. Returns one table row per comparison. Nothing is switched.
    """
    target = reference[config["scoring"]["target_column"]]
    metadata = reference[METADATA_COLUMNS]
    decision_t = config["evaluation"]["decision_t"]
    min_seeds = config["evaluation"]["min_favourable_seeds"]
    mean_projections = sweep_projections(mean_records, MODEL_COLUMN, reference, f"{position} mean")
    median_projections = sweep_projections(median_records, MODEL_COLUMN, reference, f"{position} median")

    baseline_projections = []
    for _ in median_projections:
        baseline_projections.append(reference[BASELINE_COLUMN])
    ships_baseline = predict.projector_for(config, position) == predict.PROJECTOR_BASELINE

    rows = []
    for metric in POINT_DIAGNOSTIC_METRICS:
        comparison = metric_comparison(metric, target, mean_projections, median_projections, metadata)
        check_recorded_metric(comparison, metric, mean_records, median_records)
        verdict = evaluate.seed_averaged_verdict(comparison, decision_t, min_seeds)
        rows.append(metric_table_row(position, "median vs mean", metric, comparison, verdict, MEDIAN_VERDICTS))
        if ships_baseline:
            comparison = metric_comparison(metric, target, baseline_projections, median_projections, metadata)
            verdict = evaluate.seed_averaged_verdict(comparison, decision_t, min_seeds)
            rows.append(
                metric_table_row(
                    position, "median vs baseline", metric, comparison, verdict, MEDIAN_VS_BASELINE_VERDICTS
                )
            )
    return rows


def compression_rows(
    config: dict[str, Any],
    position: str,
    reference: pd.DataFrame,
    mean_records: list[dict[str, Any]],
    median_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure projection bias by group and the calibration slope for mean, median, and baseline.

    Takes the parsed config, the position, the scored rows, and both models'
    per-seed records. Returns the group rows and the slope rows. Diagnostic
    only: nothing is corrected.

    Each model is grouped by its own seed-averaged projection. The baseline is
    repeated once per seed, so its seed terms are zero.
    """
    actual = reference[config["scoring"]["target_column"]].to_numpy(dtype=float)
    group_count = config["model"]["quantiles"]["adoption"]["projection_groups"]
    decision_t = config["evaluation"]["decision_t"]
    min_seeds = config["evaluation"]["min_favourable_seeds"]

    baseline_arrays = []
    for _ in mean_records:
        baseline_arrays.append(reference[BASELINE_COLUMN].to_numpy(dtype=float))
    models = [
        ("mean model", series_arrays(sweep_projections(mean_records, MODEL_COLUMN, reference, f"{position} mean"))),
        ("median model", series_arrays(sweep_projections(median_records, MODEL_COLUMN, reference, f"{position} median"))),
        ("baseline", baseline_arrays),
    ]

    group_rows = []
    slope_rows = []
    for model_name, projections_by_seed in models:
        groups = quantiles.projection_groups(evaluate.seed_average(projections_by_seed), group_count)
        bias_rows = evaluate.projection_group_bias(actual, projections_by_seed, groups, group_count)
        for bias_row in bias_rows:
            table_row: dict[str, Any] = {"position": position, "model": model_name}
            for key in bias_row:
                table_row[key] = bias_row[key]
            group_rows.append(table_row)

        slope = evaluate.calibration_slope(actual, projections_by_seed)
        verdict = evaluate.compression_verdict(bias_rows, slope, decision_t, min_seeds)
        slope_row: dict[str, Any] = {"position": position, "test": model_name}
        for key in slope:
            slope_row[key] = slope[key]
        slope_row["verdict"] = "not clearly compressed"
        if verdict["present"]:
            slope_row["verdict"] = "compressed"
        slope_row["reason"] = verdict["reason"]
        slope_rows.append(slope_row)
    return group_rows, slope_rows


def print_point_diagnostics(
    metric_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    slope_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Print the median comparisons and the compression diagnostic.

    Takes the metric rows, the group bias rows, the slope rows, and the parsed
    config. Returns nothing.
    """
    seed_count = len(config["model"]["tuning"]["noise_floor_seeds"])
    print_seed_averaged_table(
        f"Median against point estimates on MAE, RMSE, and within-week Spearman: {seed_count} seeds; "
        "reported only, no projector changes",
        pd.DataFrame(metric_rows),
        [
            "  difference is candidate (median) minus reference; lower is better for MAE and RMSE, "
            "higher for Spearman.",
            "  favourable_seeds counts seeds where the median did better on that metric.",
            "  units are rows for MAE and RMSE, and position-week groups for Spearman.",
        ],
    )
    print_plain_table(
        "Compression: mean projected against mean actual in five groups by each model's own projection "
        "(group 1 lowest); bias is projected minus actual",
        group_rows,
    )
    print_seed_averaged_table(
        "Calibration slope of actual on projection: 1 is calibrated, above 1 is compressed. Diagnostic only",
        pd.DataFrame(slope_rows),
        [
            "  compressed only if the lowest group is over-projected and the highest under-projected, each beyond "
            "2 seed-aware SEs in at least 8 of 10 seeds, and the slope is above 1 beyond 2 SEs.",
        ],
    )


def run_point_diagnostics(config: dict[str, Any], positions: list[str], player_weeks: pd.DataFrame) -> None:
    """Compare the median with the mean on every metric and test the point projections for compression.

    Takes the parsed config, the positions, and the player-week frame carrying
    the baseline. Returns nothing.

    Nothing is saved, switched, corrected, or tuned. Every figure comes from
    the pooled walk-forward folds; the sealed season is never scored.
    """
    seeds = config["model"]["tuning"]["noise_floor_seeds"]
    metric_rows = []
    group_rows = []
    slope_rows = []
    for position in positions:
        features = load_position_features(config, position, player_weeks)
        mean_records, median_records = point_diagnostic_sweeps(config, position, features, seeds)
        reference = mean_records[0]["scored"]

        for row in median_metric_rows(config, position, reference, mean_records, median_records):
            metric_rows.append(row)
        position_groups, position_slopes = compression_rows(config, position, reference, mean_records, median_records)
        for row in position_groups:
            group_rows.append(row)
        for row in position_slopes:
            slope_rows.append(row)

    print_point_diagnostics(metric_rows, group_rows, slope_rows, config)


# The two scopes of the final evaluation. walk-forward scores 2023 and 2024 and must
# reproduce the stated figures; sealed scores the test season, once.
FINAL_EVALUATION_SCOPES = ["walk-forward", "sealed"]

# How the shipped projector reads against its baseline in the final evaluation.
FINAL_VERDICTS = {
    evaluate.HELPS: "beats baseline",
    evaluate.HURTS: "BASELINE BETTER",
    evaluate.UNRESOLVED: "not distinguishable",
}


def final_evaluation_folds(config: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    """List the folds a final evaluation scope runs.

    Takes the parsed config and the scope. Returns the walk-forward folds, or
    the single sealed holdout fold.
    """
    if scope == "sealed":
        return [train.sealed_holdout_fold(config)]
    return train.walk_forward_folds(config)


def audit_final_folds(
    config: dict[str, Any], position: str, features: pd.DataFrame, folds: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Prove by row index that no sealed or later row reaches fitting or early stopping.

    Takes the parsed config, the position, its feature matrix, and the folds.
    Returns one audit row per fold. Raises ValueError if any test season row is
    in a training or early stopping split, or any later season row is in any
    split.

    The split guards already refuse this. The audit checks the row indices the
    models will actually be handed, before anything is trained, so the claim
    rests on the data rather than on the guard.
    """
    test_season = config["validation"]["test_season"]
    position_rows = features[features["position"] == position]
    sealed_index = position_rows.index[position_rows["season"] == test_season]
    later_index = position_rows.index[position_rows["season"] > test_season]

    audit_rows = []
    for fold in folds:
        train_rows, stop_rows, scored_rows = train.split_by_time(features, position, fold, config)
        fitted_index = train_rows.index.union(stop_rows.index)
        sealed_in_fitting = len(fitted_index.intersection(sealed_index))
        later_anywhere = len(fitted_index.union(scored_rows.index).intersection(later_index))
        if sealed_in_fitting > 0 or later_anywhere > 0:
            raise ValueError(
                f"{position} fold {fold['name']}: {sealed_in_fitting} test season rows in training or "
                f"early stopping, {later_anywhere} later-season rows in a split."
            )
        audit_rows.append(
            {
                "position": position,
                "fold": fold["name"],
                "training_rows": len(train_rows),
                "stopping_rows": len(stop_rows),
                "scored_rows": len(scored_rows),
                "scored_seasons": sorted(scored_rows["season"].unique().tolist()),
                "sealed_rows_fitted_or_stopped": sealed_in_fitting,
                "sealed_rows_scored": len(scored_rows.index.intersection(sealed_index)),
                "later_rows_anywhere": later_anywhere,
            }
        )
    return audit_rows


def final_evaluation_sweeps(
    config: dict[str, Any], position: str, features: pd.DataFrame, seeds: list[int], folds: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Train the mean model and each shipped quantile bound under every seed, on the given folds.

    Takes the parsed config, the position, its feature matrix, the seeds, and
    the folds. Returns the per-seed records keyed mean, floor, or ceiling.

    Everything is trained in memory. No model file is saved or loaded, so no
    deployment model can reach this evaluation.
    """
    records = {"mean": tune.seed_sweep(features, position, config, seeds, folds)}
    levels = quantiles.bound_levels(config)
    for bound in predict.shipped_bounds(config, position):
        level_config = quantiles.quantile_config(config, levels[bound])
        records[bound] = tune.seed_sweep(features, position, level_config, seeds, folds)
    return records


def check_model_metric(comparison: dict[str, Any], metric: str, records: list[dict[str, Any]]) -> None:
    """Fail unless each seed's model RMSE or rank correlation matches what run_evaluation recorded.

    Takes the comparison with the model as candidate, the metric, and the
    model's per-seed records. Returns nothing. Raises ValueError on a difference.
    """
    if metric == "mae":
        return
    if metric == "spearman" and comparison["groups_dropped"] > 0:
        return
    for value, record in zip(comparison["metric_b_by_seed"], records, strict=True):
        if abs(value - record[metric]) > BASELINE_MATCH_TOLERANCE:
            raise ValueError(f"Seed {record['seed']}: {metric} {value} differs from the recorded {record[metric]}.")


def final_point_rows(
    config: dict[str, Any], position: str, reference: pd.DataFrame, mean_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare the model with its baseline on MAE, RMSE, and within-week rank correlation.

    Takes the parsed config, the position, the scored rows, and the mean
    model's per-seed records. Returns one row per metric, naming the method the
    position ships. For a baseline position the shipped figure is the reference.
    """
    target = reference[config["scoring"]["target_column"]]
    metadata = reference[METADATA_COLUMNS]
    decision_t = config["evaluation"]["decision_t"]
    min_seeds = config["evaluation"]["min_favourable_seeds"]
    model_projections = sweep_projections(mean_records, MODEL_COLUMN, reference, f"{position} model")
    baseline_projections = []
    for _ in model_projections:
        baseline_projections.append(reference[BASELINE_COLUMN])

    rows = []
    for metric in POINT_DIAGNOSTIC_METRICS:
        comparison = metric_comparison(metric, target, baseline_projections, model_projections, metadata)
        check_model_metric(comparison, metric, mean_records)
        verdict = evaluate.seed_averaged_verdict(comparison, decision_t, min_seeds)
        row = metric_table_row(position, "model vs baseline", metric, comparison, verdict, FINAL_VERDICTS)
        row["ships"] = predict.projector_for(config, position)
        rows.append(row)
    return rows


def final_interval_rows(
    config: dict[str, Any], position: str, reference: pd.DataFrame, records: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Calibrate the position's shipped bounds against the pre-registered tolerances.

    Takes the parsed config, the position, the scored rows, and the per-seed
    records by model. Returns one calibration row per shipped bound, empty for
    a position that ships none.

    Groups come from the seed-averaged shipped point projection, as they did
    when the bounds were adopted.
    """
    adoption = config["model"]["quantiles"]["adoption"]
    levels = quantiles.bound_levels(config)
    actual = reference[config["scoring"]["target_column"]].to_numpy(dtype=float)
    mean_projections = sweep_projections(records["mean"], MODEL_COLUMN, reference, f"{position} model")
    points, _ = point_projections(config, position, reference, mean_projections)
    groups = quantiles.projection_groups(evaluate.seed_average(points), adoption["projection_groups"])

    rows = []
    for bound in predict.shipped_bounds(config, position):
        bound_projections = sweep_projections(records[bound], MODEL_COLUMN, reference, f"{position} {bound}")
        calibration = quantiles.level_calibration(
            actual, series_arrays(bound_projections), levels[bound], groups, adoption
        )
        calibration["verdict"] = quantiles.level_verdict(calibration, adoption)
        for row in calibration_rows(position, {"levels": {levels[bound]: calibration}}):
            row["test"] = f"{bound} (level {levels[bound]})"
            rows.append(row)
    return rows


def final_compression_rows(
    config: dict[str, Any], position: str, reference: pd.DataFrame, mean_records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Measure the calibration slope of the model and the baseline, marking which one ships.

    Takes the parsed config, the position, the scored rows, and the mean
    model's per-seed records. Returns one row per projection.
    """
    actual = reference[config["scoring"]["target_column"]].to_numpy(dtype=float)
    group_count = config["model"]["quantiles"]["adoption"]["projection_groups"]
    decision_t = config["evaluation"]["decision_t"]
    min_seeds = config["evaluation"]["min_favourable_seeds"]
    shipped = predict.projector_for(config, position)

    baseline_arrays = []
    for _ in mean_records:
        baseline_arrays.append(reference[BASELINE_COLUMN].to_numpy(dtype=float))
    model_arrays = series_arrays(sweep_projections(mean_records, MODEL_COLUMN, reference, f"{position} model"))
    projections_by_name = [(predict.PROJECTOR_MODEL, model_arrays), (predict.PROJECTOR_BASELINE, baseline_arrays)]

    rows = []
    for name, projections_by_seed in projections_by_name:
        groups = quantiles.projection_groups(evaluate.seed_average(projections_by_seed), group_count)
        bias_rows = evaluate.projection_group_bias(actual, projections_by_seed, groups, group_count)
        slope = evaluate.calibration_slope(actual, projections_by_seed)
        verdict = evaluate.compression_verdict(bias_rows, slope, decision_t, min_seeds)
        label = name
        if name == shipped:
            label = name + " (shipped)"
        reading = "not clearly compressed"
        if verdict["present"]:
            reading = "compressed"
        rows.append(
            {
                "position": position,
                "test": label,
                "slope": slope["slope"],
                "slope_se": slope["slope_se"],
                "t_against_one": slope["t_against_one"],
                "seeds_above_one": slope["seeds_above_one"],
                "lowest_group_bias": bias_rows[0]["bias"],
                "highest_group_bias": bias_rows[-1]["bias"],
                "overall_bias": slope["overall_bias"],
                "verdict": reading,
                "reason": verdict["reason"],
            }
        )
    return rows


def print_final_evaluation(scope: str, report: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> None:
    """Print the index audit, point metrics, interval calibration, and compression tables.

    Takes the scope, the collected report rows, and the parsed config. Returns nothing.
    """
    seed_count = len(config["model"]["tuning"]["noise_floor_seeds"])
    adoption = config["model"]["quantiles"]["adoption"]
    print("")
    print("=" * 100)
    print(f"FINAL EVALUATION, scope {scope}: {seed_count} seeds; every model trained in memory; no model file saved or loaded")
    print("=" * 100)
    print_plain_table("Index audit: rows each split holds, and any sealed or later-season row where it must not be", report["audit"])
    print_seed_averaged_table(
        "Point metrics: the model against its baseline; 'ships' names what the position projects with",
        pd.DataFrame(report["points"]),
        [
            "  difference is model minus baseline; lower is better for MAE and RMSE, higher for Spearman.",
            "  favourable_seeds counts seeds where the model beat the baseline on that metric.",
            "  units are rows for MAE and RMSE, and position-week groups for Spearman.",
        ],
    )
    print_seed_averaged_table(
        "Interval calibration for the shipped bounds",
        pd.DataFrame(report["intervals"]),
        [
            f"  rule: coverage within {adoption['max_overall_gap']} over all rows and within "
            f"{adoption['max_top_group_gap']} in the top group, each seed-averaged and in at least "
            f"{adoption['min_calibrated_seeds']} of {seed_count} seeds.",
        ],
    )
    print_seed_averaged_table(
        "Compression: calibration slope of actual on projection (1 is calibrated)",
        pd.DataFrame(report["compression"]),
        ["  lowest and highest group bias are projected minus actual, in fantasy points."],
    )


def run_final_evaluation(
    config: dict[str, Any], positions: list[str], player_weeks: pd.DataFrame, scope: str
) -> None:
    """Run the final evaluation of the shipped system on one scope.

    Takes the parsed config, the positions, the player-week frame carrying the
    baseline, and the scope: walk-forward, which must reproduce the stated
    figures, or sealed, which scores the test season and may be run once.
    Returns nothing.

    Every position's splits are audited by row index before any model is
    trained. Nothing is saved, no model file is read, and nothing in the
    config is changed by what the run shows.
    """
    folds = final_evaluation_folds(config, scope)
    seeds = config["model"]["tuning"]["noise_floor_seeds"]
    report: dict[str, list[dict[str, Any]]] = {"audit": [], "points": [], "intervals": [], "compression": []}

    features_by_position = {}
    for position in positions:
        features_by_position[position] = load_position_features(config, position, player_weeks)
        for row in audit_final_folds(config, position, features_by_position[position], folds):
            report["audit"].append(row)

    for position in positions:
        records = final_evaluation_sweeps(config, position, features_by_position[position], seeds, folds)
        reference = records["mean"][0]["scored"]
        for table_name, rows in [
            ("points", final_point_rows(config, position, reference, records["mean"])),
            ("intervals", final_interval_rows(config, position, reference, records)),
            ("compression", final_compression_rows(config, position, reference, records["mean"])),
        ]:
            for row in rows:
                report[table_name].append(row)

    print_final_evaluation(scope, report, config)


if __name__ == "__main__":
    sys.exit(main())
