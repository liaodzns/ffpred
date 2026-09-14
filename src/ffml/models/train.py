"""LightGBM training, one model per position.

QB, RB, WR, and TE generate fantasy points through different processes, so
each position gets its own model. A single shared model would waste capacity
learning that quarterbacks outscore tight ends, which is already known.

The split is always by time, and it has three parts. Training runs through
validation.train_through_season, minus the tail of validation.early_stopping_season
that early stopping watches, and the model is scored on validation.validation_season.
Keeping the early stopping rows separate matters: stopping on the scored season
chooses the number of rounds to suit the very rows being reported.

validation.test_season is never read here, and a guard raises if it appears in
any split, so the sealed holdout is protected by code rather than by memory.
"""

import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from ffml.features.build_dataset import feature_column_names, resolve_groups
from ffml.utils.io import ensure_directory, resolve_path

logger = logging.getLogger(__name__)

# How many features the importance report shows.
TOP_IMPORTANCE_FEATURES = 15


def resolve_parameters(position: str, config: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    """Merge the shared LightGBM parameters with this position's overrides.

    Takes the position and the parsed config. Returns the booster parameters,
    the number of boosting rounds, and the early stopping patience.

    The seed is fixed from project.random_seed so that a rerun reproduces the
    same model and the comparison against the baseline does not wobble between
    runs for reasons that have nothing to do with the features.
    """
    lightgbm_config = config["model"]["lightgbm"]

    parameters = dict(lightgbm_config["shared"])
    overrides = lightgbm_config["by_position"].get(position)
    if isinstance(overrides, dict):
        for parameter_name in overrides:
            parameters[parameter_name] = overrides[parameter_name]

    num_boost_round = parameters.pop("num_boost_round")
    early_stopping_rounds = parameters.pop("early_stopping_rounds")
    parameters["seed"] = config["project"]["random_seed"]

    return parameters, num_boost_round, early_stopping_rounds


def _guard_test_season(frame: pd.DataFrame, test_season: int, split_name: str) -> None:
    """Fail if the sealed test season turned up in a split.

    Takes the split frame, the test season, and the split's name. Returns
    nothing. Raises ValueError if the test season is present.

    The test season stays untouched until the project is finished. Looking at
    it even once, to check a number, quietly turns it into a second validation
    set and there is then no honest holdout left.
    """
    if (frame["season"] == test_season).any():
        raise ValueError(
            f"The {split_name} split contains rows from the sealed test season {test_season}. "
            "Check validation.train_through_season and validation.validation_season."
        )


def single_holdout_fold(config: dict[str, Any]) -> dict[str, Any]:
    """Describe the simple one-season holdout named in the validation config.

    Takes the parsed config. Returns a fold specification.

    A fold is just the four boundaries a split needs. Expressing the simple
    holdout in the same shape as a walk-forward fold means one split function
    serves both, so the two paths cannot drift apart.
    """
    validation_config = config["validation"]
    return {
        "name": str(validation_config["validation_season"]),
        "train_through_season": validation_config["train_through_season"],
        "early_stopping_season": validation_config["early_stopping_season"],
        "early_stopping_from_week": validation_config["early_stopping_from_week"],
        "validation_season": validation_config["validation_season"],
    }


def walk_forward_folds(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one fold per test season, in chronological order.

    Takes the parsed config. Returns the fold specifications.

    Each fold trains on everything before its test season and stops on the tail
    of the season immediately preceding it. Retraining for every season is the
    honest form of validation: it asks how the model would have performed if it
    had been built with only what was known at the time, repeatedly, rather
    than betting the whole verdict on one arbitrary holdout year.
    """
    validation_config = config["validation"]
    walk_forward_config = validation_config["walk_forward"]

    folds = []
    first_test_season = walk_forward_config["first_test_season"]
    last_test_season = walk_forward_config["last_test_season"]
    for test_season in range(first_test_season, last_test_season + 1):
        folds.append(
            {
                "name": str(test_season),
                "train_through_season": test_season - 1,
                "early_stopping_season": test_season - 1,
                "early_stopping_from_week": validation_config["early_stopping_from_week"],
                "validation_season": test_season,
            }
        )
    return folds


def _check_fold(fold: dict[str, Any], config: dict[str, Any]) -> None:
    """Fail if a fold's boundaries are not in chronological order.

    Takes the fold specification and the parsed config. Returns nothing.
    Raises ValueError describing the first problem found.
    """
    early_stopping_season = fold["early_stopping_season"]
    train_through_season = fold["train_through_season"]
    validation_season = fold["validation_season"]
    start_season = config["data"]["start_season"]

    if early_stopping_season > train_through_season:
        raise ValueError(
            f"Fold {fold['name']}: early stopping season ({early_stopping_season}) is after "
            f"the last training season ({train_through_season}). The early stopping rows "
            "must be carved out of the training seasons, not from beyond them."
        )

    if early_stopping_season >= validation_season:
        raise ValueError(
            f"Fold {fold['name']}: early stopping season ({early_stopping_season}) is not "
            f"before the scored season ({validation_season}). Stopping on the season being "
            "scored is the leak this split exists to remove."
        )

    if early_stopping_season < start_season:
        raise ValueError(
            f"Fold {fold['name']}: early stopping season ({early_stopping_season}) is before "
            f"data.start_season ({start_season}). Warmup seasons build features and are "
            "never fitted, stopped, or scored on."
        )


def split_by_time(
    features: pd.DataFrame, position: str, fold: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one position's rows into training, early stopping, and scoring sets.

    Takes the feature matrix, the position, the fold specification, and the
    parsed config. Returns the training rows, the early stopping rows, and the
    scored rows.

    The three are strictly ordered in time: training runs up to a week inside
    the fold's early stopping season, the early stopping rows are the remainder
    of that season, and the scored season comes after both. The model therefore
    never sees the scored season while fitting or while choosing how many
    rounds to fit for.

    Nothing is ever shuffled. A random split would put week 12 in the training
    set and week 5 of the same season in validation, letting the model learn
    from a future it could not have seen.
    """
    _check_fold(fold, config)

    position_rows = features[features["position"] == position]
    in_training_seasons = position_rows["season"] <= fold["train_through_season"]
    is_early_stopping = (position_rows["season"] == fold["early_stopping_season"]) & (
        position_rows["week"] >= fold["early_stopping_from_week"]
    )

    train_rows = position_rows[in_training_seasons & ~is_early_stopping]
    early_stopping_rows = position_rows[in_training_seasons & is_early_stopping]
    validation_rows = position_rows[position_rows["season"] == fold["validation_season"]]

    _check_splits(train_rows, early_stopping_rows, validation_rows, fold, config, position)
    return train_rows, early_stopping_rows, validation_rows


def _guard_warmup_seasons(frame: pd.DataFrame, start_season: int, split_name: str) -> None:
    """Fail if a warmup season turned up in a split.

    Takes the split frame, the first modelled season, and the split's name.
    Returns nothing. Raises ValueError if an earlier season is present.

    build_dataset already drops these rows, so this should be unreachable. It
    is here because a warmup row reaching a model would be invisible in the
    metrics: a season's worth of extra data would simply appear, and nothing
    would look wrong.
    """
    if (frame["season"] < start_season).any():
        present = sorted(frame[frame["season"] < start_season]["season"].unique().tolist())
        raise ValueError(
            f"The {split_name} split contains warmup seasons {present}, which are before "
            f"data.start_season ({start_season}). Warmup seasons build features and must "
            "never be fitted, stopped, or scored on."
        )


def _check_splits(
    train_rows: pd.DataFrame,
    early_stopping_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    fold: dict[str, Any],
    config: dict[str, Any],
    position: str,
) -> None:
    """Check the three splits are non-empty, disjoint, and free of forbidden seasons.

    Takes the three splits, the fold specification, the parsed config, and the
    position. Returns nothing. Raises ValueError describing the first problem
    found.

    A split error does not announce itself in the metrics; it just quietly
    makes every number in the run wrong, so all of it is checked rather than
    assumed.
    """
    test_season = config["validation"]["test_season"]
    start_season = config["data"]["start_season"]

    named_splits = [
        ("training", train_rows),
        ("early stopping", early_stopping_rows),
        ("scored", validation_rows),
    ]
    for split_name, split_rows in named_splits:
        _guard_test_season(split_rows, test_season, split_name)
        _guard_warmup_seasons(split_rows, start_season, split_name)
        if len(split_rows) == 0:
            raise ValueError(
                f"Fold {fold['name']}: the {split_name} split for {position} is empty. "
                "Check the validation config against the seasons in the feature matrix."
            )

    for left_index in range(len(named_splits)):
        for right_index in range(left_index + 1, len(named_splits)):
            left_name, left_rows = named_splits[left_index]
            right_name, right_rows = named_splits[right_index]
            overlap = left_rows.index.intersection(right_rows.index)
            if len(overlap) > 0:
                raise ValueError(
                    f"Fold {fold['name']}: the {left_name} and {right_name} splits share "
                    f"{len(overlap)} rows. The three splits must be disjoint."
                )

    logger.info(
        "%s fold %s: %s training rows, %s early stopping rows (%s from week %s), "
        "%s scored rows (%s)",
        position,
        fold["name"],
        len(train_rows),
        len(early_stopping_rows),
        fold["early_stopping_season"],
        fold["early_stopping_from_week"],
        len(validation_rows),
        fold["validation_season"],
    )


def row_counts_by_season(frame: pd.DataFrame) -> pd.DataFrame:
    """Count rows in each season.

    Takes a split frame. Returns a frame of season and rows.
    """
    return frame.groupby("season").size().reset_index(name="rows")


def row_counts_by_season_and_week(frame: pd.DataFrame) -> pd.DataFrame:
    """Count rows in each week of each season.

    Takes a split frame. Returns a table with one row per week and one column
    per season, holding zero where a week contributed nothing.

    Worth reading rather than skimming. A season whose opening weeks are empty
    is the expected shape for the earliest season in the data, where no player
    has prior history to roll over, but the same shape in a later season means
    something upstream is dropping rows it should not.
    """
    counts = frame.groupby(["season", "week"]).size().reset_index(name="rows")
    table = counts.pivot(index="week", columns="season", values="rows")
    return table.fillna(0).astype(int)


def train_model(
    features: pd.DataFrame,
    position: str,
    config: dict[str, Any],
    fold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a LightGBM model for one position.

    Takes the feature matrix, the position, the parsed config, and optionally
    a fold specification, defaulting to the simple one-season holdout. Returns
    a dictionary holding the booster, the ordered feature names, the resolved
    parameters, the best iteration, and the three splits.

    Early stopping watches the inner split, never the validation season, so the
    number of rounds is chosen without reference to the rows the model is later
    scored on. The validation season is not handed to lgb.train at all.
    """
    if fold is None:
        fold = single_holdout_fold(config)

    feature_names = feature_column_names(config, position)
    train_rows, early_stopping_rows, validation_rows = split_by_time(
        features, position, fold, config
    )

    parameters, num_boost_round, early_stopping_rounds = resolve_parameters(position, config)
    target_column = config["scoring"]["target_column"]

    train_set = lgb.Dataset(
        train_rows[feature_names], label=train_rows[target_column], feature_name=feature_names
    )
    early_stopping_set = lgb.Dataset(
        early_stopping_rows[feature_names],
        label=early_stopping_rows[target_column],
        reference=train_set,
        feature_name=feature_names,
    )

    # Annotated because the two callbacks have different types and mypy
    # otherwise widens the list to object.
    callbacks: list[Callable[..., Any]] = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[early_stopping_set],
        valid_names=["early_stopping"],
        callbacks=callbacks,
    )

    logger.info(
        "Trained %s fold %s: stopped at iteration %s of %s",
        position,
        fold["name"],
        booster.best_iteration,
        num_boost_round,
    )

    return {
        "booster": booster,
        "position": position,
        "fold": fold,
        "feature_names": feature_names,
        "parameters": parameters,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": booster.best_iteration,
        "train_rows": train_rows,
        "early_stopping_rows": early_stopping_rows,
        "validation_rows": validation_rows,
    }


def predict(result: dict[str, Any], features: pd.DataFrame) -> pd.Series:
    """Project fantasy points for a set of feature rows.

    Takes the training result and the rows to project. Returns a Series of
    projections aligned to the frame's index.

    The columns are selected by the saved feature list, so the model is always
    handed its features in the order it was fitted on.
    """
    booster = result["booster"]
    feature_names = result["feature_names"]

    values = booster.predict(features[feature_names], num_iteration=result["best_iteration"])
    return pd.Series(values, index=features.index, name="model_projection")


def gain_importance_table(result: dict[str, Any], top_n: int) -> pd.DataFrame:
    """Rank the features by the total gain their splits produced.

    Takes the training result and how many rows to return. Returns a frame of
    feature, gain, and the share of total gain.

    Raises ValueError if the booster's feature order does not match the saved
    list, since the importances would then be attributed to the wrong names.
    """
    booster = result["booster"]
    if booster.feature_name() != result["feature_names"]:
        raise ValueError(
            "The booster's feature order does not match the saved feature list. "
            "Importances cannot be attributed reliably."
        )

    table = pd.DataFrame(
        {
            "feature": result["feature_names"],
            "gain": booster.feature_importance(importance_type="gain"),
        }
    )

    total_gain = table["gain"].sum()
    if total_gain > 0:
        table["gain_share"] = table["gain"] / total_gain
    else:
        table["gain_share"] = 0.0

    table = table.sort_values("gain", ascending=False).reset_index(drop=True)
    return table.head(top_n)


def save_model(
    result: dict[str, Any], config: dict[str, Any], name_suffix: str = ""
) -> tuple[Path, Path]:
    """Save the booster and its metadata to the models directory.

    Takes the training result, the parsed config, and an optional file name
    suffix, such as _q90 for a quantile model. Returns the paths to the model
    file and the metadata file.

    The feature list is stored in order, because the prediction code must build
    its columns in exactly that order for the model to read them correctly. The
    booster is truncated to the best iteration, so the saved artifact is the
    model that was actually selected rather than the last one fitted. The
    suffix keeps a quantile model from ever overwriting its mean model.
    """
    models_directory = ensure_directory(resolve_path(config["data"]["paths"]["models"]))
    position = result["position"]

    model_path = models_directory / f"lightgbm_{position}{name_suffix}.txt"
    metadata_path = models_directory / f"lightgbm_{position}{name_suffix}_metadata.json"

    result["booster"].save_model(str(model_path), num_iteration=result["best_iteration"])

    metadata = {
        "position": position,
        "target_column": config["scoring"]["target_column"],
        "feature_names": result["feature_names"],
        "feature_groups": resolve_groups(config, position),
        "parameters": result["parameters"],
        "num_boost_round": result["num_boost_round"],
        "early_stopping_rounds": result["early_stopping_rounds"],
        "best_iteration": result["best_iteration"],
        "fold": result["fold"],
        "warmup_start_season": config["data"].get("warmup_start_season"),
        "start_season": config["data"]["start_season"],
        "train_row_count": len(result["train_rows"]),
        "early_stopping_row_count": len(result["early_stopping_rows"]),
        "validation_row_count": len(result["validation_rows"]),
        "rolling_windows": config["features"]["rolling_windows"],
        "min_prior_games": config["features"]["min_prior_games"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("Saved %s model -> %s and %s", position, model_path, metadata_path)
    return model_path, metadata_path


# ---------------------------------------------------------------------------
# Deployment training
#
# Walk-forward is how the models are evaluated. Deployment is how they are used:
# one model per position trained on every completed week, to project weeks that
# have not been played. The two share no file and no split function, so nothing
# done for live use can move an evaluation number.
# ---------------------------------------------------------------------------


def deployment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy the config with the deployment-only group switches applied.

    Takes the parsed config. Returns a deep copy in which every group named in
    predict.deployment.disabled_groups is off for every position. The original
    config, which the walk-forward evaluation reads, is left untouched.
    """
    deployed = copy.deepcopy(config)
    groups_config = deployed["features"]["groups"]
    by_position = groups_config.get("by_position") or {}

    for position in deployed["data"]["positions"]:
        overrides = by_position.get(position) or {}
        for group_name in deployed["predict"]["deployment"]["disabled_groups"]:
            overrides[group_name] = False
        by_position[position] = overrides

    groups_config["by_position"] = by_position
    return deployed


def week_completion(
    schedules: pd.DataFrame, player_weeks: pd.DataFrame, season: int, week: int
) -> dict[str, Any]:
    """Check whether one regular season week has fully arrived in the data.

    Takes the schedules, the player-week table, and the season and week.
    Returns whether it is complete, the unplayed games and their teams, and the
    teams whose stats are missing.

    Complete means every game has a final score and every team that played has
    player stats. Stats lag scores by hours to days, and history built on a
    half-loaded week gives some players one fewer game than others, silently.
    """
    is_week = (
        (schedules["season"] == season)
        & (schedules["week"] == week)
        & (schedules["game_type"] == "REG")
    )
    games = schedules[is_week]

    unplayed_games = []
    unplayed_teams = []
    played_teams = []
    for _, game in games.iterrows():
        if pd.isna(game["home_score"]):
            unplayed_games.append(f"{game['away_team']}@{game['home_team']}")
            unplayed_teams.append(game["away_team"])
            unplayed_teams.append(game["home_team"])
        else:
            played_teams.append(game["away_team"])
            played_teams.append(game["home_team"])

    is_stats_week = (player_weeks["season"] == season) & (player_weeks["week"] == week)
    teams_with_stats = set(player_weeks[is_stats_week]["team"])
    teams_missing_stats = []
    for team in sorted(set(played_teams)):
        if team not in teams_with_stats:
            teams_missing_stats.append(team)

    return {
        "complete": len(games) > 0 and len(unplayed_games) == 0 and len(teams_missing_stats) == 0,
        "unplayed_games": unplayed_games,
        "unplayed_teams": sorted(set(unplayed_teams)),
        "teams_missing_stats": teams_missing_stats,
    }


def completed_weeks(
    schedules: pd.DataFrame, player_weeks: pd.DataFrame, config: dict[str, Any]
) -> list[tuple[int, int]]:
    """List every completed regular season week from start_season, in order.

    Takes the schedules, the player-week table, and the parsed config. Returns
    the completed weeks up to the first incomplete one. Raises ValueError if a
    complete week follows an incomplete one.

    Normally the first incomplete week is simply the current one. A complete
    week after an incomplete one means a gap in the data, and training across a
    gap would skip a week without saying so.
    """
    regular = schedules[
        (schedules["game_type"] == "REG") & (schedules["season"] >= config["data"]["start_season"])
    ]
    week_pairs = regular[["season", "week"]].drop_duplicates().sort_values(["season", "week"])

    completed: list[tuple[int, int]] = []
    first_incomplete = None
    for _, pair in week_pairs.iterrows():
        season = int(pair["season"])
        week = int(pair["week"])
        is_complete = week_completion(schedules, player_weeks, season, week)["complete"]

        if first_incomplete is None and is_complete:
            completed.append((season, week))
        elif first_incomplete is None:
            first_incomplete = (season, week)
        elif is_complete:
            raise ValueError(
                f"Season {season} week {week} is complete but {first_incomplete} before it is "
                "not. The data has a gap; deployment training would skip it silently."
            )
    return completed


def _check_deployment_splits(
    train_rows: pd.DataFrame,
    stop_rows: pd.DataFrame,
    refit_rows: pd.DataFrame,
    position: str,
    config: dict[str, Any],
) -> None:
    """Check the deployment splits are non-empty, disjoint, and free of warmup rows.

    Takes the training, early stopping, and refit rows, the position, and the
    parsed config. Returns nothing. Raises ValueError on any problem.
    """
    start_season = config["data"]["start_season"]
    named_splits = [("training", train_rows), ("early stopping", stop_rows)]
    for split_name, split_rows in named_splits:
        _guard_warmup_seasons(split_rows, start_season, "deployment " + split_name)
        if len(split_rows) == 0:
            raise ValueError(f"The deployment {split_name} split for {position} is empty.")

    if len(train_rows.index.intersection(stop_rows.index)) > 0:
        raise ValueError(f"{position}: deployment training and early stopping rows overlap.")
    if len(refit_rows) != len(train_rows) + len(stop_rows):
        raise ValueError(f"{position}: the refit rows are not exactly training plus stopping rows.")


def deployment_split(
    features: pd.DataFrame,
    position: str,
    completed: list[tuple[int, int]],
    early_stopping_weeks: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one position's completed weeks for deployment training.

    Takes the feature matrix, the position, the completed weeks in order, how
    many of the latest to stop on, and the parsed config. Returns the training
    rows, the early stopping rows, and the refit rows, which are both together.
    Raises ValueError if there are too few completed weeks.

    Rows from weeks that are not complete, such as a week still being played,
    reach none of the three.
    """
    if len(completed) <= early_stopping_weeks:
        raise ValueError(f"Only {len(completed)} completed weeks; cannot hold out {early_stopping_weeks}.")

    completed_keys = []
    for season, week in completed:
        completed_keys.append(season * 100 + week)
    stop_keys = completed_keys[-early_stopping_weeks:]

    position_rows = features[features["position"] == position]
    row_keys = position_rows["season"] * 100 + position_rows["week"]
    in_completed = row_keys.isin(completed_keys)
    in_stop = row_keys.isin(stop_keys)

    train_rows = position_rows[in_completed & ~in_stop]
    stop_rows = position_rows[in_stop]
    refit_rows = position_rows[in_completed]
    _check_deployment_splits(train_rows, stop_rows, refit_rows, position, config)

    logger.info(
        "%s deployment split: %s training, %s early stopping (last %s completed weeks), "
        "%s refit; %s rows from incomplete weeks excluded",
        position,
        len(train_rows),
        len(stop_rows),
        early_stopping_weeks,
        len(refit_rows),
        len(position_rows) - len(refit_rows),
    )
    return train_rows, stop_rows, refit_rows


def _choose_deployment_rounds(
    train_rows: pd.DataFrame,
    stop_rows: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    parameters: dict[str, Any],
    round_limits: tuple[int, int],
) -> int:
    """Find the number of boosting rounds by early stopping on the latest weeks.

    Takes the training and stopping rows, the feature names, the target
    column, the booster parameters, and the maximum rounds and stopping
    patience. Returns the round count to refit with.

    The stopping loss is deliberately never logged or returned. These rows
    include the sealed test season, and reading a score off them would make it
    an evaluation.
    """
    num_boost_round, early_stopping_rounds = round_limits
    train_set = lgb.Dataset(
        train_rows[feature_names], label=train_rows[target_column], feature_name=feature_names
    )
    stop_set = lgb.Dataset(
        stop_rows[feature_names],
        label=stop_rows[target_column],
        reference=train_set,
        feature_name=feature_names,
    )

    callbacks: list[Callable[..., Any]] = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[stop_set],
        valid_names=["early_stopping"],
        callbacks=callbacks,
    )

    if booster.best_iteration > 0:
        return int(booster.best_iteration)
    return num_boost_round


def train_deployment_model(
    features: pd.DataFrame,
    position: str,
    config: dict[str, Any],
    completed: list[tuple[int, int]],
) -> dict[str, Any]:
    """Fit the live model for one position on every completed week.

    Takes the feature matrix, the position, the parsed config, and the
    completed weeks in order. Returns the refit booster and what it was trained
    on.

    Two passes. The first trains on every completed week except the latest few
    and early stops on those, which chooses the round count. The second refits
    on every completed week at that count. The latest weeks say the most about
    next week's game, and a live model that never trained on them would be
    weakest exactly where it is used. Walk-forward never refits, because it has
    to be scored honestly; this model is never scored at all.
    """
    deployed = deployment_config(config)
    feature_names = feature_column_names(deployed, position)
    early_stopping_weeks = deployed["predict"]["deployment"]["early_stopping_weeks"]
    train_rows, stop_rows, refit_rows = deployment_split(
        features, position, completed, early_stopping_weeks, deployed
    )

    parameters, num_boost_round, early_stopping_rounds = resolve_parameters(position, deployed)
    target_column = deployed["scoring"]["target_column"]
    rounds = _choose_deployment_rounds(
        train_rows, stop_rows, feature_names, target_column, parameters,
        (num_boost_round, early_stopping_rounds),
    )

    refit_set = lgb.Dataset(
        refit_rows[feature_names], label=refit_rows[target_column], feature_name=feature_names
    )
    booster = lgb.train(parameters, refit_set, num_boost_round=rounds)

    return {
        "booster": booster,
        "position": position,
        "feature_names": feature_names,
        "parameters": parameters,
        "rounds": rounds,
        "trained_through": completed[-1],
        "early_stopping_weeks": completed[-early_stopping_weeks:],
        "train_rows": train_rows,
        "stop_rows": stop_rows,
        "refit_rows": refit_rows,
    }


def deployment_model_paths(
    config: dict[str, Any], position: str, name_suffix: str = ""
) -> tuple[Path, Path]:
    """Locate a position's deployment model and metadata files.

    Takes the parsed config, the position, and an optional file name suffix,
    such as _q90 for a quantile model. Returns the two paths.

    They sit in their own subdirectory, so they can never overwrite the
    walk-forward models saved directly under data.paths.models, and the suffix
    keeps a quantile model from overwriting its position's mean model.
    """
    directory = (
        resolve_path(config["data"]["paths"]["models"])
        / config["predict"]["deployment"]["model_subdirectory"]
    )
    return (
        directory / f"lightgbm_{position}{name_suffix}.txt",
        directory / f"lightgbm_{position}{name_suffix}_metadata.json",
    )


def save_deployment_model(
    result: dict[str, Any], config: dict[str, Any], name_suffix: str = ""
) -> tuple[Path, Path]:
    """Save a deployment model and its metadata.

    Takes the deployment training result, the parsed config, and an optional
    file name suffix. Returns the model and metadata paths.
    """
    position = result["position"]
    model_path, metadata_path = deployment_model_paths(config, position, name_suffix)
    ensure_directory(model_path.parent)
    result["booster"].save_model(str(model_path))

    test_season = config["validation"]["test_season"]
    metadata = {
        "position": position,
        "purpose": "deployment: live projection only, never evaluation",
        "target_column": config["scoring"]["target_column"],
        "feature_names": result["feature_names"],
        "feature_groups": resolve_groups(deployment_config(config), position),
        "disabled_groups": config["predict"]["deployment"]["disabled_groups"],
        "parameters": result["parameters"],
        "boosting_rounds": result["rounds"],
        "trained_through": list(result["trained_through"]),
        "early_stopping_weeks": [list(pair) for pair in result["early_stopping_weeks"]],
        "train_row_count": len(result["train_rows"]),
        "early_stopping_row_count": len(result["stop_rows"]),
        "refit_row_count": len(result["refit_rows"]),
        "start_season": config["data"]["start_season"],
        "note": (
            f"Trained and early stopped on data including {test_season}. Never score these "
            f"models on {test_season}; only the walk-forward models may be evaluated there."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved %s deployment model -> %s", position, model_path)
    return model_path, metadata_path


def load_deployment_model(
    config: dict[str, Any], position: str, name_suffix: str = ""
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Load a position's deployment model and metadata.

    Takes the parsed config, the position, and an optional file name suffix,
    such as _q90 for a quantile model. Returns the booster and the metadata.
    Raises FileNotFoundError if either file is missing.
    """
    model_path, metadata_path = deployment_model_paths(config, position, name_suffix)
    if not model_path.is_file() or not metadata_path.is_file():
        command = "--deploy"
        if name_suffix:
            command = "--deploy-quantiles"
        raise FileNotFoundError(
            f"No deployment model for {position}{name_suffix} at {model_path}. "
            f"Run scripts/train_model.py {command} first."
        )
    booster = lgb.Booster(model_file=str(model_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return booster, metadata


# ---------------------------------------------------------------------------
# Scoring the walk-forward folds
#
# Shared by walk-forward training and the seed noise floor, so the numbers the
# noise floor reports come from exactly the same fold loop as every evaluation.
# ---------------------------------------------------------------------------

# Column a walk-forward projection is attached to on the scored rows.
MODEL_PROJECTION_COLUMN = "model_projection"


def folds_for(config: dict[str, Any]) -> list[dict[str, Any]]:
    """List the folds to train, honoring the walk-forward toggle.

    Takes the parsed config. Returns one fold per test season when walk-forward
    is enabled, otherwise the single holdout fold.
    """
    if config["validation"]["walk_forward"]["enabled"]:
        return walk_forward_folds(config)
    return [single_holdout_fold(config)]


def score_walk_forward(
    features: pd.DataFrame, position: str, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Train one model per fold and gather every scored row.

    Takes the feature matrix, the position, and the parsed config. Returns the
    pooled scored rows carrying the projection and a fold label, plus the
    per-fold training results.

    Each fold is fitted from scratch, so no fold's model carries what an earlier
    one learned about seasons it was not supposed to have seen yet.
    """
    scored_parts = []
    results = []

    for fold in folds_for(config):
        result = train_model(features, position, config, fold)
        scored = result["validation_rows"].copy()
        scored[MODEL_PROJECTION_COLUMN] = predict(result, result["validation_rows"])
        scored["fold"] = fold["name"]
        scored_parts.append(scored)
        results.append(result)

    return pd.concat(scored_parts, ignore_index=True), results
