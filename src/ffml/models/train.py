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


def save_model(result: dict[str, Any], config: dict[str, Any]) -> tuple[Path, Path]:
    """Save the booster and its metadata to the models directory.

    Takes the training result and the parsed config. Returns the paths to the
    model file and the metadata file.

    The feature list is stored in order, because the prediction code must build
    its columns in exactly that order for the model to read them correctly. The
    booster is truncated to the best iteration, so the saved artifact is the
    model that was actually selected rather than the last one fitted.
    """
    models_directory = ensure_directory(resolve_path(config["data"]["paths"]["models"]))
    position = result["position"]

    model_path = models_directory / f"lightgbm_{position}.txt"
    metadata_path = models_directory / f"lightgbm_{position}_metadata.json"

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
