"""Assemble one training matrix per position from the tidy player-week table.

This module decides which columns become features for each position, builds
them, drops the rows with too little history to be trainable, and saves each
position's matrix to data/processed/features_<POSITION>.parquet.

Each position gets its own file and its own columns. Quarterbacks, running
backs, receivers, and tight ends score through different processes, and one
wide matrix would hand every quarterback a block of empty receiving columns
that carry no information and invite the model to read meaning into them.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ffml.features.game_context import (
    AGE_FEATURE,
    GAME_CONTEXT_FEATURES,
    add_age_at_game,
    add_game_context_features,
)
from ffml.features.injuries import INJURY_FEATURES, INJURY_SOURCE_COLUMNS, add_injury_features
from ffml.features.opponent import OPPONENT_STRENGTH_FEATURE, add_opponent_strength
from ffml.features.rolling import (
    GAMES_PLAYED_COLUMN,
    PRIOR_GAMES_COLUMN,
    add_games_played_this_season,
    add_prior_games_career,
    add_rolling_features,
    rolling_feature_names,
)
from ffml.utils.io import read_parquet, resolve_path, write_parquet

logger = logging.getLogger(__name__)

# Groups built as lagged rolling means of the per-position columns listed under
# features.source_columns in the config, in the order they are built.
ROLLING_GROUPS = ["rolling_production", "usage_share", "opportunity"]

# Every feature group this module knows how to build.
IMPLEMENTED_GROUPS = [
    "rolling_production",
    "usage_share",
    "opportunity",
    "game_context",
    "opponent_strength",
    "player_attributes",
    "injuries",
]

# Raw columns each non-rolling group reads, checked before it is built.
GAME_CONTEXT_SOURCE_COLUMNS = [
    "total_line",
    "spread_line",
    "team",
    "home_team",
    "location",
    "home_rest",
    "away_rest",
    "roof",
    "surface",
]
OPPONENT_SOURCE_COLUMNS = ["opponent_team", "position", "season", "week"]
PLAYER_ATTRIBUTE_SOURCE_COLUMNS = ["gameday", "birth_date"]

# Columns carried alongside the features so predictions can be identified,
# joined, and grouped during evaluation. None of them is a feature.
IDENTIFIER_COLUMNS = [
    "player_id",
    "player_display_name",
    "position",
    "season",
    "week",
    "team",
    "opponent_team",
    "game_id",
]


def resolve_groups(config: dict[str, Any], position: str) -> dict[str, bool]:
    """Work out which feature groups are switched on for one position.

    Takes the parsed config and the position. Returns a mapping of group name
    to whether it is enabled. Raises ValueError if a per-position override
    names a group the default does not define.

    The default applies to every position and by_position overrides single
    groups, the same shape as the LightGBM parameters. An override naming an
    unknown group is refused rather than ignored, because a typo would
    otherwise leave that group silently at its default.
    """
    groups_config = config["features"]["groups"]
    default_groups = groups_config["default"]

    resolved = {}
    for group_name in default_groups:
        resolved[group_name] = bool(default_groups[group_name])

    by_position = groups_config.get("by_position") or {}
    overrides = by_position.get(position)
    if isinstance(overrides, dict):
        for group_name in overrides:
            if group_name not in resolved:
                raise ValueError(
                    f"features.groups.by_position.{position} sets '{group_name}', which "
                    "features.groups.default does not define."
                )
            resolved[group_name] = bool(overrides[group_name])

    return resolved


def _enabled_source_columns(config: dict[str, Any], position: str) -> list[str]:
    """List the raw columns that will become rolling features for a position.

    Takes the parsed config and the position. Returns the source column names,
    in the order their features are built. Raises ValueError if the config
    lists no columns for the position.
    """
    groups = resolve_groups(config, position)
    position_columns = config["features"]["source_columns"].get(position)
    if not isinstance(position_columns, dict):
        raise ValueError(f"features.source_columns has no entry for position {position}.")

    source_columns = []
    for group_name in ROLLING_GROUPS:
        if not groups.get(group_name, False):
            continue
        for column_name in position_columns.get(group_name) or []:
            source_columns.append(column_name)
        if group_name == "rolling_production":
            # The target is also a feature input: a player's recent fantasy
            # points are the most useful single thing known about him, and the
            # three game mean of this column is exactly the naive baseline.
            source_columns.append(config["scoring"]["target_column"])

    return source_columns


def feature_column_names(config: dict[str, Any], position: str) -> list[str]:
    """List a position's feature columns, in the exact order its model expects.

    Takes the parsed config and the position. Returns the ordered feature names.

    This is a pure function of the config and reads no data, so the order saved
    beside a trained model can be reproduced at prediction time without loading
    the training matrix. build_feature_matrix checks its own output against it.
    New groups are appended at the end, so a position's existing order does not
    move when a group is added.
    """
    source_columns = _enabled_source_columns(config, position)
    windows = config["features"]["rolling_windows"]
    names = rolling_feature_names(source_columns, windows)

    # Always on rather than under player_attributes: it arrived with the first
    # feature set, and moving it under a toggle would shift the reference point
    # every later group is measured against.
    names.append(GAMES_PLAYED_COLUMN)

    groups = resolve_groups(config, position)
    if groups.get("game_context", False):
        for feature_name in GAME_CONTEXT_FEATURES:
            names.append(feature_name)
    if groups.get("opponent_strength", False):
        names.append(OPPONENT_STRENGTH_FEATURE)
    if groups.get("player_attributes", False):
        names.append(AGE_FEATURE)
    if groups.get("injuries", False):
        for feature_name in INJURY_FEATURES:
            names.append(feature_name)

    return names


def _warn_about_unimplemented_groups(groups: dict[str, bool], position: str) -> None:
    """Name any feature group switched on in the config that this module cannot build.

    Takes the resolved groups and the position. Returns nothing.
    """
    pending_groups = []
    for group_name in groups:
        if groups[group_name] and group_name not in IMPLEMENTED_GROUPS:
            pending_groups.append(group_name)

    if len(pending_groups) > 0:
        logger.warning(
            "%s: feature groups enabled in the config but not implemented: %s",
            position,
            ", ".join(pending_groups),
        )


def _check_columns_present(
    frame: pd.DataFrame, required_columns: list[str], purpose: str
) -> None:
    """Fail if the frame is missing a column that is needed.

    Takes the frame, the required column names, and what they are needed for.
    Returns nothing. Raises ValueError naming every missing column.
    """
    missing_columns = []
    for column_name in required_columns:
        if column_name not in frame.columns:
            missing_columns.append(column_name)

    if len(missing_columns) > 0:
        raise ValueError(
            "player_weeks is missing " + purpose + ": " + ", ".join(missing_columns) + ". "
            "Rerun scripts/build_features.py to rebuild the player-week table."
        )


def _check_one_position_per_player(player_weeks: pd.DataFrame) -> None:
    """Fail if any player appears under more than one position.

    Takes the player-week table. Returns nothing. Raises ValueError otherwise.

    Each position's rolling features are built from that position's rows only.
    A player listed as a running back one week and a receiver the next would
    have his history split across two matrices, and each would silently see
    only part of it.
    """
    position_counts = player_weeks.groupby("player_id")["position"].nunique()
    split_players = position_counts[position_counts > 1]
    if len(split_players) > 0:
        raise ValueError(
            f"{len(split_players)} players appear under more than one position, for "
            f"example {split_players.index[:5].tolist()}. Per-position rolling features "
            "would split their history; decide how to assign them before building."
        )


def _drop_thin_history(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Drop player-weeks with too few prior games to have real features.

    Takes the feature frame and the parsed config. Returns the filtered frame.

    This is the same condition under which the baseline declines to project, so
    the model and the baseline are scored on exactly the same rows. It counts
    prior games rather than checking which features came back NaN: the five and
    eight game windows are legitimately NaN on rows that are kept.
    """
    min_prior_games = config["features"]["min_prior_games"]
    row_count_before = len(frame)
    frame = frame[frame[PRIOR_GAMES_COLUMN] >= min_prior_games]

    logger.info(
        "filter min_prior_games: %s -> %s rows (%s removed) fewer than %s prior games played",
        row_count_before,
        len(frame),
        row_count_before - len(frame),
        min_prior_games,
    )
    return frame


def _select_output_columns(
    frame: pd.DataFrame, feature_names: list[str], config: dict[str, Any]
) -> pd.DataFrame:
    """Keep only the identifiers, the target, and the features.

    Takes the feature frame, the ordered feature names, and the parsed config.
    Returns the narrowed frame.

    Dropping the other columns is itself a leakage guard. A current week box
    score column cannot reach the model by accident if it is not in the file.
    """
    _check_columns_present(frame, IDENTIFIER_COLUMNS, "identifier columns")

    kept_columns = []
    for column_name in IDENTIFIER_COLUMNS:
        kept_columns.append(column_name)
    kept_columns.append(config["scoring"]["target_column"])
    kept_columns.append(PRIOR_GAMES_COLUMN)
    for column_name in feature_names:
        kept_columns.append(column_name)

    logger.info(
        "Feature matrix: kept %s of %s columns (%s of them features)",
        len(kept_columns),
        len(frame.columns),
        len(feature_names),
    )
    return frame[kept_columns]


def _add_non_rolling_groups(
    frame: pd.DataFrame, player_weeks: pd.DataFrame, config: dict[str, Any], position: str
) -> tuple[pd.DataFrame, list[str]]:
    """Add the enabled feature groups that are not lagged rolling means.

    Takes the frame being built, the full player-week table, the parsed config,
    and the position. Returns the frame and the ordered names added.

    None of these needs lagging except opponent strength, which does its own.
    A betting line, a stadium roof, a player's age, and the injury report are
    all known before kickoff and may be used as of the current week.
    """
    groups = resolve_groups(config, position)
    added_names: list[str] = []

    if groups.get("game_context", False):
        _check_columns_present(frame, GAME_CONTEXT_SOURCE_COLUMNS, "game context columns")
        frame, names = add_game_context_features(frame)
        added_names = added_names + names

    if groups.get("opponent_strength", False):
        _check_columns_present(frame, OPPONENT_SOURCE_COLUMNS, "opponent strength columns")
        # Rated from the full table: a defense is judged by everything it
        # allowed, not only by the players who survive this position's filters.
        frame, names = add_opponent_strength(frame, player_weeks, config)
        added_names = added_names + names

    if groups.get("player_attributes", False):
        _check_columns_present(frame, PLAYER_ATTRIBUTE_SOURCE_COLUMNS, "player attribute columns")
        frame, names = add_age_at_game(frame)
        added_names = added_names + names

    if groups.get("injuries", False):
        _check_columns_present(frame, INJURY_SOURCE_COLUMNS, "injury report columns")
        frame, names = add_injury_features(frame)
        added_names = added_names + names

    return frame, added_names


def _build_position_features(
    player_weeks: pd.DataFrame, config: dict[str, Any], position: str
) -> tuple[pd.DataFrame, list[str]]:
    """Build every enabled feature for one position's rows.

    Takes the full player-week table, the parsed config, and the position.
    Returns the frame carrying the features and the names in build order.
    Raises ValueError if the position has no rows.
    """
    position_rows = player_weeks[player_weeks["position"] == position]
    if len(position_rows) == 0:
        raise ValueError(f"player_weeks has no rows for position {position}.")
    logger.info("%s: building features from %s player-weeks", position, len(position_rows))

    source_columns = _enabled_source_columns(config, position)
    _check_columns_present(position_rows, source_columns, "feature source columns")

    windows = config["features"]["rolling_windows"]
    frame, rolling_names = add_rolling_features(
        position_rows, "player_id", source_columns, windows
    )
    frame = add_games_played_this_season(frame)
    frame = add_prior_games_career(frame)

    frame, group_names = _add_non_rolling_groups(frame, player_weeks, config, position)
    return frame, rolling_names + [GAMES_PLAYED_COLUMN] + group_names


def season_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Report what share of each column is populated, season by season.

    Takes any frame carrying a season column. Returns a table with one row per
    column and one column per season, holding the non-null share.

    This runs before the warmup season is dropped, so a warmup year that is
    poorly covered by one of the upstream tables shows up here rather than
    silently weakening the rolling windows it exists to feed.
    """
    seasons = sorted(frame["season"].unique().tolist())

    coverage_rows = []
    for column_name in frame.columns:
        row: dict[str, Any] = {"column": column_name}
        for season in seasons:
            season_rows = frame[frame["season"] == season]
            row[str(season)] = round(float(season_rows[column_name].notna().mean()), 4)
        coverage_rows.append(row)

    return pd.DataFrame(coverage_rows)


def flag_uneven_coverage(
    coverage: pd.DataFrame, tolerance: float, config: dict[str, Any]
) -> pd.DataFrame:
    """Pick out columns that are much better covered in some seasons than others.

    Takes the coverage table, how large a gap counts as material, and the
    parsed config. Returns the flagged rows with the gap size, widest gap
    first.

    A column that is well populated in most seasons and thin in one is worse
    than a column that is thin everywhere, because the model learns to lean on
    it where it exists and then loses it exactly where it is being scored.

    The gap is measured across modelled seasons only. A warmup season is always
    thin in the longer rolling windows, since nothing precedes it, and letting
    that structural gap into the comparison would flag almost every rolling
    feature and bury the columns that are genuinely uneven.
    """
    start_season = config["data"]["start_season"]

    season_columns = []
    for column_name in coverage.columns:
        if column_name != "column" and int(column_name) >= start_season:
            season_columns.append(column_name)

    gaps = coverage[season_columns].max(axis=1) - coverage[season_columns].min(axis=1)
    flagged = coverage.copy()
    flagged["gap"] = gaps
    flagged = flagged[flagged["gap"] > tolerance]
    return flagged.sort_values("gap", ascending=False).reset_index(drop=True)


def _drop_warmup_seasons(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Drop seasons that exist only to give the rolling windows history.

    Takes the feature frame and the parsed config. Returns the filtered frame.

    The warmup rows are removed here rather than at split time, so the saved
    matrix is exactly the set of rows a model may see. There is then no way for
    a warmup row to reach training, early stopping, or scoring by mistake,
    because it is not in the file at all.
    """
    warmup_start_season = config["data"].get("warmup_start_season")
    if warmup_start_season is None:
        return frame

    start_season = config["data"]["start_season"]
    row_count_before = len(frame)
    frame = frame[frame["season"] >= start_season]

    logger.info(
        "filter warmup seasons: %s -> %s rows (%s removed) seasons %s to %s are warmup only",
        row_count_before,
        len(frame),
        row_count_before - len(frame),
        warmup_start_season,
        start_season - 1,
    )
    return frame


def check_no_empty_features(frame: pd.DataFrame, feature_names: list[str], position: str) -> None:
    """Fail if any feature column holds no values at all for this position.

    Takes the feature matrix, the feature names, and the position. Returns
    nothing. Raises ValueError naming the empty columns.

    An entirely empty feature means a column was listed for a position it does
    not apply to, such as passing efficiency for a tight end. That is the wide,
    structurally empty matrix the per-position design exists to prevent, so it
    is refused rather than trained on.
    """
    empty_features = []
    for feature_name in feature_names:
        if frame[feature_name].isna().all():
            empty_features.append(feature_name)

    if len(empty_features) > 0:
        raise ValueError(
            f"{position}: feature columns with no values at all: {', '.join(empty_features)}. "
            "Check features.source_columns for this position."
        )


def build_feature_matrix(
    player_weeks: pd.DataFrame,
    config: dict[str, Any],
    position: str,
    keep_warmup_seasons: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Build one position's training matrix from the tidy player-week table.

    Takes the player-week frame, the parsed config, the position, and whether
    to keep the warmup seasons in the result. Returns the feature matrix and
    the ordered list of feature column names.

    keep_warmup_seasons exists so the coverage report can look at the warmup
    year before it is discarded. It defaults to False, so every other caller
    gets a matrix a model may safely be trained on.

    Raises ValueError if the columns built do not match feature_column_names,
    which would mean the feature order saved beside a model no longer describes
    the matrix it was fitted on.
    """
    _warn_about_unimplemented_groups(resolve_groups(config, position), position)
    _check_one_position_per_player(player_weeks)

    frame, built_names = _build_position_features(player_weeks, config, position)

    feature_names = feature_column_names(config, position)
    if built_names != feature_names:
        raise ValueError(
            f"{position}: the features built do not match feature_column_names. Built "
            + str(built_names)
            + ", expected "
            + str(feature_names)
            + "."
        )

    frame = _drop_thin_history(frame, config)
    frame = _select_output_columns(frame, feature_names, config)
    if not keep_warmup_seasons:
        frame = _drop_warmup_seasons(frame, config)
    return frame.reset_index(drop=True), feature_names


def feature_file_path(config: dict[str, Any], position: str) -> Path:
    """Locate the saved feature matrix for one position.

    Takes the parsed config and the position. Returns the Parquet path.
    """
    processed_directory = resolve_path(config["data"]["paths"]["processed"])
    return processed_directory / f"features_{position}.parquet"


def build_and_save_features(config: dict[str, Any]) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build and save every position's feature matrix.

    Takes the parsed config. Returns, for each position, its feature matrix and
    its per-season coverage table. Each matrix is also written to
    data/processed/features_<POSITION>.parquet.

    Each matrix is built with the warmup seasons still attached so coverage can
    be measured across every season pulled, then they are dropped before saving.
    """
    processed_directory = resolve_path(config["data"]["paths"]["processed"])
    player_weeks = read_parquet(processed_directory / "player_weeks.parquet")
    logger.info(
        "Loaded player_weeks: %s rows x %s columns", len(player_weeks), len(player_weeks.columns)
    )

    built = {}
    for position in config["data"]["positions"]:
        with_warmup, feature_names = build_feature_matrix(
            player_weeks, config, position, keep_warmup_seasons=True
        )
        coverage = season_coverage(with_warmup)
        frame = _drop_warmup_seasons(with_warmup, config).reset_index(drop=True)
        check_no_empty_features(frame, feature_names, position)

        destination = feature_file_path(config, position)
        write_parquet(frame, destination)
        logger.info(
            "Saved %s features: %s rows x %s columns (%s features) -> %s",
            position,
            len(frame),
            len(frame.columns),
            len(feature_names),
            destination,
        )
        built[position] = (frame, coverage)

    return built
