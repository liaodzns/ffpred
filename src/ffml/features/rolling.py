"""Lagged rolling averages of player production.

This module owns the single rolling implementation used everywhere in the
project. The naive baseline in ffml.models.baseline imports it from here, so
there is exactly one place where a game can be let into its own prediction
window, and exactly one place to check when verifying that it cannot be.

Every function here sorts by player, season, and week before doing anything.
The incoming row order is never trusted, as CLAUDE.md requires.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that put a player's games into chronological order. Sorting by these
# before computing any rolling statistic is required by the leakage rule in
# CLAUDE.md; the incoming row order is never trusted.
SORT_COLUMNS = ["player_id", "season", "week"]

# Number of games the player has already played in the current season.
GAMES_PLAYED_COLUMN = "games_played_this_season"

# Number of games the player has already played across all seasons in the data.
# Not a feature. It exists so the min_prior_games filter can be applied and
# logged explicitly rather than inferred from which features came back NaN.
PRIOR_GAMES_COLUMN = "prior_games_career"


def shift_then_roll(
    frame: pd.DataFrame,
    group_column: str,
    value_column: str,
    window: int,
    min_periods: int,
) -> pd.Series:
    """Shift a column by one row within each group, then take a rolling mean.

    Takes the frame, the column identifying each group, the column to average,
    the number of prior games in the window, and the fewest prior games needed
    before a value is produced. Returns a Series aligned to the frame's index,
    holding NaN wherever there is too little history.

    The shift happens before the roll, never after. That ordering is what keeps
    a game out of its own prediction window, and it is the most important
    detail in this module.

    The window counts rows, and here a row is a game. The player-week table
    holds one row per game actually played, with nothing for bye weeks or
    inactive games, so three rows back is three games back no matter how many
    calendar weeks that spans. A time-based window would silently reach back
    over missed games and average whatever happened to fall inside it.
    """
    shifted = frame.groupby(group_column)[value_column].shift(1)
    rolled = shifted.groupby(frame[group_column]).rolling(window, min_periods=min_periods).mean()

    # Rolling on a grouped Series returns a frame indexed by group and then by
    # the original row, so the group level is dropped to restore alignment.
    rolled = rolled.reset_index(level=0, drop=True)
    return rolled.reindex(frame.index)


def shift_then_expand(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
    min_periods: int,
) -> pd.Series:
    """Shift a column by one row within each group, then average everything before it.

    Takes the frame, the columns identifying each group, the column to average,
    and the fewest prior rows needed before a value is produced. Returns a
    Series aligned to the frame's index.

    This is shift_then_roll with no window: every earlier row in the group
    counts, not just the last few. Opponent strength uses it, because "what
    this defense has allowed so far this season" is a growing window rather
    than a fixed one.

    It takes several group columns where shift_then_roll takes one, which is
    why it is a separate function rather than an option on that one. Splitting
    them keeps the single column path, and its tests, exactly as they were.
    """
    groupers = []
    for column_name in group_columns:
        groupers.append(frame[column_name])

    shifted = frame.groupby(groupers)[value_column].shift(1)
    expanded = shifted.groupby(groupers).expanding(min_periods=min_periods).mean()

    # Expanding on a grouped Series returns a frame indexed by the group keys
    # and then by the original row, so those levels are dropped to restore
    # alignment with the frame that was passed in.
    expanded = expanded.reset_index(level=list(range(len(group_columns))), drop=True)
    return expanded.reindex(frame.index)


def rolling_feature_name(value_column: str, window: int) -> str:
    """Build the column name for one rolling feature.

    Takes the source column name and the window length. Returns the feature
    name, such as "targets_roll3".

    The source column name is kept rather than prettified, so that every
    feature traces straight back to a real column in the data with no mapping
    layer in between.
    """
    return f"{value_column}_roll{window}"


def rolling_feature_names(columns: list[str], windows: list[int]) -> list[str]:
    """List the feature names that add_rolling_features will create.

    Takes the source columns and the window lengths. Returns the names in the
    order the features are built, column by column and window by window.

    This is a pure function of its arguments and touches no data, so the
    feature order saved with a model can be reproduced later without loading
    the training matrix.
    """
    names = []
    for column_name in columns:
        for window in windows:
            names.append(rolling_feature_name(column_name, window))
    return names


def add_rolling_features(
    frame: pd.DataFrame,
    group_column: str,
    columns: list[str],
    windows: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """Add a lagged rolling mean of every column over every window.

    Takes the player-week frame, the column identifying each player, the source
    columns to average, and the window lengths. Returns a copy of the frame
    carrying the new columns, aligned to the caller's index, and the ordered
    list of names that were added.

    min_periods is set equal to the window, so an eight game average is always
    a true eight game average and is NaN otherwise. A shorter fallback would
    make the three, five, and eight game columns hold the same number for
    players early in their careers, which quietly destroys the distinction the
    three windows exist to draw. LightGBM reads the NaN as missing, which
    CLAUDE.md prefers over filling it with a value that means something real.
    """
    ordered = frame.sort_values(SORT_COLUMNS).copy()

    feature_names = []
    for column_name in columns:
        for window in windows:
            feature_name = rolling_feature_name(column_name, window)
            ordered[feature_name] = shift_then_roll(
                ordered, group_column, column_name, window, window
            )
            feature_names.append(feature_name)

    logger.info(
        "Built %s rolling features from %s columns over windows %s",
        len(feature_names),
        len(columns),
        windows,
    )

    # Hand back the caller's own row order, so that using the result does not
    # quietly depend on the sort performed above.
    return ordered.reindex(frame.index), feature_names


def add_games_played_this_season(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the count of games the player has already played this season.

    Takes the player-week frame. Returns a copy carrying GAMES_PLAYED_COLUMN,
    aligned to the caller's index.

    cumcount is already zero based, so it counts prior games without needing a
    shift: the season opener is 0. This is games played, not week number. A
    receiver who missed weeks 2 and 3 shows 1 in week 4, not 3, which is the
    honest measure of how much this season has told us about him.
    """
    ordered = frame.sort_values(SORT_COLUMNS).copy()
    ordered[GAMES_PLAYED_COLUMN] = ordered.groupby(["player_id", "season"]).cumcount()
    return ordered.reindex(frame.index)


def add_prior_games_career(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the count of games the player has already played in the data.

    Takes the player-week frame. Returns a copy carrying PRIOR_GAMES_COLUMN,
    aligned to the caller's index.

    This counts across season boundaries, matching the baseline's window. A
    player's last three games in week 1 are the closing games of the previous
    season; those were played before kickoff, so using them is not leakage.
    """
    ordered = frame.sort_values(SORT_COLUMNS).copy()
    ordered[PRIOR_GAMES_COLUMN] = ordered.groupby("player_id").cumcount()
    return ordered.reindex(frame.index)
