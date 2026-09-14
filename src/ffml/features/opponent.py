"""How generous the upcoming defense has been to the player's position.

Some defenses give up far more to receivers than to running backs, and that
tendency persists across a season. This module rates each defense by the
fantasy points it has already allowed to each position, and attaches the rating
of the defense a player is about to face.

This is the highest leakage risk in the project. The rating must be built from
completed weeks strictly earlier than the week being predicted, and must never
include the game being predicted. A rating that includes the current week would
know how many points the defense allowed today, which is most of the answer.
tests/test_no_leakage.py exists to catch exactly that.
"""

import logging
from typing import Any

import pandas as pd

from ffml.features.rolling import shift_then_expand

logger = logging.getLogger(__name__)

# The defense a player faces. In the player-week table this is the opposing
# team's code, so the same column names the defense being rated.
DEFENSE_COLUMN = "opponent_team"

# Feature this module produces.
OPPONENT_STRENGTH_FEATURE = "opponent_points_allowed_to_position"

# Column holding the points a defense allowed in one week.
POINTS_ALLOWED_COLUMN = "points_allowed"

# Keys identifying one defense's performance against one position in one week.
DEFENSE_WEEK_KEYS = ["season", "week", DEFENSE_COLUMN, "position"]

# A defense is rated within a single season. One completed week is already real
# information about it, so a single prior week is enough to produce a rating.
MIN_PRIOR_WEEKS = 1


def defense_points_allowed(player_weeks: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Total the fantasy points each defense allowed to each position each week.

    Takes the player-week table and the name of the league target column.
    Returns one row per season, week, defense, and position.

    The league's own scoring rules are used, not a generic fantasy points
    column, because a defense that concedes receptions rather than touchdowns
    looks different under full PPR than under standard scoring, and it is this
    league's view of the matchup that matters.
    """
    weekly_totals = player_weeks.groupby(DEFENSE_WEEK_KEYS, observed=True)[target_column].sum()
    allowed = weekly_totals.reset_index()
    allowed = allowed.rename(columns={target_column: POINTS_ALLOWED_COLUMN})

    logger.info(
        "Built defense-week ratings input: %s rows across %s defenses",
        len(allowed),
        allowed[DEFENSE_COLUMN].nunique(),
    )
    return allowed


def rate_defenses(allowed: pd.DataFrame, lookback_weeks: int | None) -> pd.DataFrame:
    """Rate each defense from the weeks it has already played this season.

    Takes the defense-week table and how many prior weeks to average, where
    None means every completed week of the season. Returns the table with the
    rating column added.

    Two football specific choices are worth stating. The rating resets every
    season, unlike the player rolling features, because defensive personnel and
    scheme turn over between years and last December's rating describes a unit
    that no longer exists. And a single prior week is enough to rate a defense,
    where a player needs three games, because a defense plays every week and
    week two would otherwise be blank for every player in the league.

    The shift happens before the average, so the week being predicted is never
    inside its own rating.
    """
    ordered = allowed.sort_values([DEFENSE_COLUMN, "position", "season", "week"]).copy()
    group_columns = [DEFENSE_COLUMN, "position", "season"]

    if lookback_weeks is None:
        ratings = shift_then_expand(
            ordered, group_columns, POINTS_ALLOWED_COLUMN, MIN_PRIOR_WEEKS
        )
    else:
        groupers = []
        for column_name in group_columns:
            groupers.append(ordered[column_name])
        shifted = ordered.groupby(groupers)[POINTS_ALLOWED_COLUMN].shift(1)
        rolled = shifted.groupby(groupers).rolling(
            lookback_weeks, min_periods=MIN_PRIOR_WEEKS
        ).mean()
        ratings = rolled.reset_index(level=list(range(len(group_columns))), drop=True)
        ratings = ratings.reindex(ordered.index)

    ordered[OPPONENT_STRENGTH_FEATURE] = ratings
    return ordered


def add_opponent_strength(
    frame: pd.DataFrame, player_weeks: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Attach the rating of the defense each player is about to face.

    Takes the frame being built, the full player-week table the ratings are
    derived from, and the parsed config. Returns a copy of the frame carrying
    the rating column and the list holding its name. Raises ValueError if the
    join changes the row count.

    Ratings come from the full player-week table rather than from the frame
    being built, because the frame has already dropped players with thin
    history and a defense is rated by everything it allowed, not only by the
    players who survived that filter.
    """
    target_column = config["scoring"]["target_column"]
    lookback_weeks = config["features"]["opponent_lookback_weeks"]

    allowed = defense_points_allowed(player_weeks, target_column)
    rated = rate_defenses(allowed, lookback_weeks)
    ratings = rated[DEFENSE_WEEK_KEYS + [OPPONENT_STRENGTH_FEATURE]]

    row_count_before = len(frame)
    built = frame.merge(ratings, on=DEFENSE_WEEK_KEYS, how="left")
    if len(built) != row_count_before:
        raise ValueError(
            f"Joining opponent strength changed the row count from {row_count_before} to "
            f"{len(built)}. The defense-week key is not unique."
        )

    # Week one is unavoidably unrated: no team has played anyone yet.
    missing_count = int(built[OPPONENT_STRENGTH_FEATURE].isna().sum())
    logger.info(
        "Opponent strength: %s of %s rows unrated (%s lookback weeks), mostly week one",
        missing_count,
        len(built),
        lookback_weeks,
    )

    return built, [OPPONENT_STRENGTH_FEATURE]
