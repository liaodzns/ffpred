"""Features describing the upcoming game rather than the player's past.

Everything here is known before kickoff, so unlike the rolling features none of
it needs lagging. The betting line, the venue, the date, and the rest days are
all published days in advance.

The implied team total is the most valuable feature in this module, because the
betting market has already priced in injuries, weather, pace, and matchup
quality. It is a consensus forecast of how many points the player's offense
will score, which is the pool his fantasy points are drawn from.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Feature names this module produces, in the order they are added.
GAME_CONTEXT_FEATURES = [
    "implied_team_total",
    "home_away",
    "rest_days",
    "roof",
    "surface",
]

AGE_FEATURE = "age_at_game"

# Categories are fixed here rather than inferred from whatever happens to be in
# the data, so the encoding cannot shift between training and prediction. A
# category read off a single week of prediction rows would number the levels
# differently and silently feed the model nonsense.
HOME_AWAY_CATEGORIES = ["home", "away", "neutral"]
ROOF_CATEGORIES = ["outdoors", "dome", "closed", "open"]
SURFACE_CATEGORIES = [
    "grass",
    "fieldturf",
    "matrixturf",
    "sportturf",
    "astroturf",
    "a_turf",
]

# Value the schedules table uses for a game played at a neutral site, normally
# an international game. Neither club gets a home crowd in one.
NEUTRAL_LOCATION = "Neutral"

# Days in a year, including the leap year quarter, for converting an age in
# days into an age in years.
DAYS_PER_YEAR = 365.25


def _as_category(values: pd.Series, categories: list[str]) -> pd.Series:
    """Convert a column to a categorical with a fixed set of levels.

    Takes the raw values and the allowed categories. Returns a categorical
    Series where any value outside the list becomes NaN.

    LightGBM reads a pandas categorical natively and splits on the levels
    rather than on their numbering, which is what makes an unordered field like
    playing surface usable at all.
    """
    return pd.Series(
        pd.Categorical(values, categories=categories), index=values.index, name=values.name
    )


def implied_team_total(frame: pd.DataFrame) -> pd.Series:
    """Compute the points the betting market expects the player's team to score.

    Takes the player-week frame. Returns a Series of implied team totals, NaN
    wherever the game has no betting line.

    The market quotes two numbers: a total for both teams combined, and a
    spread. Half the total is each team's share if the game were even, and half
    the spread moves that share from one team to the other. nflverse states the
    spread from the home team's side, so a positive spread means the home team
    is favoured and takes the larger half.
    """
    half_total = frame["total_line"] / 2.0
    half_spread = frame["spread_line"] / 2.0

    is_home_team = frame["team"] == frame["home_team"]
    totals = half_total - half_spread
    totals[is_home_team] = (half_total + half_spread)[is_home_team]
    return totals


def home_away(frame: pd.DataFrame) -> pd.Series:
    """Label each row as a home, away, or neutral site game.

    Takes the player-week frame. Returns a Series of labels.

    Neutral site games get their own label rather than being forced into home
    or away. nflverse still designates one club as the home team for an
    international game, but that club has no crowd and no travel advantage, so
    calling it a home game would teach the model the wrong thing about it.
    """
    labels = pd.Series("away", index=frame.index, name="home_away")
    labels[frame["team"] == frame["home_team"]] = "home"
    labels[frame["location"] == NEUTRAL_LOCATION] = "neutral"
    return labels


def rest_days(frame: pd.DataFrame) -> pd.Series:
    """Read the days of rest since the team's previous game.

    Takes the player-week frame. Returns a Series of rest days.

    The schedule stores rest for both clubs, so the player's own side has to be
    picked out. A short week after a Monday night game and the extra week after
    a bye are both real effects on production.
    """
    days = frame["away_rest"].copy()
    days.name = "rest_days"

    is_home_team = frame["team"] == frame["home_team"]
    days[is_home_team] = frame["home_rest"][is_home_team]
    return days


def _log_missing_lines(frame: pd.DataFrame, totals: pd.Series) -> None:
    """Report how many rows have no betting line, by season.

    Takes the player-week frame and the computed implied totals. Returns
    nothing.

    Missing lines are left as NaN rather than filled. A zero implied total is a
    real number meaning the market expects a shutout, and feeding that in would
    be worse than admitting the line is unknown. Today every game in the data
    is lined, but a game that has not been posted yet is a genuine case at
    prediction time, so the count is reported every run.
    """
    missing = totals.isna()
    if not missing.any():
        logger.info("Betting lines: every row has a spread and a total")
        return

    counts = frame[missing].groupby("season").size()
    logger.warning(
        "Betting lines missing on %s of %s rows, left as NaN; by season: %s",
        int(missing.sum()),
        len(frame),
        counts.to_dict(),
    )


def add_game_context_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the game context features to the player-week frame.

    Takes the player-week frame. Returns a copy carrying the new columns and
    the ordered list of names that were added.
    """
    built = frame.copy()

    built["implied_team_total"] = implied_team_total(built)
    _log_missing_lines(built, built["implied_team_total"])

    built["home_away"] = _as_category(home_away(built), HOME_AWAY_CATEGORIES)
    built["rest_days"] = rest_days(built)

    # The surface column carries empty strings where nflverse has no value.
    # Left alone they would become a category of their own, meaning "known to
    # be blank", so they are turned into the missing value they actually are.
    #
    # It is also stripped first. The 2021 schedule spells 93 games "grass "
    # with a trailing space, which the fixed category list would reject and
    # silently turn into a missing value, losing a surface that is in fact
    # recorded. Any surrounding whitespace is a transcription artefact, never
    # part of the surface name.
    surface_values = built["surface"].str.strip().replace("", None)
    built["roof"] = _as_category(built["roof"], ROOF_CATEGORIES)
    built["surface"] = _as_category(surface_values, SURFACE_CATEGORIES)

    return built, list(GAME_CONTEXT_FEATURES)


def add_age_at_game(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the player's age in years on the day of the game.

    Takes the player-week frame. Returns a copy carrying the age column and the
    list holding its name.

    Age is computed against the game date rather than against a fixed point in
    the season, because a receiver's decline shows up over months, not years,
    and the date is known before kickoff. This lives alongside the game context
    features because the game date is what it is measured from.
    """
    built = frame.copy()

    game_dates = pd.to_datetime(built["gameday"])
    birth_dates = pd.to_datetime(built["birth_date"])
    built[AGE_FEATURE] = (game_dates - birth_dates).dt.days / DAYS_PER_YEAR

    missing_count = int(built[AGE_FEATURE].isna().sum())
    if missing_count > 0:
        logger.info(
            "Age: %s of %s rows have no birth date and are left as NaN",
            missing_count,
            len(built),
        )

    return built, [AGE_FEATURE]
