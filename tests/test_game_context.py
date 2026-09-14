"""Tests for the game context features.

The implied team total is the one worth checking hardest. It is derived from
two market numbers with a sign convention that is easy to get backwards, and
getting it backwards would hand every player his opponent's projected points
without failing anything visibly.
"""

import pandas as pd

from ffml.features.game_context import (
    add_age_at_game,
    add_game_context_features,
    home_away,
    implied_team_total,
    rest_days,
)


def schedule_frame() -> pd.DataFrame:
    """Build four player-weeks across two real games.

    Takes nothing. Returns the frame.

    The first game is CLE at CAR from 2022 week 1, spread +2.5 to the home
    side and a total of 42.5. The second is a neutral site game, so neither
    club is at home in any meaningful sense.
    """
    return pd.DataFrame(
        {
            "team": ["CAR", "CLE", "LA", "BUF"],
            "home_team": ["CAR", "CAR", "LA", "LA"],
            "location": ["Home", "Home", "Neutral", "Neutral"],
            "spread_line": [2.5, 2.5, -1.0, -1.0],
            "total_line": [42.5, 42.5, 51.5, 51.5],
            "home_rest": [7, 7, 10, 10],
            "away_rest": [7, 7, 6, 6],
            "roof": ["outdoors", "outdoors", "dome", "dome"],
            "surface": ["grass", "grass", "", ""],
        }
    )


def test_implied_total_splits_the_market_total_by_the_spread() -> None:
    """The favoured side takes the larger half of the total."""
    totals = implied_team_total(schedule_frame())

    # CAR are home and favoured by 2.5 in a 42.5 point game: 21.25 + 1.25.
    assert totals.iloc[0] == 22.50
    # CLE are the away side of the same game: 21.25 - 1.25.
    assert totals.iloc[1] == 20.00

    # The two halves must always add back to the market total.
    assert totals.iloc[0] + totals.iloc[1] == 42.5


def test_a_negative_spread_favours_the_away_side() -> None:
    """nflverse states the spread from the home side, so negative favours away."""
    totals = implied_team_total(schedule_frame())

    # LA are nominally home but are 1 point underdogs: 25.75 - 0.5.
    assert totals.iloc[2] == 25.25
    # BUF take the larger half: 25.75 + 0.5.
    assert totals.iloc[3] == 26.25
    assert totals.iloc[3] > totals.iloc[2]


def test_a_missing_line_gives_nan_rather_than_zero() -> None:
    """An unlined game must not be scored as an expected shutout."""
    frame = schedule_frame()
    frame.loc[0, "spread_line"] = None
    frame.loc[1, "total_line"] = None

    totals = implied_team_total(frame)

    assert pd.isna(totals.iloc[0])
    assert pd.isna(totals.iloc[1])
    # A zero would be a real number meaning the market expects no points at all,
    # so no row may carry one. NaN compares false here, which is the point.
    assert not (totals == 0.0).any()


def test_a_neutral_site_game_is_neither_home_nor_away() -> None:
    """The nominal home team of an international game gets no home label."""
    labels = home_away(schedule_frame())

    assert labels.tolist() == ["home", "away", "neutral", "neutral"]


def test_rest_days_come_from_the_players_own_side() -> None:
    """Each row reads the rest of its own team, not of the home team."""
    days = rest_days(schedule_frame())

    # LA are home with 10 days rest; BUF are away with 6.
    assert days.tolist() == [7, 7, 10, 6]


def test_an_empty_surface_becomes_missing_not_a_category() -> None:
    """A blank surface is unknown, not a kind of playing surface."""
    built, feature_names = add_game_context_features(schedule_frame())

    assert built["surface"].tolist()[:2] == ["grass", "grass"]
    assert built["surface"].isna().tolist() == [False, False, True, True]

    # The blank must not have become a level of its own.
    assert "" not in list(built["surface"].cat.categories)
    assert "surface" in feature_names


def test_surrounding_whitespace_does_not_lose_a_surface() -> None:
    """A surface spelled with a stray space is still that surface.

    The 2021 schedule writes 93 games as "grass " with a trailing space. The
    fixed category list would reject it and record a missing surface for a game
    whose surface is perfectly well known.
    """
    frame = schedule_frame()
    frame.loc[2, "surface"] = "grass "
    frame.loc[3, "surface"] = " fieldturf"

    built, _ = add_game_context_features(frame)

    assert built["surface"].iloc[2] == "grass"
    assert built["surface"].iloc[3] == "fieldturf"
    assert not built["surface"].iloc[2:4].isna().any()


def test_roof_and_surface_use_fixed_categories() -> None:
    """The encoding must not depend on which values happen to be present."""
    built, _ = add_game_context_features(schedule_frame())

    # Only two roof values appear in the frame, but all four levels survive, so
    # a single week of prediction rows encodes the same way training did.
    assert list(built["roof"].cat.categories) == ["outdoors", "dome", "closed", "open"]
    assert len(built["surface"].cat.categories) == 6


def test_age_is_measured_at_the_game_date() -> None:
    """Age counts from birth date to kickoff, in years."""
    frame = pd.DataFrame(
        {
            "gameday": ["2024-01-01", "2024-01-01"],
            "birth_date": pd.to_datetime(["2000-01-01", "1990-07-02"]),
        }
    )
    built, feature_names = add_age_at_game(frame)

    # 2000-01-01 to 2024-01-01 is 8766 days, which is exactly 24 years once the
    # leap day quarter is accounted for.
    assert round(float(built["age_at_game"].iloc[0]), 6) == 24.0
    assert 33.4 < float(built["age_at_game"].iloc[1]) < 33.6
    assert feature_names == ["age_at_game"]
