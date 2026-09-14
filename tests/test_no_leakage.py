"""The leakage guard: predictions may never see the game they are predicting.

This is the most important test in the project. A rolling average that
includes the current week makes validation scores look excellent and makes
real projections worthless, and it fails silently, so the property is checked
directly here on small synthetic frames with deliberately distinctive values.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from ffml.data.clean import kickoff_times, select_pre_kickoff_reports
from ffml.features.build_dataset import build_feature_matrix, feature_column_names
from ffml.features.opponent import OPPONENT_STRENGTH_FEATURE, add_opponent_strength
from ffml.features.rolling import GAMES_PLAYED_COLUMN
from ffml.models.baseline import predict_baseline
from ffml.models.predict import build_upcoming_rows, project_bound, upcoming_feature_rows

TARGET = "fantasy_points_league"
WINDOW = 3
MIN_PRIOR_GAMES = 3

# Each position's rolling source columns, as the synthetic config lists them.
# They mirror the real config so every position's builder is exercised.
SOURCE_COLUMNS_BY_POSITION = {
    "QB": {
        "rolling_production": ["attempts", "passing_epa", "rushing_yards"],
        "usage_share": ["offense_pct"],
    },
    "RB": {
        "rolling_production": ["carries", "rushing_yards", "targets"],
        "usage_share": ["offense_pct", "target_share"],
    },
    "WR": {
        "rolling_production": ["targets", "receptions", "receiving_yards", "receiving_tds"],
        "usage_share": ["offense_pct", "target_share"],
    },
    "TE": {
        "rolling_production": ["targets", "receiving_yards"],
        "usage_share": ["offense_pct", "target_share"],
    },
}

# Every source column any position reads, so one synthetic frame serves them all.
FEATURE_SOURCE_COLUMNS = [
    "attempts",
    "passing_epa",
    "rushing_yards",
    "carries",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "offense_pct",
    "target_share",
]

# Identifier columns the feature builder carries through, other than position.
IDENTIFIER_COLUMNS = [
    "player_display_name",
    "team",
    "opponent_team",
    "game_id",
]


def synthetic_player_weeks() -> pd.DataFrame:
    """Build a small player-week frame with hand-chosen, distinctive scores.

    Takes nothing. Returns a frame covering three deliberately different cases.

    The scores climb by ten each game so that any contamination shows up as an
    obviously wrong number rather than a plausible one. The three players are:

    - clean_player, who plays weeks 1 to 5 with no gaps.
    - gap_player, who plays weeks 1, 2, 3 and then 6, missing two weeks. His
      week 6 projection proves the window counts games rather than weeks.
    - rookie, who has played only twice and so has too little history.
    """
    rows = [
        ("clean_player", 2024, 1, 10.0),
        ("clean_player", 2024, 2, 20.0),
        ("clean_player", 2024, 3, 30.0),
        ("clean_player", 2024, 4, 40.0),
        ("clean_player", 2024, 5, 50.0),
        ("gap_player", 2024, 1, 10.0),
        ("gap_player", 2024, 2, 20.0),
        ("gap_player", 2024, 3, 30.0),
        ("gap_player", 2024, 6, 999.0),
        ("rookie", 2024, 1, 5.0),
        ("rookie", 2024, 2, 15.0),
    ]

    player_ids = []
    seasons = []
    weeks = []
    points = []
    for player_id, season, week, scored_points in rows:
        player_ids.append(player_id)
        seasons.append(season)
        weeks.append(week)
        points.append(scored_points)

    return pd.DataFrame(
        {"player_id": player_ids, "season": seasons, "week": weeks, TARGET: points}
    )


def baseline_for(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach baseline projections to a copy of a player-week frame.

    Takes the frame. Returns a copy carrying a projection column.
    """
    scored = frame.copy()
    scored["projection"] = predict_baseline(frame, WINDOW, MIN_PRIOR_GAMES, TARGET)
    return scored


def projection_for(scored: pd.DataFrame, player_id: str, week: int) -> float:
    """Read one player's projection for one week.

    Takes the scored frame, the player id, and the week. Returns the value.
    """
    match = scored[(scored["player_id"] == player_id) & (scored["week"] == week)]
    return float(match["projection"].iloc[0])


def test_projection_uses_only_earlier_games() -> None:
    """Week N's projection must be the mean of the three games before N."""
    scored = baseline_for(synthetic_player_weeks())

    # Weeks 1 to 3 have too little history to project.
    assert pd.isna(projection_for(scored, "clean_player", 3))

    # Mean of weeks 1, 2, 3 is 20. Had week 4 leaked into its own window the
    # answer would be 30, so the two cases cannot be confused.
    assert projection_for(scored, "clean_player", 4) == 20.0

    # Mean of weeks 2, 3, 4 is 30. Leakage here would give 40.
    assert projection_for(scored, "clean_player", 5) == 30.0


def test_changing_a_week_cannot_change_its_own_projection() -> None:
    """Mutating one week's score must leave that week's projection untouched.

    This checks the leakage property directly rather than inferring it from
    arithmetic. Only games after the mutated week may react to it.
    """
    original = synthetic_player_weeks()
    original_scored = baseline_for(original)

    mutated = original.copy()
    is_target_row = (mutated["player_id"] == "clean_player") & (mutated["week"] == 4)
    mutated.loc[is_target_row, TARGET] = -1000.0
    mutated_scored = baseline_for(mutated)

    # Week 4's own projection, and every earlier week, must be unchanged.
    assert projection_for(mutated_scored, "clean_player", 4) == projection_for(
        original_scored, "clean_player", 4
    )

    # Week 5 sits after the mutated game, so it must react to it.
    assert projection_for(mutated_scored, "clean_player", 5) != projection_for(
        original_scored, "clean_player", 5
    )


def test_window_counts_games_not_calendar_weeks() -> None:
    """A player who missed weeks 4 and 5 averages his last three games played."""
    scored = baseline_for(synthetic_player_weeks())

    # gap_player played weeks 1, 2, 3, then 6. His week 6 projection is the
    # mean of 10, 20, 30. A three calendar week lookback would have found no
    # games at all and produced NaN.
    assert projection_for(scored, "gap_player", 6) == 20.0


def test_too_little_history_gives_nan() -> None:
    """A player below min_prior_games gets NaN, never a partial average."""
    scored = baseline_for(synthetic_player_weeks())

    rookie_rows = scored[scored["player_id"] == "rookie"]
    assert rookie_rows["projection"].isna().all()

    # A partial average of his two games would have been 10.0, so returning NaN
    # is a visibly different outcome rather than a plausible looking one.
    assert not (rookie_rows["projection"] == 10.0).any()


def test_row_order_does_not_change_projections() -> None:
    """Shuffling the input must not change any projection.

    The function sorts by player, season, and week itself rather than trusting
    the order it is handed, which CLAUDE.md requires.
    """
    frame = synthetic_player_weeks()
    in_order = baseline_for(frame)

    shuffled = frame.sample(frac=1.0, random_state=0)
    shuffled_scored = baseline_for(shuffled)

    # Compare on a stable key rather than on row position.
    left = in_order.set_index(["player_id", "week"])["projection"].sort_index()
    right = shuffled_scored.set_index(["player_id", "week"])["projection"].sort_index()
    pd.testing.assert_series_equal(left, right)


def test_projection_is_aligned_to_the_caller_index() -> None:
    """The returned Series must line up with the frame that was passed in.

    A misaligned Series would attach one player's history to another player's
    row, which is leakage of the worst kind.
    """
    frame = synthetic_player_weeks().sample(frac=1.0, random_state=7)
    projections = predict_baseline(frame, WINDOW, MIN_PRIOR_GAMES, TARGET)

    assert list(projections.index) == list(frame.index)

    frame = frame.copy()
    frame["projection"] = projections
    clean_week_4 = frame[(frame["player_id"] == "clean_player") & (frame["week"] == 4)]
    assert float(clean_week_4["projection"].iloc[0]) == 20.0


# ---------------------------------------------------------------------------
# The same guarantee, for every position's feature matrix
#
# Stage seven gives each position its own source columns, so the mutation test
# is run once per position, on a column that position actually rolls.
# ---------------------------------------------------------------------------


def feature_config() -> dict:
    """Build the smallest config the per-position feature builder will accept.

    Takes nothing. Returns a config carrying only the keys it reads.
    """
    return {
        "scoring": {"target_column": TARGET},
        # The synthetic rows are all in 2024 with no warmup season, so the
        # warmup filter keeps every row. tests/test_walk_forward.py covers the
        # warmup guard.
        "data": {"start_season": 2024, "warmup_start_season": None},
        "features": {
            "rolling_windows": [WINDOW],
            "min_prior_games": MIN_PRIOR_GAMES,
            "groups": {
                "default": {"rolling_production": True, "usage_share": True},
                "by_position": {},
            },
            "source_columns": SOURCE_COLUMNS_BY_POSITION,
        },
    }


def synthetic_feature_input(position: str) -> pd.DataFrame:
    """Extend the leakage frame with every column the feature builder needs.

    Takes the position every synthetic player belongs to. Returns a player-week
    frame ready for build_feature_matrix.

    Each source column is set to a tenth of the target, so a hand computed
    check on one column carries over to all of them.
    """
    frame = synthetic_player_weeks()

    for column_name in FEATURE_SOURCE_COLUMNS:
        frame[column_name] = frame[TARGET] / 10.0

    frame["position"] = position
    for column_name in IDENTIFIER_COLUMNS:
        frame[column_name] = "x"

    return frame


@pytest.mark.parametrize(
    "position, column_name",
    [("QB", "attempts"), ("RB", "carries"), ("WR", "targets"), ("TE", "receiving_yards")],
)
def test_a_feature_cannot_see_the_game_it_describes(position: str, column_name: str) -> None:
    """Mutating one week's source value must not move that week's own rolling feature.

    The stage three mutation test, applied to each position's matrix on a column
    that position rolls. Only games after the mutated week may react to it.
    """
    feature_name = f"{column_name}_roll{WINDOW}"
    original = synthetic_feature_input(position)
    original_matrix, _ = build_feature_matrix(original, feature_config(), position)

    mutated = original.copy()
    is_target_row = (mutated["player_id"] == "clean_player") & (mutated["week"] == 4)
    mutated.loc[is_target_row, column_name] = -1000.0
    mutated_matrix, _ = build_feature_matrix(mutated, feature_config(), position)

    original_matrix = original_matrix.set_index(["player_id", "week"])
    mutated_matrix = mutated_matrix.set_index(["player_id", "week"])

    # Week 4 averages weeks 1, 2, 3, whose values are 1, 2 and 3.
    assert original_matrix.loc[("clean_player", 4), feature_name] == 2.0
    assert (
        mutated_matrix.loc[("clean_player", 4), feature_name]
        == original_matrix.loc[("clean_player", 4), feature_name]
    )

    # Week 5 sits after the mutated game, so it must react to it.
    assert (
        mutated_matrix.loc[("clean_player", 5), feature_name]
        != original_matrix.loc[("clean_player", 5), feature_name]
    )


def test_games_played_this_season_excludes_the_current_game() -> None:
    """The count is of games already played, not of games played through today."""
    matrix, _ = build_feature_matrix(synthetic_feature_input("WR"), feature_config(), "WR")
    matrix = matrix.set_index(["player_id", "week"])

    # clean_player's week 4 is his fourth game, so three are behind him.
    assert matrix.loc[("clean_player", 4), GAMES_PLAYED_COLUMN] == 3
    assert matrix.loc[("clean_player", 5), GAMES_PLAYED_COLUMN] == 4

    # gap_player missed weeks 4 and 5, so his week 6 is only his fourth game.
    assert matrix.loc[("gap_player", 6), GAMES_PLAYED_COLUMN] == 3


def test_no_current_week_statistic_is_offered_as_a_feature() -> None:
    """For every position, each feature must be a lagged rolling mean or the games count.

    A raw box score column reaching this list would be pure leakage, and it is
    the easiest mistake to make when adding a column for a new position.
    """
    for position in SOURCE_COLUMNS_BY_POSITION:
        feature_names = feature_column_names(feature_config(), position)

        for feature_name in feature_names:
            if feature_name == GAMES_PLAYED_COLUMN:
                continue
            assert feature_name.endswith(f"_roll{WINDOW}"), (position, feature_name)

        for column_name in FEATURE_SOURCE_COLUMNS:
            assert column_name not in feature_names
        assert TARGET not in feature_names


def test_a_player_split_across_positions_is_refused() -> None:
    """A player under two position labels would have his history cut in half.

    Each position's rolling features only see that position's rows, so such a
    player's windows would silently skip every game logged under the other
    label. The builder must refuse rather than build those truncated windows.
    """
    frame = synthetic_feature_input("WR")
    is_clean_week_5 = (frame["player_id"] == "clean_player") & (frame["week"] == 5)
    frame.loc[is_clean_week_5, "position"] = "TE"

    with pytest.raises(ValueError, match="more than one position"):
        build_feature_matrix(frame, feature_config(), "WR")


# ---------------------------------------------------------------------------
# Opponent strength: the highest leakage risk in the project
#
# A defensive rating that includes the week being predicted knows how many
# points that defense gave up today, which is most of the answer. The synthetic
# defense below allows nothing for three weeks and then 100 points, so a
# correct rating and a leaking one are far apart and cannot be confused.
# ---------------------------------------------------------------------------

DEFENSE = "DEF"


def opponent_config(lookback_weeks: int | None = None) -> dict:
    """Build the smallest config the opponent strength builder will accept.

    Takes the lookback in weeks, where None means the whole season so far.
    Returns the config.
    """
    return {
        "scoring": {"target_column": TARGET},
        "features": {"opponent_lookback_weeks": lookback_weeks},
    }


def synthetic_defense_weeks(week_four_points: float = 100.0) -> pd.DataFrame:
    """Build player-weeks against one defense with a sharp change in week four.

    Takes the points to allow in week four. Returns the frame.

    The defense shuts receivers out in weeks 1 to 3 and then collapses. Its
    week 4 rating must be 0.0, from the three completed weeks. A rating that
    included the current week would read 25.0 instead.
    """
    rows = [
        ("wr_a", 2024, 1, 0.0),
        ("wr_a", 2024, 2, 0.0),
        ("wr_a", 2024, 3, 0.0),
        ("wr_a", 2024, 4, week_four_points),
        ("wr_a", 2024, 5, 0.0),
    ]

    player_ids = []
    seasons = []
    weeks = []
    points = []
    for player_id, season, week, scored_points in rows:
        player_ids.append(player_id)
        seasons.append(season)
        weeks.append(week)
        points.append(scored_points)

    return pd.DataFrame(
        {
            "player_id": player_ids,
            "season": seasons,
            "week": weeks,
            "opponent_team": DEFENSE,
            "position": "WR",
            TARGET: points,
        }
    )


def rate(frame: pd.DataFrame, lookback_weeks: int | None = None) -> pd.Series:
    """Attach opponent ratings and return them indexed by week.

    Takes the player-week frame and the lookback. Returns the rating per week.
    """
    rated, _ = add_opponent_strength(frame, frame, opponent_config(lookback_weeks))
    return rated.set_index("week")[OPPONENT_STRENGTH_FEATURE]


def test_a_defense_rating_excludes_the_week_being_predicted() -> None:
    """The rating for week N must average only weeks before N."""
    ratings = rate(synthetic_defense_weeks())

    # Week 1 has nothing behind it, so no defense can be rated.
    assert pd.isna(ratings.loc[1])

    # Weeks 2, 3 and 4 all average shutouts. Week 4 is the one that matters:
    # the defense allowed 100 that week, so a leaking rating reads 25.0.
    assert ratings.loc[2] == 0.0
    assert ratings.loc[3] == 0.0
    assert ratings.loc[4] == 0.0

    # Week 5 finally sees the collapse: mean of 0, 0, 0, 100.
    assert ratings.loc[5] == 25.0


def test_changing_a_defense_week_cannot_change_its_own_rating() -> None:
    """Mutating week four's points allowed must not move week four's rating."""
    original = rate(synthetic_defense_weeks(100.0))
    mutated = rate(synthetic_defense_weeks(-500.0))

    # Week 4's rating, and every earlier week, must be untouched.
    assert mutated.loc[4] == original.loc[4]
    assert mutated.loc[3] == original.loc[3]

    # Week 5 sits after the mutated game, so it must react to it.
    assert mutated.loc[5] != original.loc[5]
    assert mutated.loc[5] == -125.0


def test_opponent_lookback_weeks_limits_the_window() -> None:
    """A lookback of two averages only the two most recent completed weeks."""
    ratings = rate(synthetic_defense_weeks(100.0), lookback_weeks=2)

    # Week 5 sees weeks 3 and 4 only: the mean of 0 and 100.
    assert ratings.loc[5] == 50.0

    # Week 4 still sees only shutouts, so the current week stays excluded no
    # matter which window is configured.
    assert ratings.loc[4] == 0.0


# ---------------------------------------------------------------------------
# The injury report: only versions on record before kickoff
#
# The report is legitimately used as of the current week because it is
# published before the game. A version modified at or after kickoff is not, so
# it must never be the one chosen.
# ---------------------------------------------------------------------------


def synthetic_schedule() -> pd.DataFrame:
    """Build one 1pm Eastern game, AAA hosting BBB.

    Takes nothing. Returns the schedule frame. Kickoff is 17:00 UTC.
    """
    return pd.DataFrame(
        {
            "season": [2024],
            "week": [1],
            "home_team": ["AAA"],
            "away_team": ["BBB"],
            "gameday": ["2024-09-08"],
            "gametime": ["13:00"],
        }
    )


def synthetic_report_versions() -> pd.DataFrame:
    """Build report versions straddling the 17:00 UTC kickoff.

    Takes nothing. Returns the injury table with date_modified in UTC.

    - early_update: Questionable on Friday, then Out on Sunday morning, both
      before kickoff. The Sunday version must win.
    - late_revision: Questionable on Friday, then Out an hour after kickoff.
      The Friday version must win, because the revision came from the game.
    - on_the_whistle: a single version stamped exactly at kickoff. Strictly
      before means it is excluded.
    - no_game: listed by a club that has no game that week, so there is no
      kickoff to compare against. Excluded.
    """
    rows = [
        ("early_update", "AAA", "Questionable", "2024-09-06 20:00"),
        ("early_update", "AAA", "Out", "2024-09-08 15:00"),
        ("late_revision", "BBB", "Questionable", "2024-09-06 20:00"),
        ("late_revision", "BBB", "Out", "2024-09-08 18:00"),
        ("on_the_whistle", "AAA", "Doubtful", "2024-09-08 17:00"),
        ("no_game", "CCC", "Questionable", "2024-09-06 20:00"),
    ]

    player_ids = []
    teams = []
    statuses = []
    modified = []
    for player_id, team, status, modified_utc in rows:
        player_ids.append(player_id)
        teams.append(team)
        statuses.append(status)
        modified.append(modified_utc)

    return pd.DataFrame(
        {
            "player_id": player_ids,
            "season": 2024,
            "week": 1,
            "team": teams,
            "report_status": statuses,
            "date_modified": pd.to_datetime(modified, utc=True),
        }
    )


def test_only_reports_filed_before_kickoff_are_used() -> None:
    """A version modified at or after kickoff may never be the one chosen."""
    chosen = select_pre_kickoff_reports(
        synthetic_report_versions(), kickoff_times(synthetic_schedule())
    )
    status_by_player = chosen.set_index("player_id")["report_status"]

    # The Sunday morning update was on record before kickoff, so it stands.
    assert status_by_player.loc["early_update"] == "Out"

    # The post-kickoff revision is discarded; the Friday report stands instead.
    assert status_by_player.loc["late_revision"] == "Questionable"

    # Exactly at kickoff is not before kickoff.
    assert "on_the_whistle" not in status_by_player.index

    # With no scheduled game, the report's timing cannot be established.
    assert "no_game" not in status_by_player.index


def test_a_report_with_no_timestamp_is_not_used() -> None:
    """A version whose date_modified is missing cannot be shown to predate kickoff.

    nflverse ships some seasons with no modification times at all. Treating a
    missing time as early enough would admit versions that may have been
    revised during or after the game, so they are excluded instead.
    """
    versions = synthetic_report_versions()
    untimed = versions[versions["player_id"] == "late_revision"].copy()
    untimed["date_modified"] = pd.to_datetime([None, None], utc=True)

    chosen = select_pre_kickoff_reports(untimed, kickoff_times(synthetic_schedule()))
    assert len(chosen) == 0


def test_kickoff_times_are_read_as_eastern_time() -> None:
    """A 13:00 gametime is 1pm Eastern, which is 17:00 UTC in September.

    Reading the schedule as UTC would move kickoff four hours earlier and
    wrongly discard every report version filed on Sunday morning.
    """
    versions = synthetic_report_versions()
    just_before = versions[versions["player_id"] == "early_update"].copy()
    just_before["date_modified"] = pd.to_datetime(["2024-09-08 16:59", "2024-09-08 16:59"], utc=True)

    chosen = select_pre_kickoff_reports(just_before, kickoff_times(synthetic_schedule()))
    assert len(chosen) == 1


# ---------------------------------------------------------------------------
# The upcoming-week row: a projection must see nothing of the week it projects
#
# A live projection is built for a game with no box score yet, but the same
# code runs for a completed week, where the answer is sitting in the table.
# These tests put it there on purpose and require that it changes nothing.
# ---------------------------------------------------------------------------

UPCOMING_SEASON = 2024
UPCOMING_WEEK = 5


def upcoming_players() -> pd.DataFrame:
    """Build the one player the upcoming-week tests project.

    Takes nothing. Returns the player frame.
    """
    return pd.DataFrame(
        {
            "player_id": ["clean_player"],
            "player_display_name": ["x"],
            "position": ["WR"],
            "team": ["x"],
        }
    )


def upcoming_sides() -> pd.DataFrame:
    """Build the one game the upcoming-week tests project.

    Takes nothing. Returns the game sides.
    """
    return pd.DataFrame({"team": ["x"], "opponent_team": ["x"], "game_id": ["upcoming"]})


def project_upcoming(frame: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Build clean_player's week 5 upcoming row from a player-week frame.

    Takes the frame. Returns the row's features and the feature names.
    """
    upcoming = build_upcoming_rows(
        frame, upcoming_players(), upcoming_sides(), UPCOMING_SEASON, UPCOMING_WEEK
    )
    rows, feature_names = upcoming_feature_rows(
        frame, upcoming, feature_config(), "WR", UPCOMING_SEASON, UPCOMING_WEEK
    )
    assert len(rows) == 1
    return rows.iloc[0], feature_names


def same_value(left: object, right: object) -> bool:
    """Compare two feature values, treating two missing values as equal.

    Takes the two values. Returns whether they match.
    """
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(left == right)


def test_an_upcoming_row_equals_the_row_its_completed_game_had() -> None:
    """Building week 5 as if unplayed gives exactly the features week 5 had once played.

    If the upcoming path computed anything differently from the historical one,
    the model would be projecting from inputs unlike any it was trained on.
    """
    frame = synthetic_feature_input("WR")
    upcoming_row, feature_names = project_upcoming(frame)

    matrix, _ = build_feature_matrix(frame, feature_config(), "WR")
    is_played = (matrix["player_id"] == "clean_player") & (matrix["week"] == UPCOMING_WEEK)
    played_row = matrix[is_played].iloc[0]

    for feature_name in feature_names:
        assert same_value(upcoming_row[feature_name], played_row[feature_name]), feature_name

    # Targets in weeks 2, 3, 4 are 2, 3, 4. Had week 5 leaked in, the window
    # would be weeks 3, 4, 5 and read 4.0.
    assert upcoming_row["targets_roll3"] == 3.0
    assert upcoming_row[GAMES_PLAYED_COLUMN] == 4


def test_the_projected_week_cannot_reach_its_own_upcoming_row() -> None:
    """Changing, deleting, or adding results at or after week 5 must change nothing.

    This is the property that matters when a projection is run for a week whose
    games are already in the data, as the smoke test is: the real result is
    right there, and it must not reach the row built to predict it.
    """
    frame = synthetic_feature_input("WR")
    reference_row, feature_names = project_upcoming(frame)
    is_projected_week = (frame["player_id"] == "clean_player") & (frame["week"] == UPCOMING_WEEK)

    mutated = frame.copy()
    for column_name in FEATURE_SOURCE_COLUMNS + [TARGET]:
        mutated.loc[is_projected_week, column_name] = -1000.0

    deleted = frame[~is_projected_week]

    future_game = frame[is_projected_week].copy()
    future_game["week"] = UPCOMING_WEEK + 1
    with_future_game = pd.concat([frame, future_game], ignore_index=True)

    for variant in [mutated, deleted, with_future_game]:
        variant_row, _ = project_upcoming(variant)
        for feature_name in feature_names:
            assert same_value(variant_row[feature_name], reference_row[feature_name]), feature_name


# ---------------------------------------------------------------------------
# The quantile path
#
# A floor and a ceiling are read off the same upcoming row as the point
# projection, by a model trained on the same features. The same guarantee must
# hold for them: nothing about the projected week can reach its own bounds.
# ---------------------------------------------------------------------------


def upcoming_rows_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build clean_player's week 5 upcoming rows as a frame a booster can read.

    Takes the player-week frame. Returns the rows and the feature names.
    """
    upcoming = build_upcoming_rows(
        frame, upcoming_players(), upcoming_sides(), UPCOMING_SEASON, UPCOMING_WEEK
    )
    return upcoming_feature_rows(frame, upcoming, feature_config(), "WR", UPCOMING_SEASON, UPCOMING_WEEK)


def tiny_quantile_booster(frame: pd.DataFrame, level: float) -> lgb.Booster:
    """Train a small quantile model on the synthetic feature rows.

    Takes the player-week frame and the quantile level. Returns the booster.

    It is deliberately allowed to split on single rows, so its predictions move
    whenever its inputs do, which is what makes the tests below able to fail.
    """
    matrix, feature_names = build_feature_matrix(frame, feature_config(), "WR")
    parameters = {
        "objective": "quantile",
        "alpha": level,
        "num_leaves": 4,
        "min_data_in_leaf": 1,
        "min_data_in_bin": 1,
        "learning_rate": 0.5,
        "verbosity": -1,
        "seed": 42,
    }
    dataset = lgb.Dataset(matrix[feature_names], label=matrix[TARGET], feature_name=feature_names)
    return lgb.train(parameters, dataset, num_boost_round=20)


def projected_week_variants(frame: pd.DataFrame) -> list[pd.DataFrame]:
    """Build the three ways the projected week's own results can be disturbed.

    Takes the player-week frame. Returns copies with week 5's results changed,
    with week 5 deleted, and with a later game added.
    """
    is_projected_week = (frame["player_id"] == "clean_player") & (frame["week"] == UPCOMING_WEEK)

    mutated = frame.copy()
    for column_name in FEATURE_SOURCE_COLUMNS + [TARGET]:
        mutated.loc[is_projected_week, column_name] = -1000.0

    future_game = frame[is_projected_week].copy()
    future_game["week"] = UPCOMING_WEEK + 1
    return [mutated, frame[~is_projected_week], pd.concat([frame, future_game], ignore_index=True)]


@pytest.mark.parametrize("level", [0.1, 0.9])
def test_the_projected_week_cannot_reach_its_own_floor_or_ceiling(level: float) -> None:
    """Changing, deleting, or adding results at or after week 5 must leave both bounds unchanged."""
    frame = synthetic_feature_input("WR")
    booster = tiny_quantile_booster(frame, level)
    reference_rows, feature_names = upcoming_rows_frame(frame)
    reference_bound = project_bound(booster, reference_rows, feature_names)

    for variant in projected_week_variants(frame):
        variant_rows, _ = upcoming_rows_frame(variant)
        assert np.array_equal(project_bound(booster, variant_rows, feature_names), reference_bound)


def test_the_quantile_bound_does_move_when_its_history_changes() -> None:
    """Guards the test above against passing vacuously on a model that ignores its inputs.

    Changing a game before week 5 changes the upcoming row's features, so it
    must be able to change the bound.
    """
    frame = synthetic_feature_input("WR")
    booster = tiny_quantile_booster(frame, 0.9)
    reference_rows, feature_names = upcoming_rows_frame(frame)
    reference_bound = project_bound(booster, reference_rows, feature_names)

    moved = False
    for new_value in [0.0, 1000.0]:
        changed = frame.copy()
        is_earlier_week = (changed["player_id"] == "clean_player") & (changed["week"] == UPCOMING_WEEK - 1)
        for column_name in FEATURE_SOURCE_COLUMNS + [TARGET]:
            changed.loc[is_earlier_week, column_name] = new_value
        changed_rows, _ = upcoming_rows_frame(changed)
        if not np.array_equal(project_bound(booster, changed_rows, feature_names), reference_bound):
            moved = True

    assert moved
