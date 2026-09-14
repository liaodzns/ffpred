"""Tests for the league fantasy point scoring function.

The most important test here reproduces the prebuilt nflverse
fantasy_points_ppr column from raw box score stats using standard full PPR
settings. If that matches, the scoring function is correct, and the league
values in config.yaml are then correct by construction.
"""

import logging

import pandas as pd
import pytest

from ffml.data.clean import score_player_weeks
from ffml.utils.io import resolve_path

# Fantasy points are sums of quarter point and tenth point terms, so a small
# tolerance covers floating point error without hiding a real scoring mistake.
TOLERANCE = 0.01


def standard_ppr_config() -> dict:
    """Build a scoring config using standard full PPR values.

    Takes nothing. Returns a scoring config dictionary shaped like the scoring
    block of config.yaml.

    These are the settings the nflverse fantasy_points_ppr column is built
    from, including six points for a kick or punt return touchdown. There are
    deliberately no yardage bonuses, because standard PPR has none.
    """
    return {
        "target_column": "fantasy_points_standard_ppr",
        "passing": {
            "yards_per_point": 25,
            "touchdown": 4,
            "interception": -2,
            "two_point_conversion": 2,
        },
        "rushing": {
            "yards_per_point": 10,
            "touchdown": 6,
            "two_point_conversion": 2,
        },
        "receiving": {
            "reception": 1,
            "yards_per_point": 10,
            "touchdown": 6,
            "two_point_conversion": 2,
        },
        "misc": {
            "fumble_lost": -2,
            "special_teams_touchdown": 6,
        },
    }


def load_raw_player_stats() -> pd.DataFrame:
    """Read the downloaded player stats table, skipping the test if absent.

    Takes nothing. Returns the raw player stats frame.
    """
    stats_path = resolve_path("data/raw/player_stats.parquet")
    if not stats_path.is_file():
        pytest.skip("data/raw/player_stats.parquet is missing; run scripts/pull_data.py first")
    return pd.read_parquet(stats_path)


def describe_mismatches(stats: pd.DataFrame, scored: pd.Series, difference: pd.Series) -> str:
    """Build a readable report of rows where the scoring did not match.

    Takes the stats frame, the computed points, and the absolute difference
    against the nflverse column. Returns a message naming the mismatch count
    and showing five example rows with the inputs that feed the scoring rules.
    """
    mismatched = stats[difference > TOLERANCE]
    report_columns = [
        "player_display_name",
        "position",
        "season",
        "week",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "rushing_yards",
        "rushing_tds",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "special_teams_tds",
        "fumble_recovery_tds",
    ]
    present_columns = []
    for column_name in report_columns:
        if column_name in mismatched.columns:
            present_columns.append(column_name)

    examples = mismatched[present_columns].head(5).copy()
    examples["computed"] = scored[mismatched.index].head(5)
    examples["nflverse"] = mismatched["fantasy_points_ppr"].head(5)
    examples["difference"] = examples["nflverse"] - examples["computed"]

    return (
        f"{len(mismatched)} of {len(stats)} rows did not match fantasy_points_ppr "
        f"within {TOLERANCE}.\n\nFive examples:\n{examples.to_string(index=False)}"
    )


def test_standard_ppr_reproduces_nflverse_column() -> None:
    """Standard PPR scoring must reproduce the prebuilt fantasy_points_ppr column.

    This is the validation that proves the scoring function correct. It runs
    against every downloaded row, with no positional or row level exclusions.
    """
    stats = load_raw_player_stats()
    scored = score_player_weeks(stats, standard_ppr_config())
    difference = (scored - stats["fantasy_points_ppr"]).abs()

    if (difference > TOLERANCE).any():
        pytest.fail(describe_mismatches(stats, scored, difference))


def test_scored_series_is_named_from_the_config() -> None:
    """The returned Series must carry the name given by scoring.target_column."""
    stats = load_raw_player_stats()
    scoring_config = standard_ppr_config()
    scored = score_player_weeks(stats.head(100), scoring_config)
    assert scored.name == scoring_config["target_column"]


def synthetic_stats(**column_values: list) -> pd.DataFrame:
    """Build a small player stats frame for the unit tests.

    Takes column names mapped to lists of values. Returns a frame containing
    every column the scoring rules look for, defaulting to zero.
    """
    all_columns = [
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "passing_2pt_conversions",
        "rushing_yards",
        "rushing_tds",
        "rushing_2pt_conversions",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "receiving_2pt_conversions",
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "fumble_recovery_tds",
        "special_teams_tds",
    ]
    row_count = 1
    for column_name in column_values:
        row_count = len(column_values[column_name])

    frame_data = {}
    for column_name in all_columns:
        if column_name in column_values:
            frame_data[column_name] = column_values[column_name]
        else:
            frame_data[column_name] = [0] * row_count
    return pd.DataFrame(frame_data)


def league_config_with_bonus(repeating: bool) -> dict:
    """Build a scoring config carrying only a rushing yardage rule and bonus.

    Takes whether the bonus repeats at every multiple of the threshold.
    Returns a scoring config dictionary.
    """
    return {
        "target_column": "points",
        "rushing": {
            "yards_per_point": 10,
            "bonus": {"threshold_yards": 100, "points": 3, "repeating": repeating},
        },
    }


def test_non_repeating_bonus_is_awarded_once() -> None:
    """A non repeating bonus pays once at 150 yards and once again at 250."""
    stats = synthetic_stats(rushing_yards=[99, 100, 150, 250])
    scored = score_player_weeks(stats, league_config_with_bonus(repeating=False))
    # 9.9 + 0 bonus, 10 + 3, 15 + 3, 25 + 3
    assert list(scored.round(2)) == [9.9, 13.0, 18.0, 28.0]


def test_repeating_bonus_is_awarded_per_multiple() -> None:
    """A repeating bonus pays twice at 250 yards and once at 150."""
    stats = synthetic_stats(rushing_yards=[99, 150, 250])
    scored = score_player_weeks(stats, league_config_with_bonus(repeating=True))
    # 9.9 + 0 bonus, 15 + 3, 25 + 6
    assert list(scored.round(2)) == [9.9, 18.0, 31.0]


def test_fumbles_lost_sums_all_three_columns() -> None:
    """Fumbles lost is the sum of the sack, rushing, and receiving columns."""
    stats = synthetic_stats(
        sack_fumbles_lost=[1, 0, 1],
        rushing_fumbles_lost=[0, 1, 1],
        receiving_fumbles_lost=[0, 0, 1],
    )
    scored = score_player_weeks(stats, {"target_column": "points", "misc": {"fumble_lost": -2}})
    assert list(scored) == [-2.0, -2.0, -6.0]


def test_yards_per_point_divides_rather_than_multiplies() -> None:
    """A 300 yard passing game is 15 points at 20 yards per point, not 12."""
    stats = synthetic_stats(passing_yards=[300])
    scoring_config = {"target_column": "points", "passing": {"yards_per_point": 20}}
    assert list(score_player_weeks(stats, scoring_config)) == [15.0]


def test_missing_column_is_warned_and_skipped(caplog: pytest.LogCaptureFixture) -> None:
    """A rule whose column is absent must warn by name and score zero for it."""
    stats = pd.DataFrame({"rushing_yards": [100]})
    scoring_config = {
        "target_column": "points",
        "rushing": {"yards_per_point": 10},
        "misc": {"fumble_return_touchdown": 6},
    }
    with caplog.at_level(logging.WARNING):
        scored = score_player_weeks(stats, scoring_config)

    assert list(scored) == [10.0]
    assert "misc.fumble_return_touchdown" in caplog.text
    assert "fumble_recovery_tds" in caplog.text
