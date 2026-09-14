"""Joining the raw nflverse tables into one tidy player-week table.

This module does two things. It computes the league's fantasy point target
from box score columns, because this league's rules do not match any prebuilt
nflverse fantasy point column, and it joins the raw tables into a single row
per player per week saved to data/processed/player_weeks.parquet.

No features are built here. Rolling averages, opponent strength, and game
context derivations belong to the feature modules in stage three.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ffml.utils.io import read_parquet, resolve_path, write_parquet

logger = logging.getLogger(__name__)

# Scoring rules that are simply points multiplied by an event count, written as
# (config section, config key, player_stats column). Keeping the mapping in one
# place means the scoring logic reads every point value from the config and
# hard codes none of them.
COUNTING_RULE_COLUMNS = [
    ("passing", "touchdown", "passing_tds"),
    ("passing", "interception", "passing_interceptions"),
    ("passing", "two_point_conversion", "passing_2pt_conversions"),
    ("rushing", "touchdown", "rushing_tds"),
    ("rushing", "two_point_conversion", "rushing_2pt_conversions"),
    ("receiving", "reception", "receptions"),
    ("receiving", "touchdown", "receiving_tds"),
    ("receiving", "two_point_conversion", "receiving_2pt_conversions"),
    ("misc", "fumble_return_touchdown", "fumble_recovery_tds"),
    ("misc", "special_teams_touchdown", "special_teams_tds"),
]

# Yardage rules, written as (config section, player_stats column). These are
# stored in the config as "yards per point", so yards are divided by the
# setting rather than multiplied by it.
YARDAGE_RULE_COLUMNS = [
    ("passing", "passing_yards"),
    ("rushing", "rushing_yards"),
    ("receiving", "receiving_yards"),
]

# Sections that can carry a 100 yard bonus.
BONUS_RULE_COLUMNS = [
    ("rushing", "rushing_yards"),
    ("receiving", "receiving_yards"),
]

# Fumbles lost is not a single column in the nflverse schema. A player can lose
# a fumble while being sacked, while rushing, or after making a catch, and each
# is counted separately, so the three have to be summed.
FUMBLE_LOST_COLUMNS = [
    "sack_fumbles_lost",
    "rushing_fumbles_lost",
    "receiving_fumbles_lost",
]


def score_player_weeks(
    player_stats_df: pd.DataFrame, scoring_config: dict[str, Any]
) -> pd.Series:
    """Compute league fantasy points for every row of a player stats frame.

    Takes the player stats frame and the scoring block of the config. Returns
    a Series of fantasy points, named after scoring.target_column and indexed
    like the input frame.

    Every point value is read from the config. A rule that has no matching
    column in the data is logged and skipped rather than guessed at.
    """
    total_points = pd.Series(0.0, index=player_stats_df.index)
    total_points = total_points + _score_counting_rules(player_stats_df, scoring_config)
    total_points = total_points + _score_yardage_rules(player_stats_df, scoring_config)
    total_points = total_points + _score_yardage_bonuses(player_stats_df, scoring_config)
    total_points = total_points + _score_fumbles_lost(player_stats_df, scoring_config)

    total_points.name = scoring_config["target_column"]
    return total_points


def _column_values(frame: pd.DataFrame, column_name: str) -> pd.Series:
    """Read one column as floats with missing values treated as zero.

    Takes the frame and a column name. Returns the column as a float Series.

    A missing box score value means the event did not happen, which is a real
    zero rather than unknown information, so filling with zero is correct here
    even though the project prefers to leave NaN alone elsewhere.
    """
    return frame[column_name].fillna(0).astype(float)


def _score_counting_rules(
    frame: pd.DataFrame, scoring_config: dict[str, Any]
) -> pd.Series:
    """Score every rule that is points multiplied by an event count.

    Takes the player stats frame and the scoring config. Returns a Series of
    points contributed by touchdowns, receptions, interceptions, and two point
    conversions.
    """
    points = pd.Series(0.0, index=frame.index)
    for section_name, rule_key, column_name in COUNTING_RULE_COLUMNS:
        section = scoring_config.get(section_name)
        if not isinstance(section, dict) or rule_key not in section:
            continue
        if not _has_column(frame, section_name, rule_key, column_name):
            continue
        points = points + section[rule_key] * _column_values(frame, column_name)
    return points


def _score_yardage_rules(
    frame: pd.DataFrame, scoring_config: dict[str, Any]
) -> pd.Series:
    """Score passing, rushing, and receiving yards.

    Takes the player stats frame and the scoring config. Returns a Series of
    points from yardage.

    The config stores these as yards per point, matching how the league site
    displays them, so yards are divided by the setting.
    """
    points = pd.Series(0.0, index=frame.index)
    for section_name, column_name in YARDAGE_RULE_COLUMNS:
        section = scoring_config.get(section_name)
        if not isinstance(section, dict) or "yards_per_point" not in section:
            continue
        if not _has_column(frame, section_name, "yards_per_point", column_name):
            continue
        yards_per_point = section["yards_per_point"]
        points = points + _column_values(frame, column_name) / yards_per_point
    return points


def _score_yardage_bonuses(
    frame: pd.DataFrame, scoring_config: dict[str, Any]
) -> pd.Series:
    """Score the 100 yard rushing and receiving bonuses.

    Takes the player stats frame and the scoring config. Returns a Series of
    bonus points.

    When the bonus is marked repeating it is awarded once per completed
    multiple of the threshold, so a 200 yard game earns it twice. Otherwise it
    is awarded once no matter how far past the threshold the player went, which
    is how Yahoo normally handles it.
    """
    points = pd.Series(0.0, index=frame.index)
    for section_name, column_name in BONUS_RULE_COLUMNS:
        section = scoring_config.get(section_name)
        if not isinstance(section, dict):
            continue
        bonus = section.get("bonus")
        if not isinstance(bonus, dict):
            continue
        if not _has_column(frame, section_name, "bonus", column_name):
            continue

        yards = _column_values(frame, column_name)
        threshold_yards = bonus["threshold_yards"]
        if bonus["repeating"]:
            bonus_count = (yards // threshold_yards).astype(float)
        else:
            bonus_count = (yards >= threshold_yards).astype(float)
        points = points + bonus["points"] * bonus_count
    return points


def _score_fumbles_lost(
    frame: pd.DataFrame, scoring_config: dict[str, Any]
) -> pd.Series:
    """Score fumbles lost, which are spread across three columns.

    Takes the player stats frame and the scoring config. Returns a Series of
    points, normally negative.
    """
    points = pd.Series(0.0, index=frame.index)
    misc_section = scoring_config.get("misc")
    if not isinstance(misc_section, dict) or "fumble_lost" not in misc_section:
        return points

    fumbles_lost = pd.Series(0.0, index=frame.index)
    for column_name in FUMBLE_LOST_COLUMNS:
        if not _has_column(frame, "misc", "fumble_lost", column_name):
            continue
        fumbles_lost = fumbles_lost + _column_values(frame, column_name)

    return points + misc_section["fumble_lost"] * fumbles_lost


def _has_column(
    frame: pd.DataFrame, section_name: str, rule_key: str, column_name: str
) -> bool:
    """Check that the column a scoring rule needs is present, warning if not.

    Takes the frame, the config section and key the rule came from, and the
    column it needs. Returns True when the column is present.

    A rule with no matching column is announced rather than dropped quietly,
    and no substitute column is invented, because a silently missing rule would
    make the target subtly wrong in a way that is very hard to notice later.
    """
    if column_name in frame.columns:
        return True
    logger.warning(
        "Scoring rule %s.%s has no matching column '%s' in the data; skipping it.",
        section_name,
        rule_key,
        column_name,
    )
    return False


# ---------------------------------------------------------------------------
# Cleaning and joining
# ---------------------------------------------------------------------------

# Columns carried from each raw table, join keys first. Only named columns are
# taken, because several raw tables repeat a column name that already exists in
# player_stats, such as position or team, and because ff_opportunity alone is
# 159 columns wide.
SCHEDULE_COLUMNS = [
    "game_id",
    "gameday",
    "weekday",
    "gametime",
    "home_team",
    "away_team",
    "location",
    "roof",
    "surface",
    "temp",
    "wind",
    "spread_line",
    "total_line",
    "div_game",
    "home_rest",
    "away_rest",
]

SNAP_COUNT_COLUMNS = [
    "pfr_id",
    "season",
    "week",
    "offense_snaps",
    "offense_pct",
    "st_snaps",
    "st_pct",
]

ROSTER_COLUMNS = [
    "player_id",
    "season",
    "week",
    "status",
    "years_exp",
    "birth_date",
]

INJURY_COLUMNS = [
    "player_id",
    "season",
    "week",
    "team",
    "injury_report_modified",
    "report_status",
    "report_primary_injury",
    "practice_status",
    "practice_primary_injury",
]

FF_OPPORTUNITY_COLUMNS = [
    "player_id",
    "season",
    "week",
    "receptions_exp",
    "pass_yards_gained_exp",
    "rec_yards_gained_exp",
    "rush_yards_gained_exp",
    "pass_touchdown_exp",
    "rec_touchdown_exp",
    "rush_touchdown_exp",
    "pass_fantasy_points_exp",
    "rec_fantasy_points_exp",
    "rush_fantasy_points_exp",
    "total_fantasy_points_exp",
    "total_yards_gained_exp",
    "total_touchdown_exp",
]

# Roster status meaning the player was on the active game day roster.
ACTIVE_ROSTER_STATUS = "ACT"


def assert_no_fanout(row_count_before: int, frame: pd.DataFrame, join_name: str) -> None:
    """Fail if a join added rows to the frame.

    Takes the row count before the join, the joined frame, and a name for the
    join. Returns nothing. Raises ValueError when rows were added.

    A silent fan-out from a join key that is not unique on the right hand side
    is the most likely bug in this stage, and it quietly duplicates player
    weeks rather than erroring, so it is checked explicitly every time.
    """
    if len(frame) > row_count_before:
        raise ValueError(
            f"Join '{join_name}' increased the row count from {row_count_before} to "
            f"{len(frame)}. The right hand key is not unique; deduplicate it before joining."
        )


def _log_step(step_name: str, row_count_before: int, row_count_after: int, reason: str) -> None:
    """Log the row count before and after a filter or join.

    Takes the step name, the two row counts, and why rows were removed.
    Returns nothing.
    """
    removed = row_count_before - row_count_after
    logger.info(
        "%s: %s -> %s rows (%s removed) %s",
        step_name,
        row_count_before,
        row_count_after,
        removed,
        reason,
    )


def _left_join_with_match_count(
    frame: pd.DataFrame, right_frame: pd.DataFrame, join_keys: list[str], join_name: str
) -> pd.DataFrame:
    """Left join a table, check for fan-out, and log how many rows matched.

    Takes the left frame, the right frame, the join keys, and a name for the
    join. Returns the joined frame.

    The match count comes from a merge indicator rather than from checking a
    payload column for nulls, because a matched row can still carry a null
    value. An injury report with no status designation is a real example.
    """
    row_count_before = len(frame)
    frame = frame.merge(right_frame, on=join_keys, how="left", indicator="_join_match")
    assert_no_fanout(row_count_before, frame, join_name)

    matched = int((frame["_join_match"] == "both").sum())
    frame = frame.drop(columns=["_join_match"])
    _log_step(
        "join " + join_name,
        row_count_before,
        len(frame),
        f"on {', '.join(join_keys)}; {matched} of {row_count_before} rows matched",
    )
    return frame


def _filter_season_type(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Keep only the season type named in the config.

    Takes the player stats frame and the parsed config. Returns the filtered
    frame.
    """
    season_type = config["data"]["season_type"]
    row_count_before = len(frame)
    frame = frame[frame["season_type"] == season_type]
    _log_step("filter season_type", row_count_before, len(frame), f"kept {season_type} only")
    return frame


def _filter_positions(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Keep only the positions listed in the config.

    Takes the player stats frame and the parsed config. Returns the filtered
    frame.
    """
    positions = config["data"]["positions"]
    row_count_before = len(frame)
    frame = frame[frame["position"].isin(positions)]
    _log_step("filter positions", row_count_before, len(frame), f"kept {positions}")
    return frame


def _join_schedules(frame: pd.DataFrame, raw_directory: Path) -> pd.DataFrame:
    """Join game level context from the schedules table.

    Takes the player week frame and the raw data directory. Returns the joined
    frame.

    game_id is one row per game in schedules, so this join cannot fan out. The
    betting line, roof, surface, and rest days all describe the upcoming game
    and are known before kickoff.
    """
    schedules = read_parquet(raw_directory / "schedules.parquet")
    return _left_join_with_match_count(frame, schedules[SCHEDULE_COLUMNS], ["game_id"], "schedules")


def _load_gsis_to_pfr_crosswalk(raw_directory: Path) -> pd.DataFrame:
    """Build a player id crosswalk from GSIS ids to Pro Football Reference ids.

    Takes the raw data directory. Returns a frame with player_id and pfr_id.

    Snap counts are the only required table keyed by Pro Football Reference
    ids rather than the GSIS ids everything else uses, so the two id systems
    have to be bridged before snap counts can be joined.
    """
    players = read_parquet(raw_directory / "players.parquet")
    crosswalk = players[["gsis_id", "pfr_id"]].dropna()
    crosswalk = crosswalk.drop_duplicates(subset=["gsis_id"])
    crosswalk = crosswalk.rename(columns={"gsis_id": "player_id"})
    logger.info("Built GSIS to PFR crosswalk: %s players", len(crosswalk))
    return crosswalk


def _join_snap_counts(frame: pd.DataFrame, raw_directory: Path) -> pd.DataFrame:
    """Join offensive and special teams snap counts.

    Takes the player week frame and the raw data directory. Returns the joined
    frame.

    Playing time is the largest single driver of fantasy scoring, so this is
    one of the more valuable joins in the pipeline.
    """
    crosswalk = _load_gsis_to_pfr_crosswalk(raw_directory)
    row_count_before = len(frame)
    frame = frame.merge(crosswalk, on="player_id", how="left")
    assert_no_fanout(row_count_before, frame, "gsis to pfr crosswalk")

    snap_counts = read_parquet(raw_directory / "snap_counts.parquet")
    snap_counts = snap_counts.rename(columns={"pfr_player_id": "pfr_id"})
    return _left_join_with_match_count(
        frame, snap_counts[SNAP_COUNT_COLUMNS], ["pfr_id", "season", "week"], "snap_counts"
    )


def _join_rosters_weekly(frame: pd.DataFrame, raw_directory: Path) -> pd.DataFrame:
    """Join weekly roster status and player attributes.

    Takes the player week frame and the raw data directory. Returns the joined
    frame.

    Rows with no GSIS id are dropped first. They are practice squad and cut
    players who never appear in player_stats, and they are the sole cause of
    the duplicate keys in this table.
    """
    rosters = read_parquet(raw_directory / "rosters_weekly.parquet")
    rosters = rosters[rosters["gsis_id"].notna()]
    rosters = rosters.rename(columns={"gsis_id": "player_id"})

    return _left_join_with_match_count(
        frame, rosters[ROSTER_COLUMNS], ["player_id", "season", "week"], "rosters_weekly"
    )


# Time zone the schedule records kickoffs in. nflverse gives every gameday and
# gametime in US Eastern regardless of venue, which is why a London game shows
# a 09:30 kickoff.
SCHEDULE_TIME_ZONE = "America/New_York"

# Keys identifying one player's entry on one team's report for one week.
INJURY_KEYS = ["player_id", "season", "week", "team"]


def kickoff_times(schedules: pd.DataFrame) -> pd.DataFrame:
    """Build one kickoff time per team per week from the schedule.

    Takes the schedules table. Returns a frame of season, week, team, and the
    kickoff as a naive US Eastern timestamp, NaT where no time is recorded.

    Each game involves two teams, so the schedule is stacked once from the home
    side and once from the away side.
    """
    home_side = schedules[["season", "week", "home_team", "gameday", "gametime"]].rename(
        columns={"home_team": "team"}
    )
    away_side = schedules[["season", "week", "away_team", "gameday", "gametime"]].rename(
        columns={"away_team": "team"}
    )
    both_sides = pd.concat([home_side, away_side], ignore_index=True)
    both_sides["kickoff"] = pd.to_datetime(
        both_sides["gameday"] + " " + both_sides["gametime"], errors="coerce"
    )
    return both_sides[["season", "week", "team", "kickoff"]]


def select_pre_kickoff_reports(injuries: pd.DataFrame, kickoffs: pd.DataFrame) -> pd.DataFrame:
    """Keep each player's latest report version filed before his game kicked off.

    Takes the injury table, with gsis_id already renamed to player_id and
    date_modified as a UTC timestamp, and the kickoff table. Returns one row
    per player, season, week, and team.

    The injury report for week N is published during the week before kickoff,
    which is why, unlike a box score, it may be used as of the current week. But
    that only holds for versions actually on record before the game began. A
    version whose date_modified is at or after kickoff is excluded: it may carry
    information from the game itself. A report with no scheduled kickoff to
    compare against is excluded too, since its timing cannot be established.

    A player can have several versions in a week, because a status is revised
    between the first practice and game day, or because a mid-week trade puts
    him on two clubs' reports. Keeping the latest admitted version per player,
    week, and team handles both.
    """
    row_count_before = len(injuries)
    timed = injuries.merge(kickoffs, on=["season", "week", "team"], how="left")
    assert_no_fanout(row_count_before, timed, "injury kickoff times")

    modified_eastern = (
        timed["date_modified"].dt.tz_convert(SCHEDULE_TIME_ZONE).dt.tz_localize(None)
    )
    has_timestamp = timed["date_modified"].notna()
    has_kickoff = timed["kickoff"].notna()
    is_before_kickoff = has_timestamp & has_kickoff & (modified_eastern < timed["kickoff"])

    # The three exclusions are counted apart because they mean different things.
    # A version modified after kickoff is a leak being blocked. A version with
    # no timestamp, or no kickoff to compare with, is a gap in the source data:
    # its timing cannot be shown to be safe, so it is excluded, but it says
    # nothing about the report having been revised late.
    no_timestamp = ~has_timestamp
    no_kickoff = has_timestamp & ~has_kickoff
    after_kickoff = has_timestamp & has_kickoff & ~is_before_kickoff

    admitted = timed[is_before_kickoff]
    _log_step(
        "filter injury reports before kickoff",
        row_count_before,
        len(admitted),
        f"{int(after_kickoff.sum())} modified at or after kickoff; "
        f"{int(no_timestamp.sum())} with no date_modified, by season "
        f"{timed[no_timestamp].groupby('season').size().to_dict()}; "
        f"{int(no_kickoff.sum())} with no scheduled kickoff to compare against",
    )

    admitted = admitted.sort_values("date_modified")
    return admitted.drop_duplicates(subset=INJURY_KEYS, keep="last")


def _join_injuries(frame: pd.DataFrame, raw_directory: Path) -> pd.DataFrame:
    """Join the latest pre-kickoff version of the weekly injury report.

    Takes the player week frame and the raw data directory. Returns the joined
    frame.

    The version's timestamp is carried through as injury_report_modified. It is
    not a feature, but it is what marks a row as having a report entry at all,
    since a report can list a player with every status field blank, and it lets
    anyone audit that the chosen version predates kickoff.
    """
    injuries = read_parquet(raw_directory / "injuries.parquet")
    injuries = injuries.rename(columns={"gsis_id": "player_id"})
    schedules = read_parquet(raw_directory / "schedules.parquet")

    reports = select_pre_kickoff_reports(injuries, kickoff_times(schedules))
    reports = reports.rename(columns={"date_modified": "injury_report_modified"})

    return _left_join_with_match_count(
        frame, reports[INJURY_COLUMNS], INJURY_KEYS, "injuries"
    )


def _join_ff_opportunity(frame: pd.DataFrame, raw_directory: Path) -> pd.DataFrame:
    """Join expected fantasy points from the ffopportunity model.

    Takes the player week frame and the raw data directory. Returns the joined
    frame.

    Expected points separate real opportunity from lucky production, which is
    the single most useful thing this table adds. Rows with no player id are
    team level totals rather than player rows, and they are the cause of every
    duplicate key here, so they are dropped first.
    """
    opportunity = read_parquet(raw_directory / "ff_opportunity.parquet")
    opportunity = opportunity[opportunity["player_id"].notna()]

    # This table types season as a string and week as a float, unlike every
    # other nflverse table, so both need casting before they can be join keys.
    opportunity["season"] = opportunity["season"].astype(int)
    opportunity["week"] = opportunity["week"].astype(int)

    return _left_join_with_match_count(
        frame, opportunity[FF_OPPORTUNITY_COLUMNS], ["player_id", "season", "week"], "ff_opportunity"
    )


def _log_depth_charts_skipped() -> None:
    """Explain why the depth chart table is not joined.

    Takes nothing. Returns nothing.
    """
    logger.warning(
        "Skipping depth_charts: 83% of its rows have a null season, the rows that do carry "
        "one still hold thousands of duplicate player weeks, its team column is entirely "
        "null, and it has no data at all for the most recent season. Joining it would add a "
        "column that is empty exactly when predictions are made."
    )


def _drop_inactive_weeks(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Drop player weeks where the player was not on the active roster.

    Takes the player week frame and the parsed config. Returns the filtered
    frame.

    An inactive player never had the chance to score, and a bye week produces
    no row at all rather than a zero. Neither is a zero point performance, and
    training on them as zeros would teach the model that good players routinely
    score nothing. Rows with no roster match are kept, because a missing roster
    row is not evidence that the player sat out.
    """
    if not config["data"]["drop_inactive_weeks"]:
        return frame

    row_count_before = len(frame)
    is_active = frame["status"] == ACTIVE_ROSTER_STATUS
    has_no_roster_row = frame["status"].isna()
    frame = frame[is_active | has_no_roster_row]
    _log_step(
        "drop inactive weeks",
        row_count_before,
        len(frame),
        f"roster status other than {ACTIVE_ROSTER_STATUS}",
    )
    return frame


def build_player_weeks(config: dict[str, Any]) -> pd.DataFrame:
    """Join the raw tables into one tidy player-week table and save it.

    Takes the parsed config. Returns the player week frame, which is also
    written to data/processed/player_weeks.parquet.
    """
    scoring_config = config["scoring"]
    if scoring_config.get("use_prebuilt_column", False):
        raise ValueError(
            "scoring.use_prebuilt_column is true, but this league's rules do not match any "
            "prebuilt nflverse fantasy point column. Set it to false so the target is "
            "computed from box score stats."
        )

    raw_directory = resolve_path(config["data"]["paths"]["raw"])
    frame = read_parquet(raw_directory / "player_stats.parquet")
    logger.info("Loaded player_stats: %s rows x %s columns", len(frame), len(frame.columns))

    frame = _filter_season_type(frame, config)
    frame = _filter_positions(frame, config)
    frame = frame.reset_index(drop=True)

    target_column = scoring_config["target_column"]
    frame[target_column] = score_player_weeks(frame, scoring_config)
    logger.info("Added target column %s", target_column)

    frame = _join_schedules(frame, raw_directory)
    frame = _join_snap_counts(frame, raw_directory)
    frame = _join_rosters_weekly(frame, raw_directory)
    frame = _join_injuries(frame, raw_directory)
    frame = _join_ff_opportunity(frame, raw_directory)
    _log_depth_charts_skipped()

    frame = _drop_inactive_weeks(frame, config)
    frame = frame.reset_index(drop=True)

    destination = resolve_path(config["data"]["paths"]["processed"]) / "player_weeks.parquet"
    write_parquet(frame, destination)
    logger.info(
        "Saved player_weeks: %s rows x %s columns -> %s",
        len(frame),
        len(frame.columns),
        destination,
    )
    return frame
