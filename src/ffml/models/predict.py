"""Projections for an upcoming week, from the deployment models.

Every row the walk-forward evaluation ever scored came from a completed game.
Projecting a game that has not been played needs a row that does not exist
yet, so this module builds one per eligible player: his history rolled forward
onto the upcoming game, plus that game's context and his opponent's rating.

It computes no feature itself. The upcoming rows are appended to the completed
history and passed through build_dataset.build_feature_matrix, the same code
that built every training row, and the target week's rows are taken back out.
Rolling means and opponent ratings only ever look backward, so appending a row
for a future game cannot change a historical feature, and the upcoming row gets
exactly the value a completed game in its place would have had.

Before anything is projected, players who will not play are excluded and listed
with a reason, and every projection carries flags for stale or missing inputs.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ffml.data.clean import (
    ACTIVE_ROSTER_STATUS,
    INJURY_COLUMNS,
    INJURY_KEYS,
    SCHEDULE_COLUMNS,
    SCHEDULE_TIME_ZONE,
    kickoff_times,
    select_pre_kickoff_reports,
)
from ffml.features.build_dataset import build_feature_matrix, resolve_groups
from ffml.features.game_context import home_away
from ffml.features.opponent import OPPONENT_STRENGTH_FEATURE
from ffml.features.rolling import GAMES_PLAYED_COLUMN, SORT_COLUMNS, rolling_feature_name
from ffml.models import baseline, quantiles, train
from ffml.utils.io import ensure_directory, read_parquet, resolve_path

# Written on every row whose interval accompanies a baseline point projection,
# because the two numbers come from different methods.
BASELINE_INTERVAL_NOTE = "point from baseline, interval from quantile model"

# What project_position returns for each projected player.
PROJECTION_COLUMNS = [
    "player_id",
    "projected_points",
    "projection_method",
    "floor",
    "ceiling",
    "interval_note",
    GAMES_PLAYED_COLUMN,
]

logger = logging.getLogger(__name__)

PROJECTED = "projected"
EXCLUDED = "excluded"

# How a position's projections are produced, chosen per position in predict.projectors.
# The model is the deployment LightGBM booster. The baseline is the naive mean of the
# player's last three games, shipped for a position whose model has not been shown to
# beat it. Every projected row records which one produced its number.
PROJECTOR_MODEL = "model"
PROJECTOR_BASELINE = "baseline"
PROJECTORS = [PROJECTOR_MODEL, PROJECTOR_BASELINE]

# The baseline projection and the feature builder's three game rolling mean of the
# target are computed by separate code paths that must agree exactly.
BASELINE_MATCH_TOLERANCE = 1e-9

# Exclusion reasons. Each player gets the first that applies, in the order they
# are listed in exclusion_reasons.
REASON_NOT_ON_ROSTER = "not_on_roster"
REASON_BYE = "bye"
REASON_INSUFFICIENT_HISTORY = "insufficient_history"
REASON_SNAP_SHARE_UNKNOWN = "snap_share_unknown"
REASON_BELOW_SNAP_SHARE = "below_snap_share"

# Fewer games than this in the current season means the rolling windows are
# still drawing on last season, when the player may have had a different team,
# quarterback, and role. Three is when the three game mean, the fastest window
# and the baseline itself, first comes entirely from this season.
EARLY_SEASON_GAMES = 3

EARLY_SEASON_NOTE = (
    "Early season: rolling averages still draw on last season's games, often with a "
    "different team or role. Less reliable than a mid-season projection."
)

DEPTH_CHART_NOTE = (
    "depth_charts is pulled but no feature group reads it, so its coverage cannot "
    "affect a projection."
)

OUTPUT_COLUMNS = [
    "player",
    "player_id",
    "position",
    "team",
    "opponent",
    "home_away",
    "kickoff_et",
    "status",
    "exclusion_reason",
    "projected_points",
    "projection_method",
    "floor",
    "ceiling",
    "interval_note",
    "injury_report_status",
    "flag_team_changed",
    "flag_early_season",
    "flag_no_betting_line",
    "flag_snap_data_missing",
    "flag_kickoff_passed",
    "flag_prior_game_missing",
    "flag_injury_features_disabled",
    "flag_projection_outside_interval",
    "note",
]


def week_key(season: int, week: int) -> int:
    """Turn a season and week into one number that sorts chronologically.

    Takes the season and week. Returns the key, such as 202610 for week 10.
    """
    return season * 100 + week


def history_before(player_weeks: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Keep only games played strictly before the target week.

    Takes the player-week table and the target season and week. Returns the
    earlier rows.

    This is the leakage boundary for projection. Any row from the target week
    or later, a real result or anything else, is cut here, so nothing about the
    game being projected can reach its own features.
    """
    keys = player_weeks["season"] * 100 + player_weeks["week"]
    return player_weeks[keys < week_key(season, week)]


def game_sides(schedules: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """List every team playing in the target week, with its opponent and game.

    Takes the schedules table and the target season and week. Returns one row
    per team, carrying the schedule columns, the kickoff, and home or away.

    Each game involves two teams, so the schedule is stacked once from each
    side, the same way the player-week table sees it.
    """
    is_target = (
        (schedules["season"] == season)
        & (schedules["week"] == week)
        & (schedules["game_type"] == "REG")
    )
    games = schedules[is_target]

    home_side = games.assign(team=games["home_team"], opponent_team=games["away_team"])
    away_side = games.assign(team=games["away_team"], opponent_team=games["home_team"])
    sides = pd.concat([home_side, away_side], ignore_index=True)
    sides = sides[["team", "opponent_team"] + SCHEDULE_COLUMNS]

    kickoffs = kickoff_times(games)[["team", "kickoff"]]
    sides = sides.merge(kickoffs, on="team", how="left")
    sides["home_away"] = home_away(sides)
    return sides


def player_history_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Summarise what is known about each player from his completed games.

    Takes the completed history. Returns one row per player, indexed by
    player_id, with his position, name, last team, career games, and recent
    snap share.

    Position comes from his own history rather than the roster, because each
    position's features are built from that position's rows, and a roster that
    lists him differently would split his history in two.
    """
    ordered = history.sort_values(SORT_COLUMNS)
    last_game = ordered.groupby("player_id").tail(1).set_index("player_id")
    last_three = ordered.groupby("player_id").tail(3)
    snap = last_three.groupby("player_id")["offense_pct"].agg(["mean", "size", "count"])

    summary = pd.DataFrame(
        {
            "history_position": last_game["position"],
            "player_display_name": last_game["player_display_name"],
            "birth_date": last_game["birth_date"],
            "last_team": last_game["team"],
            "last_season": last_game["season"],
            "last_week": last_game["week"],
            "career_games": ordered.groupby("player_id").size(),
            "snap_share_last_3": snap["mean"],
            "snap_data_missing": snap["count"] < snap["size"],
        }
    )
    summary.index.name = "player_id"
    return summary


def latest_roster(
    rosters: pd.DataFrame, season: int, week: int, positions: list[str], history_positions: pd.Series
) -> tuple[pd.DataFrame, int]:
    """Read the most recent weekly roster at or before the target week.

    Takes the weekly rosters, the target season and week, the modelled
    positions, and each known player's historical position. Returns the roster
    rows and the week they come from. Raises ValueError if the season has no
    roster yet.

    A future week's roster is not published in advance, so the latest one
    available is used and its week is reported with every run.
    """
    is_available = (
        (rosters["season"] == season) & (rosters["week"] <= week) & rosters["gsis_id"].notna()
    )
    season_rows = rosters[is_available]
    if len(season_rows) == 0:
        raise ValueError(f"No weekly roster for season {season} at or before week {week}.")

    roster_week = int(season_rows["week"].max())
    rows = season_rows[season_rows["week"] == roster_week].rename(
        columns={"gsis_id": "player_id", "position": "roster_position", "status": "roster_status"}
    )
    rows = rows.drop_duplicates(subset=["player_id"], keep="last")

    is_modelled = rows["roster_position"].isin(positions) | rows["player_id"].isin(
        history_positions.index
    )
    columns = ["player_id", "team", "roster_position", "roster_status", "full_name"]
    return rows[is_modelled][columns], roster_week


def latest_injury_statuses(injuries: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Read each player's latest report status for the target week.

    Takes the injury table and the target season and week. Returns one row per
    player with his report status.

    The report is read whether or not it carries a timestamp. This decides who
    is projected, not what a feature says, so a status revised late can only
    exclude a player who was not going to play. It cannot leak into a model.
    """
    is_target = (
        (injuries["season"] == season)
        & (injuries["week"] == week)
        & (injuries["game_type"] == "REG")
        & injuries["gsis_id"].notna()
    )
    rows = injuries[is_target]
    if "date_modified" in rows.columns:
        rows = rows.sort_values("date_modified", na_position="first")
    rows = rows.drop_duplicates(subset=["gsis_id"], keep="last")
    rows = rows.rename(columns={"gsis_id": "player_id"})
    rows["on_injury_report"] = True
    return rows[["player_id", "report_status", "on_injury_report"]]


def exclusion_reasons(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Give each candidate the first reason he cannot be projected, if any.

    Takes the candidates and the parsed config. Returns the reason per row,
    None for a player who will be projected.

    The rules are applied from last to first, so the earliest rule in this
    order is the one that sticks:

    1. roster status other than active. No non-active player played in 2022-24.
    2. ruled out: the report says Out or Doubtful. The models have essentially
       never seen either, since a player ruled out has no box score to train on.
    3. bye: his team has no game this week.
    4. insufficient history: fewer games than the models require.
    5. snap share unknown: he has games but no snap data for them.
    6. below the snap share filter in the config.
    """
    min_prior_games = config["features"]["min_prior_games"]
    threshold = config["predict"]["min_snap_share_last_3"]
    excluded_statuses = config["predict"]["excluded_report_statuses"]

    reasons = pd.Series([None] * len(candidates), index=candidates.index, dtype=object)
    reasons[candidates["snap_share_last_3"] < threshold] = REASON_BELOW_SNAP_SHARE
    is_unknown = candidates["snap_share_last_3"].isna() & (candidates["career_games"] >= 1)
    reasons[is_unknown] = REASON_SNAP_SHARE_UNKNOWN
    reasons[candidates["career_games"] < min_prior_games] = REASON_INSUFFICIENT_HISTORY
    reasons[~candidates["has_game"].astype(bool)] = REASON_BYE

    is_ruled_out = candidates["report_status"].isin(excluded_statuses)
    ruled_out_labels = "ruled_out_" + candidates["report_status"].fillna("").str.lower()
    reasons[is_ruled_out] = ruled_out_labels[is_ruled_out]

    is_inactive = candidates["roster_status"] != ACTIVE_ROSTER_STATUS
    inactive_labels = "roster_" + candidates["roster_status"].fillna("").astype(str)
    reasons[is_inactive] = inactive_labels[is_inactive]
    return reasons


def relevant_players_not_on_roster(
    summary: pd.DataFrame, roster: pd.DataFrame, season: int, config: dict[str, Any]
) -> pd.DataFrame:
    """Find recent, relevant players who appear on no roster at all.

    Takes the history summary, the roster, the target season, and the parsed
    config. Returns them as excluded candidates.

    A released or retired player is simply absent from the roster, so without
    this he would vanish from the output without explanation.
    """
    threshold = config["predict"]["min_snap_share_last_3"]
    is_recent = summary["last_season"] >= season - 1
    is_relevant = summary["snap_share_last_3"] >= threshold
    missing = summary[is_recent & is_relevant & ~summary.index.isin(roster["player_id"])]

    frame = missing.reset_index()
    frame["position"] = frame["history_position"]
    frame["team"] = frame["last_team"]
    frame["roster_status"] = None
    frame["has_game"] = False
    frame["exclusion_reason"] = REASON_NOT_ON_ROSTER
    return frame


def assemble_candidates(
    roster: pd.DataFrame,
    summary: pd.DataFrame,
    sides: pd.DataFrame,
    statuses: pd.DataFrame,
    season: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Combine roster, history, schedule, and report into one row per player.

    Takes the roster, history summary, game sides, report statuses, target
    season, and parsed config. Returns every candidate with his exclusion
    reason, including relevant players on no roster.
    """
    candidates = roster.merge(summary, left_on="player_id", right_index=True, how="left")
    candidates["position"] = candidates["history_position"].fillna(candidates["roster_position"])
    candidates["player_display_name"] = candidates["player_display_name"].fillna(
        candidates["full_name"]
    )
    candidates["career_games"] = candidates["career_games"].fillna(0).astype(int)
    candidates = candidates.merge(statuses, on="player_id", how="left")
    candidates["has_game"] = candidates["team"].isin(set(sides["team"]))
    candidates["exclusion_reason"] = exclusion_reasons(candidates, config)

    not_rostered = relevant_players_not_on_roster(summary, roster, season, config)
    candidates = pd.concat([candidates, not_rostered], ignore_index=True)
    is_modelled = candidates["position"].isin(config["data"]["positions"])
    return candidates[is_modelled].reset_index(drop=True)


def is_listed_exclusion(candidates: pd.DataFrame, season: int, config: dict[str, Any]) -> pd.Series:
    """Decide which excluded players are worth listing in the output.

    Takes the candidates, the target season, and the parsed config. Returns a
    boolean per row.

    Listed: a player excluded for a reason other than the snap filter who
    played this season or last and was playing enough to matter. Without the
    restriction the output would carry hundreds of practice squad names; with
    it, every fantasy-relevant player is either projected or explained.
    """
    threshold = config["predict"]["min_snap_share_last_3"]
    is_excluded = candidates["exclusion_reason"].notna() & (
        candidates["exclusion_reason"] != REASON_BELOW_SNAP_SHARE
    )
    is_recent = candidates["last_season"] >= season - 1
    is_relevant = (candidates["snap_share_last_3"] >= threshold) | (
        candidates["exclusion_reason"] == REASON_SNAP_SHARE_UNKNOWN
    )
    return is_excluded & is_recent & is_relevant


def pre_kickoff_reports(injuries: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Prepare injury reports for the feature builder under the timestamp rule.

    Takes the raw injury and schedule tables. Returns the admitted reports in
    the player-week injury columns.

    This is the same rule the historical rows were built with, so an upcoming
    row reads the report exactly as a training row would have. When no version
    carries a timestamp, as in 2025 and 2026, nothing is admitted.
    """
    if "date_modified" not in injuries.columns:
        logger.warning("Injury table has no date_modified column; no report can be admitted")
        return pd.DataFrame(columns=INJURY_COLUMNS)

    renamed = injuries.rename(columns={"gsis_id": "player_id"})
    reports = select_pre_kickoff_reports(renamed, kickoff_times(schedules))
    return reports.rename(columns={"date_modified": "injury_report_modified"})


def _attach_injury_reports(
    upcoming: pd.DataFrame, injury_reports: pd.DataFrame, season: int, week: int
) -> pd.DataFrame:
    """Join the admitted report for the target week onto the upcoming rows.

    Takes the upcoming rows, the admitted reports, and the target season and
    week. Returns the rows with the report columns filled where one exists.
    """
    is_target = (injury_reports["season"] == season) & (injury_reports["week"] == week)
    week_reports = injury_reports[is_target][INJURY_COLUMNS]

    payload_columns = []
    for column_name in INJURY_COLUMNS:
        if column_name not in INJURY_KEYS:
            payload_columns.append(column_name)

    base = upcoming.drop(columns=payload_columns, errors="ignore")
    return base.merge(week_reports, on=INJURY_KEYS, how="left")


def build_upcoming_rows(
    player_weeks: pd.DataFrame,
    players: pd.DataFrame,
    sides: pd.DataFrame,
    season: int,
    week: int,
    injury_reports: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one player-week row per player for a game that has not been played.

    Takes the player-week table, whose columns the rows copy; the players to
    project, with player_id, name, position, and current team; the target
    week's game sides; the target season and week; and optionally the admitted
    injury reports. Returns rows in the player-week schema with every box score
    column empty. Raises ValueError if a player has no game that week.

    The team is the player's current one, so the game context and opponent
    rating describe the game he is about to play, even when his rolling history
    was built elsewhere.
    """
    player_columns = []
    for column_name in ["player_id", "player_display_name", "position", "team", "birth_date"]:
        if column_name in players.columns:
            player_columns.append(column_name)

    upcoming = players[player_columns].merge(sides, on="team", how="inner")
    if len(upcoming) != len(players):
        raise ValueError(
            f"{len(players)} players to project but {len(upcoming)} upcoming rows: a player "
            "without exactly one game this week reached the row builder."
        )

    upcoming["season"] = season
    upcoming["week"] = week
    upcoming["season_type"] = "REG"
    if injury_reports is not None:
        upcoming = _attach_injury_reports(upcoming, injury_reports, season, week)

    return upcoming.reindex(columns=player_weeks.columns)


def upcoming_feature_rows(
    player_weeks: pd.DataFrame,
    upcoming: pd.DataFrame,
    config: dict[str, Any],
    position: str,
    season: int,
    week: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute features for upcoming rows with the historical feature builder.

    Takes the player-week table, the upcoming rows, the config to build with,
    the position, and the target season and week. Returns the upcoming rows'
    features and the ordered feature names. Raises ValueError if any upcoming
    row fails to come back.

    History is cut strictly before the target week first, so whatever the
    player-week table holds about that week, including real results when it has
    already been played, cannot reach the rows being built.
    """
    history = history_before(player_weeks, season, week)
    position_upcoming = upcoming[upcoming["position"] == position]
    combined = pd.concat([history, position_upcoming], ignore_index=True)

    matrix, feature_names = build_feature_matrix(combined, config, position)
    is_target = (matrix["season"] == season) & (matrix["week"] == week)
    rows = matrix[is_target].reset_index(drop=True)

    if len(rows) != len(position_upcoming):
        raise ValueError(
            f"{position}: {len(position_upcoming)} upcoming rows went into the feature builder "
            f"but {len(rows)} came out. Every projected player should have enough history."
        )
    return rows, feature_names


def check_upcoming_features(
    rows: pd.DataFrame, feature_names: list[str], position: str, week: int
) -> list[str]:
    """Refuse to project with a feature that is empty for every upcoming row.

    Takes the upcoming feature rows, the feature names, the position, and the
    target week. Returns the features that are empty by design. Raises
    ValueError for any other empty feature.

    A feature that is empty everywhere means the table behind it has no data for
    this week, and the model would be projecting with a silently missing input.
    The one exception is opponent strength in week 1: ratings reset each season,
    so no defense has been rated yet, and that is logged rather than raised.
    """
    expected_empty = []
    unexpected_empty = []
    for feature_name in feature_names:
        if not rows[feature_name].isna().all():
            continue
        if feature_name == OPPONENT_STRENGTH_FEATURE and week == 1:
            expected_empty.append(feature_name)
        else:
            unexpected_empty.append(feature_name)

    if len(unexpected_empty) > 0:
        raise ValueError(
            f"{position}: features empty for every upcoming row: {', '.join(unexpected_empty)}. "
            "The table behind them has no data for this week."
        )
    if len(expected_empty) > 0:
        logger.warning(
            "%s: %s is empty for every row in week 1, by design: defenses are rated within a "
            "season and none has played yet",
            position,
            ", ".join(expected_empty),
        )
    return expected_empty


def check_feature_list(
    metadata: dict[str, Any], booster: Any, expected_names: list[str], position: str
) -> None:
    """Fail unless the saved list, the booster, and the builder agree on the features.

    Takes the model metadata, the booster, the feature names the builder
    produced, and the position. Returns nothing. Raises ValueError on any
    difference in names or order.

    A model reads its inputs by position, not by name. A reordered column would
    not error; it would quietly feed the model the wrong number.
    """
    saved_names = metadata["feature_names"]
    booster_names = booster.feature_name()
    if saved_names == expected_names and booster_names == expected_names:
        return

    raise ValueError(
        f"{position}: feature list mismatch. Saved metadata has {len(saved_names)} features, "
        f"the booster {len(booster_names)}, the builder {len(expected_names)}. "
        f"Saved: {saved_names}. Built: {expected_names}. Retrain with train_model.py --deploy."
    )


def check_model_is_current(
    metadata: dict[str, Any],
    position: str,
    latest_completed: tuple[int, int],
    target: tuple[int, int],
    smoke_test: bool,
) -> None:
    """Fail if a live projection would use a stale model or one that saw the target week.

    Takes the model metadata, the position, the latest completed week before
    the target, the target week, and whether this is a smoke test. Returns
    nothing. Raises ValueError otherwise.

    The smoke test deliberately projects a week the model trained on, to
    exercise the machinery, so it skips both checks and says so.
    """
    if smoke_test:
        return

    trained = metadata["trained_through"]
    trained_key = week_key(trained[0], trained[1])
    if trained_key >= week_key(target[0], target[1]):
        raise ValueError(
            f"{position}: the deployment model was trained through {trained}, which is not "
            f"before the week being projected {target}. A live projection cannot use it."
        )
    if trained_key < week_key(latest_completed[0], latest_completed[1]):
        raise ValueError(
            f"{position}: the deployment model was trained through {trained} but "
            f"{latest_completed} is complete. Retrain with scripts/train_model.py --deploy."
        )


def previous_scheduled_week(schedules: pd.DataFrame, season: int, week: int) -> tuple[int, int]:
    """Find the regular season week scheduled immediately before the target.

    Takes the schedules and the target season and week. Returns that week.
    Raises ValueError if there is none.
    """
    regular = schedules[schedules["game_type"] == "REG"]
    keys = sorted(set((regular["season"] * 100 + regular["week"]).tolist()))

    earlier_keys = []
    for key in keys:
        if key < week_key(season, week):
            earlier_keys.append(key)
    if len(earlier_keys) == 0:
        raise ValueError(f"No scheduled week before season {season} week {week}.")

    latest = max(earlier_keys)
    return latest // 100, latest % 100


def check_history_complete(
    tables: dict[str, pd.DataFrame],
    config: dict[str, Any],
    season: int,
    week: int,
    allow_incomplete: bool,
) -> dict[str, Any]:
    """Check the week before the target has fully arrived in the data.

    Takes the loaded tables, the parsed config, the target season and week, and
    whether to allow an incomplete prior week. Returns the completion details.
    Raises ValueError for an incomplete prior week unless it is allowed.

    Projecting before last week's games and stats have all landed would build
    some players' rows from one fewer game than others, with nothing to show it.
    """
    schedules = tables["schedules"]
    player_weeks = tables["player_weeks"]
    required = previous_scheduled_week(schedules, season, week)
    status = train.week_completion(schedules, player_weeks, required[0], required[1])

    completed = train.completed_weeks(schedules, player_weeks, config)
    earlier = []
    for completed_week in completed:
        if week_key(completed_week[0], completed_week[1]) < week_key(season, week):
            earlier.append(completed_week)
    if len(earlier) == 0:
        raise ValueError(f"No completed week before season {season} week {week}.")

    if not status["complete"]:
        message = (
            f"Season {required[0]} week {required[1]} is incomplete: unplayed games "
            f"{status['unplayed_games']}, teams missing stats {status['teams_missing_stats']}."
        )
        if not allow_incomplete:
            raise ValueError(message + " Rerun once the data arrives, or pass --allow-incomplete-history.")
        logger.warning(message + " Projecting anyway; affected players are flagged.")

    missing_teams = set(status["unplayed_teams"]) | set(status["teams_missing_stats"])
    return {
        "required_week": required,
        "complete": status["complete"],
        "latest_completed": earlier[-1],
        "teams_missing_prior_game": missing_teams,
    }


def load_prediction_tables(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Read every table a projection needs.

    Takes the parsed config. Returns the tables by name.
    """
    processed_directory = resolve_path(config["data"]["paths"]["processed"])
    raw_directory = resolve_path(config["data"]["paths"]["raw"])
    return {
        "player_weeks": read_parquet(processed_directory / "player_weeks.parquet"),
        "schedules": read_parquet(raw_directory / "schedules.parquet"),
        "rosters": read_parquet(raw_directory / "rosters_weekly.parquet"),
        "injuries": read_parquet(raw_directory / "injuries.parquet"),
    }


def projector_for(config: dict[str, Any], position: str) -> str:
    """Decide how one position's projections are produced.

    Takes the parsed config and the position. Returns model or baseline: the
    position's override in predict.projectors when it has one, otherwise the
    default, and model when the block is absent. Raises ValueError for an
    unknown method or an override naming a position that is not modelled.

    This is a deployment choice made per position in the config, so no position
    is ever special-cased in code.
    """
    projector_config = config["predict"].get("projectors") or {}
    overrides = projector_config.get("by_position") or {}
    for override_position in overrides:
        if override_position not in config["data"]["positions"]:
            raise ValueError(
                f"predict.projectors.by_position names {override_position}, which is not a "
                "modelled position."
            )

    projector = projector_config.get("default", PROJECTOR_MODEL)
    if position in overrides:
        projector = overrides[position]
    if projector not in PROJECTORS:
        raise ValueError(f"Unknown projector '{projector}' for {position}. Use one of {PROJECTORS}.")
    return projector


def projection_methods(config: dict[str, Any]) -> dict[str, str]:
    """List the projection method of every modelled position.

    Takes the parsed config. Returns the method per position.
    """
    methods = {}
    for position in config["data"]["positions"]:
        methods[position] = projector_for(config, position)
    return methods


def load_current_model(context: dict[str, Any], position: str) -> tuple[Any, dict[str, Any]]:
    """Load one position's deployment model, refusing a stale one.

    Takes the run context and the position. Returns the booster and its
    metadata. Raises ValueError through check_model_is_current.
    """
    booster, metadata = train.load_deployment_model(context["config"], position)
    check_model_is_current(
        metadata, position, context["history"]["latest_completed"], context["target"], context["smoke_test"]
    )
    return booster, metadata


def baseline_projections(
    player_weeks: pd.DataFrame, upcoming: pd.DataFrame, config: dict[str, Any], season: int, week: int
) -> pd.Series:
    """Project upcoming rows with the naive baseline.

    Takes the player-week table, the upcoming rows, the parsed config, and the
    target season and week. Returns the projection per player id. Raises
    ValueError if any upcoming row has no projection.

    History is cut strictly before the target week, exactly as for the feature
    builder, and the baseline is the same function that scored the walk-forward
    folds. The upcoming rows carry no result, so they only mark where the next
    game falls; the three game mean looks back from there.
    """
    history = history_before(player_weeks, season, week)
    combined = pd.concat([history, upcoming], ignore_index=True)
    projected = baseline.run_baseline(combined, config)

    is_target = (combined["season"] == season) & (combined["week"] == week)
    projections = pd.Series(
        projected[is_target].to_numpy(), index=combined.loc[is_target, "player_id"].to_numpy()
    )

    missing = projections[projections.isna()]
    if len(missing) > 0:
        raise ValueError(
            f"{len(missing)} upcoming rows have no baseline projection, although every projected "
            f"player should have enough history: {sorted(missing.index.tolist())}."
        )
    return projections


def check_baseline_matches_rolling_mean(
    projections: pd.Series, rows: pd.DataFrame, config: dict[str, Any], position: str
) -> None:
    """Fail unless the baseline projection equals the builder's rolling mean of the target.

    Takes the baseline projection per player id, the upcoming feature rows, the
    parsed config, and the position. Returns nothing. Raises ValueError if the
    rolling mean is not built or the two disagree for any player.

    Training cross-checks the same agreement on every run. A disagreement here
    means the projected number is not the baseline the evaluation measured.
    """
    feature_name = rolling_feature_name(
        config["scoring"]["target_column"], config["model"]["baseline"]["window"]
    )
    if feature_name not in rows.columns:
        raise ValueError(f"{position}: {feature_name} is not built, so the baseline cannot be cross-checked.")

    built = rows.set_index("player_id")[feature_name]
    difference = (projections.reindex(built.index) - built).abs()
    if difference.isna().any() or float(difference.max()) > BASELINE_MATCH_TOLERANCE:
        raise ValueError(
            f"{position}: the baseline projection and {feature_name} disagree, by up to "
            f"{difference.max()}. They must be the same number."
        )


def check_projection_methods(output: pd.DataFrame, config: dict[str, Any]) -> None:
    """Fail unless every projected row names the method its position ships.

    Takes the output table and the parsed config. Returns nothing. Raises
    ValueError if a projected row's method is missing or differs from
    predict.projectors.

    A row whose number could have come from either method would make any
    comparison between players at different positions impossible to read.
    """
    projected = output[output["status"] == PROJECTED]
    problems = []
    for position in sorted(projected["position"].unique().tolist()):
        expected = projector_for(config, position)
        methods = projected.loc[projected["position"] == position, "projection_method"]
        wrong = methods.isna() | (methods != expected)
        if wrong.any():
            problems.append(f"{position}: {int(wrong.sum())} rows not labelled {expected}")

    if len(problems) > 0:
        raise ValueError("Projection methods disagree with predict.projectors: " + "; ".join(problems))


def interval_settings(config: dict[str, Any], position: str) -> dict[str, dict[str, Any]]:
    """Read whether each bound ships for one position, and why not if it does not.

    Takes the parsed config and the position. Returns, for the floor and the
    ceiling, whether it ships and its note. A position or bound with no entry
    in predict.intervals does not ship. Raises ValueError for an entry naming a
    position that is not modelled.

    The entries record the pre-registered calibration verdicts from
    scripts/train_model.py --quantiles. A bound that failed is never shown,
    because a miscalibrated ceiling is worse than none: it will be believed.
    """
    intervals = config["predict"].get("intervals") or {}
    for configured_position in intervals:
        if configured_position not in config["data"]["positions"]:
            raise ValueError(f"predict.intervals names {configured_position}, which is not a modelled position.")

    position_settings = intervals.get(position) or {}
    settings = {}
    for bound in quantiles.BOUNDS:
        bound_setting = position_settings.get(bound) or {}
        note = bound_setting.get("note")
        if not note:
            note = "no calibration result recorded"
        settings[bound] = {"ship": bool(bound_setting.get("ship", False)), "note": note}
    return settings


def shipped_bounds(config: dict[str, Any], position: str) -> list[str]:
    """List the bounds that ship for one position.

    Takes the parsed config and the position. Returns floor, ceiling, both, or
    neither, in that order.
    """
    settings = interval_settings(config, position)
    shipped = []
    for bound in quantiles.BOUNDS:
        if settings[bound]["ship"]:
            shipped.append(bound)
    return shipped


def interval_note(config: dict[str, Any], position: str) -> str:
    """Explain a position's intervals in words for its output rows.

    Takes the parsed config and the position. Returns why each missing bound
    is missing and, for a baseline position with a bound, that the point and
    the interval come from different methods. Empty when nothing needs saying.
    """
    settings = interval_settings(config, position)
    parts = []
    for bound in quantiles.BOUNDS:
        if not settings[bound]["ship"]:
            parts.append(f"{bound} not shipped: {settings[bound]['note']}")

    # A quantile interval around a baseline point mixes two methods. That is
    # stated on every such row rather than resolved silently.
    if len(shipped_bounds(config, position)) > 0 and projector_for(config, position) == PROJECTOR_BASELINE:
        parts.append(BASELINE_INTERVAL_NOTE)
    return "; ".join(parts)


def project_bound(booster: Any, rows: pd.DataFrame, feature_names: list[str]) -> Any:
    """Project one quantile bound for a set of feature rows.

    Takes the quantile booster, the rows, and the ordered feature names.
    Returns the predictions, reading the columns in the order the model was
    fitted on.
    """
    return booster.predict(rows[feature_names])


def interval_projections(
    context: dict[str, Any], rows: pd.DataFrame, feature_names: list[str], position: str
) -> pd.DataFrame:
    """Project the floor and ceiling for one position's upcoming rows.

    Takes the run context, the upcoming feature rows, the ordered feature
    names, and the position. Returns the floor, ceiling, and interval note,
    aligned to the rows. A bound that does not ship is left empty.

    A shipped bound's deployment model faces the same checks as a mean model:
    it must be current, and its feature list must match the builder's.
    """
    config = context["config"]
    levels = quantiles.bound_levels(config)
    settings = interval_settings(config, position)

    intervals = pd.DataFrame(index=rows.index)
    for bound in quantiles.BOUNDS:
        if not settings[bound]["ship"]:
            intervals[bound] = float("nan")
            continue
        suffix = quantiles.level_suffix(levels[bound])
        booster, metadata = train.load_deployment_model(config, position, suffix)
        label = f"{position} {bound} ({suffix})"
        check_model_is_current(
            metadata, label, context["history"]["latest_completed"], context["target"], context["smoke_test"]
        )
        check_feature_list(metadata, booster, feature_names, label)
        intervals[bound] = project_bound(booster, rows, feature_names)

    intervals["interval_note"] = interval_note(config, position)
    return intervals


def project_position(
    context: dict[str, Any], candidates: pd.DataFrame, position: str
) -> tuple[pd.DataFrame | None, list[str]]:
    """Project every eligible player at one position, by the method it ships.

    Takes the run context, the candidates, and the position. Returns the
    projected points and projection method per player, or None if nobody is
    eligible, and the features that were empty by design.

    Features are built for a baseline position too. They give the games played
    that the early season flag reads and the rolling mean the baseline is
    checked against, and they keep the data checks the same for every position.
    No deployment model is loaded for a baseline position.
    """
    config = context["config"]
    projector = projector_for(config, position)
    loaded_model = None
    if projector == PROJECTOR_MODEL:
        loaded_model = load_current_model(context, position)

    is_eligible = (candidates["position"] == position) & candidates["exclusion_reason"].isna()
    eligible = candidates[is_eligible]
    if len(eligible) == 0:
        return None, []

    season, week = context["target"]
    player_weeks = context["tables"]["player_weeks"]
    upcoming = build_upcoming_rows(
        player_weeks, eligible, context["sides"], season, week, context["injury_reports"]
    )
    rows, feature_names = upcoming_feature_rows(
        player_weeks, upcoming, context["deployment_config"], position, season, week
    )

    if loaded_model is None:
        projections = baseline_projections(player_weeks, upcoming, config, season, week)
        check_baseline_matches_rolling_mean(projections, rows, config, position)
        rows["projected_points"] = rows["player_id"].map(projections)
    else:
        booster, metadata = loaded_model
        check_feature_list(metadata, booster, feature_names, position)
        rows["projected_points"] = booster.predict(rows[feature_names])

    expected_empty = check_upcoming_features(rows, feature_names, position, week)
    rows["projection_method"] = projector

    intervals = interval_projections(context, rows, feature_names, position)
    for column_name in ["floor", "ceiling", "interval_note"]:
        rows[column_name] = intervals[column_name]
    return rows[PROJECTION_COLUMNS], expected_empty


def injury_display_status(frame: pd.DataFrame) -> pd.Series:
    """Describe each player's injury report entry in words.

    Takes the output rows. Returns not_listed, no_designation, or the report
    status itself.

    A player absent from the report and a player listed with no game status are
    different facts. A blank cell would show them identically.
    """
    labels = pd.Series(["not_listed"] * len(frame), index=frame.index, dtype=object)
    on_report = frame["on_injury_report"].eq(True)
    labels[on_report] = "no_designation"
    has_status = on_report & frame["injury_report_status"].notna()
    labels[has_status] = frame.loc[has_status, "injury_report_status"]
    return labels


def add_flags(
    frame: pd.DataFrame, deployment_config: dict[str, Any], now: pd.Timestamp, missing_teams: set
) -> pd.DataFrame:
    """Add the stale and missing input flags and the early season note.

    Takes the output rows, the deployment config, the current time in US
    Eastern, and the teams whose previous game has not fully arrived. Returns a
    copy with the flag columns added.
    """
    flagged = frame.copy()
    is_projected = flagged["status"] == PROJECTED

    changed = flagged["last_team"].notna() & (flagged["team"] != flagged["last_team"])
    flagged["flag_team_changed"] = changed.astype(int)
    early = is_projected & (flagged[GAMES_PLAYED_COLUMN] < EARLY_SEASON_GAMES)
    flagged["flag_early_season"] = early.astype(int)
    no_line = flagged["opponent"].notna() & (flagged["spread_line"].isna() | flagged["total_line"].isna())
    flagged["flag_no_betting_line"] = no_line.astype(int)
    flagged["flag_snap_data_missing"] = flagged["snap_data_missing"].eq(True).astype(int)
    passed = flagged["kickoff"].notna() & (flagged["kickoff"] <= now)
    flagged["flag_kickoff_passed"] = passed.astype(int)
    flagged["flag_prior_game_missing"] = flagged["team"].isin(missing_teams).astype(int)

    # A mean can sit outside its own 10th to 90th percentile range, so this is
    # flagged for the reader rather than raised.
    below_floor = flagged["floor"].notna() & (flagged["projected_points"] < flagged["floor"])
    above_ceiling = flagged["ceiling"].notna() & (flagged["projected_points"] > flagged["ceiling"])
    flagged["flag_projection_outside_interval"] = (is_projected & (below_floor | above_ceiling)).astype(int)

    disabled = []
    for position in flagged["position"]:
        groups = resolve_groups(deployment_config, position)
        disabled.append(int(not groups.get("injuries", False)))
    flagged["flag_injury_features_disabled"] = disabled

    notes = pd.Series([""] * len(flagged), index=flagged.index, dtype=object)
    notes[early] = EARLY_SEASON_NOTE
    flagged["note"] = notes
    return flagged


def assemble_output(
    context: dict[str, Any], candidates: pd.DataFrame, projections: pd.DataFrame
) -> pd.DataFrame:
    """Build the output table of projected and listed excluded players.

    Takes the run context, the candidates, and the projected points. Returns
    one row per player in the output column order, sorted by position and
    projection.
    """
    season, _ = context["target"]
    projected = candidates[candidates["exclusion_reason"].isna()].merge(
        projections, on="player_id", how="inner"
    )
    projected["status"] = PROJECTED
    excluded = candidates[is_listed_exclusion(candidates, season, context["config"])].copy()
    excluded["status"] = EXCLUDED
    frame = pd.concat([projected, excluded], ignore_index=True)

    game_columns = ["team", "opponent_team", "home_away", "kickoff", "spread_line", "total_line"]
    frame = frame.merge(context["sides"][game_columns], on="team", how="left")
    frame = frame.rename(
        columns={"player_display_name": "player", "opponent_team": "opponent", "report_status": "injury_report_status"}
    )
    frame["injury_report_status"] = injury_display_status(frame)
    frame = add_flags(frame, context["deployment_config"], context["now"], context["history"]["teams_missing_prior_game"])
    frame["kickoff_et"] = frame["kickoff"].dt.strftime("%Y-%m-%d %H:%M")
    frame["projected_points"] = frame["projected_points"].round(2)
    frame["floor"] = frame["floor"].round(2)
    frame["ceiling"] = frame["ceiling"].round(2)

    frame["status_order"] = (frame["status"] == EXCLUDED).astype(int)
    frame = frame.sort_values(
        ["position", "status_order", "projected_points"], ascending=[True, True, False], na_position="last"
    )
    return frame[OUTPUT_COLUMNS].reset_index(drop=True)


def position_ceilings(history: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    """Find the highest league score each position actually posted in training.

    Takes the completed history and the parsed config. Returns the maximum per
    position.

    A projection is an expected value, so it should sit far below a position's
    best single game. One above it means something has gone badly wrong.
    """
    modelled = history[history["season"] >= config["data"]["start_season"]]
    maxima = modelled.groupby("position")[config["scoring"]["target_column"]].max()

    ceilings = {}
    for position in maxima.index:
        ceilings[position] = float(maxima.loc[position])
    return ceilings


def sanity_check(output: pd.DataFrame, ceilings: dict[str, float]) -> pd.DataFrame:
    """Refuse output with duplicate players or implausible projections.

    Takes the output table and each position's ceiling. Returns the minimum,
    median, and maximum projection per position. Raises ValueError on a
    duplicate player, a negative projection, or one above the ceiling.
    """
    duplicates = output[output["player_id"].duplicated(keep=False)]
    if len(duplicates) > 0:
        raise ValueError(f"Players appear more than once: {sorted(set(duplicates['player']))}.")

    projected = output[output["status"] == PROJECTED]
    rows = []
    for position in sorted(projected["position"].unique().tolist()):
        points = projected[projected["position"] == position]["projected_points"]
        if (points < 0).any():
            raise ValueError(f"{position}: negative projections, minimum {points.min():.2f}.")
        if (points > ceilings[position]).any():
            raise ValueError(
                f"{position}: projection {points.max():.2f} above the position's best actual "
                f"game in training, {ceilings[position]:.1f}."
            )
        position_rows = projected[projected["position"] == position]
        methods = position_rows["projection_method"]
        rows.append(
            {
                "position": position,
                "method": ", ".join(sorted(methods.dropna().unique().tolist())),
                "projected": len(points),
                "min": round(float(points.min()), 2),
                "median": round(float(points.median()), 2),
                "max": round(float(points.max()), 2),
                "best_game": ceilings[position],
                "floor_median": _median_or_nan(position_rows["floor"]),
                "ceiling_median": _median_or_nan(position_rows["ceiling"]),
            }
        )
    return pd.DataFrame(rows)


def _median_or_nan(values: pd.Series) -> float:
    """Take the rounded median of the values present, or NaN if there are none.

    Takes the values. Returns the median to two places. A position whose bound
    does not ship has no values at all, which is expected, not an error.
    """
    present = values.dropna()
    if len(present) == 0:
        return float("nan")
    return round(float(present.median()), 2)


def position_floors(history: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    """Find the lowest league score each position actually posted in training.

    Takes the completed history and the parsed config. Returns the minimum per
    position.

    Fantasy points can go below zero through interceptions and lost fumbles,
    so a floor may be negative. A floor below the worst game any player at the
    position actually had means something has gone wrong.
    """
    modelled = history[history["season"] >= config["data"]["start_season"]]
    minima = modelled.groupby("position")[config["scoring"]["target_column"]].min()

    floors = {}
    for position in minima.index:
        floors[position] = float(minima.loc[position])
    return floors


def _interval_problems(
    rows: pd.DataFrame, settings: dict[str, dict[str, Any]], position: str, lowest: float, highest: float
) -> list[str]:
    """List everything wrong with one position's floors and ceilings.

    Takes the position's projected rows, its interval settings, the position,
    and its worst and best actual games. Returns the problems, empty if none.
    """
    problems = []
    for bound in quantiles.BOUNDS:
        values = rows[bound]
        if settings[bound]["ship"] and values.isna().any():
            problems.append(f"{position}: {int(values.isna().sum())} rows missing a shipped {bound}")
        if not settings[bound]["ship"] and values.notna().any():
            problems.append(f"{position}: {bound} present but not shipped")

    crossed = rows["floor"].notna() & rows["ceiling"].notna() & (rows["floor"] > rows["ceiling"])
    if crossed.any():
        problems.append(f"{position}: {int(crossed.sum())} floors above their ceiling")
    if (rows["floor"] < lowest).any():
        problems.append(f"{position}: a floor below the position's worst actual game, {lowest:.1f}")
    if (rows["ceiling"] > highest).any():
        problems.append(f"{position}: a ceiling above the position's best actual game, {highest:.1f}")
    return problems


def check_intervals(
    output: pd.DataFrame, config: dict[str, Any], lowest: dict[str, float], highest: dict[str, float]
) -> None:
    """Refuse floors and ceilings that should not be shown.

    Takes the output table, the parsed config, and each position's worst and
    best actual game in training. Returns nothing. Raises ValueError if a
    shipped bound is missing, an unshipped one is present, a floor sits above
    its ceiling, a bound falls outside anything the position has ever scored,
    or an excluded player carries a bound.

    A bound that failed calibration must never reach the output, because it
    would be believed.
    """
    projected = output[output["status"] == PROJECTED]
    problems = []
    for position in sorted(projected["position"].unique().tolist()):
        rows = projected[projected["position"] == position]
        settings = interval_settings(config, position)
        for problem in _interval_problems(rows, settings, position, lowest[position], highest[position]):
            problems.append(problem)

    excluded = output[output["status"] == EXCLUDED]
    if excluded["floor"].notna().any() or excluded["ceiling"].notna().any():
        problems.append("excluded players carry a floor or ceiling")

    if len(problems) > 0:
        raise ValueError("Interval checks failed: " + "; ".join(problems))


def data_freshness(
    tables: dict[str, pd.DataFrame], config: dict[str, Any], season: int
) -> dict[str, Any]:
    """Describe how current the data behind a projection is.

    Takes the loaded tables, the parsed config, and the target season. Returns
    the freshness details.

    The injury report carries no timestamp in recent seasons, so the time the
    file was pulled is the best available statement of how current it is.
    """
    raw_directory = resolve_path(config["data"]["paths"]["raw"])
    schedules = tables["schedules"]
    scored = schedules[(schedules["game_type"] == "REG") & schedules["home_score"].notna()]

    injuries = tables["injuries"]
    season_reports = injuries[injuries["season"] == season]
    timestamped = 0
    if "date_modified" in season_reports.columns:
        timestamped = int(season_reports["date_modified"].notna().sum())

    return {
        "latest_completed_gameday": str(scored["gameday"].max()),
        "schedule_pulled": _file_time(raw_directory / "schedules.parquet"),
        "injury_report_pulled": _file_time(raw_directory / "injuries.parquet"),
        "roster_pulled": _file_time(raw_directory / "rosters_weekly.parquet"),
        "injury_rows_this_season": len(season_reports),
        "injury_rows_timestamped": timestamped,
        "injury_weeks_this_season": sorted(season_reports["week"].unique().tolist()),
    }


def _file_time(path: Path) -> str:
    """Format a file's modification time.

    Takes the path. Returns the local time it was last written.
    """
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def missing_inputs(sides: pd.DataFrame) -> dict[str, int]:
    """Count the games this week with missing context.

    Takes the game sides. Returns the counts, per game rather than per team.
    """
    per_game = sides.drop_duplicates(subset=["game_id"])
    no_line = per_game["spread_line"].isna() | per_game["total_line"].isna()
    blank_surface = per_game["surface"].fillna("").astype(str).str.strip() == ""
    return {
        "games": len(per_game),
        "games_without_betting_line": int(no_line.sum()),
        "games_with_blank_surface": int(blank_surface.sum()),
    }


def count_active_rookies(candidates: pd.DataFrame) -> int:
    """Count active players with no NFL games.

    Takes the candidates. Returns the count.

    They have no snap share to test against the filter, so they are neither
    projected nor listed. This count is how they stay visible. Practice squad
    and released players with no games are left out of it, since they could not
    play this week regardless.
    """
    is_active = candidates["roster_status"] == ACTIVE_ROSTER_STATUS
    return int((is_active & (candidates["career_games"] == 0)).sum())


def exclusion_counts(candidates: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
    """Count exclusions by reason, and how many of each are listed.

    Takes the candidates and the output table. Returns one row per reason.
    """
    excluded = candidates[candidates["exclusion_reason"].notna()]
    totals = excluded.groupby("exclusion_reason").size().rename("all_candidates")
    listed = output[output["status"] == EXCLUDED].groupby("exclusion_reason").size().rename("listed")
    table = pd.concat([totals, listed], axis=1).fillna(0).astype(int)
    return table.sort_values("all_candidates", ascending=False).reset_index()


def write_projections(
    output: pd.DataFrame, config: dict[str, Any], season: int, week: int, smoke_test: bool
) -> Path:
    """Write the output table to CSV, named by season and week.

    Takes the output table, the parsed config, the target season and week, and
    whether this is a smoke test. Returns the path written.
    """
    directory = resolve_path(config["data"]["paths"]["predictions"])
    if smoke_test:
        directory = directory / "smoke"
    ensure_directory(directory)

    path = directory / f"projections_{season}_week{week:02d}.csv"
    output.to_csv(path, index=False)
    logger.info("Wrote %s rows to %s", len(output), path)
    return path


def build_context(
    config: dict[str, Any], season: int, week: int, smoke_test: bool, allow_incomplete_history: bool
) -> dict[str, Any]:
    """Load and check everything a projection run needs.

    Takes the parsed config, the target season and week, and the two run
    options. Returns the run context.
    """
    tables = load_prediction_tables(config)
    history_status = check_history_complete(tables, config, season, week, allow_incomplete_history)
    sides = game_sides(tables["schedules"], season, week)
    if len(sides) == 0:
        raise ValueError(f"No regular season games scheduled for season {season} week {week}.")

    return {
        "config": config,
        "deployment_config": train.deployment_config(config),
        "tables": tables,
        "target": (season, week),
        "smoke_test": smoke_test,
        "history": history_status,
        "sides": sides,
        "injury_reports": pre_kickoff_reports(tables["injuries"], tables["schedules"]),
        "now": pd.Timestamp.now(tz=SCHEDULE_TIME_ZONE).tz_localize(None),
    }


def run_prediction(
    config: dict[str, Any],
    season: int,
    week: int,
    smoke_test: bool = False,
    allow_incomplete_history: bool = False,
) -> dict[str, Any]:
    """Project one week for every position and write the output.

    Takes the parsed config, the target season and week, whether this is a
    smoke test, and whether to allow an incomplete prior week. Returns the
    output table and everything the run summary reports.
    """
    context = build_context(config, season, week, smoke_test, allow_incomplete_history)
    history = history_before(context["tables"]["player_weeks"], season, week)
    summary = player_history_summary(history)
    roster, roster_week = latest_roster(
        context["tables"]["rosters"], season, week, config["data"]["positions"], summary["history_position"]
    )
    statuses = latest_injury_statuses(context["tables"]["injuries"], season, week)
    candidates = assemble_candidates(roster, summary, context["sides"], statuses, season, config)

    projection_parts = []
    expected_empty = {}
    for position in config["data"]["positions"]:
        projected, empty_by_design = project_position(context, candidates, position)
        expected_empty[position] = empty_by_design
        if projected is not None:
            projection_parts.append(projected)

    output = assemble_output(context, candidates, pd.concat(projection_parts, ignore_index=True))
    ceilings = position_ceilings(history, config)
    distribution = sanity_check(output, ceilings)
    check_projection_methods(output, config)
    check_intervals(output, config, position_floors(history, config), ceilings)

    return {
        "output": output,
        "distribution": distribution,
        "exclusions": exclusion_counts(candidates, output),
        "rookies_without_games": count_active_rookies(candidates),
        "freshness": data_freshness(context["tables"], config, season),
        "missing_inputs": missing_inputs(context["sides"]),
        "roster_week": roster_week,
        "history": context["history"],
        "expected_empty": expected_empty,
        "projection_methods": projection_methods(config),
        "interval_bounds": interval_bounds_by_position(config),
        "trained_through": _trained_through(config),
        "path": write_projections(output, config, season, week, smoke_test),
    }


def interval_bounds_by_position(config: dict[str, Any]) -> dict[str, list[str]]:
    """List the bounds every modelled position ships.

    Takes the parsed config. Returns the shipped bounds per position.
    """
    bounds = {}
    for position in config["data"]["positions"]:
        bounds[position] = shipped_bounds(config, position)
    return bounds


def _trained_through(config: dict[str, Any]) -> dict[str, Any]:
    """Read which week each deployment model in use was trained through.

    Takes the parsed config. Returns the week per position, or a note for a
    position that ships the baseline, whose model is not read.
    """
    trained: dict[str, Any] = {}
    for position in config["data"]["positions"]:
        if projector_for(config, position) == PROJECTOR_BASELINE:
            trained[position] = "baseline: model not used"
            continue
        _, metadata = train.load_deployment_model(config, position)
        trained[position] = metadata["trained_through"]
    return trained


def top_projections(output: pd.DataFrame, count: int) -> dict[str, pd.DataFrame]:
    """Pick the highest projections at each position for an eyeball check.

    Takes the output table and how many per position. Returns them by position.
    """
    projected = output[output["status"] == PROJECTED]
    tops = {}
    for position in sorted(projected["position"].unique().tolist()):
        rows = projected[projected["position"] == position].head(count)
        tops[position] = rows[
            [
                "player",
                "team",
                "opponent",
                "projected_points",
                "projection_method",
                "floor",
                "ceiling",
                "flag_team_changed",
                "injury_report_status",
            ]
        ]
    return tops
