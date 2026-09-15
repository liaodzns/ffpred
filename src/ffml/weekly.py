"""Weekly operation: refuse to project on stale inputs, and log every projection run.

A start/sit decision is made once, before kickoff, on whatever data is on disk.
If the injury report is a day old, last week's games have not all landed, or a
betting line is missing, the projection is quietly worse and nothing on screen
says so. The checks here refuse to project at all in that state. The row flags
already in the pipeline remain for what cannot be refused, such as a Thursday
game that has already kicked off.

Every projection run is also logged with its time, and a log is never
overwritten, so what was projected before kickoff can later be scored against
what happened.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ffml.models import predict, train
from ffml.utils.io import ensure_directory, read_parquet, resolve_path

logger = logging.getLogger(__name__)

# The raw tables whose age decides whether a projection may run.
AGED_TABLES = ["schedules", "injuries", "rosters_weekly"]

LOG_TIME_FORMAT = "%Y%m%dT%H%M"
LOGGED_AT_FORMAT = "%Y-%m-%d %H:%M"
PROJECTIONS_SUFFIX = "_projections.parquet"
CANDIDATES_SUFFIX = "_candidates.parquet"
CANDIDATE_LOG_COLUMNS = ["player_id", "player_display_name", "position", "team", "roster_status", "exclusion_reason"]


def data_pull_ages(raw_directory: Path, now: datetime) -> dict[str, float]:
    """Measure how long ago each time-sensitive raw table was pulled.

    Takes the raw data directory and the current local time. Returns the age in
    hours per table, infinite for a table that has never been pulled.

    The injury report carries no timestamp in recent seasons, so the time its
    file was written is the only available statement of how current it is.
    """
    ages = {}
    for table_name in AGED_TABLES:
        path = raw_directory / f"{table_name}.parquet"
        if not path.is_file():
            ages[table_name] = float("inf")
            continue
        written = datetime.fromtimestamp(path.stat().st_mtime)
        ages[table_name] = (now - written).total_seconds() / 3600.0
    return ages


def count_week_rows(frame: pd.DataFrame, season: int, week: int) -> int:
    """Count a table's rows for one season and week.

    Takes the table and the season and week. Returns the row count.
    """
    return int(((frame["season"] == season) & (frame["week"] == week)).sum())


def _age_problems(ages: dict[str, float], max_age_hours: float) -> list[str]:
    """List the tables pulled longer ago than the limit.

    Takes the ages in hours and the limit. Returns the problems.
    """
    problems = []
    for table_name in ages:
        if ages[table_name] > max_age_hours:
            problems.append(
                f"{table_name} was pulled {ages[table_name]:.0f} hours ago, over the {max_age_hours:.0f} hour "
                "limit. Run scripts/pull_data.py."
            )
    return problems


def _game_problems(sides: pd.DataFrame, season: int, week: int) -> list[str]:
    """Check this week's games are scheduled and every one has a betting line.

    Takes the target week's game sides and the season and week. Returns the
    problems.

    The spread and total feed the implied team total, which carries what the
    betting market knows about injuries and weather. A missing line would leave
    that feature empty for both teams.
    """
    if len(sides) == 0:
        return [f"No regular season games are scheduled for {season} week {week}; the schedule is not current."]

    per_game = sides.drop_duplicates(subset=["game_id"])
    missing = per_game[per_game["spread_line"].isna() | per_game["total_line"].isna()]
    if len(missing) == 0:
        return []
    games = sorted(missing["game_id"].astype(str).tolist())
    return [f"{len(missing)} of {len(per_game)} games this week have no spread or total yet: {', '.join(games)}."]


def freshness_problems(inputs: dict[str, Any], season: int, week: int, max_age_hours: float) -> list[str]:
    """List every reason the data is not fresh enough to project the week.

    Takes the gathered inputs (table ages, game sides, the prior week and its
    completion status, and this week's injury and roster row counts), the
    target season and week, and the age limit. Returns the problems, empty
    when the data is current.
    """
    problems = _age_problems(inputs["ages"], max_age_hours)
    for problem in _game_problems(inputs["sides"], season, week):
        problems.append(problem)

    prior_season, prior_week = inputs["prior_week"]
    status = inputs["prior_status"]
    if not status["complete"]:
        problems.append(
            f"Last week ({prior_season} week {prior_week}) has not fully arrived: unplayed games "
            f"{status['unplayed_games']}, teams missing stats {status['teams_missing_stats']}."
        )
    if inputs["injury_rows"] == 0:
        problems.append(
            f"The injury report has no rows for {season} week {week} yet. Reports are published from "
            "midweek; run later in the week."
        )
    if inputs["roster_rows"] == 0:
        problems.append(f"The weekly roster for {season} week {week} has not been published yet; run later in the week.")
    return problems


def check_freshness(config: dict[str, Any], season: int, week: int, now: datetime | None = None) -> dict[str, Any]:
    """Refuse to project a week whose inputs are stale or incomplete.

    Takes the parsed config, the target season and week, and optionally the
    current local time. Returns the gathered inputs. Raises ValueError listing
    every problem at once.
    """
    if now is None:
        now = datetime.now()
    tables = predict.load_prediction_tables(config)
    schedules = tables["schedules"]
    prior_week = predict.previous_scheduled_week(schedules, season, week)

    inputs = {
        "ages": data_pull_ages(resolve_path(config["data"]["paths"]["raw"]), now),
        "sides": predict.game_sides(schedules, season, week),
        "prior_week": prior_week,
        "prior_status": train.week_completion(schedules, tables["player_weeks"], prior_week[0], prior_week[1]),
        "injury_rows": count_week_rows(tables["injuries"], season, week),
        "roster_rows": count_week_rows(tables["rosters"], season, week),
    }
    problems = freshness_problems(inputs, season, week, config["weekly"]["max_data_age_hours"])
    if len(problems) > 0:
        raise ValueError(
            f"The data is not fresh enough to project {season} week {week}:\n  " + "\n  ".join(problems)
        )
    return inputs


def log_directory(config: dict[str, Any]) -> Path:
    """Locate the projection log directory.

    Takes the parsed config. Returns the directory.
    """
    return resolve_path(config["weekly"]["log_directory"])


def _stamp(frame: pd.DataFrame, season: int, week: int, logged_at: datetime) -> None:
    """Mark logged rows with their target week and the time they were logged.

    Takes the frame, the season and week, and the log time. Returns nothing.
    """
    frame["season"] = season
    frame["week"] = week
    frame["logged_at"] = logged_at.strftime(LOGGED_AT_FORMAT)


def write_projection_log(
    result: dict[str, Any], directory: Path, season: int, week: int, logged_at: datetime
) -> tuple[Path, Path]:
    """Log one projection run's output and candidates, never overwriting an earlier log.

    Takes the run result from predict.run_prediction, the log directory, the
    target season and week, and the log time in US Eastern. Returns the paths
    of the projections log and the candidates log. Raises ValueError if a log
    for the same minute already exists.

    The candidates are logged too, so a rostered player who was not projected
    can later be shown with his reason.
    """
    ensure_directory(directory)
    stem = f"{season}_week{week:02d}_{logged_at.strftime(LOG_TIME_FORMAT)}"
    projections_path = directory / (stem + PROJECTIONS_SUFFIX)
    candidates_path = directory / (stem + CANDIDATES_SUFFIX)
    if projections_path.exists() or candidates_path.exists():
        raise ValueError(f"A projection log {stem} already exists; logs are never overwritten. Rerun in a minute.")

    projections = result["output"].copy()
    _stamp(projections, season, week, logged_at)
    trained_through = []
    for position in projections["position"]:
        trained_through.append(str(result["trained_through"].get(position)))
    projections["trained_through"] = trained_through

    candidate_columns = []
    for column in CANDIDATE_LOG_COLUMNS:
        if column in result["candidates"].columns:
            candidate_columns.append(column)
    candidates = result["candidates"][candidate_columns].copy()
    _stamp(candidates, season, week, logged_at)

    projections.to_parquet(projections_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    logger.info("Logged %s projection rows to %s", len(projections), projections_path)
    return projections_path, candidates_path


def read_latest_log(directory: Path, season: int, week: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read the most recent projection log for one week.

    Takes the log directory and the season and week. Returns the logged
    projections, the logged candidates, and the log's name. Raises ValueError if
    the week has no log.
    """
    paths = sorted(directory.glob(f"{season}_week{week:02d}_*{PROJECTIONS_SUFFIX}"))
    if len(paths) == 0:
        raise ValueError(
            f"No logged projection for {season} week {week} in {directory}. Run scripts/weekly.py first."
        )
    projections_path = paths[-1]
    candidates_path = directory / projections_path.name.replace(PROJECTIONS_SUFFIX, CANDIDATES_SUFFIX)
    return read_parquet(projections_path), read_parquet(candidates_path), projections_path.name
