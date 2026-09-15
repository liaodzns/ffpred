"""Scoring logged projections against what happened: the in-season check.

Each week's projections are logged before kickoff. Once a week is complete, they
can be joined to the actual scores and compared with the naive baseline, which is
the question every model in this project has had to answer. Running this through
the season shows whether the deployed models are holding up.

It is a monitoring figure, not a decision rule. It describes one deployed model,
not a seed-averaged comparison, on part of a season. And the season being
monitored is the next clean holdout: nothing may change during it in response
to these numbers, or it stops being one.

Only projections logged before a player's kickoff are scored, since one made
after could reflect late information the decision did not have.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ffml import weekly
from ffml.models import evaluate, predict, train

logger = logging.getLogger(__name__)

MONITORING_NOTE = (
    "One deployed model on part of a season: a monitoring figure, not a decision rule and not a "
    "seed-averaged comparison."
)

PLAYER_WEEK_KEYS = ["season", "week", "player_id"]


def holdout_note(season: int) -> str:
    """Remind the reader that the monitored season is the next clean holdout.

    Takes the season. Returns the reminder.
    """
    return (
        f"{season} is the next clean holdout. Nothing (models, features, projectors, intervals) may change "
        f"during {season} in response to these figures, or it stops being one."
    )


def load_logs(directory: Path, season: int) -> pd.DataFrame:
    """Read every projection log for a season.

    Takes the log directory and the season. Returns all logged rows, empty when
    there are no logs.
    """
    frames = []
    for path in sorted(directory.glob(f"{season}_week*{weekly.PROJECTIONS_SUFFIX}")):
        frames.append(pd.read_parquet(path))
    if len(frames) == 0:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pre_kickoff_projections(logs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep each player-week's latest projection logged before his kickoff.

    Takes every logged row. Returns one projected row per player-week, and how
    many projected player-weeks had no log before kickoff and so are not scored.
    """
    projected = logs[logs["status"] == predict.PROJECTED].copy()
    projected["logged_time"] = pd.to_datetime(projected["logged_at"])
    projected["kickoff_time"] = pd.to_datetime(projected["kickoff_et"])

    # A missing kickoff time compares as not before, so such a row is never
    # treated as a pre-kickoff projection.
    before = projected[projected["logged_time"] < projected["kickoff_time"]]
    latest = before.sort_values("logged_time").drop_duplicates(subset=PLAYER_WEEK_KEYS, keep="last")

    all_player_weeks = projected.drop_duplicates(subset=PLAYER_WEEK_KEYS)
    missed = len(all_player_weeks) - len(latest)
    return latest.reset_index(drop=True), missed


def attach_actuals(
    projections: pd.DataFrame, player_weeks: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, int]:
    """Join each projection to the points the player actually scored.

    Takes the projections, the player-week table, and the target column.
    Returns the rows with an actual score, and how many projected players did
    not play and so have none.

    Inactive weeks are dropped when the player-week table is built, so a
    player who was projected and then sat simply has no row.
    """
    actuals = player_weeks[PLAYER_WEEK_KEYS + [target_column]]
    merged = projections.drop(columns=[target_column], errors="ignore").merge(actuals, on=PLAYER_WEEK_KEYS, how="left")
    played = merged[merged[target_column].notna()]
    return played.reset_index(drop=True), len(merged) - len(played)


def attach_baseline(scored: pd.DataFrame, player_weeks: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add the naive baseline each scored player had before his game.

    Takes the scored rows, the player-week table, and the parsed config.
    Returns the rows with a baseline_points column.

    predict.baseline_projections cuts history strictly before the week, so the
    week's own result cannot reach the baseline it is compared with.
    """
    target_column = config["scoring"]["target_column"]
    parts = []
    for _, rows in scored.groupby(["season", "week"]):
        season = int(rows["season"].iloc[0])
        week = int(rows["week"].iloc[0])
        upcoming = pd.DataFrame(
            {
                "player_id": rows["player_id"].to_numpy(),
                "position": rows["position"].to_numpy(),
                "season": season,
                "week": week,
                target_column: float("nan"),
            }
        )
        baseline = predict.baseline_projections(player_weeks, upcoming, config, season, week)
        with_baseline = rows.copy()
        with_baseline["baseline_points"] = with_baseline["player_id"].map(baseline)
        parts.append(with_baseline)
    if len(parts) == 0:
        return scored.assign(baseline_points=pd.Series(dtype=float))
    return pd.concat(parts, ignore_index=True)


def _position_score(position: str, rows: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Score one position's projections and baseline against the actual points.

    Takes the position, its scored rows, and the target column. Returns the
    row counts, both MAEs with the paired difference, and both within-week rank
    correlations.
    """
    rows = rows.reset_index(drop=True)
    actual = rows[target_column]
    comparison = evaluate.paired_error_comparison(actual, rows["baseline_points"], rows["projected_points"])
    metadata = rows[["position", "season", "week"]]
    projection_rank, _ = evaluate.compute_spearman_within_position_week(actual, rows["projected_points"], metadata)
    baseline_rank, _ = evaluate.compute_spearman_within_position_week(actual, rows["baseline_points"], metadata)
    return {
        "position": position,
        "weeks": int(rows[["season", "week"]].drop_duplicates().shape[0]),
        "rows": len(rows),
        "projection_mae": float((actual - rows["projected_points"]).abs().mean()),
        "baseline_mae": float((actual - rows["baseline_points"]).abs().mean()),
        "mae_difference": comparison["delta_mae"],
        "difference_se": comparison["se"],
        "t": comparison["t"],
        "projection_rank_correlation": projection_rank,
        "baseline_rank_correlation": baseline_rank,
    }


def score_positions(scored: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Score every position, season to date.

    Takes the scored rows with actual and baseline points, and the parsed
    config. Returns one row per position; a position with no scored rows says so.

    The MAE difference is projection minus baseline, so negative favours the
    projection. For a position that projects with the baseline the two are the
    same number.
    """
    target_column = config["scoring"]["target_column"]
    rows = []
    for position in config["data"]["positions"]:
        position_rows = scored[scored["position"] == position]
        if len(position_rows) == 0:
            rows.append({"position": position, "weeks": 0, "rows": 0})
            continue
        rows.append(_position_score(position, position_rows, target_column))
    return pd.DataFrame(rows)


def run_scoring(
    config: dict[str, Any], season: int, logs: pd.DataFrame, player_weeks: pd.DataFrame, schedules: pd.DataFrame
) -> dict[str, Any]:
    """Score every logged projection for a season's completed weeks.

    Takes the parsed config, the season, its logged rows, the player-week
    table, and the schedules. Returns the per-position scores and the counts of
    what could not be scored.
    """
    if len(logs) == 0:
        return {"logged_rows": 0}

    completed = train.completed_weeks(schedules, player_weeks, config)
    completed_keys = []
    for completed_season, completed_week in completed:
        if completed_season == season:
            completed_keys.append(completed_week)
    in_completed = logs[logs["week"].isin(completed_keys)]

    projections, missed_kickoff = pre_kickoff_projections(in_completed)
    target_column = config["scoring"]["target_column"]
    played, did_not_play = attach_actuals(projections, player_weeks, target_column)
    scored = attach_baseline(played, player_weeks, config)
    return {
        "logged_rows": len(logs),
        "completed_weeks_logged": sorted(in_completed["week"].unique().tolist()),
        "not_logged_before_kickoff": missed_kickoff,
        "did_not_play": did_not_play,
        "scores": score_positions(scored, config),
    }
