"""Tests for the weekly freshness checks and the projection log."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from ffml import weekly

SEASON = 2026
WEEK = 2


def game_sides(missing_line: bool = False) -> pd.DataFrame:
    """Build two games' sides for the target week.

    Takes whether one game should lack its total. Returns the frame.
    """
    total = 44.5
    if missing_line:
        total = float("nan")
    return pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2", "g2"],
            "team": ["BUF", "MIA", "KC", "DEN"],
            "spread_line": [3.0, 3.0, -2.5, -2.5],
            "total_line": [47.5, 47.5, total, total],
        }
    )


def fresh_inputs() -> dict:
    """Build inputs that pass every freshness check.

    Takes nothing. Returns the inputs.
    """
    return {
        "ages": {"schedules": 2.0, "injuries": 2.0, "rosters_weekly": 2.0},
        "sides": game_sides(),
        "prior_week": (SEASON, 1),
        "prior_status": {"complete": True, "unplayed_games": [], "teams_missing_stats": []},
        "injury_rows": 150,
        "roster_rows": 900,
    }


def test_current_data_passes() -> None:
    """Nothing is reported when every input is fresh and complete."""
    assert weekly.freshness_problems(fresh_inputs(), SEASON, WEEK, 24) == []


def test_each_stale_or_missing_input_is_reported() -> None:
    """Every condition that makes a projection quietly worse is its own failure."""
    stale = fresh_inputs()
    stale["ages"]["injuries"] = 30.0
    assert "injuries was pulled 30 hours ago" in weekly.freshness_problems(stale, SEASON, WEEK, 24)[0]

    no_line = fresh_inputs()
    no_line["sides"] = game_sides(missing_line=True)
    assert "1 of 2 games this week have no spread or total yet: g2" in weekly.freshness_problems(no_line, SEASON, WEEK, 24)[0]

    no_games = fresh_inputs()
    no_games["sides"] = game_sides().iloc[0:0]
    assert "schedule is not current" in weekly.freshness_problems(no_games, SEASON, WEEK, 24)[0]

    incomplete = fresh_inputs()
    incomplete["prior_status"] = {"complete": False, "unplayed_games": ["KC@DEN"], "teams_missing_stats": []}
    assert "Last week (2026 week 1) has not fully arrived" in weekly.freshness_problems(incomplete, SEASON, WEEK, 24)[0]

    early = fresh_inputs()
    early["injury_rows"] = 0
    assert "injury report has no rows for 2026 week 2" in weekly.freshness_problems(early, SEASON, WEEK, 24)[0]

    no_roster = fresh_inputs()
    no_roster["roster_rows"] = 0
    assert "weekly roster for 2026 week 2" in weekly.freshness_problems(no_roster, SEASON, WEEK, 24)[0]


def test_all_problems_are_reported_together() -> None:
    """One run shows everything that needs fixing, not one problem per attempt."""
    inputs = fresh_inputs()
    inputs["ages"]["schedules"] = 99.0
    inputs["sides"] = game_sides(missing_line=True)
    inputs["injury_rows"] = 0
    assert len(weekly.freshness_problems(inputs, SEASON, WEEK, 24)) == 3


def run_result() -> dict:
    """Build a minimal projection run result to log.

    Takes nothing. Returns the result.
    """
    output = pd.DataFrame({"player_id": ["p1"], "position": ["WR"], "status": ["projected"], "projected_points": [12.5]})
    candidates = pd.DataFrame(
        {
            "player_id": ["p1", "p2"],
            "player_display_name": ["One", "Two"],
            "position": ["WR", "RB"],
            "team": ["BUF", "MIA"],
            "roster_status": ["ACT", "RES"],
            "exclusion_reason": [None, "roster_RES"],
        }
    )
    return {"output": output, "candidates": candidates, "trained_through": {"WR": [2026, 1]}}


def test_a_log_is_written_with_its_time_and_never_overwritten(tmp_path: Path) -> None:
    """A second run in the same minute refuses rather than replacing the first log."""
    logged_at = datetime(2026, 9, 18, 21, 5)
    projections_path, candidates_path = weekly.write_projection_log(run_result(), tmp_path, SEASON, WEEK, logged_at)

    assert projections_path.name == "2026_week02_20260918T2105_projections.parquet"
    logged = pd.read_parquet(projections_path)
    assert logged["logged_at"].iloc[0] == "2026-09-18 21:05"
    assert logged["trained_through"].iloc[0] == "[2026, 1]"
    assert pd.read_parquet(candidates_path)["exclusion_reason"].tolist()[1] == "roster_RES"

    with pytest.raises(ValueError, match="never overwritten"):
        weekly.write_projection_log(run_result(), tmp_path, SEASON, WEEK, logged_at)


def test_the_latest_log_for_the_week_is_read(tmp_path: Path) -> None:
    """A later run in the week supersedes an earlier one for the comparison."""
    weekly.write_projection_log(run_result(), tmp_path, SEASON, WEEK, datetime(2026, 9, 16, 9, 0))
    later = run_result()
    later["output"]["projected_points"] = [9.0]
    weekly.write_projection_log(later, tmp_path, SEASON, WEEK, datetime(2026, 9, 18, 21, 0))

    projections, candidates, name = weekly.read_latest_log(tmp_path, SEASON, WEEK)
    assert name.startswith("2026_week02_20260918T2100")
    assert projections["projected_points"].iloc[0] == 9.0
    assert len(candidates) == 2

    with pytest.raises(ValueError, match="No logged projection"):
        weekly.read_latest_log(tmp_path, SEASON, 3)
