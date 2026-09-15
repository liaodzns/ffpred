"""Tests for scoring logged projections against actual results."""

import pandas as pd
import pytest

from ffml import monitor

TARGET = "fantasy_points_league"


def monitor_config() -> dict:
    """Build the settings the scorer reads.

    Takes nothing. Returns the config.
    """
    return {
        "data": {"positions": ["WR", "TE"]},
        "scoring": {"target_column": TARGET},
        "model": {"baseline": {"method": "rolling_mean", "window": 3}},
        "features": {"min_prior_games": 3},
    }


def log_row(player_id: str, projected: float, logged_at: str, kickoff: str) -> dict:
    """Build one logged projection row for a receiver in 2026 week 4.

    Takes the player id, the projection, the log time, and the kickoff time.
    Returns the row.
    """
    return {
        "player_id": player_id,
        "position": "WR",
        "season": 2026,
        "week": 4,
        "status": "projected",
        "projected_points": projected,
        "logged_at": logged_at,
        "kickoff_et": kickoff,
    }


def test_the_latest_projection_logged_before_kickoff_is_the_one_scored() -> None:
    """A later pre-kickoff run supersedes an earlier one; a run after kickoff never counts."""
    logs = pd.DataFrame(
        [
            log_row("p1", 10.0, "2026-09-30 09:00", "2026-10-04 13:00"),
            log_row("p1", 15.0, "2026-10-03 21:00", "2026-10-04 13:00"),
            log_row("p1", 99.0, "2026-10-04 14:00", "2026-10-04 13:00"),
            log_row("p2", 6.0, "2026-10-03 21:00", "2026-10-04 13:00"),
            log_row("p3", 7.0, "2026-10-02 09:00", "2026-10-01 20:15"),
        ]
    )
    projections, missed = monitor.pre_kickoff_projections(logs)

    by_player = projections.set_index("player_id")["projected_points"].to_dict()
    assert by_player == {"p1": 15.0, "p2": 6.0}
    assert missed == 1


def player_weeks() -> pd.DataFrame:
    """Build three prior games and the week 4 game for two receivers.

    Takes nothing. Returns the frame. p1 scored 10, 20, 30 then 12; p2 scored
    5, 5, 5 then 4.
    """
    rows = []
    for player_id, scores in [("p1", [10.0, 20.0, 30.0, 12.0]), ("p2", [5.0, 5.0, 5.0, 4.0])]:
        for week in range(1, 5):
            rows.append({"player_id": player_id, "position": "WR", "season": 2026, "week": week, TARGET: scores[week - 1]})
    return pd.DataFrame(rows)


def test_projections_are_scored_against_actuals_and_the_prior_baseline() -> None:
    """Projection errors 3 and 2 average 2.5; baseline errors 8 and 1 average 4.5.

    The baseline is the three games before week 4: 20 for p1 and 5 for p2. p3
    was projected but did not play, so he is counted and not scored.
    """
    projections = pd.DataFrame(
        [
            log_row("p1", 15.0, "2026-10-03 21:00", "2026-10-04 13:00"),
            log_row("p2", 6.0, "2026-10-03 21:00", "2026-10-04 13:00"),
            log_row("p3", 9.0, "2026-10-03 21:00", "2026-10-04 13:00"),
        ]
    )
    played, did_not_play = monitor.attach_actuals(projections, player_weeks(), TARGET)
    assert did_not_play == 1

    scored = monitor.attach_baseline(played, player_weeks(), monitor_config())
    assert scored.set_index("player_id")["baseline_points"].to_dict() == {"p1": 20.0, "p2": 5.0}

    scores = monitor.score_positions(scored, monitor_config()).set_index("position")
    assert scores.loc["WR", "rows"] == 2
    assert scores.loc["WR", "projection_mae"] == pytest.approx(2.5)
    assert scores.loc["WR", "baseline_mae"] == pytest.approx(4.5)
    assert scores.loc["WR", "mae_difference"] == pytest.approx(-2.0)
    assert scores.loc["WR", "projection_rank_correlation"] == pytest.approx(1.0)
    assert scores.loc["TE", "rows"] == 0


def test_the_holdout_reminder_names_the_season() -> None:
    """Every scoring run carries the no-decisions rule for the season it looks at."""
    assert "Nothing (models, features, projectors, intervals) may change during 2026" in monitor.holdout_note(2026)
