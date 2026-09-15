"""Tests for the roster start/sit comparison."""

import pandas as pd

from ffml import start_sit
from ffml.models import predict

POSITIONS = ["QB", "RB", "WR", "TE"]


def comparison_config() -> dict:
    """Build the settings the comparison reads.

    Takes nothing. Returns the config.
    """
    return {
        "data": {"positions": POSITIONS},
        "predict": {
            "projectors": {"default": "model", "by_position": {"QB": "baseline"}},
            "intervals": {"WR": {"floor": {"ship": True}, "ceiling": {"ship": True}}},
        },
        "start_sit": {
            "distinguishing_mae": {"QB": 6.9807, "RB": 4.7552, "WR": 4.7414, "TE": 3.6150},
            "trust_notes": {"QB": "qb trust", "RB": "rb trust", "WR": "wr trust", "TE": "te trust"},
        },
    }


def logged_projections() -> pd.DataFrame:
    """Build a logged projection with three receivers, a running back, and an excluded receiver.

    Takes nothing. Returns the frame.
    """
    return pd.DataFrame(
        {
            "player": ["Wide One", "Wide Two", "Wide Three", "Back One", "Hurt Receiver"],
            "player_id": ["w1", "w2", "w3", "r1", "w4"],
            "position": ["WR", "WR", "WR", "RB", "WR"],
            "team": ["BUF", "MIA", "KC", "DET", "SEA"],
            "status": ["projected", "projected", "projected", "projected", "excluded"],
            "exclusion_reason": [None, None, None, None, "ruled_out_out"],
            "projected_points": [12.0, 15.0, 8.0, 14.0, float("nan")],
            "flag_early_season": [1, 0, 0, 0, 0],
            "flag_team_changed": [0, 1, 0, 0, 0],
            "flag_kickoff_passed": [0, 0, 0, 1, 0],
        }
    )


def logged_candidates() -> pd.DataFrame:
    """Build logged candidates including a running back below the snap share filter.

    Takes nothing. Returns the frame.
    """
    return pd.DataFrame(
        {
            "player_id": ["w1", "w2", "w3", "r1", "w4", "r2"],
            "position": ["WR", "WR", "WR", "RB", "WR", "RB"],
            "team": ["BUF", "MIA", "KC", "DET", "SEA", "NYG"],
            "exclusion_reason": [None, None, None, None, "ruled_out_out", "below_snap_share"],
        }
    )


def roster_players(ids: list[str]) -> list[dict]:
    """Build resolved roster players for the given ids.

    Takes the ids. Returns the players.
    """
    positions = {"w1": "WR", "w2": "WR", "w3": "WR", "w4": "WR", "r1": "RB", "r2": "RB", "q1": "QB"}
    players = []
    for player_id in ids:
        players.append({"player_id": player_id, "player": f"name {player_id}", "position": positions[player_id], "team": "X"})
    return players


def sections_by_position(ids: list[str]) -> dict:
    """Run the comparison and key its sections by position.

    Takes the roster ids. Returns the sections.
    """
    sections = start_sit.roster_comparison(roster_players(ids), logged_projections(), logged_candidates(), comparison_config())
    keyed = {}
    for section in sections:
        keyed[section["position"]] = section
    return keyed


def test_projected_players_are_ranked_by_projection_within_position() -> None:
    """Receivers print highest projection first, regardless of roster order."""
    wide = sections_by_position(["w1", "w2", "w3"])["WR"]

    names = []
    for row in wide["projected"]:
        names.append(row["player"])
    assert names == ["Wide Two", "Wide One", "Wide Three"]
    assert wide["projected"][0]["flags"] == "team change"
    assert wide["projected"][1]["flags"] == "early season"


def test_no_rostered_player_is_dropped_and_each_gets_a_reason() -> None:
    """Listed exclusions, unlisted exclusions, and non-candidates all appear with why."""
    sections = sections_by_position(["w4", "r1", "r2", "q1"])

    assert sections["WR"]["not_projected"][0]["reason"] == "ruled_out_out"
    assert sections["RB"]["not_projected"][0]["reason"] == "below_snap_share"
    assert sections["QB"]["not_projected"][0]["reason"] == start_sit.NOT_A_CANDIDATE


def test_the_close_call_line_uses_one_mae_as_the_yardstick() -> None:
    """Three points apart is inside WR's 4.74 MAE; seven points is beyond it."""
    inside = sections_by_position(["w1", "w2"])["WR"]["close_call"]
    assert "3.0 points apart, inside one MAE (4.74)" in inside
    assert "does not distinguish" in inside

    beyond = sections_by_position(["w2", "w3"])["WR"]["close_call"]
    assert "leads Wide Three by 7.0 points, beyond one MAE (4.74)" in beyond
    assert "prefers Wide Two" in beyond

    single = sections_by_position(["r1"])["RB"]["close_call"]
    assert single.startswith("Only 1 projected RB")


def test_every_position_carries_its_trust_and_interval_notes() -> None:
    """The notes appear at the point of decision, even for a position with no rostered players."""
    sections = sections_by_position(["w1"])

    assert sections["QB"]["trust"] == "qb trust"
    assert sections["WR"]["trust"] == "wr trust"
    assert sections["WR"]["interval_note"] == ""
    assert "floor not shipped" in sections["TE"]["interval_note"]
    assert predict.EXCLUDED == "excluded"


def test_section_tables_hold_the_displayed_columns() -> None:
    """The ranked table starts at rank 1 and carries every column the decision needs."""
    ranked, not_projected = start_sit.section_tables(sections_by_position(["w1", "w2", "w4"])["WR"])

    assert ranked["rank"].tolist() == [1, 2]
    for column in start_sit.DISPLAY_COLUMNS:
        assert column in ranked.columns
    assert not_projected["reason"].tolist() == ["ruled_out_out"]
