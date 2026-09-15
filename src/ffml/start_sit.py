"""Ranking my own roster's players by position, with the project's measured error as the yardstick.

The comparison projects nothing itself. It reads the logged projection for the
week and puts the user's players in order beside everything the pipeline
already says about them: the method behind each number, floor and ceiling where
those are calibrated, the injury report, and every stale or missing input flag.

Two lines per position keep the ranking honest. The close-call line uses the
position's measured MAE as the yardstick: two players within one MAE of each
other are not told apart by the model, whatever order they print in. The trust
line says how much the projection has earned at that position, copied from the
stated performance table rather than written fresh.

No lineup is built. Players are compared within a position, by hand.
"""

from typing import Any

import pandas as pd

from ffml.models import predict

NOT_A_CANDIDATE = "not a projection candidate this week"
NOT_PROJECTED = "not projected"

# Pipeline flags shown for each player, with the words they print as. The
# disabled injury features flag is left out: it is set on every row this season.
FLAG_LABELS = [
    ("flag_early_season", "early season"),
    ("flag_team_changed", "team change"),
    ("flag_no_betting_line", "no betting line"),
    ("flag_snap_data_missing", "snap data missing"),
    ("flag_prior_game_missing", "prior game missing"),
    ("flag_kickoff_passed", "kickoff passed"),
    ("flag_projection_outside_interval", "outside its interval"),
]

DISPLAY_COLUMNS = [
    "player",
    "team",
    "opponent",
    "home_away",
    "kickoff_et",
    "projected_points",
    "floor",
    "ceiling",
    "projection_method",
    "injury_report_status",
    "flags",
]
NOT_PROJECTED_COLUMNS = ["player", "team", "reason"]


def flag_text(row: dict[str, Any]) -> str:
    """List a player's raised pipeline flags in words.

    Takes the player's output row. Returns the flags joined by commas, empty if
    none is raised.
    """
    labels = []
    for column, label in FLAG_LABELS:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        if int(value) == 1:
            labels.append(label)
    return ", ".join(labels)


def _player_row(player: dict[str, Any], projections: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    """Find one rostered player in the logged projection, or say why he is not in it.

    Takes the resolved roster player, the logged projections, and the logged
    candidates. Returns his row, with a reason whenever he was not projected.
    """
    player_id = player["player_id"]
    in_output = projections[projections["player_id"] == player_id]
    if len(in_output) > 0:
        row = in_output.iloc[0].to_dict()
        row["reason"] = row.get("exclusion_reason")
        row["flags"] = flag_text(row)
        return row

    # Not in the output at all: either excluded without being listed, such as a
    # player below the snap share filter, or not a candidate this week.
    position = player["position"]
    team = player["team"]
    reason = NOT_A_CANDIDATE
    in_candidates = candidates[candidates["player_id"] == player_id]
    if len(in_candidates) > 0:
        candidate = in_candidates.iloc[0]
        position = candidate["position"]
        team = candidate["team"]
        reason = candidate["exclusion_reason"]
        if not isinstance(reason, str):
            reason = NOT_PROJECTED
    return {
        "player": player["player"],
        "player_id": player_id,
        "position": position,
        "team": team,
        "status": predict.EXCLUDED,
        "reason": reason,
        "flags": "",
    }


def roster_player_rows(
    players: list[dict[str, Any]], projections: pd.DataFrame, candidates: pd.DataFrame
) -> list[dict[str, Any]]:
    """Look up every rostered player in the logged projection.

    Takes the resolved roster players, the logged projections, and the logged
    candidates. Returns one row per player; none is dropped.
    """
    rows = []
    for player in players:
        rows.append(_player_row(player, projections, candidates))
    return rows


def _projected_points(row: dict[str, Any]) -> float:
    """Read a row's projected points, for sorting.

    Takes the row. Returns the projection.
    """
    return float(row["projected_points"])


def close_call_line(position: str, projected: list[dict[str, Any]], mae: float) -> str:
    """Say whether the model tells the top two projected players at a position apart.

    Takes the position, its projected rows in ranked order, and the position's
    measured MAE. Returns the line.

    Within one MAE, the ordering is inside the model's typical miss, so it is
    not a distinction the model can be said to make.
    """
    if len(projected) < 2:
        return f"Only {len(projected)} projected {position} on the roster: nothing to compare."

    first = projected[0]
    second = projected[1]
    gap = _projected_points(first) - _projected_points(second)
    if gap <= mae:
        return (
            f"{first['player']} and {second['player']} are {gap:.1f} points apart, inside one MAE ({mae:.2f}): "
            "the model does not distinguish them."
        )
    return (
        f"{first['player']} leads {second['player']} by {gap:.1f} points, beyond one MAE ({mae:.2f}): "
        f"the model prefers {first['player']}."
    )


def position_section(position: str, rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Build one position's part of the comparison.

    Takes the position, every rostered player's row, and the parsed config.
    Returns the ranked projected players, the players not projected with their
    reasons, the close-call line, the trust note, and the interval note.
    """
    projected = []
    not_projected = []
    for row in rows:
        if row["position"] != position:
            continue
        if row["status"] == predict.PROJECTED:
            projected.append(row)
        else:
            not_projected.append(row)
    projected = sorted(projected, key=_projected_points, reverse=True)

    start_sit_config = config["start_sit"]
    return {
        "position": position,
        "projected": projected,
        "not_projected": not_projected,
        "close_call": close_call_line(position, projected, start_sit_config["distinguishing_mae"][position]),
        "trust": start_sit_config["trust_notes"][position],
        "interval_note": predict.interval_note(config, position),
    }


def roster_comparison(
    players: list[dict[str, Any]], projections: pd.DataFrame, candidates: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compare the rostered players within each position.

    Takes the resolved roster players, the logged projections and candidates,
    and the parsed config. Returns one section per modelled position, in the
    config's position order.
    """
    rows = roster_player_rows(players, projections, candidates)
    sections = []
    for position in config["data"]["positions"]:
        sections.append(position_section(position, rows, config))
    return sections


def section_tables(section: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lay out one section's projected and not-projected players as tables.

    Takes the section. Returns the ranked table and the not-projected table.
    """
    projected_rows = []
    for rank, row in enumerate(section["projected"], start=1):
        table_row: dict[str, Any] = {"rank": rank}
        for column in DISPLAY_COLUMNS:
            table_row[column] = row.get(column)
        projected_rows.append(table_row)

    not_projected_rows = []
    for row in section["not_projected"]:
        table_row = {}
        for column in NOT_PROJECTED_COLUMNS:
            table_row[column] = row.get(column)
        not_projected_rows.append(table_row)
    return pd.DataFrame(projected_rows), pd.DataFrame(not_projected_rows)
