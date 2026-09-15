"""Tests for matching a hand-typed roster to player ids."""

from pathlib import Path

import pandas as pd
import pytest

from ffml import roster

POSITIONS = ["QB", "RB", "WR", "TE"]
SEASON = 2026
TODAY = "2026-09-15"


def synthetic_rosters() -> pd.DataFrame:
    """Build one week of weekly roster rows with the name variations seen in real data.

    Takes nothing. Returns the frame.
    """
    rows = [
        ("a1", "A.J. Brown", "A.J.", "Brown", "A.J.", "WR", "PHI", "ACT"),
        ("a2", "Tutu Atwell", "Chatarius", "Atwell", "Tutu", "WR", "LA", "ACT"),
        ("a3", "Mike Williams", "Mike", "Williams", "Mike", "WR", "PIT", "ACT"),
        ("a4", "Mike Williams", "Mike", "Williams", "Mike", "WR", "NYJ", "ACT"),
        ("a5", "LeQuint Allen Jr.", "LeQuint", "Allen", "LeQuint", "RB", "JAX", "ACT"),
        ("a6", "Audric Estimé", "Audric", "Estimé", "Audric", "RB", "DEN", "RES"),
        ("a7", "Jaxon Smith-Njigba", "Jaxon", "Smith-Njigba", "Jaxon", "WR", "SEA", "ACT"),
        ("a8", "Released Receiver", "Released", "Receiver", "Released", "WR", "FA", "CUT"),
        ("a9", "De'Von Achane", "De'Von", "Achane", "Devon", "RB", "MIA", "ACT"),
    ]
    records = []
    for gsis_id, full_name, first_name, last_name, football_name, position, team, status in rows:
        records.append(
            {
                "season": SEASON,
                "week": 1,
                "gsis_id": gsis_id,
                "full_name": full_name,
                "first_name": first_name,
                "last_name": last_name,
                "football_name": football_name,
                "position": position,
                "team": team,
                "status": status,
            }
        )
    return pd.DataFrame(records)


def synthetic_players() -> pd.DataFrame:
    """Build the player table rows for the synthetic roster.

    Takes nothing. Returns the frame.
    """
    rosters = synthetic_rosters()
    return pd.DataFrame(
        {
            "gsis_id": rosters["gsis_id"],
            "display_name": rosters["full_name"],
            "first_name": rosters["first_name"],
            "last_name": rosters["last_name"],
            "football_name": rosters["football_name"],
            "common_first_name": rosters["football_name"],
        }
    )


def pool() -> pd.DataFrame:
    """Build the candidate pool from the synthetic tables.

    Takes nothing. Returns the pool.
    """
    return roster.build_candidate_pool(synthetic_rosters(), synthetic_players(), SEASON, 1, POSITIONS)


def entries(lines: list[str]) -> list[dict]:
    """Parse roster lines the way the roster file is read.

    Takes the typed lines. Returns the entries.
    """
    parsed = []
    for line in lines:
        entry = roster.parse_roster_line(line, POSITIONS)
        if entry is not None:
            parsed.append(entry)
    return parsed


def resolve(lines: list[str], cache: dict | None = None) -> dict:
    """Resolve typed lines against the synthetic pool.

    Takes the lines and an optional cache. Returns the resolution.
    """
    if cache is None:
        cache = {}
    return roster.resolve_roster(entries(lines), pool(), cache, SEASON, TODAY)


def resolved_ids(result: dict) -> list[str]:
    """List the player ids a resolution produced, in roster order.

    Takes the resolution. Returns the ids.
    """
    ids = []
    for player in result["players"]:
        ids.append(player["player_id"])
    return ids


def test_names_normalise_suffixes_initials_apostrophes_accents_and_hyphens() -> None:
    """The variations real rosters carry all reduce to one comparable form."""
    assert roster.normalize_name("A.J. Brown") == "aj brown"
    assert roster.normalize_name("LeQuint Allen Jr.") == "lequint allen"
    assert roster.normalize_name("Kenneth Walker III") == "kenneth walker"
    assert roster.normalize_name("De'Von Achane") == "devon achane"
    assert roster.normalize_name("Audric Estimé") == "audric estime"
    assert roster.normalize_name("  Jaxon   Smith-Njigba ") == "jaxon smith njigba"


def test_exact_unique_names_resolve_without_comment() -> None:
    """Punctuation, suffix, and accent differences still count as the exact name."""
    result = resolve(["AJ Brown", "Jaxon Smith Njigba", "LeQuint Allen", "Audric Estime"])

    assert resolved_ids(result) == ["a1", "a7", "a5", "a6"]
    assert result["notes"] == []


def test_a_nickname_resolves_through_the_variant_tier_with_a_note() -> None:
    """Tutu Atwell's given name matches only by first name and last name, which is noted."""
    result = resolve(["Chatarius Atwell"])

    assert resolved_ids(result) == ["a2"]
    assert len(result["notes"]) == 1
    assert "Tutu Atwell" in result["notes"][0]


def test_a_hint_resolves_two_players_with_one_name() -> None:
    """Position and team hints pick one Mike Williams out of two."""
    result = resolve(["Mike Williams | WR PIT"])
    assert resolved_ids(result) == ["a3"]


def test_an_ambiguous_name_fails_and_lists_every_candidate() -> None:
    """Two Mike Williams on current rosters is never guessed at."""
    with pytest.raises(ValueError) as raised:
        resolve(["Mike Williams"])

    message = str(raised.value)
    assert "2 players match" in message
    assert "PIT" in message and "NYJ" in message


def test_an_unmatched_name_fails_with_its_closest_candidates() -> None:
    """A misspelling is reported with the player it was probably meant to be."""
    with pytest.raises(ValueError) as raised:
        resolve(["Jaxon Smith Njiba"])

    assert "no player on a current roster matches" in str(raised.value)
    assert "Jaxon Smith-Njigba" in str(raised.value)


def test_every_problem_is_reported_in_one_error_and_nothing_is_dropped() -> None:
    """A good line does not hide the bad ones, and a bad line is never silently skipped."""
    with pytest.raises(ValueError) as raised:
        resolve(["AJ Brown", "Mike Williams", "Jaxon Smith Njiba"])

    message = str(raised.value)
    assert message.startswith("2 roster line(s)")
    assert "'Mike Williams'" in message
    assert "'Jaxon Smith Njiba'" in message


def test_a_player_who_is_no_longer_with_a_team_fails() -> None:
    """A released player resolves by name but cannot be compared."""
    with pytest.raises(ValueError, match="listed CUT"):
        resolve(["Released Receiver"])


def test_the_same_player_twice_in_the_roster_file_fails(tmp_path: Path) -> None:
    """A duplicated line would double-count a player in the comparison."""
    roster_file = tmp_path / "roster.txt"
    roster_file.write_text("# my team\nAJ Brown\n\nA.J. Brown\n", encoding="utf-8")

    with pytest.raises(ValueError, match="more than once"):
        roster.read_roster(roster_file, POSITIONS)


def test_a_line_naming_two_teams_is_refused() -> None:
    """A hint may give one position and one team."""
    with pytest.raises(ValueError, match="two teams"):
        roster.parse_roster_line("Mike Williams | WR PIT NYJ", POSITIONS)


def test_a_cached_line_is_used_without_being_resolved_again() -> None:
    """The cache answers for its line even when the typed name would no longer match on its own."""
    cache = {"JSN": {"gsis_id": "a7", "tier": "exact", "resolved_on": "2026-09-01"}}
    result = resolve(["JSN"], cache)

    assert resolved_ids(result) == ["a7"]
    assert result["cache"]["JSN"]["resolved_on"] == "2026-09-01"


def test_a_cached_player_who_left_a_roster_or_was_released_fails() -> None:
    """The cache is re-verified every run against the current roster."""
    gone = {"Old Name": {"gsis_id": "zz", "tier": "exact", "resolved_on": "2026-09-01"}}
    with pytest.raises(ValueError, match="not on any 2026 roster"):
        resolve(["Old Name"], gone)

    released = {"Released Receiver": {"gsis_id": "a8", "tier": "exact", "resolved_on": "2026-09-01"}}
    with pytest.raises(ValueError, match="listed CUT"):
        resolve(["Released Receiver"], released)


def test_an_edited_line_is_resolved_afresh_and_the_old_entry_is_pruned() -> None:
    """The cache is keyed by the typed line, so an edit is a new lookup."""
    cache = {"AJ Brown": {"gsis_id": "a1", "tier": "exact", "resolved_on": "2026-09-01"}}
    result = resolve(["A.J. Brown | WR"], cache)

    assert resolved_ids(result) == ["a1"]
    assert list(result["cache"].keys()) == ["A.J. Brown | WR"]
    assert result["cache"]["A.J. Brown | WR"]["resolved_on"] == TODAY


def test_an_injured_reserve_player_still_resolves() -> None:
    """He is on a roster, so he is shown with his exclusion reason rather than failing to match."""
    result = resolve(["Audric Estimé"])
    assert result["players"][0]["roster_status"] == "RES"
