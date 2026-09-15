"""Matching a hand-typed fantasy roster to player ids, failing loudly on anything uncertain.

A start/sit comparison is only as good as the players it compares. A name that
quietly matches the wrong Mike Williams, or silently matches nobody, would drop
or swap a player without any sign. So every typed name resolves to exactly one
player on a current NFL roster, or the run stops and lists every problem at once
with the closest candidates.

Matching is deliberately narrow:

- Only current-season weekly rosters at the modelled positions are searched.
  Many skill-position names repeat across NFL history, but very few repeat on
  one season's rosters, so this removes most ambiguity at the source.
- Names are compared after normalisation: accents removed, lowercase, hyphens as
  spaces, periods and apostrophes dropped, and a trailing Jr., Sr., II, III, IV,
  or V dropped. "A.J. Brown" and "AJ Brown" meet; so do "Audric Estimé" and
  "Audric Estime".
- An exact name is tried first: the player table's display name or the roster's
  full name. Only if nothing matches exactly are variants tried: first, football,
  or common first name with the last name, which covers nicknames.
- A match must be unique. Two players behind one name is an error, never a guess.

Resolved names are cached, keyed by the typed line, so a name is not re-resolved
every week. Each cached id is still re-checked against the current roster.
"""

import difflib
import json
import logging
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

EXACT_TIER = "exact"
VARIANT_TIER = "variant"

# Roster statuses that mean a player is no longer with a team. Injured reserve,
# inactive, and practice squad players are still on a roster, so they resolve and
# are shown with their exclusion reason instead of failing.
REMOVED_STATUSES = ["CUT", "RET"]

# Name suffixes dropped before comparing, so "Kenneth Walker III" and
# "Kenneth Walker" meet. Dropped only from names of three or more words, so a
# two-word name is never cut to one.
NAME_SUFFIXES = ["jr", "sr", "ii", "iii", "iv", "v"]

CLOSEST_CANDIDATE_COUNT = 5
CLOSEST_CANDIDATE_CUTOFF = 0.6

# Name columns read from the weekly roster and the player table. Roster copies are
# prefixed so they can sit beside the player table's own columns.
FIRST_NAME_COLUMNS = ["first_name", "football_name", "common_first_name", "roster_first_name", "roster_football_name"]
LAST_NAME_COLUMNS = ["last_name", "roster_last_name"]
EXACT_NAME_COLUMNS = ["display_name", "full_name"]


def normalize_name(name: str) -> str:
    """Reduce a player name to the form names are compared in.

    Takes the name. Returns it with accents removed, lowercased, hyphens turned
    into spaces, periods, apostrophes, and commas dropped, whitespace collapsed,
    and a trailing suffix such as Jr. or III removed.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = decomposed.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_name.lower().replace("-", " ")
    for character in [".", "'", ","]:
        lowered = lowered.replace(character, "")

    tokens = lowered.split()
    if len(tokens) > 2 and tokens[-1] in NAME_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def parse_roster_line(raw_line: str, positions: list[str]) -> dict[str, Any] | None:
    """Read one line of the roster file.

    Takes the raw line and the modelled positions. Returns the typed line, the
    name, and the optional position and team hints, or None for a blank or
    comment line. Raises ValueError for a hint with no name or two teams.

    A hint follows a vertical bar: "Mike Williams | WR PIT".
    """
    content = raw_line.split("#", 1)[0].strip()
    if content == "":
        return None

    name_part = content
    hint_part = ""
    if "|" in content:
        name_part, hint_part = content.split("|", 1)
    name = name_part.strip()
    if name == "":
        raise ValueError(f"Roster line '{content}' has a hint but no player name.")

    position = None
    team = None
    for token in hint_part.split():
        hint = token.strip().upper()
        if hint in positions:
            position = hint
        elif team is None:
            team = hint
        else:
            raise ValueError(f"Roster line '{content}' names two teams, {team} and {hint}.")
    return {"line": content, "name": name, "position": position, "team": team}


def read_roster(path: Path, positions: list[str]) -> list[dict[str, Any]]:
    """Read every player line of the roster file.

    Takes the roster file path and the modelled positions. Returns one entry per
    player line. Raises ValueError if the file is missing, lists no players, or
    lists the same player line twice.
    """
    if not path.is_file():
        raise ValueError(
            f"No roster file at {path}. Copy config/roster.example.txt to {path} and list your players."
        )

    entries = []
    seen_keys = []
    duplicates = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        entry = parse_roster_line(raw_line, positions)
        if entry is None:
            continue
        key = f"{normalize_name(entry['name'])}|{entry['position']}|{entry['team']}"
        if key in seen_keys:
            duplicates.append(entry["line"])
            continue
        seen_keys.append(key)
        entries.append(entry)

    if len(duplicates) > 0:
        raise ValueError("The roster lists the same player more than once: " + "; ".join(duplicates))
    if len(entries) == 0:
        raise ValueError(f"The roster file {path} lists no players.")
    return entries


def build_candidate_pool(
    rosters: pd.DataFrame, players: pd.DataFrame, season: int, week: int, positions: list[str]
) -> pd.DataFrame:
    """Gather every player a roster name may match.

    Takes the weekly rosters, the player table, the season and week, and the
    modelled positions. Returns one row per player on a roster at a modelled
    position in that season, from his latest roster week up to the given week,
    with every name column the matcher reads. Raises ValueError if there are none.

    Every roster status is kept, so a player on injured reserve still resolves.
    """
    in_season = rosters[
        (rosters["season"] == season) & (rosters["week"] <= week) & rosters["position"].isin(positions)
    ]
    in_season = in_season[in_season["gsis_id"].notna()]
    if len(in_season) == 0:
        raise ValueError(
            f"No {season} weekly roster rows at {', '.join(positions)} up to week {week}. Pull the data first."
        )

    latest = in_season.sort_values(["gsis_id", "week"]).drop_duplicates("gsis_id", keep="last")
    pool = latest[
        ["gsis_id", "full_name", "first_name", "last_name", "football_name", "position", "team", "status", "week"]
    ].rename(
        columns={
            "first_name": "roster_first_name",
            "last_name": "roster_last_name",
            "football_name": "roster_football_name",
            "week": "roster_week",
        }
    )
    player_names = players[["gsis_id", "display_name", "first_name", "last_name", "football_name", "common_first_name"]]
    return pool.merge(player_names, on="gsis_id", how="left").reset_index(drop=True)


def _name_values(row: dict[str, Any], columns: list[str]) -> list[str]:
    """Collect the normalised, non-empty names in the given columns of one row.

    Takes the row and the column names. Returns the names.
    """
    values = []
    for column in columns:
        value = row.get(column)
        if isinstance(value, str) and value.strip() != "":
            values.append(normalize_name(value))
    return values


def _add_to_index(index: dict[str, list[str]], key: str, gsis_id: str) -> None:
    """Record that a name key points at a player, once.

    Takes the index, the key, and the player id. Returns nothing.
    """
    if key not in index:
        index[key] = []
    if gsis_id not in index[key]:
        index[key].append(gsis_id)


def _row_dicts(pool: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn the candidate pool into one dictionary per row, keyed by column name.

    Takes the pool. Returns the rows.
    """
    rows = []
    for record in pool.to_dict("records"):
        row: dict[str, Any] = {}
        for key in record:
            row[str(key)] = record[key]
        rows.append(row)
    return rows


def name_indexes(pool: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Index every candidate by the names he may be typed as.

    Takes the candidate pool. Returns the exact-name index and the variant-name
    index, each mapping a normalised name to the player ids behind it.
    """
    exact_index: dict[str, list[str]] = {}
    variant_index: dict[str, list[str]] = {}
    for row in _row_dicts(pool):
        gsis_id = row["gsis_id"]
        for key in _name_values(row, EXACT_NAME_COLUMNS):
            _add_to_index(exact_index, key, gsis_id)
        for first_name in _name_values(row, FIRST_NAME_COLUMNS):
            for last_name in _name_values(row, LAST_NAME_COLUMNS):
                _add_to_index(variant_index, f"{first_name} {last_name}", gsis_id)
    return exact_index, variant_index


def pool_records(pool: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Key the candidate pool by player id.

    Takes the pool. Returns each player's row by id.
    """
    records = {}
    for row in _row_dicts(pool):
        records[str(row["gsis_id"])] = row
    return records


def describe_candidate(row: dict[str, Any]) -> str:
    """Describe a candidate well enough to tell him apart from a namesake.

    Takes the candidate's row. Returns the description.
    """
    name = row.get("display_name")
    if not isinstance(name, str):
        name = row.get("full_name")
    return f"{name} ({row['position']} {row['team']}, {row['status']}, {row['gsis_id']})"


def _apply_hint(ids: list[str], entry: dict[str, Any], records: dict[str, dict[str, Any]]) -> list[str]:
    """Keep the candidates that agree with the line's position and team hints.

    Takes the candidate ids, the roster entry, and the pool records. Returns the
    ids that match every hint given.
    """
    kept = []
    for gsis_id in ids:
        row = records[gsis_id]
        if entry["position"] is not None and row["position"] != entry["position"]:
            continue
        if entry["team"] is not None and row["team"] != entry["team"]:
            continue
        kept.append(gsis_id)
    return kept


def resolve_entry(
    entry: dict[str, Any],
    records: dict[str, dict[str, Any]],
    exact_index: dict[str, list[str]],
    variant_index: dict[str, list[str]],
) -> dict[str, Any]:
    """Match one roster line to a player, by exact name first and variants only if needed.

    Takes the roster entry, the pool records, and the two name indexes. Returns
    the outcome, matched, ambiguous, hint_mismatch, or unmatched, with the
    matched id and tier or the candidates involved.
    """
    key = normalize_name(entry["name"])
    tier = EXACT_TIER
    ids = exact_index.get(key, [])
    if len(ids) == 0:
        tier = VARIANT_TIER
        ids = variant_index.get(key, [])
    if len(ids) == 0:
        return {"outcome": "unmatched", "candidates": []}

    hinted = _apply_hint(ids, entry, records)
    if len(hinted) == 1:
        return {"outcome": "matched", "gsis_id": hinted[0], "tier": tier}
    if len(hinted) == 0:
        return {"outcome": "hint_mismatch", "candidates": ids}
    return {"outcome": "ambiguous", "candidates": hinted}


def closest_candidates(
    name: str,
    records: dict[str, dict[str, Any]],
    exact_index: dict[str, list[str]],
    variant_index: dict[str, list[str]],
) -> list[str]:
    """Find the players whose names are nearest to an unmatched one.

    Takes the typed name, the pool records, and the two name indexes. Returns up
    to five candidate descriptions, nearest first.
    """
    all_keys = []
    for key in exact_index:
        all_keys.append(key)
    for key in variant_index:
        all_keys.append(key)

    close_keys = difflib.get_close_matches(
        normalize_name(name), all_keys, n=CLOSEST_CANDIDATE_COUNT * 2, cutoff=CLOSEST_CANDIDATE_CUTOFF
    )
    described_ids: list[str] = []
    descriptions: list[str] = []
    for key in close_keys:
        ids = exact_index.get(key, []) + variant_index.get(key, [])
        for gsis_id in ids:
            if gsis_id in described_ids or len(descriptions) >= CLOSEST_CANDIDATE_COUNT:
                continue
            described_ids.append(gsis_id)
            descriptions.append(describe_candidate(records[gsis_id]))
    return descriptions


def _problem_text(
    entry: dict[str, Any],
    result: dict[str, Any],
    records: dict[str, dict[str, Any]],
    indexes: tuple[dict[str, list[str]], dict[str, list[str]]],
) -> str:
    """Describe why a roster line did not resolve, with the candidates to choose from.

    Takes the roster entry, its resolution result, the pool records, and the
    name indexes. Returns the problem text.
    """
    candidates = []
    for gsis_id in result["candidates"]:
        candidates.append(describe_candidate(records[gsis_id]))

    if result["outcome"] == "ambiguous":
        header = f"'{entry['line']}': {len(candidates)} players match. Add a hint such as '| WR PIT'"
    elif result["outcome"] == "hint_mismatch":
        header = f"'{entry['line']}': the name matches, but no match fits the hint"
    else:
        header = f"'{entry['line']}': no player on a current roster matches. Closest"
        candidates = closest_candidates(entry["name"], records, indexes[0], indexes[1])
        if len(candidates) == 0:
            candidates = ["(no close names)"]

    lines = [header + ":"]
    for candidate in candidates:
        lines.append("      " + candidate)
    return "\n".join(lines)


def _status_problem(entry: dict[str, Any], row: dict[str, Any] | None, gsis_id: str, season: int) -> str | None:
    """Check a resolved player is still with a team.

    Takes the roster entry, the player's pool row or None, his id, and the
    season. Returns the problem text, or None if he is still on a roster.
    """
    if row is None:
        return (
            f"'{entry['line']}': cached player {gsis_id} is not on any {season} roster at a modelled "
            "position. Update or remove the line."
        )
    if row["status"] in REMOVED_STATUSES:
        return (
            f"'{entry['line']}': {describe_candidate(row)} is listed {row['status']}, no longer with a team. "
            "Update or remove the line."
        )
    return None


def _resolve_line(
    entry: dict[str, Any],
    cached: dict[str, Any] | None,
    records: dict[str, dict[str, Any]],
    indexes: tuple[dict[str, list[str]], dict[str, list[str]]],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one roster line, from the cache when it holds the line.

    Takes the roster entry, its cache entry or None, the pool records, the name
    indexes, and the season and date. Returns the player id, tier, resolution
    date, a problem, and a note; the problem is None when the line resolved.
    """
    if cached is not None:
        gsis_id = cached["gsis_id"]
        problem = _status_problem(entry, records.get(gsis_id), gsis_id, context["season"])
        return {"gsis_id": gsis_id, "tier": cached["tier"], "resolved_on": cached["resolved_on"], "problem": problem, "note": None}

    result = resolve_entry(entry, records, indexes[0], indexes[1])
    if result["outcome"] != "matched":
        return {"problem": _problem_text(entry, result, records, indexes), "note": None}

    gsis_id = result["gsis_id"]
    note = None
    if result["tier"] == VARIANT_TIER:
        note = f"'{entry['line']}' matched {describe_candidate(records[gsis_id])} by a name variant."
    problem = _status_problem(entry, records[gsis_id], gsis_id, context["season"])
    return {"gsis_id": gsis_id, "tier": result["tier"], "resolved_on": context["today"], "problem": problem, "note": note}


def resolve_roster(
    entries: list[dict[str, Any]], pool: pd.DataFrame, cache: dict[str, Any], season: int, today: str
) -> dict[str, Any]:
    """Resolve every roster line to one player, or fail with every problem listed.

    Takes the roster entries, the candidate pool, the cache of earlier
    resolutions, the season, and today's date. Returns the resolved players, the
    updated cache holding only the current lines, and notes on variant matches.
    Raises ValueError listing every unmatched, ambiguous, or departed player.

    Nothing is dropped. Every line is resolved before anything is raised, so one
    run shows all that needs fixing.
    """
    records = pool_records(pool)
    indexes = name_indexes(pool)
    context = {"season": season, "today": today}

    resolved_players = []
    new_cache = {}
    problems = []
    notes = []
    for entry in entries:
        outcome = _resolve_line(entry, cache.get(entry["line"]), records, indexes, context)
        if outcome["note"] is not None:
            notes.append(outcome["note"])
        if outcome["problem"] is not None:
            problems.append(outcome["problem"])
            continue

        row = records[outcome["gsis_id"]]
        resolved_players.append(_resolved_player(entry, row, outcome["tier"]))
        new_cache[entry["line"]] = {
            "gsis_id": outcome["gsis_id"],
            "display_name": row.get("display_name"),
            "position": row["position"],
            "team": row["team"],
            "tier": outcome["tier"],
            "resolved_on": outcome["resolved_on"],
        }

    if len(problems) > 0:
        raise ValueError(
            f"{len(problems)} roster line(s) did not resolve to one current player. Fix them in the roster "
            "file; nothing was compared.\n  " + "\n  ".join(problems)
        )
    return {"players": resolved_players, "cache": new_cache, "notes": notes}


def _resolved_player(entry: dict[str, Any], row: dict[str, Any], tier: str) -> dict[str, Any]:
    """Describe a resolved roster player for the comparison.

    Takes the roster entry, his pool row, and the match tier. Returns the player.
    """
    name = row.get("display_name")
    if not isinstance(name, str):
        name = row.get("full_name")
    return {
        "line": entry["line"],
        "player_id": row["gsis_id"],
        "player": name,
        "position": row["position"],
        "team": row["team"],
        "roster_status": row["status"],
        "tier": tier,
    }


def load_cache(path: Path) -> dict[str, Any]:
    """Read the cache of earlier roster resolutions.

    Takes the cache path. Returns the cache, empty if the file does not exist.
    """
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    """Write the cache of roster resolutions.

    Takes the cache path and the cache. Returns nothing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Saved %s roster resolutions to %s", len(cache), path)
