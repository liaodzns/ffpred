"""Compare my own roster's players within each position for one week.

Reads the roster file, resolves every name to exactly one current player or
stops with every problem listed, then prints the latest logged projection for
the week, ranked by position, with the close-call and trust lines. It projects
nothing itself; scripts/weekly.py or predict_week.py --log writes the log.

This script stays thin. The matching lives in src/ffml/roster.py and the
comparison in src/ffml/start_sit.py.
"""

import argparse
import logging
import sys
from datetime import date
from typing import Any

import pandas as pd

from ffml import roster, start_sit, weekly
from ffml.config import ConfigError, load_config
from ffml.data import ingest
from ffml.utils.io import read_parquet, resolve_path, use_utf8_output


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Takes nothing. Returns the parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Rank my roster's players by position for one week.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--season", type=int, default=None, help="Season to compare. Defaults to the current one.")
    parser.add_argument("--week", type=int, default=None, help="Week to compare. Defaults to the current one.")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve the roster names, failing loudly on any problem. Compares nothing.",
    )
    return parser.parse_args()


def resolve_target(arguments: argparse.Namespace) -> tuple[int, int]:
    """Decide which season and week to compare.

    Takes the arguments. Returns the season and week, filling any not given
    from nflreadpy's current season and week.
    """
    season = arguments.season
    week = arguments.week
    if season is None or week is None:
        current_season, current_week = ingest.current_season_and_week()
        if season is None:
            season = current_season
        if week is None:
            week = current_week
    return int(season), int(week)


def resolve_roster_players(config: dict[str, Any], season: int, week: int) -> list[dict[str, Any]]:
    """Resolve the roster file to current players and update the cache.

    Takes the parsed config and the season and week. Returns the resolved
    players. Raises ValueError listing every name that did not resolve.
    """
    raw_directory = resolve_path(config["data"]["paths"]["raw"])
    positions = config["data"]["positions"]
    pool = roster.build_candidate_pool(
        read_parquet(raw_directory / "rosters_weekly.parquet"),
        read_parquet(raw_directory / "players.parquet"),
        season,
        week,
        positions,
    )
    entries = roster.read_roster(resolve_path(config["start_sit"]["roster_file"]), positions)
    cache_path = resolve_path(config["start_sit"]["roster_cache"])
    result = roster.resolve_roster(entries, pool, roster.load_cache(cache_path), season, date.today().isoformat())
    roster.save_cache(cache_path, result["cache"])

    for note in result["notes"]:
        print("  note: " + note)
    return result["players"]


def print_section(section: dict[str, Any]) -> None:
    """Print one position's ranking, notes, and close call.

    Takes the position section. Returns nothing.
    """
    ranked, not_projected = start_sit.section_tables(section)
    print("")
    print(section["position"])
    print("-" * 100)
    print("  trust: " + section["trust"])
    if len(ranked) == 0:
        print("  no projected players on the roster at this position")
    else:
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(ranked.to_string(index=False))
    if section["interval_note"] != "":
        print("  intervals: " + section["interval_note"])
    if len(not_projected) > 0:
        print("  not projected:")
        print(not_projected.to_string(index=False))
    print("  close call: " + section["close_call"])


def main() -> int:
    """Run the roster comparison.

    Takes nothing. Returns a process exit code, 0 on success.
    """
    use_utf8_output()
    arguments = parse_arguments()
    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print("Config error: " + str(error), file=sys.stderr)
        return 1
    logging.basicConfig(level=config["logging"]["level"].upper(), format="%(levelname)s %(name)s: %(message)s")

    try:
        season, week = resolve_target(arguments)
        players = resolve_roster_players(config, season, week)
        if arguments.resolve_only:
            print(f"All {len(players)} roster names resolved to one current player each.")
            return 0
        projections, candidates, log_name = weekly.read_latest_log(weekly.log_directory(config), season, week)
        sections = start_sit.roster_comparison(players, projections, candidates, config)
    except ValueError as error:
        print("Cannot compare the roster: " + str(error), file=sys.stderr)
        return 1

    print("")
    print("=" * 100)
    print(f"Start/sit: season {season} week {week}, from {log_name}. Players are compared within a position only.")
    print("=" * 100)
    for section in sections:
        print_section(section)
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
