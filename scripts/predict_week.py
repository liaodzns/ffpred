"""Project an upcoming week with the deployment models.

By default this projects the current week, as nflreadpy reports it. Pass
--season and --week to choose one. --smoke-test runs the whole machinery for a
completed week as a check that it works end to end; its output is not a
forecast and no accuracy is ever computed from it.

This script stays thin. It reads the config, calls into ffml, and prints a
summary. All of the real work lives in src/ffml/models/predict.py.
"""

import argparse
import logging
import sys

import pandas as pd

from ffml.config import ConfigError, load_config
from ffml.data import ingest
from ffml.models import predict

# How many players per position the eyeball check prints.
TOP_COUNT = 10


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Takes nothing. Returns the parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Project an upcoming week of fantasy points.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--season", type=int, default=None, help="Season to project.")
    parser.add_argument("--week", type=int, default=None, help="Week to project.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the machinery for a completed week. The output is not a forecast.",
    )
    parser.add_argument(
        "--allow-incomplete-history",
        action="store_true",
        help="Project even if last week's games or stats have not all arrived; flags affected players.",
    )
    return parser.parse_args()


def configure_logging(config: dict) -> None:
    """Set up log output using the level named in the config.

    Takes the parsed config. Returns nothing.
    """
    logging.basicConfig(
        level=config["logging"]["level"].upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )


def resolve_target(arguments: argparse.Namespace, config: dict) -> tuple[int, int]:
    """Decide which season and week to project.

    Takes the arguments and the parsed config. Returns the season and week.
    Raises ValueError if a smoke test does not name its week.

    The command line wins, then predict.season and predict.week in the config,
    then the current season and week from nflreadpy.
    """
    season = arguments.season
    week = arguments.week
    if season is None:
        season = config["predict"]["season"]
    if week is None:
        week = config["predict"]["week"]

    if arguments.smoke_test and (season is None or week is None):
        raise ValueError("A smoke test needs an explicit completed --season and --week.")

    if season is None or week is None:
        current_season, current_week = ingest.current_season_and_week()
        if season is None:
            season = current_season
        if week is None:
            week = current_week
    return int(season), int(week)


def print_banner(result: dict, season: int, week: int, smoke_test: bool) -> None:
    """Print the run header, including the warnings that apply to every row.

    Takes the run result, the target season and week, and whether this is a
    smoke test. Returns nothing.
    """
    print("")
    print("=" * 88)
    if smoke_test:
        print(f"SMOKE TEST: season {season} week {week}. NOT A FORECAST.")
        print("The deployment models trained on this week. This run checks the machinery only;")
        print("no accuracy is computed from it and none should be.")
    else:
        print(f"Projections: season {season} week {week}")
    print("=" * 88)

    output = result["output"]
    projected = output[output["status"] == predict.PROJECTED]
    early = int(projected["flag_early_season"].sum())
    if early > 0:
        print(f"WARNING: {early} of {len(projected)} projections are early season.")
        print("  " + predict.EARLY_SEASON_NOTE)
    print("  " + predict.DEPTH_CHART_NOTE)
    print("")


def print_data_state(result: dict) -> None:
    """Print how current the data is and what is missing from it.

    Takes the run result. Returns nothing.
    """
    freshness = result["freshness"]
    history = result["history"]
    missing = result["missing_inputs"]

    print("Data")
    print("-" * 88)
    print(f"  latest completed gameday in schedule   {freshness['latest_completed_gameday']}")
    print(f"  schedule pulled                        {freshness['schedule_pulled']}")
    print(f"  injury report pulled                   {freshness['injury_report_pulled']}  (no timestamps in source)")
    print(f"  injury rows this season                {freshness['injury_rows_this_season']}, "
          f"timestamped {freshness['injury_rows_timestamped']}, weeks {freshness['injury_weeks_this_season']}")
    print(f"  roster week used                       {result['roster_week']}")
    print(f"  models trained through                 {result['trained_through']}")
    print(f"  prior week {history['required_week']} complete     {history['complete']}")
    print(f"  games this week                        {missing['games']}")
    print(f"  games without a betting line           {missing['games_without_betting_line']}  (left NaN)")
    print(f"  games with a blank surface             {missing['games_with_blank_surface']}  (left NaN)")
    for position in result["expected_empty"]:
        if result["expected_empty"][position]:
            print(f"  {position} empty by design                    {result['expected_empty'][position]}")
    print("")


def print_exclusions(result: dict) -> None:
    """Print how many players each exclusion removed.

    Takes the run result. Returns nothing.
    """
    output = result["output"]
    projected = output[output["status"] == predict.PROJECTED]

    print("Exclusions (all candidates, and how many are listed in the output)")
    print("-" * 88)
    print(result["exclusions"].to_string(index=False))
    print(f"  active players with no NFL games, not listed: {result['rookies_without_games']}")
    print(f"  projected players with a team change: {int(projected['flag_team_changed'].sum())}")
    print(
        f"  projected players whose game has already kicked off: "
        f"{int(projected['flag_kickoff_passed'].sum())} of {len(projected)}"
    )
    print("")


def print_checks(result: dict) -> None:
    """Print the projection distribution and the top players per position.

    Takes the run result. Returns nothing.
    """
    print("Projection distribution per position (checked: no duplicates, none negative, none above ceiling)")
    print("-" * 88)
    print(result["distribution"].to_string(index=False))
    print("")

    tops = predict.top_projections(result["output"], TOP_COUNT)
    for position in tops:
        print(f"Top {TOP_COUNT} {position}")
        print("-" * 88)
        with pd.option_context("display.width", 160):
            print(tops[position].to_string(index=False))
        print("")


def main() -> int:
    """Run a projection for one week.

    Takes nothing. Returns a process exit code, 0 on success.
    """
    arguments = parse_arguments()
    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print("Config error: " + str(error), file=sys.stderr)
        return 1

    configure_logging(config)
    try:
        season, week = resolve_target(arguments, config)
        result = predict.run_prediction(
            config, season, week, arguments.smoke_test, arguments.allow_incomplete_history
        )
    except ValueError as error:
        print("Cannot project: " + str(error), file=sys.stderr)
        return 1

    print_banner(result, season, week, arguments.smoke_test)
    print_data_state(result)
    print_exclusions(result)
    print_checks(result)
    print(f"Wrote {len(result['output'])} rows to {result['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
