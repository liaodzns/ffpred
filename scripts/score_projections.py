"""Score logged projections against what happened: the in-season check.

For every completed week with a projection log, the latest projection made
before each player's kickoff is joined to his actual points and compared with
the naive baseline, position by position, season to date.

This is monitoring, not a decision rule. The season being monitored is the next
clean holdout, and nothing may change during it in response to these figures.

This script stays thin. The scoring lives in src/ffml/monitor.py.
"""

import argparse
import logging
import sys

import pandas as pd

from ffml import monitor, weekly
from ffml.config import ConfigError, load_config
from ffml.data import ingest
from ffml.models import predict
from ffml.utils.io import read_parquet, resolve_path


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Takes nothing. Returns the parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Score logged projections against actual results.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--season", type=int, default=None, help="Season to score. Defaults to the current one.")
    return parser.parse_args()


def print_report(result: dict, config: dict, season: int) -> None:
    """Print the season-to-date scores and what could not be scored.

    Takes the scoring result, the parsed config, and the season. Returns nothing.
    """
    print("")
    print("=" * 100)
    print(f"In-season check: logged projections against actual results, {season} season to date")
    print("=" * 100)
    print("  " + monitor.MONITORING_NOTE)
    print("  " + monitor.holdout_note(season))
    print(f"  completed weeks with logs:          {result['completed_weeks_logged']}")
    print(f"  player-weeks not logged before kickoff (not scored): {result['not_logged_before_kickoff']}")
    print(f"  projected players who did not play (not scored):     {result['did_not_play']}")
    print("  mae_difference is projection minus baseline; negative favours the projection.")
    for position in config["data"]["positions"]:
        if predict.projector_for(config, position) == predict.PROJECTOR_BASELINE:
            print(f"  {position} projects with the baseline, so its projection and baseline figures are the same.")
    print("")

    scores = result["scores"]
    if int(scores["rows"].sum()) == 0:
        print("  No completed week has a projection logged before kickoff yet, so nothing is scored.")
    else:
        with pd.option_context("display.width", 250, "display.max_columns", 20):
            print(scores.round(4).to_string(index=False))
    print("")


def main() -> int:
    """Run the in-season scoring.

    Takes nothing. Returns a process exit code, 0 on success.
    """
    arguments = parse_arguments()
    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print("Config error: " + str(error), file=sys.stderr)
        return 1
    logging.basicConfig(level=config["logging"]["level"].upper(), format="%(levelname)s %(name)s: %(message)s")

    season = arguments.season
    if season is None:
        season, _ = ingest.current_season_and_week()

    directory = weekly.log_directory(config)
    logs = monitor.load_logs(directory, season)
    if len(logs) == 0:
        print(f"No projection logs for {season} in {directory} yet. Run scripts/weekly.py each week; nothing to score.")
        return 0

    try:
        player_weeks = read_parquet(resolve_path(config["data"]["paths"]["processed"]) / "player_weeks.parquet")
        schedules = read_parquet(resolve_path(config["data"]["paths"]["raw"]) / "schedules.parquet")
        result = monitor.run_scoring(config, season, logs, player_weeks, schedules)
    except ValueError as error:
        print("Cannot score: " + str(error), file=sys.stderr)
        return 1

    print_report(result, config, season)
    return 0


if __name__ == "__main__":
    sys.exit(main())
