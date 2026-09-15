"""Run the whole weekly routine: pull, build, check, refit, project, and compare my roster.

Each step is the project's existing script, run in order, so nothing here is a
second implementation of any of them. The routine stops at the first step that
fails and says which one.

The freshness check and the roster name check run before the live models are
refit, so a stale injury report or a misspelled name fails in seconds rather
than after the slowest step. Run this late in the week; see the weekly routine in
README.md for why.

This script stays thin. The freshness checks live in src/ffml/weekly.py.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from ffml import weekly
from ffml.config import ConfigError, load_config
from ffml.data import ingest

SCRIPTS_DIRECTORY = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Takes nothing. Returns the parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Pull, build, refit, project, and compare my roster for one week.")
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--season", type=int, default=None, help="Season to project. Defaults to the current one.")
    parser.add_argument("--week", type=int, default=None, help="Week to project. Defaults to the current one.")
    return parser.parse_args()


def run_step(label: str, script_and_arguments: list[str]) -> None:
    """Run one existing pipeline script, stopping the routine if it fails.

    Takes a label for the step and the script name followed by its arguments.
    Returns nothing. Raises RuntimeError if the script exits with an error.
    """
    print("")
    print("#" * 100)
    print(f"# {label}")
    print("#" * 100)
    # Flushed so the banner lands before the step's own output when this run is
    # redirected to a file, where Python buffers its prints but the child does not.
    sys.stdout.flush()
    command = [sys.executable, str(SCRIPTS_DIRECTORY / script_and_arguments[0])] + script_and_arguments[1:]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"step '{label}' failed with exit code {completed.returncode}")


def main() -> int:
    """Run the weekly routine.

    Takes nothing. Returns a process exit code, 0 on success.
    """
    arguments = parse_arguments()
    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print("Config error: " + str(error), file=sys.stderr)
        return 1

    season = arguments.season
    week = arguments.week
    if season is None or week is None:
        current_season, current_week = ingest.current_season_and_week()
        if season is None:
            season = current_season
        if week is None:
            week = current_week

    config_arguments = []
    if arguments.config is not None:
        config_arguments = ["--config", arguments.config]
    target_arguments = ["--season", str(season), "--week", str(week)]

    try:
        run_step("1. Pull data", ["pull_data.py"] + config_arguments)
        run_step("2. Clean and build features", ["build_features.py"] + config_arguments)
        print("")
        print(f"# 3. Freshness check for {season} week {week}")
        weekly.check_freshness(config, season, week)
        print("  schedule, last week's results, injury report, weekly roster, and betting lines are current")
        run_step("4. Check roster names", ["start_sit.py", "--resolve-only"] + target_arguments + config_arguments)
        run_step("5. Refit the live models", ["train_model.py", "--deploy"] + config_arguments)
        run_step("6. Refit the live quantile models", ["train_model.py", "--deploy-quantiles"] + config_arguments)
        run_step("7. Project and log", ["predict_week.py", "--require-fresh", "--log"] + target_arguments + config_arguments)
        run_step("8. Compare my roster", ["start_sit.py"] + target_arguments + config_arguments)
    except (RuntimeError, ValueError) as error:
        print("Weekly run stopped: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
