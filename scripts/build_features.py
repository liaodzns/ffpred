"""Build the player-week table and one feature matrix per position.

This runs two stages in order. First the raw nflverse tables are cleaned and
joined into data/processed/player_weeks.parquet, then each position's feature
matrix is built from it and saved to data/processed/features_<POSITION>.parquet.

This script stays thin. It reads the config, calls into ffml, and prints a
summary. The real work lives in src/ffml/data/clean.py and
src/ffml/features/build_dataset.py.
"""

import argparse
import logging
import sys

import pandas as pd

from ffml.config import ConfigError, load_config
from ffml.data import clean
from ffml.features import build_dataset
from ffml.utils.io import use_utf8_output

# A column whose non-null share differs by more than this between its best and
# worst modelled season is called out. A tenth is large enough to ignore
# ordinary year-to-year wobble and small enough to catch a genuinely thin season.
COVERAGE_GAP_TOLERANCE = 0.10


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Takes nothing. Returns the parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Clean the raw tables and build one feature matrix per position."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to config/config.yaml at the project root.",
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


def print_summary(frame: pd.DataFrame, target_column: str) -> None:
    """Print the shape of the player week table and its target distribution.

    Takes the player week frame and the name of the target column. Returns
    nothing.
    """
    print("")
    print("player_weeks")
    print("-" * 72)
    print(f"  rows            {len(frame):,}")
    print(f"  columns         {len(frame.columns)}")
    print(f"  seasons         {sorted(frame['season'].unique().tolist())}")
    print(f"  positions       {sorted(frame['position'].unique().tolist())}")
    print("")

    print(f"{target_column} by position")
    print("-" * 72)
    summary = frame.groupby("position")[target_column].agg(["count", "mean", "std", "max"])
    print(summary.round(2).to_string())
    print("")


def print_feature_summary(frame: pd.DataFrame, feature_names: list[str], position: str) -> None:
    """Print one position's matrix shape and how full each feature is.

    Takes the feature matrix, its ordered feature names, and the position.
    Returns nothing.

    The longer windows are NaN for players who have not played that many games
    yet, which is intended, but a feature that is mostly empty will not help.
    """
    print("")
    print(f"features_{position}")
    print("=" * 72)
    print(f"  rows            {len(frame):,}")
    print(f"  columns         {len(frame.columns)} ({len(feature_names)} of them features)")
    print("  rows by season  " + str(frame.groupby("season").size().to_dict()))
    print("")

    coverage_rows = []
    for feature_name in feature_names:
        present = int(frame[feature_name].notna().sum())
        coverage_rows.append(
            {"feature": feature_name, "present": present, "share": round(present / len(frame), 4)}
        )
    print(pd.DataFrame(coverage_rows).to_string(index=False))
    print("")


def print_season_coverage(coverage: pd.DataFrame, config: dict, position: str) -> None:
    """Print per-season coverage for one position and flag uneven columns.

    Takes the per-season coverage table, the parsed config, and the position.
    Returns nothing.

    The table includes the warmup season, so a warmup year an upstream table
    covers poorly is visible. The comparison that flags a column excludes it,
    because a first season is always thin in the longer windows.
    """
    print(f"{position}: per-season coverage, share non-null (warmup season shown)")
    print("-" * 72)
    print(coverage.to_string(index=False))
    print("")

    flagged = build_dataset.flag_uneven_coverage(coverage, COVERAGE_GAP_TOLERANCE, config)
    print(
        f"{position}: columns whose coverage varies by more than "
        f"{COVERAGE_GAP_TOLERANCE:.0%} across modelled seasons"
    )
    print("-" * 72)
    if len(flagged) == 0:
        print("  none")
    else:
        print(flagged.to_string(index=False))
    print("")


def print_rows_by_week(built: dict, config: dict) -> None:
    """Print surviving rows per week of the first modelled season, by position.

    Takes the built matrices keyed by position and the parsed config. Returns
    nothing.

    The opening weeks are what the warmup season is for. Without one they are
    empty, because no player has enough prior games yet.
    """
    start_season = config["data"]["start_season"]

    season_parts = []
    for position in built:
        frame = built[position][0]
        season_parts.append(frame[frame["season"] == start_season])
    season_rows = pd.concat(season_parts, ignore_index=True)

    print("")
    print(f"{start_season} rows by week, by position (the weeks the warmup season rescues)")
    print("-" * 72)
    counts = season_rows.groupby(["week", "position"]).size().reset_index(name="rows")
    table = counts.pivot(index="week", columns="position", values="rows")
    print(table.fillna(0).astype(int).to_string())
    print("")


def main() -> int:
    """Run the cleaning stage and then build every position's feature matrix.

    Takes nothing. Returns a process exit code, 0 on success.
    """
    use_utf8_output()
    arguments = parse_arguments()

    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print("Config error: " + str(error), file=sys.stderr)
        return 1

    configure_logging(config)
    player_weeks = clean.build_player_weeks(config)
    print_summary(player_weeks, config["scoring"]["target_column"])

    built = build_dataset.build_and_save_features(config)
    for position in built:
        features, coverage = built[position]
        feature_names = build_dataset.feature_column_names(config, position)
        print_feature_summary(features, feature_names, position)
        print_season_coverage(coverage, config, position)

    print_rows_by_week(built, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
