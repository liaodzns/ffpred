"""Download the raw nflverse tables listed in config.yaml into data/raw/.

This script stays thin. It reads the config, calls into ffml, and prints a
summary. All of the real work lives in src/ffml/data/ingest.py.
"""

import argparse
import logging
import sys

import pandas as pd

from ffml.config import ConfigError, load_config
from ffml.data import ingest
from ffml.utils.io import use_utf8_output


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments.

    Takes nothing. Returns the parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Download raw nflverse tables into data/raw/ as Parquet."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml. Defaults to config/config.yaml at the project root.",
    )
    parser.add_argument(
        "--tables",
        default=None,
        help=(
            "Comma separated table names to pull for this run only, overriding the "
            "data.tables toggles in the config. Known tables: " + ", ".join(ingest.KNOWN_TABLES)
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Clear the nflreadpy download cache first, so every table is fetched fresh from "
            "nflverse. Parquet files in data/raw are overwritten on every run either way."
        ),
    )
    return parser.parse_args()


def parse_table_names(tables_argument: str | None) -> list[str] | None:
    """Turn the --tables argument into a list of table names.

    Takes the raw comma separated string, or None when the flag was not given.
    Returns the list of names, or None meaning "use the config toggles".
    Raises SystemExit on an unknown or empty name list.
    """
    if tables_argument is None:
        return None

    table_names = []
    for raw_name in tables_argument.split(","):
        table_name = raw_name.strip()
        if table_name == "":
            continue
        if table_name not in ingest.KNOWN_TABLES:
            raise SystemExit(
                "Unknown table: " + table_name + ". Known tables: "
                + ", ".join(ingest.KNOWN_TABLES)
            )
        table_names.append(table_name)

    if len(table_names) == 0:
        raise SystemExit("--tables was given but listed no table names.")
    return table_names


def configure_logging(config: dict) -> None:
    """Set up log output using the level named in the config.

    Takes the parsed config. Returns nothing.
    """
    logging.basicConfig(
        level=config["logging"]["level"].upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )


def print_summary(frames: dict[str, pd.DataFrame]) -> None:
    """Print one line per downloaded table with its row and column counts.

    Takes the dictionary of table name to DataFrame. Returns nothing.
    """
    print("")
    print("Downloaded tables")
    print("-" * 72)
    for table_name in frames:
        frame = frames[table_name]
        row_count = format(len(frame), ",")
        column_count = len(frame.columns)
        print(f"  {table_name:<16} {row_count:>10} rows   {column_count:>3} columns")
    print("")


def print_player_stats_columns(frame: pd.DataFrame) -> None:
    """Print every column of the player stats table alongside its dtype.

    Takes the player stats DataFrame. Returns nothing.

    This league's scoring rules do not match any prebuilt nflverse fantasy
    point column, so the target has to be built from box score columns by
    hand. The exact names have changed between schema versions, and there is
    no single fumbles lost column, so the real names are printed here to be
    checked before the scoring function is written.
    """
    print("player_stats columns (" + str(len(frame.columns)) + " total)")
    print("-" * 72)
    column_number = 1
    for column_name in frame.columns:
        column_dtype = frame[column_name].dtype
        print(f"  {column_number:>3}. {column_name:<44} {column_dtype}")
        column_number += 1
    print("")


def main() -> int:
    """Run the raw data pull.

    Takes nothing. Returns a process exit code, 0 on success.
    """
    use_utf8_output()
    arguments = parse_arguments()
    table_names = parse_table_names(arguments.tables)

    try:
        config = load_config(arguments.config)
    except ConfigError as error:
        print("Config error: " + str(error), file=sys.stderr)
        return 1

    configure_logging(config)
    frames = ingest.pull_all_tables(config, table_names=table_names, force=arguments.force)

    print_summary(frames)
    if "player_stats" in frames:
        print_player_stats_columns(frames["player_stats"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
