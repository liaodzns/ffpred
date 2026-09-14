"""Downloading raw nflverse tables and saving them to data/raw/ as Parquet.

This is the only module in the project that imports nflreadpy. nflreadpy
returns Polars DataFrames; every frame is converted with .to_pandas() the
moment it arrives, so no Polars object ever leaves this file.
"""

import logging
from pathlib import Path
from typing import Any

import nflreadpy
import pandas as pd

from ffml.utils.io import resolve_path, write_parquet

logger = logging.getLogger(__name__)

# Tables this module knows how to load, in the same order as the data.tables
# block of config.yaml so that log and summary output is predictable.
KNOWN_TABLES = [
    "player_stats",
    "schedules",
    "snap_counts",
    "rosters_weekly",
    "players",
    "ff_opportunity",
    "depth_charts",
    "injuries",
    "pbp",
]


def resolve_seasons(config: dict[str, Any]) -> list[int]:
    """Work out which seasons to download.

    Takes the parsed config. Returns the inclusive list of season years to pull.

    The range begins at data.warmup_start_season when one is set, otherwise at
    data.start_season. The warmup season is downloaded like any other, and is
    dropped later in build_dataset once its only job, giving the rolling windows
    something to reach back into, is done.

    A null end_season means "through the current season". nflreadpy's
    get_current_season() does not roll over to the new year until the Thursday
    after Labor Day, so during the offseason it correctly returns the season
    just completed rather than one that has not kicked off yet.
    """
    start_season = config["data"]["start_season"]
    end_season = config["data"]["end_season"]
    if end_season is None:
        end_season = nflreadpy.get_current_season()
        logger.info("data.end_season is null, using current season %s", end_season)

    first_season = start_season
    warmup_start_season = config["data"].get("warmup_start_season")
    if warmup_start_season is not None:
        if warmup_start_season >= start_season:
            raise ValueError(
                f"data.warmup_start_season ({warmup_start_season}) must be before "
                f"data.start_season ({start_season}). A warmup season that is also a "
                "modelled season is not a warmup."
            )
        first_season = warmup_start_season
        logger.info(
            "Pulling warmup seasons %s to %s; they build features but are never modelled",
            warmup_start_season,
            start_season - 1,
        )

    seasons = []
    for season in range(first_season, end_season + 1):
        seasons.append(season)
    return seasons


def enabled_tables(config: dict[str, Any]) -> list[str]:
    """List the tables whose config toggle is set to true.

    Takes the parsed config. Returns the enabled table names in KNOWN_TABLES
    order. A name in the config that this module does not recognise is logged
    as a warning rather than ignored silently, so a typo does not quietly pull
    nothing.
    """
    toggles = config["data"]["tables"]

    for table_name in toggles:
        if table_name not in KNOWN_TABLES:
            logger.warning("Ignoring unknown table name in data.tables: %s", table_name)

    selected_tables = []
    for table_name in KNOWN_TABLES:
        if toggles.get(table_name, False):
            selected_tables.append(table_name)
    return selected_tables


def load_table(table_name: str, seasons: list[int]) -> pd.DataFrame:
    """Download one nflverse table and convert it to pandas.

    Takes the table name and the seasons to pull. Returns the table as a
    pandas DataFrame. Raises ValueError for an unrecognised name.
    """
    if table_name == "player_stats":
        # summary_level="week" is the default, but it is passed explicitly
        # because weekly rows are the backbone of this project. Season totals
        # would be useless for predicting a single game.
        polars_frame = nflreadpy.load_player_stats(seasons=seasons, summary_level="week")
    elif table_name == "schedules":
        # load_schedules defaults to every season ever played, so the season
        # list has to be passed or the pull is far larger than needed.
        polars_frame = nflreadpy.load_schedules(seasons=seasons)
    elif table_name == "snap_counts":
        polars_frame = nflreadpy.load_snap_counts(seasons=seasons)
    elif table_name == "rosters_weekly":
        polars_frame = nflreadpy.load_rosters_weekly(seasons=seasons)
    elif table_name == "players":
        # The player master table has no season dimension. It is one row per
        # player, used for identifier joins and birthdate.
        polars_frame = nflreadpy.load_players()
    elif table_name == "ff_opportunity":
        polars_frame = nflreadpy.load_ff_opportunity(
            seasons=seasons, stat_type="weekly", model_version="latest"
        )
    elif table_name == "depth_charts":
        polars_frame = nflreadpy.load_depth_charts(seasons=seasons)
    elif table_name == "injuries":
        polars_frame = nflreadpy.load_injuries(seasons=seasons)
    elif table_name == "pbp":
        raise NotImplementedError(
            "Play by play data is deferred to a later version; see the data sources section "
            "of CLAUDE.md. Set data.tables.pbp to false in config.yaml."
        )
    else:
        raise ValueError(
            "Unknown table: " + table_name + ". Known tables: " + ", ".join(KNOWN_TABLES)
        )

    return polars_frame.to_pandas()


def warn_if_incomplete(frame: pd.DataFrame, table_name: str, seasons: list[int]) -> None:
    """Log a warning if a downloaded table is empty or short the newest season.

    Takes the frame, its table name, and the seasons that were requested.
    Returns nothing.

    This warns rather than raises on purpose. Before the season opens, and in
    its first days, the newest year genuinely has no rows yet. That is normal
    football timing, not a broken download.
    """
    if len(frame) == 0:
        logger.warning("Table %s came back empty for seasons %s", table_name, seasons)
        return

    # The players master table has no season column, so there is nothing to check.
    if "season" not in frame.columns:
        return

    # The season column is not consistently typed across nflverse tables:
    # player_stats returns it as an integer, ff_opportunity as a string, and
    # depth_charts as a float with nulls. Compare numerically rather than
    # trusting the dtype, or every string keyed table looks like it is missing
    # the newest season.
    newest_requested_season = max(seasons)
    seasons_present = pd.to_numeric(frame["season"], errors="coerce").dropna().unique()
    if float(newest_requested_season) not in seasons_present:
        logger.warning(
            "Table %s has no rows for season %s. That is expected if the season has not "
            "started yet, but it can also mean the nflverse source for this table has not "
            "been updated for that season.",
            table_name,
            newest_requested_season,
        )


def save_table(frame: pd.DataFrame, table_name: str, raw_directory: Path) -> Path:
    """Save one table to the raw data directory as Parquet and log its size.

    Takes the frame, its table name, and the raw data directory. Returns the
    path that was written.
    """
    destination = raw_directory / (table_name + ".parquet")
    write_parquet(frame, destination)
    logger.info(
        "Saved %s: %s rows x %s columns -> %s",
        table_name,
        len(frame),
        len(frame.columns),
        destination,
    )
    return destination


def clear_download_cache() -> None:
    """Clear nflreadpy's local download cache.

    Takes nothing. Returns nothing. This wrapper lives here because ingest.py
    is the only module allowed to import nflreadpy, so scripts cannot call
    clear_cache directly.
    """
    nflreadpy.clear_cache()
    logger.info("Cleared the nflreadpy download cache")


def pull_all_tables(
    config: dict[str, Any],
    table_names: list[str] | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download the selected tables and write each one to data/raw.

    Takes the parsed config, an optional explicit list of table names that
    overrides the config toggles, and a force flag that clears the nflreadpy
    download cache first. Returns a dictionary of table name to DataFrame, so
    the caller can inspect columns without reading the files back off disk.
    """
    if force:
        clear_download_cache()

    if table_names is None:
        table_names = enabled_tables(config)

    seasons = resolve_seasons(config)
    raw_directory = resolve_path(config["data"]["paths"]["raw"])
    logger.info(
        "Pulling %s tables for seasons %s into %s", len(table_names), seasons, raw_directory
    )

    frames = {}
    for table_name in table_names:
        logger.info("Downloading %s", table_name)
        frame = load_table(table_name, seasons)
        warn_if_incomplete(frame, table_name, seasons)
        save_table(frame, table_name, raw_directory)
        frames[table_name] = frame

    return frames
