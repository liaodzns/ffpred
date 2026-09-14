"""Path handling and Parquet read and write helpers.

Every pipeline stage writes its output to disk so that later stages can be
rerun without repeating earlier work. These helpers keep the path resolution
and Parquet details in one place.

This module deliberately does not import ffml.config. config.py imports this
module for path handling, so importing it back would create a cycle.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the root directory of the project.

    Takes nothing. Returns the directory that contains config/, data/, and
    src/. This file lives at <root>/src/ffml/utils/io.py, so the root is four
    levels up from it.
    """
    return Path(__file__).resolve().parents[3]


def resolve_path(path_from_config: str) -> Path:
    """Turn a path read from config.yaml into an absolute path.

    Takes a path string such as "data/raw". Returns it unchanged if it is
    already absolute, otherwise resolved against the project root so that the
    pipeline behaves the same no matter which directory it is run from.
    """
    path = Path(path_from_config)
    if path.is_absolute():
        return path
    return project_root() / path


def ensure_directory(directory: Path) -> Path:
    """Create a directory, and its parents, if it does not already exist.

    Takes a directory path. Returns the same path, now guaranteed to exist.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_parquet(frame: pd.DataFrame, destination: Path) -> Path:
    """Write a DataFrame to a Parquet file, creating parent directories.

    Takes the frame and the destination file path. Returns the destination.

    The index is not written. Every table in this project is keyed by real
    columns such as player id, season, and week rather than by row position,
    so a saved index would only be noise on the next read.
    """
    ensure_directory(destination.parent)
    frame.to_parquet(destination, index=False)
    return destination


def read_parquet(source: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame.

    Takes the file path. Returns the frame. Raises FileNotFoundError naming
    the first pipeline script, because a missing file here nearly always means
    an earlier stage has not been run yet.
    """
    if not source.is_file():
        raise FileNotFoundError(
            f"Parquet file not found: {source}. An earlier pipeline stage has probably not "
            "been run yet. The first stage is: python scripts/pull_data.py"
        )
    return pd.read_parquet(source)
