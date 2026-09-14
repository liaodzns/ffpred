"""Loading and validation for config/config.yaml.

Every tunable setting in this project lives in the YAML file rather than in
source code, so experiments are run by editing configuration. This module
reads that file, checks the parts the pipeline depends on, and returns a plain
dictionary.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from ffml.utils.io import project_root

logger = logging.getLogger(__name__)

# Location of the config file relative to the project root.
DEFAULT_CONFIG_RELATIVE_PATH = "config/config.yaml"

# Top level sections the file is expected to contain. Only the data section is
# validated in depth, because that is all the ingest stage reads. The others
# are checked for presence so that a truncated or half edited file fails right
# away rather than several stages later.
REQUIRED_SECTIONS = [
    "project",
    "data",
    "scoring",
    "features",
    "validation",
    "model",
    "evaluation",
    "predict",
    "logging",
]

# Directory keys that data.paths must define.
REQUIRED_PATH_KEYS = [
    "raw",
    "processed",
    "predictions",
    "models",
]


class ConfigError(Exception):
    """Raised when config.yaml is missing, unreadable, or fails validation."""


def default_config_path() -> Path:
    """Return the path of the project's config file.

    Takes nothing. Returns <project root>/config/config.yaml.
    """
    return project_root() / DEFAULT_CONFIG_RELATIVE_PATH


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Read config.yaml from disk and validate it.

    Takes an optional path to the config file, defaulting to
    config/config.yaml at the project root. Returns the parsed configuration
    as a dictionary. Raises ConfigError if the file is missing, does not parse
    into a mapping, or fails validation.
    """
    if config_path is None:
        config_path = default_config_path()
    config_path = Path(config_path)

    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ConfigError(f"Config file did not parse into a mapping: {config_path}")

    validate_config(config)
    logger.debug("Loaded config from %s", config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Check that the configuration contains what the pipeline needs.

    Takes the parsed config dictionary. Returns nothing. Raises ConfigError
    describing the first problem found.
    """
    missing_sections = []
    for section_name in REQUIRED_SECTIONS:
        if section_name not in config:
            missing_sections.append(section_name)

    if missing_sections:
        raise ConfigError(
            "Config is missing required top level sections: " + ", ".join(missing_sections)
        )

    _validate_data_section(config["data"])


def _validate_data_section(data: Any) -> None:
    """Validate the data section, which is what the ingest stage reads.

    Takes the value of config["data"]. Returns nothing. Raises ConfigError if
    the season range or the position list is unusable.
    """
    if not isinstance(data, dict):
        raise ConfigError("Config section 'data' must be a mapping.")

    start_season = data.get("start_season")
    if not _is_integer(start_season):
        raise ConfigError(f"data.start_season must be an integer, got {start_season!r}.")

    # A null end_season means "through the current season". It is resolved at
    # runtime in ingest.py, which is the only module allowed to ask nflreadpy
    # which season is current.
    end_season = data.get("end_season")
    if end_season is not None:
        if not _is_integer(end_season):
            raise ConfigError(
                f"data.end_season must be an integer or null, got {end_season!r}."
            )
        if end_season < start_season:
            raise ConfigError(
                f"data.end_season ({end_season}) is before data.start_season ({start_season})."
            )

    positions = data.get("positions")
    if not isinstance(positions, list) or len(positions) == 0:
        raise ConfigError("data.positions must be a non-empty list of position codes.")

    _validate_tables(data.get("tables"))
    _validate_paths(data.get("paths"))


def _validate_tables(tables: Any) -> None:
    """Validate the data.tables download toggles.

    Takes the value of config["data"]["tables"]. Returns nothing. Raises
    ConfigError unless it is a non-empty mapping of table name to boolean.

    The set of valid table names is not checked here. ingest.py owns that,
    because it is the module that knows how to load each table.
    """
    if not isinstance(tables, dict) or len(tables) == 0:
        raise ConfigError(
            "data.tables must be a non-empty mapping of table name to true or false."
        )

    for table_name in tables:
        toggle = tables[table_name]
        if not isinstance(toggle, bool):
            raise ConfigError(f"data.tables.{table_name} must be true or false, got {toggle!r}.")


def _validate_paths(paths: Any) -> None:
    """Validate the data.paths directory settings.

    Takes the value of config["data"]["paths"]. Returns nothing. Raises
    ConfigError if a required directory key is missing or empty.
    """
    if not isinstance(paths, dict):
        raise ConfigError("data.paths must be a mapping.")

    missing_keys = []
    for path_key in REQUIRED_PATH_KEYS:
        if path_key not in paths:
            missing_keys.append(path_key)

    if missing_keys:
        raise ConfigError("data.paths is missing required keys: " + ", ".join(missing_keys))

    for path_key in REQUIRED_PATH_KEYS:
        path_value = paths[path_key]
        if not isinstance(path_value, str) or path_value.strip() == "":
            raise ConfigError(
                f"data.paths.{path_key} must be a non-empty string, got {path_value!r}."
            )


def _is_integer(value: Any) -> bool:
    """Report whether a value is a plain integer.

    Takes any value. Returns True only for integers. Python treats bool as a
    subclass of int, so without the explicit check True would pass as a valid
    season number.
    """
    if isinstance(value, bool):
        return False
    return isinstance(value, int)
