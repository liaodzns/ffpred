"""Tests for loading and validating config/config.yaml."""

from pathlib import Path

import pytest
import yaml

from ffml.config import ConfigError, default_config_path, load_config
from ffml.features.build_dataset import feature_column_names, resolve_groups
from ffml.features.game_context import GAME_CONTEXT_FEATURES
from ffml.features.opponent import OPPONENT_STRENGTH_FEATURE
from ffml.features.rolling import GAMES_PLAYED_COLUMN, rolling_feature_names


def read_real_config_dict() -> dict:
    """Read the project's own config.yaml without validating it.

    Takes nothing. Returns the parsed dictionary, used as a starting point for
    building deliberately broken variants.
    """
    with open(default_config_path(), encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def write_config(directory: Path, config: dict) -> Path:
    """Write a config dictionary to a temporary YAML file.

    Takes a directory and the config dictionary. Returns the path written.
    """
    config_path = directory / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file)
    return config_path


def test_real_config_loads() -> None:
    """The config file committed to the repository must load and validate."""
    config = load_config()
    assert isinstance(config["data"]["start_season"], int)
    assert len(config["data"]["positions"]) > 0
    assert isinstance(config["data"]["tables"]["player_stats"], bool)
    assert config["data"]["paths"]["raw"] != ""


def test_null_end_season_is_allowed(tmp_path: Path) -> None:
    """A null end_season means "through the current season" and must validate."""
    config = read_real_config_dict()
    config["data"]["end_season"] = None
    config_path = write_config(tmp_path, config)

    loaded_config = load_config(config_path)
    assert loaded_config["data"]["end_season"] is None


def test_missing_section_is_rejected(tmp_path: Path) -> None:
    """Dropping a required top level section must raise ConfigError."""
    config = read_real_config_dict()
    del config["scoring"]
    config_path = write_config(tmp_path, config)

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_end_season_before_start_season_is_rejected(tmp_path: Path) -> None:
    """A backwards season range must raise ConfigError rather than pull nothing."""
    config = read_real_config_dict()
    config["data"]["start_season"] = 2022
    config["data"]["end_season"] = 2021
    config_path = write_config(tmp_path, config)

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_non_boolean_table_toggle_is_rejected(tmp_path: Path) -> None:
    """A table toggle that is not true or false must raise ConfigError."""
    config = read_real_config_dict()
    config["data"]["tables"]["player_stats"] = "yes"
    config_path = write_config(tmp_path, config)

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_missing_path_key_is_rejected(tmp_path: Path) -> None:
    """Dropping a required data.paths directory must raise ConfigError."""
    config = read_real_config_dict()
    del config["data"]["paths"]["raw"]
    config_path = write_config(tmp_path, config)

    with pytest.raises(ConfigError):
        load_config(config_path)


def test_missing_config_file_is_rejected(tmp_path: Path) -> None:
    """Pointing at a file that does not exist must raise ConfigError."""
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")


# ---------------------------------------------------------------------------
# Per-position feature groups and columns
# ---------------------------------------------------------------------------


def groups_config(by_position: dict) -> dict:
    """Build a config holding only a groups block.

    Takes the by_position overrides. Returns the config.
    """
    return {
        "features": {
            "groups": {
                "default": {"game_context": True, "injuries": False},
                "by_position": by_position,
            }
        }
    }


def test_a_position_with_no_override_gets_the_default() -> None:
    """An empty override inherits every group from the default."""
    resolved = resolve_groups(groups_config({"QB": {}}), "QB")
    assert resolved == {"game_context": True, "injuries": False}


def test_an_override_changes_only_its_own_group_and_position() -> None:
    """Turning injuries on for QB must leave QB's other groups and every other position alone."""
    config = groups_config({"QB": {"injuries": True}, "RB": {}})

    assert resolve_groups(config, "QB") == {"game_context": True, "injuries": True}
    assert resolve_groups(config, "RB") == {"game_context": True, "injuries": False}


def test_an_override_naming_an_unknown_group_is_rejected() -> None:
    """A misspelt group must raise rather than silently leave the real one at its default."""
    with pytest.raises(ValueError, match="does not define"):
        resolve_groups(groups_config({"QB": {"injury": True}}), "QB")


def test_wr_keeps_exactly_its_stage_six_features() -> None:
    """Restructuring the config must not move a single WR feature or its order."""
    config = load_config()

    stage_six_sources = [
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "fantasy_points_league",
        "offense_pct",
        "target_share",
        "total_fantasy_points_exp",
        "rec_touchdown_exp",
    ]
    expected = rolling_feature_names(stage_six_sources, config["features"]["rolling_windows"])
    expected.append(GAMES_PLAYED_COLUMN)
    for feature_name in GAME_CONTEXT_FEATURES:
        expected.append(feature_name)
    expected.append(OPPONENT_STRENGTH_FEATURE)

    assert feature_column_names(config, "WR") == expected


def test_each_position_rolls_only_columns_that_apply_to_it() -> None:
    """No quarterback receiving columns, and no pass-catcher passing columns."""
    config = load_config()

    quarterback_features = feature_column_names(config, "QB")
    for feature_name in quarterback_features:
        assert not feature_name.startswith("target"), feature_name
        assert not feature_name.startswith("receiving"), feature_name
        assert not feature_name.startswith("receptions"), feature_name

    for position in ["RB", "WR", "TE"]:
        for feature_name in feature_column_names(config, position):
            assert not feature_name.startswith("attempts"), (position, feature_name)
            assert not feature_name.startswith("passing"), (position, feature_name)

    # Every position keeps the target's own roll, which is the baseline cross-check.
    for position in config["data"]["positions"]:
        assert "fantasy_points_league_roll3" in feature_column_names(config, position)
