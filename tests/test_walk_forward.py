"""Tests for walk-forward fold construction and the split guards.

A fold that is subtly wrong does not announce itself. The metrics still come
out, they are simply measuring something other than what was intended, so the
structure is asserted here rather than inferred from whether a run looks
plausible.
"""

import pandas as pd
import pytest

from ffml.models.train import (
    sealed_holdout_fold,
    single_holdout_fold,
    split_by_time,
    walk_forward_folds,
)

WARMUP_SEASON = 2021
START_SEASON = 2022
TEST_SEASON = 2025


def walk_forward_config() -> dict:
    """Build a config covering the split settings only.

    Takes nothing. Returns the config.
    """
    return {
        "data": {"start_season": START_SEASON, "warmup_start_season": WARMUP_SEASON},
        "validation": {
            "train_through_season": 2023,
            "early_stopping_season": 2023,
            "early_stopping_from_week": 14,
            "validation_season": 2024,
            "test_season": TEST_SEASON,
            "walk_forward": {
                "enabled": True,
                "first_test_season": 2023,
                "last_test_season": 2024,
            },
        },
    }


def synthetic_features(include_warmup: bool = False) -> pd.DataFrame:
    """Build one WR row per week of each season the feature matrix holds.

    Takes whether to include the warmup season. Returns the frame.

    The default mirrors production: build_dataset drops the warmup season, so
    features.parquet starts at start_season. The sealed test season is present,
    because it is in the file even though no split may select it.
    """
    modelled_seasons = [2022, 2023, 2024, TEST_SEASON]
    if include_warmup:
        modelled_seasons = [WARMUP_SEASON] + modelled_seasons

    seasons = []
    weeks = []
    for season in modelled_seasons:
        for week in range(1, 19):
            seasons.append(season)
            weeks.append(week)

    return pd.DataFrame({"season": seasons, "week": weeks, "position": "WR"})


def test_folds_cover_each_test_season_in_order() -> None:
    """One fold per test season, oldest first, stopping on the year before."""
    folds = walk_forward_folds(walk_forward_config())

    assert len(folds) == 2
    assert folds[0]["validation_season"] == 2023
    assert folds[1]["validation_season"] == 2024

    for fold in folds:
        # Stopping happens on the season immediately before the scored one, so
        # the scored season is never seen while choosing the round count.
        assert fold["early_stopping_season"] == fold["validation_season"] - 1
        assert fold["train_through_season"] < fold["validation_season"]


def test_splits_are_disjoint_and_ordered_in_time() -> None:
    """No row may appear in two splits, and training must precede scoring."""
    config = walk_forward_config()
    features = synthetic_features()

    for fold in walk_forward_folds(config):
        train_rows, stop_rows, scored_rows = split_by_time(features, "WR", fold, config)

        assert len(train_rows.index.intersection(stop_rows.index)) == 0
        assert len(train_rows.index.intersection(scored_rows.index)) == 0
        assert len(stop_rows.index.intersection(scored_rows.index)) == 0

        # Every training and stopping row predates every scored row.
        assert train_rows["season"].max() < scored_rows["season"].min()
        assert stop_rows["season"].max() < scored_rows["season"].min()


def test_the_warmup_season_never_reaches_a_split() -> None:
    """2021 exists to build features and must never be fitted or scored."""
    config = walk_forward_config()
    features = synthetic_features()

    for fold in walk_forward_folds(config):
        for split_rows in split_by_time(features, "WR", fold, config):
            assert not (split_rows["season"] < START_SEASON).any()


def test_the_sealed_test_season_never_reaches_a_split() -> None:
    """2025 stays untouched until the project is finished."""
    config = walk_forward_config()
    features = synthetic_features()

    for fold in walk_forward_folds(config):
        for split_rows in split_by_time(features, "WR", fold, config):
            assert not (split_rows["season"] == TEST_SEASON).any()


def test_a_warmup_row_that_slips_through_is_caught() -> None:
    """The guard must fire if the matrix still holds warmup rows.

    build_dataset drops them, so this should be unreachable. The guard exists
    because a warmup row reaching a model would look like nothing more than a
    larger training set.
    """
    config = walk_forward_config()
    leaking_features = synthetic_features(include_warmup=True)
    fold = walk_forward_folds(config)[0]

    with pytest.raises(ValueError, match="warmup"):
        split_by_time(leaking_features, "WR", fold, config)


def test_stopping_on_the_scored_season_is_rejected() -> None:
    """The leak stage six exists to remove must not be reachable by config."""
    config = walk_forward_config()
    features = synthetic_features()

    bad_fold = {
        "name": "bad",
        "train_through_season": 2024,
        "early_stopping_season": 2024,
        "early_stopping_from_week": 14,
        "validation_season": 2024,
    }
    with pytest.raises(ValueError, match="scored"):
        split_by_time(features, "WR", bad_fold, config)


def test_the_sealed_holdout_trains_through_2024_and_scores_only_2025() -> None:
    """The final fold keeps the walk-forward shape, and no 2025 row reaches fitting or stopping."""
    config = walk_forward_config()
    features = synthetic_features()
    fold = sealed_holdout_fold(config)

    train_rows, stop_rows, scored_rows = split_by_time(features, "WR", fold, config)
    sealed_index = features.index[features["season"] == TEST_SEASON]

    assert fold["sealed_holdout"] is True
    assert train_rows["season"].max() == 2024
    assert train_rows[train_rows["season"] == 2024]["week"].max() == 13
    assert stop_rows["season"].unique().tolist() == [2024]
    assert stop_rows["week"].min() == 14
    assert scored_rows["season"].unique().tolist() == [TEST_SEASON]
    assert len(scored_rows) == 18

    assert len(train_rows.index.intersection(sealed_index)) == 0
    assert len(stop_rows.index.intersection(sealed_index)) == 0


def test_a_sealed_fold_cannot_train_or_stop_on_the_sealed_season_or_score_another() -> None:
    """The sealed mark opens the scored split to the test season, and nothing else."""
    config = walk_forward_config()
    features = synthetic_features()

    trains_on_sealed = sealed_holdout_fold(config)
    trains_on_sealed["train_through_season"] = TEST_SEASON
    with pytest.raises(ValueError, match="Training must end"):
        split_by_time(features, "WR", trains_on_sealed, config)

    stops_on_sealed = sealed_holdout_fold(config)
    stops_on_sealed["train_through_season"] = TEST_SEASON
    stops_on_sealed["early_stopping_season"] = TEST_SEASON
    with pytest.raises(ValueError, match="scored"):
        split_by_time(features, "WR", stops_on_sealed, config)

    scores_another = sealed_holdout_fold(config)
    scores_another["train_through_season"] = 2023
    scores_another["early_stopping_season"] = 2023
    scores_another["validation_season"] = 2024
    with pytest.raises(ValueError, match="may only score the test season"):
        split_by_time(features, "WR", scores_another, config)


def test_scoring_the_sealed_season_without_the_sealed_mark_is_refused() -> None:
    """A fold that merely names 2025 as its scored season is still stopped by the guard."""
    config = walk_forward_config()
    features = synthetic_features()

    unmarked = sealed_holdout_fold(config)
    unmarked.pop("sealed_holdout")
    with pytest.raises(ValueError, match="sealed test season"):
        split_by_time(features, "WR", unmarked, config)


def test_the_single_holdout_fold_has_the_same_shape() -> None:
    """One split function serves both paths, so they cannot drift apart."""
    fold = single_holdout_fold(walk_forward_config())

    assert fold["validation_season"] == 2024
    assert fold["early_stopping_season"] == 2023
    assert fold["early_stopping_from_week"] == 14
    assert sorted(fold.keys()) == sorted(walk_forward_folds(walk_forward_config())[0].keys())
