"""Tests for the seed noise floor and the gate that decides whether tuning is measurable."""

import pandas as pd
import pytest

from ffml.models import train, tune

TARGET = "fantasy_points_league"


def test_seeded_config_changes_only_the_seed() -> None:
    """A seeded copy differs from the original in the seed and nothing else."""
    config = {"project": {"random_seed": 42}, "model": {"lightgbm": {"shared": {"num_leaves": 31}}}}
    seeded = tune.seeded_config(config, 7)

    assert seeded["project"]["random_seed"] == 7
    assert config["project"]["random_seed"] == 42
    assert seeded["model"] == config["model"]

    # A deep copy: changing the copy's parameters cannot reach the original.
    seeded["model"]["lightgbm"]["shared"]["num_leaves"] = 7
    assert config["model"]["lightgbm"]["shared"]["num_leaves"] == 31


def seed_record(seed: int, projections: list[float], rmse: float) -> dict:
    """Build one per-seed record with hand-chosen projections.

    Takes the seed, the projections for four rows whose actual score is 10, and
    an RMSE to report. Returns the record.
    """
    scored = pd.DataFrame({TARGET: [10.0, 10.0, 10.0, 10.0], train.MODEL_PROJECTION_COLUMN: projections})
    errors = (scored[TARGET] - scored[train.MODEL_PROJECTION_COLUMN]).abs()
    return {"seed": seed, "scored": scored, "mae": float(errors.mean()), "rmse": rmse, "spearman": 0.5}


def hand_made_records() -> list[dict]:
    """Build three seeds with known MAEs.

    Takes nothing. Returns the records. Seed 42 scores MAE 0.5, seed 1 scores
    1.0, and seed 2 scores 0.25, so seed 2 is best, seed 1 worst, and seed 42
    ranks second.
    """
    return [
        seed_record(42, [9.0, 11.0, 10.0, 10.0], 1.0),
        seed_record(1, [8.0, 12.0, 10.0, 10.0], 2.0),
        seed_record(2, [10.0, 10.0, 10.0, 9.0], 3.0),
    ]


def test_summary_statistics_match_hand_computed_values() -> None:
    """Mean, spread, range, and the shipped seed's rank are computed correctly."""
    summary = tune.summarize_noise_floor(hand_made_records(), TARGET, 42)

    assert summary["mae_best"] == 0.25
    assert summary["mae_worst"] == 1.0
    assert summary["mae_range"] == 0.75
    assert summary["mae_mean"] == pytest.approx(0.583333, abs=1e-6)
    assert summary["mae_sd"] == pytest.approx(0.381881, abs=1e-6)
    assert summary["rmse_sd"] == 1.0
    assert summary["spearman_sd"] == 0.0
    assert summary["best_seed"] == 2
    assert summary["worst_seed"] == 1
    assert summary["reference_rank"] == 2


def test_best_and_worst_seeds_are_compared_row_by_row() -> None:
    """The paired comparison uses the two seeds' errors on the same rows.

    Best errors are 0, 0, 0, 1 and worst errors 2, 2, 0, 0, so the differences
    are 2, 2, 0, -1: mean 0.75, standard deviation 1.5, standard error 0.75.
    """
    summary = tune.summarize_noise_floor(hand_made_records(), TARGET, 42)

    assert summary["worst_vs_best_se"] == pytest.approx(0.75)
    assert summary["worst_vs_best_t"] == pytest.approx(1.0)


def test_ranking_a_seed_that_was_not_measured_is_refused() -> None:
    """The shipped seed has to be among those measured, or its rank means nothing."""
    with pytest.raises(ValueError, match="not among"):
        tune.seed_rank([0.5, 1.0], [1, 2], 42)


def test_the_gate_skips_when_seed_noise_reaches_the_effects_sought() -> None:
    """At or above the smallest effect in the band, a search would rank seeds, not parameters."""
    band = [0.02, 0.04]

    assert tune.gate_verdict({"mae_range": 0.157}, band)["verdict"] == tune.SKIP
    assert tune.gate_verdict({"mae_range": 0.036}, band)["verdict"] == tune.SKIP
    assert tune.gate_verdict({"mae_range": 0.02}, band)["verdict"] == tune.SKIP

    below = tune.gate_verdict({"mae_range": 0.019}, band)
    assert below["verdict"] == tune.SEARCH
    assert "below" in below["reason"]
