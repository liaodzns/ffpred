"""Measuring how far a model moves when only its random seed changes.

LightGBM samples rows and features on every boosting round whenever the bagging
and feature fractions are below one, as they are here. Two runs identical in
every respect except the seed therefore build different trees and score
differently. No parameter, feature, or data change produced that difference; it
is noise.

That noise sets a floor under every comparison in the project. An effect
smaller than the spread seeds alone produce cannot be told apart from a lucky
seed, however large its paired t, because the paired standard error counts the
sampling of rows but not the randomness of training.

Tuning is gated on this measurement. A position is only worth searching if its
seed spread is smaller than the effects a search would be looking for. In stage
nine no position cleared that bar, so this module measures and records the
floor and runs no search.
"""

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd

from ffml.models import evaluate, train

logger = logging.getLogger(__name__)

# Columns the evaluation groups by, including season so pooled folds do not
# merge the same week number from different seasons into one ranking.
METADATA_COLUMNS = ["position", "week", "season"]

SKIP = "skip"
SEARCH = "search"


def seeded_config(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Copy the config with only the random seed changed.

    Takes the parsed config and the seed. Returns a deep copy, so no run can
    alter the config the next run reads.
    """
    seeded = copy.deepcopy(config)
    seeded["project"]["random_seed"] = seed
    return seeded


def seed_sweep(
    features: pd.DataFrame,
    position: str,
    config: dict[str, Any],
    seeds: list[int],
    folds: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Train and score one position's walk-forward model once per seed.

    Takes the feature matrix, the position, the parsed config, the seeds, and
    optionally the folds to run, defaulting to the walk-forward folds. Returns
    one record per seed with its scored rows, pooled MAE, RMSE, rank
    correlation, and each fold's best iteration.

    Every run uses the same folds, the same features, and the same parameters.
    The seed is the only thing that differs.
    """
    target_column = config["scoring"]["target_column"]

    records = []
    for seed in seeds:
        seeded = seeded_config(config, seed)
        scored, results = train.score_walk_forward(features, position, seeded, folds)
        evaluation = evaluate.run_evaluation(
            scored[target_column],
            scored[train.MODEL_PROJECTION_COLUMN],
            scored[METADATA_COLUMNS],
            seeded,
        )

        best_iterations = []
        for result in results:
            best_iterations.append(result["best_iteration"])

        records.append(
            {
                "seed": seed,
                "scored": scored,
                "mae": evaluation["overall"]["mae"],
                "rmse": evaluation["overall"]["rmse"],
                "spearman": evaluation["overall"]["spearman"],
                "best_iterations": best_iterations,
            }
        )
        logger.info("%s seed %s: pooled MAE %.4f", position, seed, evaluation["overall"]["mae"])
    return records


def seed_rank(maes: list[float], seeds: list[int], seed: int) -> int:
    """Rank one seed among the others by MAE.

    Takes the per-seed MAEs, the seeds in the same order, and the seed to rank.
    Returns its rank, where 1 is the lowest MAE. Raises ValueError if the seed
    was not measured.
    """
    if seed not in seeds:
        raise ValueError(f"Seed {seed} was not among the measured seeds {seeds}.")

    own_mae = maes[seeds.index(seed)]
    better = 0
    for mae in maes:
        if mae < own_mae:
            better += 1
    return better + 1


def summarize_noise_floor(
    records: list[dict[str, Any]], target_column: str, reference_seed: int
) -> dict[str, Any]:
    """Summarise how much the seed alone moved one position's results.

    Takes the per-seed records, the target column, and the seed the shipped
    model uses. Returns the spread statistics, where the shipped seed ranks,
    and the paired comparison between the best and worst seed.

    The range, best to worst, is the bar used to decide whether a position can
    be tuned: an improvement has to beat the largest difference seed luck
    produced. The paired t between those two seeds shows how misleading a
    single-seed significance test is on its own; the two models are identical
    except for the seed.
    """
    seeds = []
    maes = []
    rmses = []
    spearmans = []
    for record in records:
        seeds.append(record["seed"])
        maes.append(record["mae"])
        rmses.append(record["rmse"])
        spearmans.append(record["spearman"])

    best = records[int(np.argmin(maes))]
    worst = records[int(np.argmax(maes))]
    comparison = evaluate.paired_error_comparison(
        best["scored"][target_column],
        best["scored"][train.MODEL_PROJECTION_COLUMN],
        worst["scored"][train.MODEL_PROJECTION_COLUMN],
    )

    return {
        "seeds": seeds,
        "maes": maes,
        "scored_rows": len(best["scored"]),
        "mae_mean": float(np.mean(maes)),
        "mae_sd": float(np.std(maes, ddof=1)),
        "mae_best": float(np.min(maes)),
        "mae_worst": float(np.max(maes)),
        "mae_range": float(np.max(maes) - np.min(maes)),
        "rmse_sd": float(np.std(rmses, ddof=1)),
        "spearman_sd": float(np.std(spearmans, ddof=1)),
        "best_seed": best["seed"],
        "worst_seed": worst["seed"],
        "worst_vs_best_t": comparison["t"],
        "worst_vs_best_se": comparison["se"],
        "reference_seed": reference_seed,
        "reference_rank": seed_rank(maes, seeds, reference_seed),
    }


def gate_verdict(summary: dict[str, Any], effect_band: list[float]) -> dict[str, str]:
    """Decide whether tuning a position could be measured at all.

    Takes the noise floor summary and the smallest and largest effects a search
    would be looking for. Returns the verdict, skip or search, and the reason.

    A position is skipped when its seed range reaches the smallest effect in
    the band. Below that, a search could in principle find an improvement larger
    than seed luck. At or above it, the ranking of candidates would largely be a
    ranking of seeds.
    """
    smallest_effect = min(effect_band)
    largest_effect = max(effect_band)
    seed_range = summary["mae_range"]
    ratio = seed_range / largest_effect

    if seed_range >= smallest_effect:
        return {
            "verdict": SKIP,
            "reason": (
                f"seed range {seed_range:.4f} is at or above the smallest effect sought "
                f"({smallest_effect:.2f}), {ratio:.1f}x the largest ({largest_effect:.2f}); "
                "candidate differences would not be separable from seed luck"
            ),
        }
    return {
        "verdict": SEARCH,
        "reason": (
            f"seed range {seed_range:.4f} is below the smallest effect sought "
            f"({smallest_effect:.2f}), so a search could detect an improvement"
        ),
    }
