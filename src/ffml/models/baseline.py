"""The naive baseline predictor: a player's mean over his last N games.

Every model built later is judged against this number. Sports models often
fail to beat a simple average, and without the baseline there is no way to
tell whether a model is genuinely good or merely complicated.

The rolling arithmetic itself lives in ffml.features.rolling, so the baseline
and the model features are computed by the same code and cannot drift apart.
"""

import logging
from typing import Any

import pandas as pd

from ffml.features.rolling import SORT_COLUMNS, shift_then_roll

logger = logging.getLogger(__name__)


def predict_baseline(
    player_weeks: pd.DataFrame,
    window: int,
    min_prior_games: int,
    target_column: str,
) -> pd.Series:
    """Project each player's mean fantasy points over his last N games.

    Takes the player-week table, the number of games in the window, the number
    of prior games required before a projection is made, and the name of the
    target column. Returns a Series of projections aligned to the input frame's
    index, with NaN for players who have too little history.
    """
    ordered = player_weeks.sort_values(SORT_COLUMNS)

    # A player's window runs across season boundaries, so his last three games
    # in week 1 are the closing games of the previous season. Those were played
    # before kickoff, so using them is not leakage, and resetting at every
    # season would leave the first three weeks of every year unprojected.
    projections = shift_then_roll(
        ordered, "player_id", target_column, window, min_prior_games
    )

    _log_missing_projections(projections, window, min_prior_games)

    # Hand back the caller's own row order, so that assigning this to a column
    # does not quietly depend on the sort performed above.
    return projections.reindex(player_weeks.index)


def _log_missing_projections(
    projections: pd.Series, window: int, min_prior_games: int
) -> None:
    """Report how many rows have no baseline because of thin history.

    Takes the projections, the window size, and the minimum prior games.
    Returns nothing.
    """
    row_count = len(projections)
    if row_count == 0:
        logger.info("Baseline over the last %s games: no rows to project", window)
        return

    missing_count = int(projections.isna().sum())
    missing_share = 100.0 * missing_count / row_count
    logger.info(
        "Baseline over the last %s games: %s of %s rows have fewer than %s prior games "
        "and are left as NaN (%.1f%% of rows)",
        window,
        missing_count,
        row_count,
        min_prior_games,
        missing_share,
    )


def run_baseline(player_weeks: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Run the baseline using the settings in the config.

    Takes the player-week table and the parsed config. Returns the projections.
    Raises ValueError for a baseline method this module does not implement.
    """
    baseline_config = config["model"]["baseline"]
    method = baseline_config["method"]
    if method != "rolling_mean":
        raise ValueError(
            f"Unsupported baseline method '{method}'. Only 'rolling_mean' is implemented."
        )

    return predict_baseline(
        player_weeks,
        baseline_config["window"],
        config["features"]["min_prior_games"],
        config["scoring"]["target_column"],
    )
