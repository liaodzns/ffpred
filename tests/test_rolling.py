"""Tests for the shift-then-roll helper.

These check the arithmetic directly with hand-computed values, so that the
rolling mean is verified independently of the leakage behavior exercised in
tests/test_no_leakage.py.
"""

import pandas as pd

from ffml.features.rolling import add_rolling_features, shift_then_roll


def single_group_frame() -> pd.DataFrame:
    """Build a one-group frame with easily checked values.

    Takes nothing. Returns a frame of six rows scoring 10 through 60.
    """
    return pd.DataFrame(
        {
            "player_id": ["a", "a", "a", "a", "a", "a"],
            "points": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


def test_first_rows_are_nan_until_min_periods_is_met() -> None:
    """With a window of 3, the first three rows cannot be projected."""
    frame = single_group_frame()
    rolled = shift_then_roll(frame, "player_id", "points", window=3, min_periods=3)

    assert rolled.isna().tolist() == [True, True, True, False, False, False]


def test_rolling_means_match_hand_computed_values() -> None:
    """Each value is the mean of the three rows before it."""
    frame = single_group_frame()
    rolled = shift_then_roll(frame, "player_id", "points", window=3, min_periods=3)

    # Row 3 averages 10, 20, 30. Row 4 averages 20, 30, 40. Row 5 averages
    # 30, 40, 50. The row's own value never appears in its window.
    assert rolled.iloc[3] == 20.0
    assert rolled.iloc[4] == 30.0
    assert rolled.iloc[5] == 40.0


def test_min_periods_below_window_allows_partial_windows() -> None:
    """A lower min_periods produces averages before the window is full."""
    frame = single_group_frame()
    rolled = shift_then_roll(frame, "player_id", "points", window=3, min_periods=1)

    # Row 0 still has nothing before it. Row 1 sees only row 0, and row 2 sees
    # rows 0 and 1, so both are partial averages.
    assert pd.isna(rolled.iloc[0])
    assert rolled.iloc[1] == 10.0
    assert rolled.iloc[2] == 15.0
    assert rolled.iloc[3] == 20.0


def test_groups_do_not_bleed_into_each_other() -> None:
    """One group's history must never enter another group's window."""
    frame = pd.DataFrame(
        {
            "player_id": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "points": [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    rolled = shift_then_roll(frame, "player_id", "points", window=3, min_periods=3)

    # Player b's fourth row averages 1, 2, 3 and knows nothing of player a,
    # whose values are an order of magnitude larger.
    assert rolled.iloc[3] == 20.0
    assert rolled.iloc[7] == 2.0

    # Player b's first three rows have no history of their own.
    assert rolled.iloc[4:7].isna().all()


def test_result_is_aligned_to_the_input_index() -> None:
    """The returned Series must carry the frame's own index."""
    frame = single_group_frame()
    frame.index = [100, 101, 102, 103, 104, 105]
    rolled = shift_then_roll(frame, "player_id", "points", window=3, min_periods=3)

    assert list(rolled.index) == [100, 101, 102, 103, 104, 105]
    assert rolled.loc[103] == 20.0


def test_window_of_one_returns_the_previous_value() -> None:
    """A window of one is just the previous game, which is a useful sanity case."""
    frame = single_group_frame()
    rolled = shift_then_roll(frame, "player_id", "points", window=1, min_periods=1)

    assert pd.isna(rolled.iloc[0])
    assert rolled.iloc[1] == 10.0
    assert rolled.iloc[5] == 50.0


def multi_column_frame() -> pd.DataFrame:
    """Build a one-player frame carrying two source columns.

    Takes nothing. Returns six games with points 10 to 60 and targets 1 to 6.

    add_rolling_features sorts by player, season, and week, so those columns
    have to be present even for a single player.
    """
    return pd.DataFrame(
        {
            "player_id": ["a", "a", "a", "a", "a", "a"],
            "season": [2024, 2024, 2024, 2024, 2024, 2024],
            "week": [1, 2, 3, 4, 5, 6],
            "points": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "targets": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_add_rolling_features_names_columns_in_a_fixed_order() -> None:
    """Names run column by column, then window by window, and are returned in order.

    The order matters beyond tidiness: it is saved beside the trained model and
    the prediction code has to rebuild the columns in exactly this sequence.
    """
    frame = multi_column_frame()
    built, feature_names = add_rolling_features(
        frame, "player_id", ["points", "targets"], [3, 5]
    )

    assert feature_names == [
        "points_roll3",
        "points_roll5",
        "targets_roll3",
        "targets_roll5",
    ]

    for feature_name in feature_names:
        assert feature_name in built.columns

    # The source columns must survive untouched alongside the new ones.
    assert built["points"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


def test_each_window_holds_a_true_average_of_that_many_games() -> None:
    """A five game column is a five game mean or NaN, never a shorter average."""
    frame = multi_column_frame()
    built, _ = add_rolling_features(frame, "player_id", ["points"], [3, 5])

    # Row 3 has three prior games, so the three game window is full and the
    # five game window is not yet defined.
    assert built["points_roll3"].iloc[3] == 20.0
    assert pd.isna(built["points_roll5"].iloc[3])

    # Row 4 has four prior games. Still not five, so it stays NaN rather than
    # quietly becoming a four game average.
    assert pd.isna(built["points_roll5"].iloc[4])

    # Row 5 finally has five prior games: the mean of 10 through 50.
    assert built["points_roll5"].iloc[5] == 30.0
    assert built["points_roll3"].iloc[5] == 40.0
