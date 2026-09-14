"""Tests for the injury report features.

The distinction that matters most here is between a player who is not on the
report, which is a real and common value, and a player who is listed but whose
field is blank, which is genuinely unknown. Collapsing the two would either
invent missing data for every healthy player or hide real gaps.
"""

import pandas as pd

from ffml.features.injuries import (
    NOT_LISTED,
    PRACTICE_STATUS_CATEGORIES,
    REPORT_STATUS_CATEGORIES,
    add_injury_features,
)


def report_frame() -> pd.DataFrame:
    """Build one row for each case the encoding must handle.

    Takes nothing. Returns the frame, indexed 0 to 6:

    0. not on the report at all
    1. listed with no game status, practised fully
    2. listed Questionable, limited in practice
    3. listed Out, did not practise
    4. listed with no game status and a whitespace-only practice field
    5. listed as a resting veteran who did not practise
    6. listed with only a "Note" and no practice field
    """
    listed_at = pd.Timestamp("2024-09-06 20:00", tz="UTC")
    return pd.DataFrame(
        {
            "injury_report_modified": [
                pd.NaT, listed_at, listed_at, listed_at, listed_at, listed_at, listed_at,
            ],
            "report_status": [None, None, "Questionable", "Out", None, None, "Note"],
            "practice_status": [
                None,
                "Full Participation in Practice",
                "Limited Participation in Practice",
                "Did Not Participate In Practice",
                "\n    ",
                "Did Not Participate In Practice",
                None,
            ],
            "report_primary_injury": [None, "Ankle", "Hamstring", "Knee", "Toe", None, None],
            "practice_primary_injury": [
                None,
                "Ankle",
                "Hamstring",
                "Knee",
                "Toe",
                "Not injury related - resting player",
                None,
            ],
        }
    )


def test_not_being_listed_is_a_value_not_a_gap() -> None:
    """A healthy player absent from the report gets not_listed, never NaN."""
    built, _ = add_injury_features(report_frame())

    assert built["injury_report_status"].iloc[0] == NOT_LISTED
    assert built["injury_practice_status"].iloc[0] == NOT_LISTED
    assert not pd.isna(built["injury_practice_status"].iloc[0])


def test_report_statuses_map_to_their_categories() -> None:
    """Listed without a status is distinct from not listed, and Note is no designation."""
    built, _ = add_injury_features(report_frame())

    assert built["injury_report_status"].tolist() == [
        "not_listed",
        "no_designation",
        "questionable",
        "out",
        "no_designation",
        "no_designation",
        "no_designation",
    ]


def test_a_blank_practice_field_on_a_listed_player_is_unknown() -> None:
    """Whitespace or nothing in the practice field of a listed player is NaN.

    The report mentions him but does not say how he practised, so the honest
    value is missing, not a guess and not not_listed.
    """
    built, _ = add_injury_features(report_frame())
    practice = built["injury_practice_status"]

    assert practice.iloc[1] == "full"
    assert practice.iloc[2] == "limited"
    assert practice.iloc[3] == "did_not_participate"
    assert pd.isna(practice.iloc[4])
    assert pd.isna(practice.iloc[6])


def test_a_rest_day_is_flagged_apart_from_an_injury() -> None:
    """Only the resting veteran carries the not-injury-related flag.

    Rows 3 and 5 both did not practise. Only row 5 was resting, and the flag is
    what lets the model tell a planned rest day from a real injury.
    """
    built, _ = add_injury_features(report_frame())

    assert built["injury_not_injury_related"].tolist() == [0, 0, 0, 0, 0, 1, 0]
    assert built["injury_practice_status"].iloc[3] == built["injury_practice_status"].iloc[5]


def test_categories_are_fixed_even_when_absent() -> None:
    """Every level is kept, so one week of prediction rows encodes like training."""
    built, feature_names = add_injury_features(report_frame())

    # Doubtful never appears in the frame, but it must remain a level.
    assert list(built["injury_report_status"].cat.categories) == REPORT_STATUS_CATEGORIES
    assert list(built["injury_practice_status"].cat.categories) == PRACTICE_STATUS_CATEGORIES
    assert feature_names == [
        "injury_report_status",
        "injury_practice_status",
        "injury_not_injury_related",
    ]
