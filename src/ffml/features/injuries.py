"""The weekly injury report, turned into features.

NFL clubs publish an injury report through the week: practice participation
from each practice session, and a game status on the final report, normally
Friday. It is the one signal in this project that describes a player's
condition for the upcoming game rather than his past games.

Leakage. The report for week N is published before week N kicks off, so unlike
a box score it may be used as of the current week. clean.py enforces that
strictly: it keeps only the latest version of each report whose date_modified
is before that game's kickoff, so a status revised after the game began never
reaches a feature.

Censoring. A player ruled out does not play, so he produces no box score row
and never enters the player-week table. Almost no Out or Doubtful player
appears in training. A model built here can learn Questionable against not
designated, and it can learn from practice participation, but it has never
seen what Out means. A prediction step must exclude ruled-out players rather
than score them.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Feature names this module produces, in the order they are added.
INJURY_FEATURES = [
    "injury_report_status",
    "injury_practice_status",
    "injury_not_injury_related",
]

# Timestamp of the chosen report version, carried through from clean.py. A row
# with a value here has a report entry, even if every status field is blank.
REPORT_MODIFIED_COLUMN = "injury_report_modified"

# Raw columns this module reads.
INJURY_SOURCE_COLUMNS = [
    REPORT_MODIFIED_COLUMN,
    "report_status",
    "practice_status",
    "report_primary_injury",
    "practice_primary_injury",
]

# The value for a player who does not appear on his club's report at all. Most
# healthy players are simply never listed, so this is a real and informative
# value, not missing data.
NOT_LISTED = "not_listed"

# The value for a player who is listed but given no game status. That is the
# normal case for a player who practised fully with a minor ailment.
NO_DESIGNATION = "no_designation"

# Categories are fixed rather than read off the data, so the encoding cannot
# shift between training and prediction. Doubtful and out are kept even though
# the training rows almost never contain them, for exactly that reason.
REPORT_STATUS_CATEGORIES = [NOT_LISTED, NO_DESIGNATION, "questionable", "doubtful", "out"]
PRACTICE_STATUS_CATEGORIES = [NOT_LISTED, "full", "limited", "did_not_participate"]

# Raw report statuses and what they map to. "Note" is a comment on the report
# rather than a game status, so it counts as listed without a designation.
REPORT_STATUS_VALUES = {
    "Questionable": "questionable",
    "Doubtful": "doubtful",
    "Out": "out",
    "Note": NO_DESIGNATION,
}

# Raw practice statuses and what they map to. Anything else on a listed player,
# including the whitespace-only entries nflverse sometimes carries, is unknown
# and becomes NaN rather than being guessed at.
PRACTICE_STATUS_VALUES = {
    "Full Participation in Practice": "full",
    "Limited Participation in Practice": "limited",
    "Did Not Participate In Practice": "did_not_participate",
}

# Text marking a listing that is not about an injury, such as a resting veteran
# or a personal matter. Matched case-insensitively.
NOT_INJURY_RELATED_TEXT = "not injury related"
NOT_INJURY_REASON_COLUMNS = ["report_primary_injury", "practice_primary_injury"]


def is_on_report(frame: pd.DataFrame) -> pd.Series:
    """Mark the rows whose player appears on his club's injury report.

    Takes the player-week frame. Returns a boolean Series.
    """
    return frame[REPORT_MODIFIED_COLUMN].notna()


def _empty_labels(frame: pd.DataFrame) -> pd.Series:
    """Make an object Series of missing values aligned to the frame.

    Takes the frame. Returns the Series, ready to be filled in by mask.
    """
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def _as_fixed_category(labels: pd.Series, categories: list[str], name: str) -> pd.Series:
    """Convert labels to a categorical with a fixed set of levels.

    Takes the labels, the allowed categories, and the column name. Returns the
    categorical Series, with any label outside the list as NaN.
    """
    return pd.Series(pd.Categorical(labels, categories=categories), index=labels.index, name=name)


def injury_report_status(frame: pd.DataFrame) -> pd.Series:
    """Encode the game status from the final pre-kickoff report.

    Takes the player-week frame. Returns a categorical Series.

    A player who is not listed gets not_listed, and a player listed with no game
    status gets no_designation. Those two are different facts and are kept
    apart: one says the club did not mention him, the other that it did.
    """
    listed = is_on_report(frame)
    raw_values = frame["report_status"]

    labels = _empty_labels(frame)
    labels[~listed] = NOT_LISTED
    labels[listed & raw_values.isna()] = NO_DESIGNATION
    for raw_value in REPORT_STATUS_VALUES:
        labels[listed & (raw_values == raw_value)] = REPORT_STATUS_VALUES[raw_value]

    return _as_fixed_category(labels, REPORT_STATUS_CATEGORIES, "injury_report_status")


def injury_practice_status(frame: pd.DataFrame) -> pd.Series:
    """Encode practice participation from the final pre-kickoff report.

    Takes the player-week frame. Returns a categorical Series.

    A listed player whose practice field is blank gets NaN, because the report
    mentions him but does not say how he practised. That is genuinely unknown,
    unlike an unlisted player, whose participation is known to be unremarkable.
    """
    listed = is_on_report(frame)
    raw_values = frame["practice_status"].fillna("").astype(str).str.strip()

    labels = _empty_labels(frame)
    labels[~listed] = NOT_LISTED
    for raw_value in PRACTICE_STATUS_VALUES:
        labels[listed & (raw_values == raw_value)] = PRACTICE_STATUS_VALUES[raw_value]

    return _as_fixed_category(labels, PRACTICE_STATUS_CATEGORIES, "injury_practice_status")


def injury_not_injury_related(frame: pd.DataFrame) -> pd.Series:
    """Flag listings that are about something other than an injury.

    Takes the player-week frame. Returns 1 for a listed player whose reason is
    marked not injury related, otherwise 0.

    A veteran resting on a Wednesday shows up as did not participate, exactly
    like an injured player, but it is the opposite signal: clubs rest players
    they are protecting for Sunday. Without this flag the model would read a
    planned rest day as a health problem.
    """
    listed = is_on_report(frame)

    mentions_not_injury = pd.Series(False, index=frame.index)
    for column_name in NOT_INJURY_REASON_COLUMNS:
        reasons = frame[column_name].fillna("").astype(str).str.lower()
        mentions_not_injury = mentions_not_injury | reasons.str.contains(
            NOT_INJURY_RELATED_TEXT, regex=False
        )

    return (listed & mentions_not_injury).astype(int)


def add_injury_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add the injury report features to the player-week frame.

    Takes the player-week frame, carrying the report columns joined in
    clean.py. Returns a copy with the three features added and their names.
    """
    built = frame.copy()
    built["injury_report_status"] = injury_report_status(built)
    built["injury_practice_status"] = injury_practice_status(built)
    built["injury_not_injury_related"] = injury_not_injury_related(built)

    logger.info(
        "Injury features: %s of %s rows on the report; report status %s; practice status %s",
        int(is_on_report(built).sum()),
        len(built),
        built["injury_report_status"].value_counts(dropna=False).to_dict(),
        built["injury_practice_status"].value_counts(dropna=False).to_dict(),
    )
    return built, list(INJURY_FEATURES)
