"""Tests for the prediction path: exclusions, flags, guards, and deployment training splits."""

import pandas as pd
import pytest

from ffml.models import predict, train


def exclusion_config() -> dict:
    """Build the config keys the exclusion rules read.

    Takes nothing. Returns the config.
    """
    return {
        "features": {"min_prior_games": 3},
        "predict": {"min_snap_share_last_3": 0.25, "excluded_report_statuses": ["Out", "Doubtful"]},
    }


def candidate_frame() -> pd.DataFrame:
    """Build candidates that each trip a different combination of rules.

    Takes nothing. Returns the frame. The expected reason per row is the
    earliest rule that applies, not the only one.
    """
    rows = [
        # roster, report, has_game, games, snap share, expected reason
        ("ACT", None, True, 10, 0.60, None),
        ("RES", "Out", False, 0, None, "roster_RES"),
        ("ACT", "Out", False, 10, 0.60, "ruled_out_out"),
        ("ACT", "Doubtful", True, 10, 0.60, "ruled_out_doubtful"),
        ("ACT", "Questionable", False, 10, 0.60, "bye"),
        ("ACT", None, True, 2, 0.70, "insufficient_history"),
        ("ACT", None, True, 5, None, "snap_share_unknown"),
        ("ACT", None, True, 5, 0.10, "below_snap_share"),
        ("ACT", "Questionable", True, 10, 0.60, None),
    ]
    columns = ["roster_status", "report_status", "has_game", "career_games", "snap_share_last_3", "expected"]
    return pd.DataFrame(rows, columns=columns)


def test_each_player_gets_the_earliest_exclusion_that_applies() -> None:
    """A player on injured reserve who is also listed Out is excluded for the roster, first."""
    candidates = candidate_frame()
    reasons = predict.exclusion_reasons(candidates, exclusion_config())

    for position in range(len(candidates)):
        expected = candidates["expected"].iloc[position]
        actual = reasons.iloc[position]
        if expected is None:
            assert actual is None, (position, actual)
        else:
            assert actual == expected, (position, actual)


def test_questionable_players_are_projected() -> None:
    """Questionable players played 57% of the time, so they are not excluded."""
    reasons = predict.exclusion_reasons(candidate_frame(), exclusion_config())
    assert reasons.iloc[8] is None


def output_frame(points: list[float], player_ids: list[str]) -> pd.DataFrame:
    """Build a minimal output table of projected players.

    Takes the projected points and player ids. Returns the frame.
    """
    return pd.DataFrame(
        {
            "player": player_ids,
            "player_id": player_ids,
            "position": "WR",
            "status": predict.PROJECTED,
            "projected_points": points,
            "projection_method": predict.PROJECTOR_MODEL,
        }
    )


def test_a_player_appearing_twice_is_refused() -> None:
    """Duplicate rows would double-count a player in any lineup decision."""
    with pytest.raises(ValueError, match="more than once"):
        predict.sanity_check(output_frame([10.0, 11.0], ["a", "a"]), {"WR": 56.4})


def test_negative_and_implausible_projections_are_refused() -> None:
    """A negative or above-ceiling projection means something is broken, so it raises."""
    with pytest.raises(ValueError, match="negative"):
        predict.sanity_check(output_frame([-0.5, 11.0], ["a", "b"]), {"WR": 56.4})
    with pytest.raises(ValueError, match="above"):
        predict.sanity_check(output_frame([60.0, 11.0], ["a", "b"]), {"WR": 56.4})

    distribution = predict.sanity_check(output_frame([4.0, 11.0, 20.0], ["a", "b", "c"]), {"WR": 56.4})
    assert distribution["median"].iloc[0] == 11.0


class FakeBooster:
    """A stand-in for a LightGBM booster that only reports its feature names."""

    def __init__(self, names: list[str]) -> None:
        """Store the feature names.

        Takes the names. Returns nothing.
        """
        self.names = names

    def feature_name(self) -> list[str]:
        """Report the stored feature names.

        Takes nothing. Returns the names.
        """
        return self.names


def test_a_feature_list_mismatch_raises_rather_than_warns() -> None:
    """A reordered column would silently feed the model the wrong number."""
    names = ["a_roll3", "b_roll3"]
    predict.check_feature_list({"feature_names": names}, FakeBooster(names), names, "WR")

    with pytest.raises(ValueError, match="mismatch"):
        predict.check_feature_list({"feature_names": names}, FakeBooster(names), ["b_roll3", "a_roll3"], "WR")
    with pytest.raises(ValueError, match="mismatch"):
        predict.check_feature_list({"feature_names": names}, FakeBooster(["a_roll3"]), names, "WR")


def test_history_is_cut_strictly_before_the_target_week() -> None:
    """The target week's own rows, and anything later, never reach an upcoming row."""
    frame = pd.DataFrame({"season": [2025, 2025, 2026, 2026, 2026], "week": [17, 18, 1, 2, 3]})
    kept = predict.history_before(frame, 2026, 2)
    assert kept[["season", "week"]].values.tolist() == [[2025, 17], [2025, 18], [2026, 1]]


def live_model_metadata(season: int, week: int) -> dict:
    """Build metadata for a model trained through one week.

    Takes the season and week. Returns the metadata.
    """
    return {"trained_through": [season, week]}


def test_a_live_projection_refuses_a_stale_model_or_one_that_saw_the_week() -> None:
    """A stale model misses a completed week; a model that trained on the target is leakage."""
    predict.check_model_is_current(live_model_metadata(2025, 18), "WR", (2025, 18), (2026, 1), False)

    with pytest.raises(ValueError, match="Retrain"):
        predict.check_model_is_current(live_model_metadata(2025, 17), "WR", (2025, 18), (2026, 1), False)
    with pytest.raises(ValueError, match="not before"):
        predict.check_model_is_current(live_model_metadata(2026, 1), "WR", (2025, 18), (2026, 1), False)

    # The smoke test deliberately projects a week its model trained on.
    predict.check_model_is_current(live_model_metadata(2025, 18), "WR", (2025, 9), (2025, 10), True)


def projector_config(by_position: dict | None) -> dict:
    """Build a config with a projector choice per position.

    Takes the per-position overrides, or None to leave the block out entirely.
    Returns the config.
    """
    predict_section: dict = {}
    if by_position is not None:
        predict_section["projectors"] = {"default": predict.PROJECTOR_MODEL, "by_position": by_position}
    return {"data": {"positions": ["QB", "RB"]}, "predict": predict_section}


def test_each_position_gets_its_projector_and_the_default_is_the_model() -> None:
    """An override applies to its own position only, and a config without the block uses models."""
    config = projector_config({"QB": predict.PROJECTOR_BASELINE})

    assert predict.projector_for(config, "QB") == predict.PROJECTOR_BASELINE
    assert predict.projector_for(config, "RB") == predict.PROJECTOR_MODEL
    assert predict.projector_for(projector_config(None), "QB") == predict.PROJECTOR_MODEL


def test_an_unknown_projector_or_position_is_refused() -> None:
    """A typo must not quietly fall back to the model."""
    with pytest.raises(ValueError, match="Unknown projector"):
        predict.projector_for(projector_config({"QB": "ensemble"}), "QB")
    with pytest.raises(ValueError, match="not a modelled position"):
        predict.projector_for(projector_config({"K": predict.PROJECTOR_BASELINE}), "RB")


def labelled_output(methods: list[str | None]) -> pd.DataFrame:
    """Build an output table with a projected QB, a projected RB, and an excluded RB.

    Takes the projection method of each row in that order. Returns the frame.
    """
    return pd.DataFrame(
        {
            "player": ["qb", "rb", "out"],
            "player_id": ["qb", "rb", "out"],
            "position": ["QB", "RB", "RB"],
            "status": [predict.PROJECTED, predict.PROJECTED, predict.EXCLUDED],
            "projected_points": [18.0, 12.0, None],
            "projection_method": methods,
        }
    )


def test_every_projected_row_must_name_the_method_its_position_ships() -> None:
    """Mixed methods pass when each matches the config; a wrong or missing label raises."""
    config = projector_config({"QB": predict.PROJECTOR_BASELINE})

    # The excluded row has no number, so it carries no method.
    predict.check_projection_methods(labelled_output(["baseline", "model", None]), config)

    with pytest.raises(ValueError, match="QB"):
        predict.check_projection_methods(labelled_output(["model", "model", None]), config)
    with pytest.raises(ValueError, match="RB"):
        predict.check_projection_methods(labelled_output(["baseline", None, None]), config)


TARGET = "fantasy_points_league"


def baseline_config() -> dict:
    """Build the settings the baseline reads.

    Takes nothing. Returns the config.
    """
    return {
        "model": {"baseline": {"method": "rolling_mean", "window": 3}},
        "features": {"min_prior_games": 3},
        "scoring": {"target_column": TARGET},
    }


def quarterback_history() -> pd.DataFrame:
    """Build five weeks for one quarterback, the fifth already played for 100 points.

    Takes nothing. Returns the frame.
    """
    return pd.DataFrame(
        {
            "player_id": ["qb", "qb", "qb", "qb", "qb"],
            "position": ["QB", "QB", "QB", "QB", "QB"],
            "season": [2025, 2025, 2025, 2025, 2025],
            "week": [1, 2, 3, 4, 5],
            TARGET: [10.0, 20.0, 30.0, 40.0, 100.0],
        }
    )


def upcoming_quarterback(week: int) -> pd.DataFrame:
    """Build the quarterback's upcoming row for one week, with no result.

    Takes the week. Returns the frame.
    """
    return pd.DataFrame(
        {"player_id": ["qb"], "position": ["QB"], "season": [2025], "week": [week], TARGET: [float("nan")]}
    )


def test_the_baseline_projects_the_three_games_before_the_target_week() -> None:
    """Weeks 2 to 4 average 30. Week 5's own 100 points, already in the table, must not reach it."""
    projections = predict.baseline_projections(
        quarterback_history(), upcoming_quarterback(5), baseline_config(), 2025, 5
    )
    assert projections.loc["qb"] == pytest.approx(30.0)


def test_a_player_without_enough_history_has_no_baseline_and_is_refused() -> None:
    """Before week 3 he has two games, below the three a projection needs."""
    with pytest.raises(ValueError, match="no baseline"):
        predict.baseline_projections(quarterback_history(), upcoming_quarterback(3), baseline_config(), 2025, 3)


def test_a_baseline_that_disagrees_with_the_rolling_feature_is_refused() -> None:
    """The projected number must be the same three game mean the evaluation measured."""
    projections = pd.Series([30.0], index=["qb"])
    agreeing = pd.DataFrame({"player_id": ["qb"], "fantasy_points_league_roll3": [30.0]})
    disagreeing = pd.DataFrame({"player_id": ["qb"], "fantasy_points_league_roll3": [29.5]})

    predict.check_baseline_matches_rolling_mean(projections, agreeing, baseline_config(), "QB")
    with pytest.raises(ValueError, match="disagree"):
        predict.check_baseline_matches_rolling_mean(projections, disagreeing, baseline_config(), "QB")


def deployment_groups_config() -> dict:
    """Build a config whose TE override turns injuries on, as stage seven's config did.

    Takes nothing. Returns the config.
    """
    return {
        "data": {"positions": ["QB", "TE"]},
        "features": {
            "groups": {
                "default": {"rolling_production": True, "injuries": False},
                "by_position": {"QB": {}, "TE": {"injuries": True}},
            }
        },
        "predict": {"deployment": {"disabled_groups": ["injuries"], "early_stopping_weeks": 2}},
    }


def test_deployment_disables_groups_without_touching_the_evaluation_config() -> None:
    """Injuries go off for deployment; the walk-forward config keeps TE's injuries on."""
    config = deployment_groups_config()
    deployed = train.deployment_config(config)

    assert deployed["features"]["groups"]["by_position"]["TE"]["injuries"] is False
    assert deployed["features"]["groups"]["by_position"]["QB"]["injuries"] is False
    assert config["features"]["groups"]["by_position"]["TE"]["injuries"] is True


def completion_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a schedule and player-week table spanning complete and incomplete weeks.

    Takes nothing. Returns the schedules and player weeks.

    Weeks 1 and 2 are complete. Week 3 has a finished game but team DDD has no
    stats yet. Week 4 has not been played.
    """
    schedules = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 2, 3, 4],
            "game_type": ["REG", "REG", "REG", "REG"],
            "home_team": ["AAA", "AAA", "CCC", "AAA"],
            "away_team": ["BBB", "BBB", "DDD", "BBB"],
            "home_score": [21.0, 17.0, 24.0, None],
        }
    )
    player_weeks = pd.DataFrame(
        {
            "season": [2025, 2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2, 3],
            "team": ["AAA", "BBB", "AAA", "BBB", "CCC"],
        }
    )
    return schedules, player_weeks


def test_a_week_needs_every_score_and_every_team_stats_to_be_complete() -> None:
    """A final score is not enough: the stats have to have landed too."""
    schedules, player_weeks = completion_tables()

    assert train.week_completion(schedules, player_weeks, 2025, 2)["complete"]

    missing_stats = train.week_completion(schedules, player_weeks, 2025, 3)
    assert not missing_stats["complete"]
    assert missing_stats["teams_missing_stats"] == ["DDD"]

    unplayed = train.week_completion(schedules, player_weeks, 2025, 4)
    assert not unplayed["complete"]
    assert unplayed["unplayed_games"] == ["BBB@AAA"]


def test_completed_weeks_stop_at_the_first_incomplete_week() -> None:
    """Weeks after a gap are not trained on, even if they look complete."""
    schedules, player_weeks = completion_tables()
    config = {"data": {"start_season": 2025}}
    assert train.completed_weeks(schedules, player_weeks, config) == [(2025, 1), (2025, 2)]


def deployment_features() -> pd.DataFrame:
    """Build one WR row per week across two seasons and an incomplete week.

    Takes nothing. Returns the frame.
    """
    seasons = []
    weeks = []
    for season, week in [(2024, 17), (2024, 18), (2025, 1), (2025, 2), (2025, 3), (2025, 4)]:
        seasons.append(season)
        weeks.append(week)
    return pd.DataFrame({"season": seasons, "week": weeks, "position": "WR"})


def test_deployment_stops_on_the_latest_weeks_and_refits_on_everything() -> None:
    """The stop set is the most recent completed weeks; the refit set is all of them."""
    completed = [(2024, 17), (2024, 18), (2025, 1), (2025, 2), (2025, 3)]
    config = {"data": {"start_season": 2022}}
    train_rows, stop_rows, refit_rows = train.deployment_split(
        deployment_features(), "WR", completed, 2, config
    )

    assert stop_rows[["season", "week"]].values.tolist() == [[2025, 2], [2025, 3]]
    assert train_rows[["season", "week"]].values.tolist() == [[2024, 17], [2024, 18], [2025, 1]]
    assert len(train_rows.index.intersection(stop_rows.index)) == 0

    # The incomplete week 4 reaches no set at all.
    assert not (refit_rows["week"] == 4).any()
    assert len(refit_rows) == len(train_rows) + len(stop_rows)
