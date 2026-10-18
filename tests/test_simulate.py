import numpy as np
import pandas as pd
import pytest

from src import poisson, simulate


def toy_model():
    return poisson.PoissonModel(
        intercept=0.1,
        home_advantage=0.2,
        attack={"Strong": 0.5, "Weak": -0.5},
        defence={"Strong": -0.3, "Weak": 0.3},
    )


def toy_fixtures():
    return pd.DataFrame(
        {"home_team": ["Strong", "Weak"], "away_team": ["Weak", "Strong"]}
    )


def test_add_unseen_teams_gives_average_strength_and_reports_them():
    model, added = simulate.add_unseen_teams(toy_model(), ["Strong", "Ipswich", "Leeds"])

    assert added == ["Ipswich", "Leeds"]
    assert model.attack["Ipswich"] == 0.0
    assert model.defence["Leeds"] == 0.0
    assert model.attack["Strong"] == 0.5  # known teams untouched
    assert sum(model.attack.values()) == pytest.approx(0)  # still sums to zero


def test_add_unseen_teams_does_not_modify_the_original_model():
    original = toy_model()

    simulate.add_unseen_teams(original, ["Ipswich"])

    assert "Ipswich" not in original.attack
    assert "Ipswich" not in original.defence


def test_add_unseen_teams_with_only_known_teams_changes_nothing():
    model, added = simulate.add_unseen_teams(toy_model(), ["Strong", "Weak"])

    assert added == []
    assert model == toy_model()


def test_fixture_lambdas_match_expected_goals_row_by_row():
    lam_home, lam_away = simulate.fixture_lambdas(toy_model(), toy_fixtures())

    for i, (home, away) in enumerate(zip(toy_fixtures()["home_team"], toy_fixtures()["away_team"])):
        expected = poisson.expected_goals(toy_model(), home, away)
        assert (lam_home[i], lam_away[i]) == pytest.approx(expected)


def test_fixture_lambdas_unknown_team_raises():
    fixtures = pd.DataFrame({"home_team": ["Strong"], "away_team": ["Newcomer"]})

    with pytest.raises(ValueError, match="Unknown team"):
        simulate.fixture_lambdas(toy_model(), fixtures)


def test_simulate_goals_shape_and_type():
    home, away = simulate.simulate_goals(np.array([1.0, 2.0, 0.5]), np.array([1.0, 1.0, 1.0]), n_sims=7)

    assert home.shape == away.shape == (7, 3)
    assert home.min() >= 0
    assert np.issubdtype(home.dtype, np.integer)


def test_simulate_goals_same_seed_is_reproducible_and_different_seed_differs():
    lam = np.array([1.5, 1.0, 2.0])

    a1, b1 = simulate.simulate_goals(lam, lam, n_sims=100, seed=1)
    a2, b2 = simulate.simulate_goals(lam, lam, n_sims=100, seed=1)
    a3, _ = simulate.simulate_goals(lam, lam, n_sims=100, seed=2)

    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)
    assert not np.array_equal(a1, a3)


def test_simulate_goals_averages_converge_to_lambdas():
    lam_home = np.array([0.8, 1.5, 2.5])
    lam_away = np.array([1.2, 1.0, 0.6])

    home, away = simulate.simulate_goals(lam_home, lam_away, n_sims=50_000, seed=0)

    assert home.mean(axis=0) == pytest.approx(lam_home, abs=0.03)
    assert away.mean(axis=0) == pytest.approx(lam_away, abs=0.03)


TEAMS = ["A", "B", "C"]


def table_fixtures():
    return pd.DataFrame({"home_team": ["A", "B"], "away_team": ["B", "C"]})


def table_goals():
    # Two simulations of the two fixtures A-B and B-C.
    # sim 0: A 2-1 B, B 0-0 C      sim 1: A 0-3 B, B 1-2 C
    home = np.array([[2, 0], [0, 1]])
    away = np.array([[1, 0], [3, 2]])
    return home, away


def test_build_tables_matches_hand_computed_values():
    home, away = table_goals()

    points, goal_diff, goals_for = simulate.build_tables(home, away, table_fixtures(), TEAMS)

    assert np.array_equal(points, [[3, 1, 1], [0, 3, 3]])
    assert np.array_equal(goal_diff, [[1, -1, 0], [-3, 2, 1]])
    assert np.array_equal(goals_for, [[2, 1, 0], [0, 4, 2]])


def test_build_tables_conserves_points_and_goal_difference():
    rng = np.random.default_rng(0)
    fixtures = pd.DataFrame({"home_team": ["A", "B", "C"], "away_team": ["B", "C", "A"]})
    home, away = rng.poisson(1.4, (50, 3)), rng.poisson(1.1, (50, 3))

    points, goal_diff, goals_for = simulate.build_tables(home, away, fixtures, TEAMS)

    assert np.all(goal_diff.sum(axis=1) == 0)  # every goal for one team is against another
    assert np.array_equal(goals_for.sum(axis=1), (home + away).sum(axis=1))
    draws = (home == away).sum(axis=1)
    assert np.array_equal(points.sum(axis=1), 3 * (3 - draws) + 2 * draws)


def test_build_tables_unknown_team_raises():
    fixtures = pd.DataFrame({"home_team": ["A"], "away_team": ["Z"]})

    with pytest.raises(ValueError):
        simulate.build_tables(np.array([[1]]), np.array([[0]]), fixtures, TEAMS)


def test_final_positions_rank_by_points_then_goal_difference():
    points = np.array([[3, 1, 1], [0, 3, 3]])
    goal_diff = np.array([[1, -1, 0], [-3, 2, 1]])
    goals_for = np.array([[2, 1, 0], [0, 4, 2]])

    positions = simulate.final_positions(points, goal_diff, goals_for)

    assert np.array_equal(positions, [[1, 3, 2], [3, 1, 2]])


def test_final_positions_uses_goals_for_when_points_and_goal_difference_tie():
    points = np.array([[10, 10]])
    goal_diff = np.array([[2, 2]])
    goals_for = np.array([[20, 25]])

    assert np.array_equal(simulate.final_positions(points, goal_diff, goals_for), [[2, 1]])


def test_final_positions_random_tiebreak_is_fair_and_reproducible():
    tied = np.zeros((2000, 2), dtype=int)

    first = simulate.final_positions(tied, tied, tied, seed=3)
    again = simulate.final_positions(tied, tied, tied, seed=3)

    assert np.array_equal(first, again)
    assert (first[:, 0] == 1).mean() == pytest.approx(0.5, abs=0.05)


def test_final_positions_each_row_is_a_permutation():
    rng = np.random.default_rng(1)
    points = rng.integers(0, 6, (100, 6))

    positions = simulate.final_positions(points, points * 0, points * 0)

    assert all(sorted(row) == [1, 2, 3, 4, 5, 6] for row in positions)


POSITIONS = np.array([[1, 2, 3, 4], [2, 1, 4, 3], [1, 3, 2, 4], [4, 1, 3, 2]])
FOUR = ["T0", "T1", "T2", "T3"]


def test_summarize_probabilities_match_hand_counts():
    points = np.array([[9, 6, 3, 0], [6, 9, 0, 3], [9, 3, 6, 0], [0, 9, 3, 6]])

    summary = simulate.summarize(POSITIONS, points, FOUR, n_top=2, n_relegated=1).set_index("team")

    assert summary["p_title"].to_dict() == {"T0": 0.5, "T1": 0.5, "T2": 0.0, "T3": 0.0}
    assert summary.loc["T0", "p_top2"] == 0.75
    assert summary["p_relegation"].to_dict() == {"T0": 0.25, "T1": 0.0, "T2": 0.25, "T3": 0.5}
    assert summary.loc["T1", "expected_points"] == pytest.approx(6.75)


def test_summarize_is_sorted_by_expected_points_and_totals_are_consistent():
    points = np.array([[9, 6, 3, 0], [6, 9, 0, 3], [9, 3, 6, 0], [0, 9, 3, 6]])

    summary = simulate.summarize(POSITIONS, points, FOUR, n_top=2, n_relegated=1)

    assert summary["expected_points"].is_monotonic_decreasing
    assert summary["p_title"].sum() == pytest.approx(1)  # exactly one champion per season
    assert summary["p_relegation"].sum() == pytest.approx(1)  # n_relegated = 1
    assert summary["p_top2"].sum() == pytest.approx(2)


def test_position_distribution_rows_and_columns_sum_to_one():
    dist = simulate.position_distribution(POSITIONS, FOUR)

    assert dist.shape == (4, 4)
    assert dist.sum(axis=1).tolist() == pytest.approx([1, 1, 1, 1])
    assert dist.sum(axis=0).tolist() == pytest.approx([1, 1, 1, 1])
    assert dist.loc["T0", 1] == 0.5


def double_round_robin(teams):
    return pd.DataFrame(
        [(h, a) for h in teams for a in teams if h != a], columns=["home_team", "away_team"]
    )


def test_simulate_season_stronger_team_wins_title_more_often():
    result = simulate.simulate_season(toy_model(), double_round_robin(["Strong", "Weak"]), n_sims=5000)

    summary = result.summary.set_index("team")
    assert summary.loc["Strong", "p_title"] > summary.loc["Weak", "p_title"]
    assert summary["p_title"].sum() == pytest.approx(1)
    assert result.added_teams == []


def test_simulate_season_is_reproducible():
    fixtures = double_round_robin(["Strong", "Weak"])

    a = simulate.simulate_season(toy_model(), fixtures, n_sims=500, seed=7)
    b = simulate.simulate_season(toy_model(), fixtures, n_sims=500, seed=7)

    pd.testing.assert_frame_equal(a.summary, b.summary)


def test_simulate_season_reports_and_handles_unseen_teams():
    fixtures = double_round_robin(["Strong", "Newcomer"])

    result = simulate.simulate_season(toy_model(), fixtures, n_sims=200)

    assert result.added_teams == ["Newcomer"]
    assert set(result.summary["team"]) == {"Strong", "Newcomer"}


def test_simulate_season_adds_points_from_played_matches():
    remaining = pd.DataFrame({"home_team": ["Weak"], "away_team": ["Strong"]})
    played = pd.DataFrame(
        {"home_team": ["Strong"], "away_team": ["Weak"], "home_goals": [0], "away_goals": [5]}
    )

    without = simulate.simulate_season(toy_model(), remaining, n_sims=2000, seed=1)
    with_played = simulate.simulate_season(toy_model(), remaining, n_sims=2000, seed=1, played=played)

    pts = lambda r, t: r.summary.set_index("team").loc[t, "expected_points"]
    assert pts(with_played, "Weak") == pytest.approx(pts(without, "Weak") + 3)
    assert pts(with_played, "Strong") == pytest.approx(pts(without, "Strong"))


def season_matches():
    """Two seasons of a 3-team league; 2023/24 has a clear table."""
    rows = [
        # season, date, home, away, hg, ag, result
        ("2022/23", "2022-08-01", "A", "B", 1, 0, "H"),
        ("2022/23", "2022-08-08", "B", "C", 2, 2, "D"),
        ("2022/23", "2022-08-15", "C", "A", 0, 1, "A"),
        ("2023/24", "2023-08-01", "A", "B", 3, 0, "H"),
        ("2023/24", "2023-08-08", "B", "C", 0, 1, "A"),
        ("2023/24", "2023-08-15", "C", "A", 1, 1, "D"),
    ]
    df = pd.DataFrame(rows, columns=["season", "date", "home_team", "away_team", "home_goals", "away_goals", "result"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_season_fixtures_returns_only_that_season():
    fixtures = simulate.season_fixtures(season_matches(), "2023/24")

    assert list(fixtures.columns) == ["home_team", "away_team"]
    assert len(fixtures) == 3
    assert fixtures.iloc[0].tolist() == ["A", "B"]


def test_season_functions_reject_unknown_season():
    with pytest.raises(ValueError):
        simulate.season_fixtures(season_matches(), "1999/00")
    with pytest.raises(ValueError):
        simulate.fit_before_season(season_matches(), "1999/00")


def test_fit_before_season_ignores_that_season_and_later(monkeypatch):
    seen = {}

    def fake_fit(matches, as_of_date, **settings):
        seen["as_of"] = as_of_date
        seen["settings"] = settings
        seen["max_date"] = matches[matches["date"] < as_of_date]["date"].max()
        return "model"

    monkeypatch.setattr(simulate, "fit_poisson", fake_fit)

    assert simulate.fit_before_season(season_matches(), "2023/24") == "model"
    assert seen["as_of"] == pd.Timestamp("2023-08-01")
    assert seen["max_date"] == pd.Timestamp("2022-08-15")  # nothing from 2023/24 is usable
    assert seen["settings"] == {"half_life": poisson.MODEL_HALF_LIFE, "ridge": poisson.MODEL_RIDGE}


def test_actual_table_ranks_by_points_then_goal_difference():
    table = simulate.actual_table(season_matches(), "2023/24")

    # A and C both have 4 points; A wins the tie on goal difference (+3 vs +1). B has 0.
    assert table["team"].tolist() == ["A", "C", "B"]
    assert table["actual_points"].tolist() == [4, 4, 0]
    assert table["actual_position"].tolist() == [1, 2, 3]
