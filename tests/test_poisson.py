import numpy as np
import pandas as pd
import pytest

from src import poisson


def matches_frame():
    return pd.DataFrame(
        {
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Everton"],
            "home_goals": [2, 0],
            "away_goals": [1, 3],
        }
    )


def test_to_long_format_gives_two_rows_per_match():
    long = poisson.to_long_format(matches_frame())

    assert len(long) == 4
    assert list(long.columns) == ["team", "opponent", "is_home", "goals"]


def test_to_long_format_assigns_goals_and_home_flag_correctly():
    long = poisson.to_long_format(matches_frame())

    # First match: Arsenal (home) 2 - 1 Chelsea (away)
    home_row = long[(long["team"] == "Arsenal") & (long["opponent"] == "Chelsea")].iloc[0]
    away_row = long[(long["team"] == "Chelsea") & (long["opponent"] == "Arsenal")].iloc[0]

    assert (home_row["is_home"], home_row["goals"]) == (1, 2)
    assert (away_row["is_home"], away_row["goals"]) == (0, 1)


def test_to_long_format_conserves_total_goals():
    matches = matches_frame()
    long = poisson.to_long_format(matches)

    assert long["goals"].sum() == matches["home_goals"].sum() + matches["away_goals"].sum()


def simulate_matches(n_seasons=8, seed=0):
    """Simulate double round-robin seasons from known true parameters."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(20)]
    attack = rng.normal(0, 0.25, 20)
    defence = rng.normal(0, 0.25, 20)
    attack -= attack.mean()
    defence -= defence.mean()
    true_attack = dict(zip(teams, attack))
    true_defence = dict(zip(teams, defence))
    intercept, home_adv = 0.2, 0.3

    rows = []
    day = 0
    for _ in range(n_seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam_h = np.exp(intercept + home_adv + true_attack[h] + true_defence[a])
                lam_a = np.exp(intercept + true_attack[a] + true_defence[h])
                rows.append(
                    {
                        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=day // 10),
                        "home_team": h,
                        "away_team": a,
                        "home_goals": rng.poisson(lam_h),
                        "away_goals": rng.poisson(lam_a),
                    }
                )
                day += 1
    return pd.DataFrame(rows), true_attack, true_defence, intercept, home_adv


def test_fit_poisson_recovers_known_parameters():
    matches, true_attack, true_defence, intercept, home_adv = simulate_matches()

    model = poisson.fit_poisson(matches)

    assert model.intercept == pytest.approx(intercept, abs=0.05)
    assert model.home_advantage == pytest.approx(home_adv, abs=0.05)
    for team in true_attack:
        assert model.attack[team] == pytest.approx(true_attack[team], abs=0.15)
        assert model.defence[team] == pytest.approx(true_defence[team], abs=0.15)


def test_fit_poisson_effects_sum_to_zero():
    matches, *_ = simulate_matches(n_seasons=2)

    model = poisson.fit_poisson(matches)

    assert sum(model.attack.values()) == pytest.approx(0, abs=1e-9)
    assert sum(model.defence.values()) == pytest.approx(0, abs=1e-9)


def small_dated_matches():
    """Three teams, two rounds of every ordered pairing: 12 matches, weekly."""
    pairs = [("A", "B"), ("B", "A"), ("A", "C"), ("C", "A"), ("B", "C"), ("C", "B")] * 2
    goals = [(2, 1), (1, 1), (3, 0), (0, 2), (1, 0), (2, 2), (0, 1), (1, 3), (2, 0), (1, 1), (0, 0), (3, 1)]
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-08-01", periods=12, freq="7D"),
            "home_team": [h for h, _ in pairs],
            "away_team": [a for _, a in pairs],
            "home_goals": [g[0] for g in goals],
            "away_goals": [g[1] for g in goals],
        }
    )


def test_fit_poisson_ignores_matches_on_or_after_as_of_date():
    base = small_dated_matches()
    cutoff = pd.Timestamp("2023-10-02")  # 9 base matches fall before it
    # A wild result ON the cutoff date and one after it: neither may be used.
    extra = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-10-02", "2023-12-01"]),
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_goals": [9, 9],
            "away_goals": [0, 0],
        }
    )

    without = poisson.fit_poisson(base, as_of_date=cutoff)
    with_future = poisson.fit_poisson(pd.concat([base, extra]), as_of_date=cutoff)

    assert with_future.intercept == pytest.approx(without.intercept)
    assert with_future.home_advantage == pytest.approx(without.home_advantage)
    assert with_future.attack == pytest.approx(without.attack)


def test_fit_poisson_raises_when_no_matches_before_date():
    with pytest.raises(ValueError):
        poisson.fit_poisson(small_dated_matches(), as_of_date=pd.Timestamp("2023-08-01"))


def toy_model(home_advantage=0.2):
    return poisson.PoissonModel(
        intercept=0.1,
        home_advantage=home_advantage,
        attack={"Strong": 0.5, "Weak": -0.5},
        defence={"Strong": -0.3, "Weak": 0.3},
    )


def test_expected_goals_matches_formula():
    lam_home, lam_away = poisson.expected_goals(toy_model(), "Strong", "Weak")

    assert lam_home == pytest.approx(np.exp(0.1 + 0.2 + 0.5 + 0.3))
    assert lam_away == pytest.approx(np.exp(0.1 + (-0.5) + (-0.3)))


def test_expected_goals_unknown_team_raises():
    with pytest.raises(ValueError, match="Unknown team"):
        poisson.expected_goals(toy_model(), "Strong", "Newly Promoted")


def test_score_matrix_cell_matches_poisson_formula():
    matrix = poisson.score_matrix(1.0, 2.0)

    # P(home=0) = e^-1, P(away=2) = e^-2 * 2^2 / 2!
    assert matrix[0, 2] == pytest.approx(np.exp(-1) * np.exp(-2) * 4 / 2)
    assert matrix.shape == (11, 11)


def test_score_matrix_total_is_almost_one():
    assert poisson.score_matrix(1.5, 1.2).sum() == pytest.approx(1.0, abs=1e-3)


def test_outcome_probabilities_sum_to_one():
    probs = poisson.outcome_probabilities(poisson.score_matrix(1.5, 1.2))

    assert sum(probs) == pytest.approx(1.0)


def test_outcome_probabilities_rows_are_home_goals():
    # 2-0 is the only possible score: a home win.
    matrix = np.zeros((3, 3))
    matrix[2, 0] = 1.0

    assert poisson.outcome_probabilities(matrix) == (1.0, 0.0, 0.0)


def test_equal_lambdas_give_symmetric_home_and_away():
    p_home, p_draw, p_away = poisson.outcome_probabilities(poisson.score_matrix(1.3, 1.3))

    assert p_home == pytest.approx(p_away)
    assert p_draw > 0.2


def test_stronger_team_is_favoured_and_home_advantage_helps():
    model = toy_model()
    p_home, _, p_away = poisson.outcome_probabilities(
        poisson.score_matrix(*poisson.expected_goals(model, "Strong", "Weak"))
    )
    assert p_home > p_away

    # Same team vs itself-like symmetric strengths: only home advantage differs.
    flat = poisson.PoissonModel(0.1, 0.3, {"A": 0.0, "B": 0.0}, {"A": 0.0, "B": 0.0})
    p_home_flat, _, p_away_flat = poisson.outcome_probabilities(
        poisson.score_matrix(*poisson.expected_goals(flat, "A", "B"))
    )
    assert p_home_flat > p_away_flat


def test_predict_match_equals_manual_chain():
    model = toy_model()
    lam_home, lam_away = poisson.expected_goals(model, "Strong", "Weak")
    expected = poisson.outcome_probabilities(poisson.score_matrix(lam_home, lam_away))

    assert poisson.predict_match(model, "Strong", "Weak") == pytest.approx(expected)


def test_predict_match_probabilities_sum_to_one():
    assert sum(poisson.predict_match(toy_model(), "Weak", "Strong")) == pytest.approx(1.0)


def test_predict_match_unknown_team_raises():
    with pytest.raises(ValueError, match="Unknown team"):
        poisson.predict_match(toy_model(), "Nobody", "Weak")


def test_recency_weights_halve_every_half_life():
    dates = pd.Series(pd.to_datetime(["2024-01-31", "2024-01-01", "2023-12-02"]))

    weights = poisson.recency_weights(dates, pd.Timestamp("2024-01-31"), half_life=30)

    assert weights.tolist() == pytest.approx([1.0, 0.5, 0.25])


def test_fit_poisson_half_life_must_be_positive():
    with pytest.raises(ValueError, match="half_life"):
        poisson.fit_poisson(small_dated_matches(), half_life=0)


def test_fit_poisson_huge_half_life_matches_unweighted_fit():
    matches, *_ = simulate_matches(n_seasons=2)

    plain = poisson.fit_poisson(matches)
    barely_weighted = poisson.fit_poisson(matches, half_life=1e9)

    assert barely_weighted.home_advantage == pytest.approx(plain.home_advantage, abs=1e-6)
    assert barely_weighted.attack == pytest.approx(plain.attack, abs=1e-6)


def test_fit_poisson_short_half_life_follows_the_recent_strengths():
    early, early_attack, *_ = simulate_matches(n_seasons=4, seed=1)
    late, late_attack, *_ = simulate_matches(n_seasons=4, seed=2)
    late["date"] += pd.Timedelta(days=400)  # the "late" strengths replace the early ones
    matches = pd.concat([early, late], ignore_index=True)
    teams = sorted(early_attack)

    def distance(model, truth):
        return np.mean([abs(model.attack[t] - truth[t]) for t in teams])

    weighted = poisson.fit_poisson(matches, half_life=60)
    unweighted = poisson.fit_poisson(matches)

    assert distance(weighted, late_attack) < distance(weighted, early_attack)
    assert distance(weighted, late_attack) < distance(unweighted, late_attack)
