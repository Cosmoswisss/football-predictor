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


def history_data():
    """Eight simulated seasons (about 10 months) with known true strengths."""
    matches, true_attack, true_defence, *_ = simulate_matches(n_seasons=8)
    return matches, true_attack, true_defence


def test_month_starts_covers_first_to_last_match():
    matches = pd.DataFrame({"date": pd.to_datetime(["2023-08-12", "2023-10-20"])})

    assert poisson.month_starts(matches).tolist() == [pd.Timestamp("2023-09-01"), pd.Timestamp("2023-10-01")]


def test_strength_history_shape_columns_and_strength_definition():
    matches, *_ = history_data()
    dates = poisson.month_starts(matches)

    history = poisson.strength_history(matches, dates, min_matches=500)

    assert list(history.columns) == ["date", "team", "attack", "defence", "strength"]
    assert history["date"].nunique() > 1
    assert (history["strength"] == history["attack"] - history["defence"]).all()


def test_strength_history_skips_snapshots_with_too_little_data():
    matches, *_ = history_data()

    history = poisson.strength_history(matches, poisson.month_starts(matches), min_matches=10**9)

    assert history.empty
    assert list(history.columns) == ["date", "team", "attack", "defence", "strength"]


def test_strength_history_snapshot_ignores_matches_on_or_after_its_date():
    matches, *_ = history_data()
    snapshot = pd.DatetimeIndex([matches["date"].max() - pd.Timedelta(days=60)])
    altered = matches.copy()
    after = altered["date"] >= snapshot[0]
    altered.loc[after, ["home_goals", "away_goals"]] = [9, 0]  # absurd results from the snapshot date on

    original = poisson.strength_history(matches, snapshot)
    changed = poisson.strength_history(altered, snapshot)

    pd.testing.assert_frame_equal(original, changed)


def test_strength_history_follows_true_strengths_and_sums_to_zero():
    matches, true_attack, true_defence = history_data()
    last = pd.DatetimeIndex([matches["date"].max()])

    history = poisson.strength_history(matches, last, half_life=10_000)  # almost no decay

    assert history["attack"].sum() == pytest.approx(0, abs=1e-6)
    truth = pd.Series({t: true_attack[t] - true_defence[t] for t in true_attack})
    fitted = history.set_index("team")["strength"]
    assert np.corrcoef(truth[fitted.index], fitted)[0, 1] > 0.9


def test_strength_history_drops_teams_inactive_for_a_long_time():
    matches, *_ = history_data()
    matches = matches.copy()
    # "T00" stops playing after the first 100 days
    gone = (matches["home_team"] == "T00") | (matches["away_team"] == "T00")
    matches = matches[~gone | (matches["date"] < matches["date"].min() + pd.Timedelta(days=100))]
    last = pd.DatetimeIndex([matches["date"].max()])

    history = poisson.strength_history(matches, last, active_days=100)

    assert "T00" not in set(history["team"])


def test_strength_history_file_round_trip(tmp_path):
    matches, *_ = history_data()
    path = tmp_path / "results" / "strength.csv"

    written = poisson.save_strength_history(matches, path)  # creates the folder too
    loaded = poisson.load_strength_history(path)

    pd.testing.assert_frame_equal(loaded, written)
    assert pd.api.types.is_datetime64_any_dtype(loaded["date"])


def newcomer_matches(base):
    """Add a newly promoted team 'NEW' that scores 0 goals in its first 3 matches."""
    start = base["date"].max() + pd.Timedelta(days=7)
    rows = [
        {"date": start, "home_team": "NEW", "away_team": "T00", "home_goals": 0, "away_goals": 1},
        {"date": start + pd.Timedelta(days=7), "home_team": "T01", "away_team": "NEW", "home_goals": 2, "away_goals": 0},
        {"date": start + pd.Timedelta(days=14), "home_team": "NEW", "away_team": "T02", "home_goals": 0, "away_goals": 1},
    ]
    return pd.concat([base, pd.DataFrame(rows)], ignore_index=True)


def test_without_ridge_a_scoreless_newcomer_makes_the_attack_estimate_diverge():
    base, *_ = simulate_matches(n_seasons=3)

    model = poisson.fit_poisson(newcomer_matches(base))  # ridge = 0 is the old behaviour

    assert model.attack["NEW"] < -8  # the known defect: "scored nothing" drives attack toward -infinity


def test_ridge_keeps_a_scoreless_newcomer_finite_and_leaves_other_teams_alone():
    base, *_ = simulate_matches(n_seasons=3)

    without_new = poisson.fit_poisson(base, ridge=15)
    with_new = poisson.fit_poisson(newcomer_matches(base), ridge=15)

    assert -3 < with_new.attack["NEW"] < 0  # pulled below average, but moderately
    shift = np.mean([abs(with_new.attack[t] - without_new.attack[t]) for t in without_new.attack])
    assert shift < 0.05  # the other 20 teams barely move (no league-wide spike)


def small_league():
    rng = np.random.default_rng(5)
    teams = ["A", "B", "C", "D"]
    rows = []
    for _ in range(2):
        for h in teams:
            for a in teams:
                if h != a:
                    rows.append((h, a, rng.poisson(1.5), rng.poisson(1.1)))
    df = pd.DataFrame(rows, columns=["home_team", "away_team", "home_goals", "away_goals"])
    df["date"] = pd.date_range("2023-01-01", periods=len(df), freq="3D")
    return df


def test_ridge_fit_matches_a_general_purpose_optimizer_on_the_same_objective():
    from scipy.optimize import minimize

    matches = small_league()
    ridge, half_life = 3.0, 60.0
    model = poisson.fit_poisson(matches, ridge=ridge, half_life=half_life)

    # Independent check: write the objective out explicitly and hand it to scipy.
    teams = ["A", "B", "C", "D"]
    weights = poisson.recency_weights(matches["date"], matches["date"].max(), half_life)

    def negative_objective(p):
        intercept, home, attack, defence = p[0], p[1], p[2:6], p[6:10]
        total = 0.0
        for row, w in zip(matches.itertuples(), weights):
            h, a = teams.index(row.home_team), teams.index(row.away_team)
            eta_home = intercept + home + attack[h] + defence[a]
            eta_away = intercept + attack[a] + defence[h]
            total += w * (row.home_goals * eta_home - np.exp(eta_home))
            total += w * (row.away_goals * eta_away - np.exp(eta_away))
        return -(total - 0.5 * ridge * (np.sum(attack**2) + np.sum(defence**2)))

    start = np.zeros(10)
    start[0] = 0.2
    best = minimize(negative_objective, start, method="BFGS", options={"gtol": 1e-10}).x

    assert model.intercept == pytest.approx(best[0], abs=1e-3)
    assert model.home_advantage == pytest.approx(best[1], abs=1e-3)
    assert [model.attack[t] for t in teams] == pytest.approx(best[2:6], abs=1e-3)
    assert [model.defence[t] for t in teams] == pytest.approx(best[6:10], abs=1e-3)


def test_ridge_effects_sum_to_zero():
    matches, *_ = simulate_matches(n_seasons=2)

    model = poisson.fit_poisson(matches, ridge=10)

    assert sum(model.attack.values()) == pytest.approx(0, abs=1e-6)
    assert sum(model.defence.values()) == pytest.approx(0, abs=1e-6)


def test_tiny_ridge_is_almost_the_unpenalized_fit():
    matches, *_ = simulate_matches(n_seasons=4)

    plain = poisson.fit_poisson(matches)
    barely = poisson.fit_poisson(matches, ridge=1e-6)

    assert barely.home_advantage == pytest.approx(plain.home_advantage, abs=1e-3)
    assert barely.attack == pytest.approx(plain.attack, abs=1e-3)
    assert barely.defence == pytest.approx(plain.defence, abs=1e-3)


def test_huge_ridge_shrinks_team_effects_to_zero_but_still_fits_home_advantage():
    matches, _, _, _, home_adv = simulate_matches(n_seasons=4)

    model = poisson.fit_poisson(matches, ridge=1e8)

    assert max(abs(v) for v in model.attack.values()) < 1e-3
    assert max(abs(v) for v in model.defence.values()) < 1e-3
    assert model.home_advantage == pytest.approx(home_adv, abs=0.1)


def test_ridge_shrinks_more_when_the_penalty_is_larger():
    matches, *_ = simulate_matches(n_seasons=2)

    spread = lambda m: np.std(list(m.attack.values()))

    assert spread(poisson.fit_poisson(matches, ridge=50)) < spread(poisson.fit_poisson(matches, ridge=5))
    assert spread(poisson.fit_poisson(matches, ridge=5)) < spread(poisson.fit_poisson(matches))


def test_ridge_fit_respects_as_of_date_and_works_with_half_life():
    base = small_dated_matches()
    cutoff = pd.Timestamp("2023-10-02")
    extra = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-10-02", "2023-12-01"]),
            "home_team": ["A", "B"],
            "away_team": ["B", "A"],
            "home_goals": [9, 9],
            "away_goals": [0, 0],
        }
    )

    without = poisson.fit_poisson(base, as_of_date=cutoff, ridge=5, half_life=90)
    with_future = poisson.fit_poisson(pd.concat([base, extra]), as_of_date=cutoff, ridge=5, half_life=90)

    assert with_future.attack == pytest.approx(without.attack)
    assert with_future.home_advantage == pytest.approx(without.home_advantage)


def test_negative_ridge_is_rejected():
    with pytest.raises(ValueError, match="ridge"):
        poisson.fit_poisson(small_dated_matches(), ridge=-1)


def test_strength_history_stays_sane_when_a_newcomer_has_scored_nothing():
    base, *_ = simulate_matches(n_seasons=3)
    matches = newcomer_matches(base)
    snapshot = pd.DatetimeIndex([matches["date"].max() + pd.Timedelta(days=1)])

    history = poisson.strength_history(matches, snapshot, min_matches=50)

    assert history["attack"].abs().max() < 3  # no -20 for the scoreless newcomer, no league-wide jump
    assert "NEW" in set(history["team"])
