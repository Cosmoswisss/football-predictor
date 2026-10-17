import numpy as np
import pandas as pd
import pytest

from src import backtest


def test_outcome_index_maps_h_d_a():
    result = pd.Series(["H", "D", "A", "H"])

    assert backtest.outcome_index(result).tolist() == [0, 1, 2, 0]


def test_remove_margin_gives_probabilities_summing_to_one():
    probs = backtest.remove_margin(np.array([2.0]), np.array([3.5]), np.array([4.0]))

    # implied: 0.5, 0.2857, 0.25 -> booksum 1.0357 (a 3.6% margin)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, 0] == pytest.approx(0.5 / (0.5 + 1 / 3.5 + 0.25))


def test_remove_margin_keeps_ordering_and_fair_odds_unchanged():
    # Odds that already have no margin: 1/2 + 1/4 + 1/4 = 1
    probs = backtest.remove_margin(np.array([2.0]), np.array([4.0]), np.array([4.0]))

    assert probs[0].tolist() == pytest.approx([0.5, 0.25, 0.25])


def test_remove_margin_works_row_by_row_and_keeps_nan():
    probs = backtest.remove_margin(
        np.array([2.0, np.nan]), np.array([4.0, 3.0]), np.array([4.0, 3.0])
    )

    assert probs.shape == (2, 3)
    assert probs[0].sum() == pytest.approx(1.0)
    assert np.isnan(probs[1]).all()


def test_remove_margin_rejects_non_positive_odds():
    with pytest.raises(ValueError):
        backtest.remove_margin(np.array([0.0]), np.array([3.0]), np.array([3.0]))


def test_log_loss_hand_value_and_uniform_baseline():
    probs = np.array([[0.5, 0.3, 0.2]])

    assert backtest.log_loss(probs, np.array([0])) == pytest.approx(-np.log(0.5))
    uniform = np.full((4, 3), 1 / 3)
    assert backtest.log_loss(uniform, np.array([0, 1, 2, 0])) == pytest.approx(np.log(3))


def test_log_loss_perfect_prediction_is_zero_and_wrong_one_is_large_but_finite():
    perfect = np.array([[1.0, 0.0, 0.0]])
    confidently_wrong = np.array([[0.0, 0.0, 1.0]])

    assert backtest.log_loss(perfect, np.array([0])) == pytest.approx(0.0)
    assert 30 < backtest.log_loss(confidently_wrong, np.array([0])) < 40  # -ln(1e-15) ~ 34.5


def test_brier_score_hand_value_and_uniform_baseline():
    probs = np.array([[0.5, 0.3, 0.2]])

    # (0.5-1)^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04
    assert backtest.brier_score(probs, np.array([0])) == pytest.approx(0.38)
    uniform = np.full((3, 3), 1 / 3)
    assert backtest.brier_score(uniform, np.array([0, 1, 2])) == pytest.approx(2 / 3)


def test_brier_score_perfect_and_worst():
    assert backtest.brier_score(np.array([[1.0, 0.0, 0.0]]), np.array([0])) == 0.0
    assert backtest.brier_score(np.array([[0.0, 0.0, 1.0]]), np.array([0])) == pytest.approx(2.0)


def test_better_forecast_scores_lower_on_both_metrics():
    outcomes = np.array([0, 0, 1, 2])
    sharp = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    vague = np.full((4, 3), 1 / 3)

    assert backtest.log_loss(sharp, outcomes) < backtest.log_loss(vague, outcomes)
    assert backtest.brier_score(sharp, outcomes) < backtest.brier_score(vague, outcomes)


def test_metrics_reject_bad_shapes_and_unnormalised_rows():
    with pytest.raises(ValueError):
        backtest.log_loss(np.array([[0.5, 0.5, 0.5]]), np.array([0]))  # sums to 1.5
    with pytest.raises(ValueError):
        backtest.brier_score(np.array([[0.5, 0.5]]), np.array([0]))  # only 2 columns


def walk_forward_data(n_seasons=3):
    """Simulated league history; the last 3 match days are the evaluation set."""
    from tests.test_poisson import simulate_matches

    matches, *_ = simulate_matches(n_seasons=n_seasons)
    matches = matches.sort_values("date", kind="mergesort").reset_index(drop=True)
    last_dates = sorted(matches["date"].unique())[-3:]
    return matches, matches["date"].isin(last_dates)


def test_poisson_walk_forward_returns_valid_probabilities_for_eval_matches_only():
    matches, mask = walk_forward_data()

    probs = backtest.poisson_walk_forward(matches, mask)

    assert probs.shape == (mask.sum(), 3)
    assert probs.sum(axis=1) == pytest.approx(np.ones(mask.sum()))
    assert (probs > 0).all()


def test_poisson_walk_forward_never_uses_the_match_day_itself_or_later():
    matches, mask = walk_forward_data()
    last_date = matches["date"].max()
    altered = matches.copy()
    on_last_day = altered["date"] == last_date
    altered.loc[on_last_day, ["home_goals", "away_goals"]] = [9, 0]  # absurd scores on the final day

    original = backtest.poisson_walk_forward(matches, mask)
    after = backtest.poisson_walk_forward(altered, mask)

    in_eval_last_day = (matches.loc[mask, "date"] == last_date).to_numpy()
    # Predictions for the final day must be identical: its own results are not allowed in.
    assert np.array_equal(original[in_eval_last_day], after[in_eval_last_day])
    # Earlier days are untouched too.
    assert np.array_equal(original[~in_eval_last_day], after[~in_eval_last_day])


def test_poisson_walk_forward_earlier_results_do_change_later_predictions():
    matches, mask = walk_forward_data()
    first_eval_date = matches.loc[mask, "date"].min()
    altered = matches.copy()
    # Make a past day's results absurd: it IS allowed to influence later days.
    before = altered["date"] == sorted(matches["date"].unique())[-4]
    altered.loc[before, ["home_goals", "away_goals"]] = [9, 0]

    original = backtest.poisson_walk_forward(matches, mask)
    after = backtest.poisson_walk_forward(altered, mask)

    assert not np.array_equal(original, after)
    assert first_eval_date > matches.loc[before, "date"].max()  # the altered day really is earlier


def test_poisson_walk_forward_handles_team_unseen_before_the_eval_day():
    matches, mask = walk_forward_data(n_seasons=2)
    newcomer = matches[mask].iloc[[0]].copy()
    newcomer["home_team"] = "Newcomer"  # appears for the first time on an evaluation day
    matches = pd.concat([matches, newcomer], ignore_index=True)
    matches = matches.sort_values("date", kind="mergesort").reset_index(drop=True)
    mask = matches["date"].isin(sorted(matches["date"].unique())[-3:])

    probs = backtest.poisson_walk_forward(matches, mask)

    assert np.isfinite(probs).all()
    assert probs.sum(axis=1) == pytest.approx(np.ones(mask.sum()))


def test_poisson_walk_forward_accepts_half_life():
    matches, mask = walk_forward_data()

    plain = backtest.poisson_walk_forward(matches, mask)
    weighted = backtest.poisson_walk_forward(matches, mask, half_life=60)

    assert weighted.shape == plain.shape
    assert not np.array_equal(plain, weighted)


def probs_with_losses(losses):
    """Probability rows (outcome 0 happened) whose log-loss equals the given losses."""
    p = np.exp(-np.array(losses))
    rest = (1 - p) / 2
    return np.column_stack([p, rest, rest])


def test_per_match_losses_average_to_the_scores():
    probs = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
    outcomes = np.array([0, 1])

    assert backtest.per_match_log_loss(probs, outcomes).mean() == pytest.approx(
        backtest.log_loss(probs, outcomes)
    )
    assert backtest.per_match_brier(probs, outcomes).mean() == pytest.approx(
        backtest.brier_score(probs, outcomes)
    )


def test_compare_models_paired_difference_and_standard_error():
    outcomes = np.zeros(4, dtype=int)
    reference = probs_with_losses([1.0, 1.0, 1.0, 1.0])
    model = probs_with_losses([0.2, 0.4, 0.6, 0.8])

    table = backtest.compare_models({"bookmaker": reference, "m": model}, outcomes).set_index("model")

    # per-match differences are -0.8, -0.6, -0.4, -0.2: mean -0.5, sample std 0.2582
    assert table.loc["m", "ll_diff"] == pytest.approx(-0.5)
    assert table.loc["m", "ll_se"] == pytest.approx(0.2 * np.sqrt(5 / 3) / 2)
    assert table.loc["m", "log_loss"] == pytest.approx(0.5)
    assert np.isnan(table.loc["bookmaker", "ll_diff"])


def test_compare_models_sorts_best_first_and_needs_the_reference():
    outcomes = np.array([0, 1, 2])
    good = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    vague = np.full((3, 3), 1 / 3)

    table = backtest.compare_models({"bookmaker": vague, "good": good}, outcomes)

    assert table["model"].tolist() == ["good", "bookmaker"]
    with pytest.raises(ValueError, match="Reference"):
        backtest.compare_models({"a": good}, outcomes)


def with_result(matches):
    matches = matches.copy()
    matches["result"] = np.where(
        matches["home_goals"] > matches["away_goals"],
        "H",
        np.where(matches["home_goals"] == matches["away_goals"], "D", "A"),
    )
    return matches


def test_tune_half_life_returns_one_sorted_row_per_value():
    matches, mask = walk_forward_data()

    table = backtest.tune_half_life(with_result(matches), mask, [30, None])

    assert len(table) == 2
    assert 30 in table["half_life"].tolist()
    assert table["half_life"].isna().sum() == 1  # NaN stands for "no weighting"
    assert table["log_loss"].is_monotonic_increasing


def test_tune_elo_returns_one_sorted_row_per_combination():
    matches, mask = walk_forward_data()

    table = backtest.tune_elo(with_result(matches), mask, [10, 30], [0, 65], [0.5])

    assert len(table) == 4
    assert list(table.columns) == ["k", "home_bonus", "draw_width", "log_loss", "brier"]
    assert table["log_loss"].is_monotonic_increasing


def test_final_test_predictions_scores_all_models_on_the_same_matches():
    matches, mask = walk_forward_data()
    matches = with_result(matches)
    matches["season"] = np.where(mask, "TEST", "TRAIN")
    matches["odds_home"], matches["odds_draw"], matches["odds_away"] = 2.4, 3.3, 3.1
    matches.loc[matches.index[mask][0], "odds_draw"] = np.nan  # one match without odds is skipped

    preds, outcomes, test_matches = backtest.final_test_predictions(matches, ["TEST"])

    assert len(test_matches) == mask.sum() - 1
    assert set(preds) == {"bookmaker", "elo", "poisson (plain)", "poisson (half-life 365d)"}
    for probs in preds.values():
        assert probs.shape == (len(test_matches), 3)
        assert probs.sum(axis=1) == pytest.approx(np.ones(len(test_matches)))
    assert len(outcomes) == len(test_matches)


def test_frozen_settings_and_season_split_do_not_overlap():
    assert not set(backtest.VALIDATION_SEASONS) & set(backtest.TEST_SEASONS)
    assert backtest.FROZEN_ELO == {"k": 25.0, "home_bonus": 55.0, "draw_width": 0.5}
    assert backtest.FROZEN_HALF_LIFE == 365
