import numpy as np
import pandas as pd
import pytest

from src import elo


def test_expected_score_known_values_and_symmetry():
    assert elo.expected_score(0) == pytest.approx(0.5)
    assert elo.expected_score(400) == pytest.approx(10 / 11)
    assert elo.expected_score(120) + elo.expected_score(-120) == pytest.approx(1.0)


def test_update_ratings_equal_teams_home_win_moves_k_over_two():
    home, away = elo.update_ratings(1500, 1500, "H", k=20, home_bonus=0)

    assert (home, away) == (pytest.approx(1510), pytest.approx(1490))


def test_update_ratings_draw_between_equals_changes_nothing():
    assert elo.update_ratings(1500, 1500, "D", k=20, home_bonus=0) == (1500, 1500)


def test_update_ratings_home_bonus_means_a_draw_costs_the_home_side():
    home, away = elo.update_ratings(1500, 1500, "D", k=20, home_bonus=65)

    assert home < 1500 < away  # the home side was expected to do better than a draw


def test_update_ratings_conserves_rating_points_and_upsets_move_more():
    home, away = elo.update_ratings(1400, 1600, "H", k=20, home_bonus=65)
    assert home + away == pytest.approx(3000)

    upset = elo.update_ratings(1400, 1600, "H")[0] - 1400
    expected_win = elo.update_ratings(1600, 1400, "H")[0] - 1600
    assert upset > expected_win


def test_elo_probabilities_sum_to_one_and_are_non_negative():
    probs = elo.elo_probabilities(np.array([1300.0, 1500, 1800]), np.array([1700.0, 1500, 1400]))

    assert probs.shape == (3, 3)
    assert probs.sum(axis=1) == pytest.approx([1, 1, 1])
    assert (probs >= 0).all()


def test_elo_probabilities_equal_teams_without_bonus_are_symmetric():
    p_home, p_draw, p_away = elo.elo_probabilities(1500, 1500, home_bonus=0)

    assert p_home == pytest.approx(p_away)
    assert p_draw > 0.2


def test_elo_probabilities_stronger_team_and_home_bonus_help():
    weak_home = elo.elo_probabilities(1400, 1600, home_bonus=0)
    strong_home = elo.elo_probabilities(1600, 1400, home_bonus=0)
    assert strong_home[0] > weak_home[0]
    assert strong_home[2] < weak_home[2]

    without = elo.elo_probabilities(1500, 1500, home_bonus=0)
    with_bonus = elo.elo_probabilities(1500, 1500, home_bonus=65)
    assert with_bonus[0] > without[0]


def test_elo_probabilities_draws_are_most_likely_between_equal_teams():
    close = elo.elo_probabilities(1500, 1500, home_bonus=0)[1]
    far = elo.elo_probabilities(1800, 1300, home_bonus=0)[1]

    assert close > far


def test_elo_probabilities_zero_draw_width_means_no_draws():
    assert elo.elo_probabilities(1500, 1500, draw_width=0.0)[1] == pytest.approx(0.0)


def sample_matches():
    rows = [
        ("2023-08-01", "A", "B", "H"),
        ("2023-08-01", "C", "D", "D"),  # same day, different teams
        ("2023-08-08", "A", "C", "H"),
        ("2023-08-08", "B", "D", "A"),
        ("2023-08-15", "A", "D", "H"),
        ("2023-08-15", "B", "C", "A"),
    ]
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "result"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_run_elo_first_prediction_uses_initial_ratings():
    probs, _ = elo.run_elo(sample_matches())

    assert probs[0] == pytest.approx(elo.elo_probabilities(1500, 1500))
    assert probs.shape == (6, 3)
    assert probs.sum(axis=1) == pytest.approx(np.ones(6))


def test_run_elo_conserves_total_rating_and_rewards_winners():
    _, ratings = elo.run_elo(sample_matches())

    assert sum(ratings.values()) == pytest.approx(4 * 1500)
    assert max(ratings, key=ratings.get) == "A"  # A won all its matches


def test_run_elo_predictions_never_use_the_match_itself_or_later_ones():
    matches = sample_matches()
    changed = matches.copy()
    changed.loc[changed.index[-1], "result"] = "H"  # alter only the LAST match's result

    original_probs, _ = elo.run_elo(matches)
    changed_probs, _ = elo.run_elo(changed)

    assert np.array_equal(original_probs, changed_probs)  # even the last prediction is unchanged


def test_run_elo_same_day_order_does_not_matter():
    matches = sample_matches()
    swapped = matches.copy()
    swapped.iloc[[0, 1]] = matches.iloc[[1, 0]].to_numpy()  # swap the two matches on day 1

    original_probs, ratings = elo.run_elo(matches)
    swapped_probs, swapped_ratings = elo.run_elo(swapped)

    assert ratings == pytest.approx(swapped_ratings)
    assert original_probs[[0, 1]] == pytest.approx(swapped_probs[[1, 0]])


def test_run_elo_rejects_unsorted_matches():
    with pytest.raises(ValueError, match="sorted"):
        elo.run_elo(sample_matches().iloc[::-1])
