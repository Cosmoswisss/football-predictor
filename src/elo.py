"""Elo rating baseline: ratings updated match by match, turned into H/D/A probabilities."""

import numpy as np
import pandas as pd

# Conventional starting values, to be tuned on the validation seasons (not on the test seasons).
INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_BONUS = 65.0  # rating points added to the home side
DRAW_WIDTH = 0.5  # larger = more draws

SCORE = {"H": 1.0, "D": 0.5, "A": 0.0}


def expected_score(rating_diff: float | np.ndarray) -> float | np.ndarray:
    """Expected score (win = 1, draw = 0.5, loss = 0) of a side rated rating_diff higher.

    1 / (1 + 10^(-diff/400)): a 400-point edge means about a 91% expected score.
    """
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(rating_diff) / 400.0))


def update_ratings(
    rating_home: float,
    rating_away: float,
    result: str,
    k: float = K_FACTOR,
    home_bonus: float = HOME_BONUS,
) -> tuple[float, float]:
    """Return the new (home, away) ratings after one match.

    The home side moves by k * (actual score - expected score), the away
    side by the opposite amount, so the rating points in the league are
    conserved. An upset moves ratings more than an expected result.
    """
    expected_home = float(expected_score(rating_home + home_bonus - rating_away))
    change = k * (SCORE[result] - expected_home)
    return rating_home + change, rating_away - change


def elo_probabilities(
    rating_home: float | np.ndarray,
    rating_away: float | np.ndarray,
    home_bonus: float = HOME_BONUS,
    draw_width: float = DRAW_WIDTH,
) -> np.ndarray:
    """Return probabilities [home win, draw, away win], shape (..., 3).

    Ordered-logit form with x = rating difference (incl. home bonus) on the
    logistic scale: P(home) = sigmoid(x - c), P(away) = sigmoid(-x - c) and
    P(draw) is what is left. c = draw_width; c = 0 means no draws at all.
    """
    diff = np.asarray(rating_home) + home_bonus - np.asarray(rating_away)
    x = diff * np.log(10.0) / 400.0  # same scale as expected_score
    p_home = 1.0 / (1.0 + np.exp(-(x - draw_width)))
    p_away = 1.0 / (1.0 + np.exp(x + draw_width))
    return np.stack([p_home, 1.0 - p_home - p_away, p_away], axis=-1)


def run_elo(
    matches: pd.DataFrame,
    k: float = K_FACTOR,
    home_bonus: float = HOME_BONUS,
    draw_width: float = DRAW_WIDTH,
    initial_rating: float = INITIAL_RATING,
) -> tuple[np.ndarray, dict[str, float]]:
    """Walk through matches in date order; return pre-match probabilities and final ratings.

    Row i of the (n_matches, 3) probability array is the prediction made
    BEFORE match i was played: it only uses ratings updated by earlier
    matches (no data leakage). New teams start at initial_rating.
    matches must be sorted by date and have home_team, away_team, result.
    """
    if not matches["date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date")

    ratings: dict[str, float] = {}
    probs = np.empty((len(matches), 3))
    for i, (home, away, result) in enumerate(
        zip(matches["home_team"], matches["away_team"], matches["result"])
    ):
        r_home = ratings.get(home, initial_rating)
        r_away = ratings.get(away, initial_rating)
        probs[i] = elo_probabilities(r_home, r_away, home_bonus, draw_width)  # predict first...
        ratings[home], ratings[away] = update_ratings(r_home, r_away, result, k, home_bonus)  # ...then learn
    return probs, ratings
