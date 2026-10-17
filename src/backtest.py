"""Evaluation of match predictions: margin removal, scoring metrics, backtests.

Probabilities are arrays of shape (n_matches, 3) with columns
[home win, draw, away win]. Outcomes are integers 0 (home), 1 (draw), 2 (away).
"""

import numpy as np
import pandas as pd

from itertools import product

from src.elo import run_elo
from src.poisson import fit_poisson, predict_match
from src.simulate import add_unseen_teams

OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}


def outcome_index(result: pd.Series) -> np.ndarray:
    """Convert the result column ('H', 'D', 'A') to 0, 1, 2."""
    return result.map(OUTCOME_INDEX).to_numpy()


def remove_margin(
    odds_home: np.ndarray, odds_draw: np.ndarray, odds_away: np.ndarray
) -> np.ndarray:
    """Turn decimal odds into probabilities with the bookmaker margin removed.

    1/odds is the "implied probability", but the three implied probabilities
    add up to more than 1 (the booksum, e.g. 1.05): the excess is the
    bookmaker's margin. Dividing each by the booksum rescales them to sum to
    exactly 1. Rows with missing odds stay NaN.
    """
    odds = np.column_stack([odds_home, odds_draw, odds_away]).astype(float)
    if np.any(odds <= 0):
        raise ValueError("Odds must be positive")
    implied = 1.0 / odds
    return implied / implied.sum(axis=1, keepdims=True)


def _check_probabilities(probs: np.ndarray, outcomes: np.ndarray) -> None:
    """Raise ValueError unless probs is (n, 3) with rows summing to 1."""
    if probs.ndim != 2 or probs.shape[1] != 3 or len(probs) != len(outcomes):
        raise ValueError("probs must have shape (n_matches, 3), one outcome per row")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each row of probs must sum to 1")


def per_match_log_loss(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """-ln(probability given to what actually happened), one value per match.

    Probabilities are clipped at 1e-15 to avoid ln(0).
    """
    _check_probabilities(probs, outcomes)
    chosen = probs[np.arange(len(outcomes)), outcomes]
    return -np.log(np.clip(chosen, 1e-15, 1.0))


def per_match_brier(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """sum_k (p_k - 1[k happened])^2, one value per match."""
    _check_probabilities(probs, outcomes)
    actual = np.zeros_like(probs)
    actual[np.arange(len(outcomes)), outcomes] = 1.0
    return ((probs - actual) ** 2).sum(axis=1)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean of -ln(probability given to what actually happened). Lower is better.

    Always saying 1/3 scores ln(3) = 1.0986. Confident wrong predictions are
    punished heavily.
    """
    return float(per_match_log_loss(probs, outcomes).mean())


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean over matches of sum_k (p_k - 1[k happened])^2. Lower is better.

    Ranges from 0 (perfect) to 2 (confidently wrong); always saying 1/3 gives 2/3.
    Unlike log-loss, it does not blow up on a single very wrong prediction.
    """
    return float(per_match_brier(probs, outcomes).mean())


def compare_models(
    predictions: dict[str, np.ndarray], outcomes: np.ndarray, reference: str = "bookmaker"
) -> pd.DataFrame:
    """Score several models on the same matches and compare each with a reference.

    predictions maps a model name to its (n, 3) probabilities. Besides the mean
    log-loss and Brier score, each non-reference model gets the *paired*
    difference to the reference: mean of (model loss - reference loss) over
    the same matches, with its standard error std(d) / sqrt(n). A difference
    smaller than about two standard errors is indistinguishable from luck.
    (Matches are not fully independent, so the true uncertainty is a bit larger.)
    """
    if reference not in predictions:
        raise ValueError(f"Reference model {reference!r} not in predictions")

    def per_match(probs):
        return per_match_log_loss(probs, outcomes), per_match_brier(probs, outcomes)

    ref_ll, ref_br = per_match(predictions[reference])
    n = len(outcomes)
    rows = []
    for name, probs in predictions.items():
        ll, br = per_match(probs)
        row = {"model": name, "log_loss": ll.mean(), "brier": br.mean()}
        for label, loss, ref_loss in [("ll", ll, ref_ll), ("brier", br, ref_br)]:
            if name == reference:
                row[f"{label}_diff"] = row[f"{label}_se"] = np.nan
            else:
                diff = loss - ref_loss
                row[f"{label}_diff"] = diff.mean()
                row[f"{label}_se"] = diff.std(ddof=1) / np.sqrt(n)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def poisson_walk_forward(
    matches: pd.DataFrame,
    eval_mask: pd.Series,
    half_life: float | None = None,
    max_goals: int = 10,
) -> np.ndarray:
    """Predict every evaluation match with a model that has only seen the past.

    For each distinct match date among the evaluation matches, the Poisson
    model is refitted on matches strictly before that date (all matches that
    day share one fit, since none of them is "before" another) and used to
    predict that day's matches. Teams the model has never seen (just
    promoted) get average strength.

    matches: all matches, sorted by date, with goals and teams.
    eval_mask: boolean Series aligned with matches, True for matches to predict.
    Returns an (n_eval, 3) array of [home, draw, away] probabilities, in the
    order the evaluation matches appear in matches.
    """
    eval_matches = matches[eval_mask].reset_index(drop=True)
    probs = np.empty((len(eval_matches), 3))

    for date, day in eval_matches.groupby("date"):
        model = fit_poisson(matches, as_of_date=date, half_life=half_life)
        teams = list(set(day["home_team"]) | set(day["away_team"]))
        model, _ = add_unseen_teams(model, teams)
        for row, home, away in zip(day.index, day["home_team"], day["away_team"]):
            probs[row] = predict_match(model, home, away, max_goals)
    return probs


def tune_elo(
    matches: pd.DataFrame,
    eval_mask: pd.Series,
    k_values: list[float],
    home_bonuses: list[float],
    draw_widths: list[float],
) -> pd.DataFrame:
    """Grid-search the Elo hyperparameters on the evaluation (validation) matches.

    Elo is run over ALL matches in date order (earlier seasons are warm-up);
    only the matches in eval_mask are scored. Returns one row per combination,
    best log-loss first. Use validation seasons here, never the test seasons.
    """
    outcomes = outcome_index(matches.loc[eval_mask, "result"])
    mask = eval_mask.to_numpy()
    rows = []
    for k, bonus, width in product(k_values, home_bonuses, draw_widths):
        probs, _ = run_elo(matches, k=k, home_bonus=bonus, draw_width=width)
        rows.append(
            {
                "k": k,
                "home_bonus": bonus,
                "draw_width": width,
                "log_loss": log_loss(probs[mask], outcomes),
                "brier": brier_score(probs[mask], outcomes),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def tune_half_life(
    matches: pd.DataFrame, eval_mask: pd.Series, half_lives: list[float | None]
) -> pd.DataFrame:
    """Score the walk-forward Poisson model for each half-life (None = no weighting).

    Returns one row per half-life, best log-loss first. In the table, a
    half_life of NaN means "no weighting" (pandas turns None into NaN).
    """
    outcomes = outcome_index(matches.loc[eval_mask, "result"])
    rows = []
    for half_life in half_lives:
        probs = poisson_walk_forward(matches, eval_mask, half_life=half_life)
        rows.append(
            {
                "half_life": half_life,
                "log_loss": log_loss(probs, outcomes),
                "brier": brier_score(probs, outcomes),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


# Data split and settings frozen after tuning on the validation seasons only.
# The test seasons were scored once with exactly these values.
VALIDATION_SEASONS = ["2018/19", "2019/20", "2020/21"]
TEST_SEASONS = ["2021/22", "2022/23", "2023/24", "2024/25"]
FROZEN_ELO = {"k": 25.0, "home_bonus": 55.0, "draw_width": 0.5}
FROZEN_HALF_LIFE = 365


def final_test_predictions(
    matches: pd.DataFrame, test_seasons: list[str] = TEST_SEASONS
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Predict the test seasons with every model, using the frozen settings.

    Returns (predictions, outcomes, test_matches), all aligned row by row.
    Only matches with complete bookmaker odds are scored, so every model is
    judged on exactly the same matches. matches must be sorted by date.
    """
    odds = matches[["odds_home", "odds_draw", "odds_away"]]
    mask = matches["season"].isin(test_seasons) & odds.notna().all(axis=1)
    test_matches = matches[mask].reset_index(drop=True)
    outcomes = outcome_index(test_matches["result"])
    keep = mask.to_numpy()

    predictions = {
        "bookmaker": remove_margin(
            test_matches["odds_home"], test_matches["odds_draw"], test_matches["odds_away"]
        ),
        "elo": run_elo(matches, **FROZEN_ELO)[0][keep],
        "poisson (plain)": poisson_walk_forward(matches, mask, half_life=None),
        f"poisson (half-life {FROZEN_HALF_LIFE}d)": poisson_walk_forward(
            matches, mask, half_life=FROZEN_HALF_LIFE
        ),
    }
    return predictions, outcomes, test_matches
