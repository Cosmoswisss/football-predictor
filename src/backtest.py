"""Evaluation of match predictions: margin removal, scoring metrics, backtests.

Probabilities are arrays of shape (n_matches, 3) with columns
[home win, draw, away win]. Outcomes are integers 0 (home), 1 (draw), 2 (away).
"""

import numpy as np
import pandas as pd

from itertools import product
from pathlib import Path

from src.elo import run_elo
from src.poisson import MODEL_HALF_LIFE, MODEL_RIDGE, fit_poisson, predict_match
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
    ridge: float = 0.0,
    max_goals: int = 10,
) -> np.ndarray:
    """Predict every evaluation match with a model that has only seen the past.

    For each distinct match date among the evaluation matches, the Poisson
    model is refitted on matches strictly before that date (all matches that
    day share one fit, since none of them is "before" another) and used to
    predict that day's matches. Teams the model has never seen (just
    promoted) get average strength. half_life and ridge are passed to fit_poisson.

    matches: all matches, sorted by date, with goals and teams.
    eval_mask: boolean Series aligned with matches, True for matches to predict.
    Returns an (n_eval, 3) array of [home, draw, away] probabilities, in the
    order the evaluation matches appear in matches.
    """
    eval_matches = matches[eval_mask].reset_index(drop=True)
    probs = np.empty((len(eval_matches), 3))

    for date, day in eval_matches.groupby("date"):
        model = fit_poisson(matches, as_of_date=date, half_life=half_life, ridge=ridge)
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


def tune_poisson(
    matches: pd.DataFrame,
    eval_mask: pd.Series,
    half_lives: list[float | None],
    ridges: list[float],
) -> pd.DataFrame:
    """Score the walk-forward Poisson model for every half-life x ridge combination.

    half_life None means no time weighting; ridge 0 means no penalty. Returns
    one row per combination, best log-loss first. In the table a half_life of
    NaN means "no weighting" (pandas turns None into NaN).
    """
    outcomes = outcome_index(matches.loc[eval_mask, "result"])
    rows = []
    for half_life, ridge in product(half_lives, ridges):
        probs = poisson_walk_forward(matches, eval_mask, half_life=half_life, ridge=ridge)
        rows.append(
            {
                "half_life": half_life,
                "ridge": ridge,
                "log_loss": log_loss(probs, outcomes),
                "brier": brier_score(probs, outcomes),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def tune_half_life(
    matches: pd.DataFrame, eval_mask: pd.Series, half_lives: list[float | None]
) -> pd.DataFrame:
    """Score the walk-forward Poisson model (no penalty) for each half-life.

    Returns one row per half-life, best log-loss first. In the table, a
    half_life of NaN means "no weighting" (pandas turns None into NaN).
    """
    table = tune_poisson(matches, eval_mask, half_lives, [0.0])
    return table.drop(columns="ridge")


# Data split and settings frozen after tuning on the validation seasons only.
# The test seasons were scored once with exactly these values.
VALIDATION_SEASONS = ["2018/19", "2019/20", "2020/21"]
TEST_SEASONS = ["2021/22", "2022/23", "2023/24", "2024/25"]
# Never looked at before the v2 settings were frozen: the clean test for v1 versus v2.
CLEAN_TEST_SEASONS = ["2025/26"]
FROZEN_ELO = {"k": 25.0, "home_bonus": 55.0, "draw_width": 0.5}
FROZEN_HALF_LIFE = MODEL_HALF_LIFE  # v1 and v2 both use this half-life
FROZEN_RIDGE = MODEL_RIDGE  # v2 only: v1 (scored on 2021/22-2024/25 before the fix) had no penalty


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
        f"poisson (half-life {FROZEN_HALF_LIFE}d + ridge {FROZEN_RIDGE:g})": poisson_walk_forward(
            matches, mask, half_life=FROZEN_HALF_LIFE, ridge=FROZEN_RIDGE
        ),
    }
    return predictions, outcomes, test_matches


def calibration_table(
    probs: np.ndarray, outcomes: np.ndarray, outcome: int = 0, n_bins: int = 10
) -> pd.DataFrame:
    """Compare predicted probabilities of one outcome with how often it happened.

    Predictions for `outcome` (0 = home win, 1 = draw, 2 = away win) are
    sorted into n_bins equal-width bins on [0, 1]. For each non-empty bin we
    return the mean predicted probability, the observed frequency of the
    outcome, and the number of matches. A well-calibrated model has
    mean_predicted close to observed_frequency in every bin.
    """
    _check_probabilities(probs, outcomes)
    if outcome not in (0, 1, 2) or n_bins < 1:
        raise ValueError("outcome must be 0, 1 or 2 and n_bins at least 1")

    predicted = probs[:, outcome]
    happened = (outcomes == outcome).astype(float)
    # floor(p * n_bins) gives the bin number; p = 1.0 goes in the last bin
    bin_index = np.minimum(np.floor(predicted * n_bins).astype(int), n_bins - 1)

    rows = []
    for b in range(n_bins):
        in_bin = bin_index == b
        if in_bin.any():
            rows.append(
                {
                    "bin_low": b / n_bins,
                    "bin_high": (b + 1) / n_bins,
                    "mean_predicted": predicted[in_bin].mean(),
                    "observed_frequency": happened[in_bin].mean(),
                    "count": int(in_bin.sum()),
                }
            )
    return pd.DataFrame(rows)


RESULTS_PATH = Path("results/test_predictions.csv")
OUTCOME_NAMES = ["home", "draw", "away"]
MATCH_COLUMNS = ["date", "season", "home_team", "away_team", "result"]


def predictions_to_frame(
    predictions: dict[str, np.ndarray], test_matches: pd.DataFrame
) -> pd.DataFrame:
    """One table: the match columns plus <model>_home/_draw/_away for every model."""
    frame = test_matches[MATCH_COLUMNS].reset_index(drop=True).copy()
    for model, probs in predictions.items():
        for i, outcome in enumerate(OUTCOME_NAMES):
            frame[f"{model}_{outcome}"] = probs[:, i]
    return frame


def frame_to_predictions(
    frame: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Inverse of predictions_to_frame: (predictions, outcomes, test_matches)."""
    models = [c[: -len("_home")] for c in frame.columns if c.endswith("_home")]
    predictions = {
        model: frame[[f"{model}_{o}" for o in OUTCOME_NAMES]].to_numpy() for model in models
    }
    test_matches = frame[MATCH_COLUMNS].copy()
    return predictions, outcome_index(test_matches["result"]), test_matches


def save_test_predictions(
    matches: pd.DataFrame,
    path: Path = RESULTS_PATH,
    test_seasons: list[str] = TEST_SEASONS + CLEAN_TEST_SEASONS,
) -> pd.DataFrame:
    """Compute the frozen models' test-season predictions and write them to a CSV.

    Slow (about two minutes: the Poisson model is refitted on every match date),
    so the dashboard reads the saved file instead. Returns the table written.
    """
    predictions, _, test_matches = final_test_predictions(matches, test_seasons)
    frame = predictions_to_frame(predictions, test_matches)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_test_predictions(
    path: Path = RESULTS_PATH,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Read a file written by save_test_predictions: (predictions, outcomes, test_matches)."""
    return frame_to_predictions(pd.read_csv(path, parse_dates=["date"]))


def log_loss_by_group(
    predictions: dict[str, np.ndarray], outcomes: np.ndarray, groups: pd.Series
) -> pd.DataFrame:
    """Log-loss of every model within each group (e.g. each season).

    Returns a table with one row per group and one column per model.
    """
    groups = np.asarray(groups)
    rows = {}
    for group in sorted(set(groups)):
        in_group = groups == group
        rows[group] = {
            model: log_loss(probs[in_group], outcomes[in_group])
            for model, probs in predictions.items()
        }
    return pd.DataFrame(rows).T
