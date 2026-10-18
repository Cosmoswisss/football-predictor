"""Poisson model: fit team strengths and compute match outcome probabilities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson as poisson_dist

# Model settings ("v2"), frozen after tuning on the validation seasons 2018/19-2020/21.
MODEL_HALF_LIFE = 365  # days: a match this old counts half as much as one played today
MODEL_RIDGE = 4.0  # penalty on attack/defence effects, a Normal prior with sd 1/sqrt(4) = 0.5


def to_long_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn one row per match into two rows per match, one per team's goals.

    Input needs home_team, away_team, home_goals, away_goals. Output columns:
    team, opponent, is_home (1 or 0) and goals. The Poisson regression models
    "goals scored by a team against an opponent", so each match gives two
    observations: the home side's goals and the away side's goals.
    """
    home = pd.DataFrame(
        {
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "is_home": 1,
            "goals": matches["home_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": 0,
            "goals": matches["away_goals"],
        }
    )
    return pd.concat([home, away], ignore_index=True)


@dataclass(frozen=True)
class PoissonModel:
    """Fitted parameters (all on the log scale).

    Expected goals of team A against team B:
        exp(intercept + attack[A] + defence[B] (+ home_advantage if A is home))
    attack and defence each sum to zero over all teams, so 0 means "average".
    A larger defence value means a team concedes more (a weaker defence).
    """

    intercept: float
    home_advantage: float
    attack: dict[str, float]
    defence: dict[str, float]


def sum_to_zero_dummies(values: pd.Series, levels: list[str]) -> pd.DataFrame:
    """Encode a categorical column so the fitted effects sum to zero.

    One column per level except the last. A row of the last level gets -1 in
    every column, which forces the last effect to equal minus the sum of the
    others, hence all effects sum to zero.
    """
    dummies = pd.DataFrame({lvl: (values == lvl).astype(float) for lvl in levels[:-1]})
    dummies.loc[values == levels[-1], :] = -1.0
    return dummies


def recency_weights(
    dates: pd.Series, reference_date: pd.Timestamp, half_life: float
) -> np.ndarray:
    """Weight 0.5 ** (age / half_life): a match half_life days old counts half as much.

    age is the number of days between the match and reference_date.
    """
    age_days = (pd.Timestamp(reference_date) - dates).dt.days.to_numpy().clip(min=0)
    return 0.5 ** (age_days / half_life)


def fit_penalized_poisson(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    penalty: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> np.ndarray:
    """Maximize a ridge-penalized weighted Poisson log-likelihood.

    Finds beta maximizing  sum_i w_i * (y_i*eta_i - exp(eta_i)) - 0.5 * sum_j penalty_j * beta_j**2
    with eta = X @ beta. Column 0 of X must be the intercept. Newton's method
    (the same thing as iteratively reweighted least squares): at each step solve
    (X' W X + P) step = gradient, where W = diag(w * mu) and P = diag(penalty),
    then halve the step until the objective does not get worse.
    The penalty makes the objective strictly concave, so a finite maximum
    exists even when an unpenalized fit would diverge.
    """

    def objective(beta: np.ndarray) -> float:
        eta = X @ beta
        return float(np.sum(weights * (y * eta - np.exp(eta))) - 0.5 * np.sum(penalty * beta**2))

    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(np.average(y, weights=weights), 1e-3))  # start at the overall scoring rate

    for _ in range(max_iter):
        mu = np.exp(X @ beta)
        gradient = X.T @ (weights * (y - mu)) - penalty * beta
        hessian = (X * (weights * mu)[:, None]).T @ X + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)

        current, scale = objective(beta), 1.0
        while scale > 1e-8:
            candidate = objective(beta + scale * step)
            if np.isfinite(candidate) and candidate >= current - 1e-12:
                break
            scale /= 2
        beta = beta + scale * step
        if np.max(np.abs(scale * step)) < tol:
            break
    return beta


def fit_ridge_poisson(
    long: pd.DataFrame, teams: list[str], weights: np.ndarray | None, ridge: float
) -> PoissonModel:
    """Penalized fit on the long-format table; used by fit_poisson when ridge > 0.

    One attack and one defence column per team (no sum-to-zero trick needed).
    Only the attack and defence effects are penalized, not the intercept or
    the home advantage. The effects come out summing to zero by themselves:
    shifting all of them by c and the intercept by -c changes nothing in the
    likelihood but increases the penalty, so the optimum is the centered one.
    """
    index = {team: i for i, team in enumerate(teams)}
    n, n_teams = len(long), len(teams)
    rows = np.arange(n)

    X = np.zeros((n, 2 + 2 * n_teams))
    X[:, 0] = 1.0  # intercept
    X[:, 1] = long["is_home"].to_numpy()
    X[rows, 2 + long["team"].map(index).to_numpy()] = 1.0  # attack of the scoring team
    X[rows, 2 + n_teams + long["opponent"].map(index).to_numpy()] = 1.0  # defence of the opponent

    penalty = np.concatenate([[0.0, 0.0], np.full(2 * n_teams, ridge)])
    w = np.ones(n) if weights is None else weights
    beta = fit_penalized_poisson(X, long["goals"].to_numpy(dtype=float), w, penalty)

    return PoissonModel(
        intercept=float(beta[0]),
        home_advantage=float(beta[1]),
        attack={team: float(beta[2 + i]) for team, i in index.items()},
        defence={team: float(beta[2 + n_teams + i]) for team, i in index.items()},
    )


def fit_poisson(
    matches: pd.DataFrame,
    as_of_date: pd.Timestamp | None = None,
    half_life: float | None = None,
    ridge: float = 0.0,
) -> PoissonModel:
    """Fit the Poisson model on matches played strictly before as_of_date.

    The date filter lives here so a prediction can never use the match it is
    predicting or anything later (no data leakage). If as_of_date is None,
    all given matches are used. Raises ValueError if no match is left.

    half_life (in days) makes recent matches count more: a match played
    half_life days before as_of_date (or before the latest match if
    as_of_date is None) gets half the weight of one played that day.
    None means every match counts equally.

    ridge (>= 0) penalizes large attack/defence values: the fit maximizes the
    weighted log-likelihood minus ridge/2 * sum(effects squared). This stops a
    team with very little data (say 0 goals in 3 matches) from getting an
    absurd strength. Equivalent to a Normal prior with standard deviation
    1/sqrt(ridge) on every effect. 0 means no penalty (the plain maximum likelihood fit).
    """
    if half_life is not None and half_life <= 0:
        raise ValueError("half_life must be positive")
    if ridge < 0:
        raise ValueError("ridge must not be negative")
    if as_of_date is not None:
        matches = matches[matches["date"] < pd.Timestamp(as_of_date)]
    if matches.empty:
        raise ValueError("No matches available before as_of_date")

    weights = None
    if half_life is not None:
        reference = as_of_date if as_of_date is not None else matches["date"].max()
        match_weights = recency_weights(matches["date"], reference, half_life)
        weights = np.concatenate([match_weights, match_weights])  # long format: home rows, then away rows

    long = to_long_format(matches)
    teams = sorted(set(long["team"]) | set(long["opponent"]))

    if ridge > 0:
        return fit_ridge_poisson(long, teams, weights, ridge)

    attack_cols = sum_to_zero_dummies(long["team"], teams).add_prefix("attack_")
    defence_cols = sum_to_zero_dummies(long["opponent"], teams).add_prefix("defence_")
    exog = pd.concat([attack_cols, defence_cols], axis=1)
    exog.insert(0, "is_home", long["is_home"].astype(float))
    exog.insert(0, "intercept", 1.0)

    # Maximum likelihood fit of: goals ~ Poisson(exp(exog @ params)).
    # freq_weights multiplies each observation's log-likelihood by its weight.
    result = sm.GLM(long["goals"], exog, family=sm.families.Poisson(), freq_weights=weights).fit()
    params = result.params

    def effects(prefix: str) -> dict[str, float]:
        fitted = {t: float(params[f"{prefix}{t}"]) for t in teams[:-1]}
        fitted[teams[-1]] = -sum(fitted.values())  # the omitted level
        return fitted

    return PoissonModel(
        intercept=float(params["intercept"]),
        home_advantage=float(params["is_home"]),
        attack=effects("attack_"),
        defence=effects("defence_"),
    )


def expected_goals(model: PoissonModel, home_team: str, away_team: str) -> tuple[float, float]:
    """Return (lambda_home, lambda_away): each side's expected goals.

    lambda_home = exp(intercept + home_advantage + attack[home] + defence[away])
    lambda_away = exp(intercept + attack[away] + defence[home])
    Raises ValueError for a team the model has never seen (e.g. just promoted).
    """
    for team in (home_team, away_team):
        if team not in model.attack:
            raise ValueError(f"Unknown team: {team!r} (not in the training data)")

    log_home = model.intercept + model.home_advantage + model.attack[home_team] + model.defence[away_team]
    log_away = model.intercept + model.attack[away_team] + model.defence[home_team]
    return float(np.exp(log_home)), float(np.exp(log_away))


def score_matrix(lambda_home: float, lambda_away: float, max_goals: int = 10) -> np.ndarray:
    """Return P(home scores i, away scores j) as a (max_goals+1)^2 array.

    Home and away goals are assumed independent, so each cell is the product
    of two Poisson probabilities. Rows are home goals, columns away goals.
    The grid stops at max_goals, so its total is slightly below 1.
    """
    goals = np.arange(max_goals + 1)
    home_probs = poisson_dist.pmf(goals, lambda_home)
    away_probs = poisson_dist.pmf(goals, lambda_away)
    return np.outer(home_probs, away_probs)


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    """Return (P_home_win, P_draw, P_away_win) from a score matrix.

    Home wins are the cells below the diagonal (home goals > away goals),
    draws the diagonal, away wins above it. The three are divided by their
    total so they sum to exactly 1 despite the truncated tail of the grid.
    """
    home_win = np.tril(matrix, k=-1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, k=1).sum()
    total = home_win + draw + away_win
    return float(home_win / total), float(draw / total), float(away_win / total)


def predict_match(
    model: PoissonModel, home_team: str, away_team: str, max_goals: int = 10
) -> tuple[float, float, float]:
    """Return (P_home_win, P_draw, P_away_win) for one match.

    Chains the three steps: expected goals, scoreline grid, outcome sums.
    """
    lambda_home, lambda_away = expected_goals(model, home_team, away_team)
    return outcome_probabilities(score_matrix(lambda_home, lambda_away, max_goals))


def month_starts(matches: pd.DataFrame) -> pd.DatetimeIndex:
    """First day of every month from the first to the last match date."""
    return pd.date_range(matches["date"].min(), matches["date"].max(), freq="MS")


def strength_history(
    matches: pd.DataFrame,
    dates: pd.DatetimeIndex,
    half_life: float | None = MODEL_HALF_LIFE,
    min_matches: int = 100,
    active_days: int = 400,
    ridge: float = MODEL_RIDGE,
) -> pd.DataFrame:
    """Team strengths as the model saw them on each snapshot date.

    For every date, the model is fitted only on matches strictly before it
    (no leakage). Returns one row per date and team with columns date, team,
    attack, defence and strength = attack - defence (higher is better; a
    positive defence value means conceding more). The model uses the frozen
    half-life and ridge penalty unless told otherwise.

    Snapshots with fewer than min_matches earlier matches are skipped (the
    fit would be too noisy). Only teams that played within the last
    active_days days are reported, so relegated teams drop out of the chart.
    """
    rows = []
    for date in dates:
        earlier = matches[matches["date"] < date]
        if len(earlier) < min_matches:
            continue
        model = fit_poisson(matches, as_of_date=date, half_life=half_life, ridge=ridge)

        recent = earlier[earlier["date"] >= date - pd.Timedelta(days=active_days)]
        active = set(recent["home_team"]) | set(recent["away_team"])
        for team in sorted(active):
            rows.append(
                {
                    "date": date,
                    "team": team,
                    "attack": model.attack[team],
                    "defence": model.defence[team],
                    "strength": model.attack[team] - model.defence[team],
                }
            )
    return pd.DataFrame(rows, columns=["date", "team", "attack", "defence", "strength"])


STRENGTH_PATH = Path("results/strength_history.csv")


def save_strength_history(matches: pd.DataFrame, path: Path = STRENGTH_PATH) -> pd.DataFrame:
    """Compute monthly team strengths for all matches and write them to a CSV.

    Takes about 15 seconds (one model fit per month), so the dashboard reads
    the saved file instead. Returns the table written.
    """
    history = strength_history(matches, month_starts(matches))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False)
    return history


def load_strength_history(path: Path = STRENGTH_PATH) -> pd.DataFrame:
    """Read a file written by save_strength_history."""
    return pd.read_csv(path, parse_dates=["date"])
