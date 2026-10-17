"""Monte Carlo season simulation based on the Poisson model."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.poisson import PoissonModel, expected_goals, fit_poisson


def add_unseen_teams(
    model: PoissonModel, teams: list[str]
) -> tuple[PoissonModel, list[str]]:
    """Give teams the model has never seen (e.g. promoted ones) average strength.

    Attack and defence are 0 on the log scale, i.e. exactly the league average.
    Because the existing effects sum to zero, adding zeros keeps that true.
    Returns the extended model (the input is not modified) and the sorted list
    of teams that were added, so the app can warn about them.
    """
    unseen = sorted(set(teams) - set(model.attack))
    extended = replace(
        model,
        attack={**model.attack, **{t: 0.0 for t in unseen}},
        defence={**model.defence, **{t: 0.0 for t in unseen}},
    )
    return extended, unseen


def fixture_lambdas(
    model: PoissonModel, fixtures: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lambda_home, lambda_away) arrays with one entry per fixture.

    fixtures needs home_team and away_team columns. Every team must be known
    to the model (use add_unseen_teams first for promoted teams).
    """
    lambdas = [
        expected_goals(model, home, away)
        for home, away in zip(fixtures["home_team"], fixtures["away_team"])
    ]
    lam_home, lam_away = np.array(lambdas).T
    return lam_home, lam_away


def simulate_goals(
    lam_home: np.ndarray, lam_away: np.ndarray, n_sims: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Draw random scores for every fixture in every simulated season.

    Home and away goals are independent Poisson draws. Both returned arrays
    have shape (n_sims, n_fixtures). The same seed always gives the same
    draws, so results are reproducible.
    """
    rng = np.random.default_rng(seed)
    shape = (n_sims, len(lam_home))
    home_goals = rng.poisson(lam_home, size=shape)  # lam_home broadcasts over rows
    away_goals = rng.poisson(lam_away, size=shape)
    return home_goals, away_goals


def build_tables(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    fixtures: pd.DataFrame,
    teams: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (points, goal_diff, goals_for), each of shape (n_sims, n_teams).

    Column t of every array belongs to teams[t]. Instead of looping over
    matches, we use two one-hot matrices H and A of shape (n_fixtures, n_teams)
    that mark each fixture's home and away team. A matrix product such as
    home_points @ H then adds every fixture's points to the right team.
    """
    index = {team: i for i, team in enumerate(teams)}
    home_idx = fixtures["home_team"].map(index).to_numpy()  # unknown team -> NaN
    away_idx = fixtures["away_team"].map(index).to_numpy()
    if pd.isna(home_idx).any() or pd.isna(away_idx).any():
        raise ValueError("Fixture contains a team that is not in the teams list")

    n_fixtures = len(fixtures)
    home_onehot = np.zeros((n_fixtures, len(teams)))
    away_onehot = np.zeros((n_fixtures, len(teams)))
    home_onehot[np.arange(n_fixtures), home_idx.astype(int)] = 1.0
    away_onehot[np.arange(n_fixtures), away_idx.astype(int)] = 1.0

    home_points = 3 * (home_goals > away_goals) + 1 * (home_goals == away_goals)
    away_points = 3 * (away_goals > home_goals) + 1 * (home_goals == away_goals)

    points = home_points @ home_onehot + away_points @ away_onehot
    goal_diff = (home_goals - away_goals) @ home_onehot + (away_goals - home_goals) @ away_onehot
    goals_for = home_goals @ home_onehot + away_goals @ away_onehot
    return points, goal_diff, goals_for


def final_positions(
    points: np.ndarray, goal_diff: np.ndarray, goals_for: np.ndarray, seed: int = 42
) -> np.ndarray:
    """Return each team's final league position (1 = champion) per simulation.

    Teams are ranked by points, then goal difference, then goals scored. This
    simplifies the real Premier League rules (head-to-head is ignored). Any
    teams still tied are ordered randomly, reproducibly from the seed.
    Output has shape (n_sims, n_teams): positions[s, t] is team t's rank in
    simulation s.
    """
    tiebreak = np.random.default_rng(seed).random(points.shape)
    # lexsort sorts ascending with the LAST key as the most important one.
    worst_to_best = np.lexsort((tiebreak, goals_for, goal_diff, points), axis=1)
    best_to_worst = worst_to_best[:, ::-1]  # best_to_worst[s, k] = team in place k+1
    # argsort of a permutation is its inverse: from "who is in place k" to "where is team t".
    return np.argsort(best_to_worst, axis=1) + 1


def summarize(
    positions: np.ndarray,
    points: np.ndarray,
    teams: list[str],
    n_top: int = 4,
    n_relegated: int = 3,
) -> pd.DataFrame:
    """Return one row per team: expected points and key finishing probabilities.

    Each probability is the fraction of simulations in which the event
    happened, e.g. p_title = share of seasons where the team finished first.
    The table is sorted by expected points, best first.
    """
    n_teams = len(teams)
    summary = pd.DataFrame(
        {
            "team": teams,
            "expected_points": points.mean(axis=0),
            "p_title": (positions == 1).mean(axis=0),
            f"p_top{n_top}": (positions <= n_top).mean(axis=0),
            "p_relegation": (positions > n_teams - n_relegated).mean(axis=0),
        }
    )
    return summary.sort_values("expected_points", ascending=False).reset_index(drop=True)


def position_distribution(positions: np.ndarray, teams: list[str]) -> pd.DataFrame:
    """Return a teams x positions table: P(team finishes in that position).

    Rows are teams, columns are positions 1..n. Every row and every column sums to 1.
    """
    n_teams = len(teams)
    shares = np.stack([(positions == place).mean(axis=0) for place in range(1, n_teams + 1)], axis=1)
    return pd.DataFrame(shares, index=teams, columns=range(1, n_teams + 1))


@dataclass(frozen=True)
class SeasonSimulation:
    """Result of simulate_season."""

    summary: pd.DataFrame
    positions: pd.DataFrame  # position distribution, teams x places
    added_teams: list[str]  # teams the model did not know (given average strength)


def simulate_season(
    model: PoissonModel,
    fixtures: pd.DataFrame,
    n_sims: int = 10_000,
    seed: int = 42,
    played: pd.DataFrame | None = None,
) -> SeasonSimulation:
    """Simulate the rest of a season n_sims times and summarize the outcomes.

    fixtures: matches still to be played (home_team, away_team).
    played: optional matches already played (home_team, away_team,
    home_goals, away_goals). Their real results are added to every simulated table.
    Teams unknown to the model get average strength and are reported in
    added_teams. The same seed always gives the same result.
    """
    teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
    if played is not None:
        teams |= set(played["home_team"]) | set(played["away_team"])
    teams = sorted(teams)

    model, added = add_unseen_teams(model, teams)
    lam_home, lam_away = fixture_lambdas(model, fixtures)
    home_goals, away_goals = simulate_goals(lam_home, lam_away, n_sims, seed)
    points, goal_diff, goals_for = build_tables(home_goals, away_goals, fixtures, teams)

    if played is not None and len(played) > 0:
        # Real results: one "simulation" whose numbers are added to all of them.
        real = build_tables(
            played[["home_goals"]].to_numpy().T, played[["away_goals"]].to_numpy().T, played, teams
        )
        points, goal_diff, goals_for = points + real[0], goal_diff + real[1], goals_for + real[2]

    # seed + 1 so the tie-break randomness is independent of the goal draws
    positions = final_positions(points, goal_diff, goals_for, seed + 1)
    return SeasonSimulation(
        summary=summarize(positions, points, teams),
        positions=position_distribution(positions, teams),
        added_teams=added,
    )


def season_fixtures(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Return the home_team/away_team pairs of one season, e.g. season='2024/25'."""
    in_season = matches[matches["season"] == season]
    if in_season.empty:
        raise ValueError(f"No matches for season {season!r}")
    return in_season[["home_team", "away_team"]].reset_index(drop=True)


def fit_before_season(matches: pd.DataFrame, season: str) -> PoissonModel:
    """Fit the Poisson model using only matches before the season's first match.

    This is what makes a season "replay" honest: the model knows nothing about
    the season it is asked to predict (no data leakage).
    """
    in_season = matches[matches["season"] == season]
    if in_season.empty:
        raise ValueError(f"No matches for season {season!r}")
    return fit_poisson(matches, as_of_date=in_season["date"].min())


def actual_table(matches: pd.DataFrame, season: str) -> pd.DataFrame:
    """Return the real final table of a season: team, actual_points, actual_position.

    Uses the same ranking rule as the simulation: points, goal difference, goals scored.
    """
    games = matches[matches["season"] == season]
    home = pd.DataFrame(
        {
            "team": games["home_team"],
            "points": games["result"].map({"H": 3, "D": 1, "A": 0}),
            "gf": games["home_goals"],
            "ga": games["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "team": games["away_team"],
            "points": games["result"].map({"H": 0, "D": 1, "A": 3}),
            "gf": games["away_goals"],
            "ga": games["home_goals"],
        }
    )
    table = pd.concat([home, away]).groupby("team").sum()
    table["gd"] = table["gf"] - table["ga"]
    table = table.sort_values(["points", "gd", "gf"], ascending=False)
    table["actual_position"] = np.arange(1, len(table) + 1)
    return table.reset_index().rename(columns={"points": "actual_points"})[
        ["team", "actual_points", "actual_position"]
    ]
