"""Streamlit dashboard: season simulation, match predictor, team strength, model evaluation.

This file only displays. All the logic lives in src/.
Run with:  python -m streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import backtest, data, poisson, simulate

st.set_page_config(page_title="Football Predictor", layout="wide")


@st.cache_data(show_spinner="Downloading and preparing match data (first run only)...")
def load_matches() -> pd.DataFrame:
    """Cleaned matches; downloaded and built on first run if the file does not exist yet."""
    return data.ensure_dataset()


@st.cache_resource
def get_model(season: str) -> poisson.PoissonModel:
    """Poisson model fitted only on matches before the season starts."""
    return simulate.fit_before_season(load_matches(), season)


@st.cache_data
def run_simulation(season: str, n_sims: int, seed: int) -> simulate.SeasonSimulation:
    """Simulate the chosen season n_sims times."""
    fixtures = simulate.season_fixtures(load_matches(), season)
    return simulate.simulate_season(get_model(season), fixtures, n_sims, seed)


@st.cache_data
def load_strengths() -> pd.DataFrame:
    """Monthly team strengths saved by poisson.save_strength_history."""
    return poisson.load_strength_history()


@st.cache_data
def load_evaluation():
    """Test-season predictions saved by backtest.save_test_predictions."""
    return backtest.load_test_predictions()


st.title("Football Predictor")
st.caption(
    "Poisson model + Monte Carlo simulation of the Premier League. "
    "A season is replayed: the model only sees matches played before it started."
)

matches = load_matches()
seasons = sorted(matches["season"].unique())

# The first season has no earlier matches to learn from, so it cannot be replayed.
st.sidebar.header("Settings")
season = st.sidebar.selectbox("Season", seasons[1:], index=len(seasons) - 2)
n_sims = st.sidebar.select_slider("Simulations", [1_000, 5_000, 10_000, 20_000], value=10_000)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

tab_season, tab_match, tab_strength, tab_eval = st.tabs(
    ["Season simulation", "Match predictor", "Team strength", "Model evaluation"]
)

with tab_season:
    result = run_simulation(season, n_sims, int(seed))
    if result.added_teams:
        st.warning(
            f"No earlier data for {', '.join(result.added_teams)} (newly promoted). "
            "They are given average strength, which probably overrates them."
        )

    table = result.summary.merge(simulate.actual_table(matches, season), on="team")
    for col in ["p_title", "p_top4", "p_relegation"]:
        table[col] *= 100
    st.subheader(f"{season}: simulated vs. real")
    st.dataframe(
        table.rename(
            columns={
                "team": "Team",
                "expected_points": "Expected pts",
                "p_title": "Title %",
                "p_top4": "Top 4 %",
                "p_relegation": "Relegation %",
                "actual_points": "Real pts",
                "actual_position": "Real position",
            }
        ),
        hide_index=True,
        column_config={
            "Expected pts": st.column_config.NumberColumn(format="%.1f"),
            "Title %": st.column_config.NumberColumn(format="%.1f"),
            "Top 4 %": st.column_config.NumberColumn(format="%.1f"),
            "Relegation %": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    left, right = st.columns(2)
    with left:
        contenders = table[table["p_title"] > 0.05].sort_values("p_title")
        fig = px.bar(contenders, x="p_title", y="team", orientation="h",
                     labels={"p_title": "Title probability (%)", "team": ""},
                     title="Title race")
        st.plotly_chart(fig)
    with right:
        order = result.summary["team"].tolist()
        fig = px.imshow(
            result.positions.loc[order] * 100,
            labels={"x": "Final position", "y": "", "color": "%"},
            aspect="auto",
            color_continuous_scale="Blues",
            title="Probability of each final position (%)",
        )
        fig.update_yaxes(dtick=1)  # label every team, not every second one
        fig.update_layout(height=600)
        st.plotly_chart(fig)

with tab_match:
    model = get_model(season)
    teams = sorted(set(matches.loc[matches["season"] == season, "home_team"]))
    col_home, col_away = st.columns(2)
    home = col_home.selectbox("Home team", teams, index=0)
    away = col_away.selectbox("Away team", teams, index=1)

    if home == away:
        st.info("Pick two different teams.")
    else:
        model, added = simulate.add_unseen_teams(model, [home, away])
        if added:
            st.warning(f"{', '.join(added)}: no earlier data, average strength used.")
        lam_home, lam_away = poisson.expected_goals(model, home, away)
        p_home, p_draw, p_away = poisson.predict_match(model, home, away)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{home} win", f"{p_home:.0%}")
        c2.metric("Draw", f"{p_draw:.0%}")
        c3.metric(f"{away} win", f"{p_away:.0%}")
        st.caption(f"Expected goals: {home} {lam_home:.2f} – {lam_away:.2f} {away}")

        grid = poisson.score_matrix(lam_home, lam_away)[:6, :6] * 100
        fig = px.imshow(
            grid,
            x=list(range(6)),
            y=list(range(6)),
            labels={"x": f"{away} goals", "y": f"{home} goals", "color": "%"},
            color_continuous_scale="Blues",
            title="Scoreline probabilities (%)",
            text_auto=".1f",
        )
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=1)
        st.plotly_chart(fig)

with tab_strength:
    if not poisson.STRENGTH_PATH.exists():
        st.info(
            f"{poisson.STRENGTH_PATH} not found. Create it with:\n\n"
            "python -c \"import pandas as pd; from src import data, poisson; "
            "poisson.save_strength_history(pd.read_csv(data.PROCESSED_PATH, parse_dates=['date']))\""
        )
    else:
        history = load_strengths()
        latest = history[history["date"] == history["date"].max()]
        top_now = latest.sort_values("strength", ascending=False)["team"].head(4).tolist()

        measure = st.radio(
            "Measure",
            ["Overall strength", "Attack", "Defence (higher = better)"],
            horizontal=True,
        )
        picked = st.multiselect("Teams", sorted(history["team"].unique()), default=top_now)

        shown = history[history["team"].isin(picked)].copy()
        shown["Defence (higher = better)"] = -shown["defence"]  # flip so up always means better
        column = {
            "Overall strength": "strength",
            "Attack": "attack",
            "Defence (higher = better)": "Defence (higher = better)",
        }[measure]
        fig = px.line(shown, x="date", y=column, color="team", labels={"date": "", column: measure, "team": "Team"})
        st.plotly_chart(fig)
        st.caption(
            "Values are on a log scale: +0.1 means roughly 10% more goals than an average team "
            "(0 = league average). Each month's value comes from a model fitted only on matches "
            "before that date, with recent matches weighted more (half-life 365 days). "
            "Overall strength = attack minus defence weakness."
        )

MODEL_LABELS = {
    "bookmaker": "Bookmaker (margin removed)",
    "elo": "Elo",
    "poisson (plain)": "Poisson, no weighting",
    "poisson (half-life 365d)": "Poisson v1 (time-weighted)",
    "poisson (half-life 365d + ridge 4)": "Poisson v2 (time-weighted + ridge)",
}
CLEAN_SET = f"Clean test: {', '.join(backtest.CLEAN_TEST_SEASONS)}"
SEEN_SET = f"Seen seasons: {backtest.TEST_SEASONS[0]} to {backtest.TEST_SEASONS[-1]} (informational)"

with tab_eval:
    if not backtest.RESULTS_PATH.exists():
        st.info(
            f"{backtest.RESULTS_PATH} not found. Create it (a few minutes) with:\n\n"
            "python -c \"import pandas as pd; from src import backtest, data; "
            "backtest.save_test_predictions(pd.read_csv(data.PROCESSED_PATH, parse_dates=['date']))\""
        )
    else:
        raw_predictions, outcomes, test_matches = load_evaluation()
        predictions = {MODEL_LABELS.get(name, name): probs for name, probs in raw_predictions.items()}
        is_clean = test_matches["season"].isin(backtest.CLEAN_TEST_SEASONS).to_numpy()

        st.markdown(
            "**How this was evaluated.** Version 1 of the Poisson model was tuned on validation seasons "
            f"(2018/19 to 2020/21) and scored once on {backtest.TEST_SEASONS[0]} to {backtest.TEST_SEASONS[-1]}: "
            "the bookmaker was clearly best. A chart then exposed an instability (a team with no goals in its "
            "first matches got an absurd strength), so version 2 adds a ridge penalty, chosen on the validation "
            f"seasons only. {', '.join(backtest.CLEAN_TEST_SEASONS)} had never been looked at when version 2 was "
            "frozen, so it is the **clean test**. The earlier seasons are *seen*: informational only. "
            "Every prediction uses only matches played before it, and the bookmaker probabilities have the margin removed."
        )

        choice = st.radio("Evaluation set", [CLEAN_SET, SEEN_SET, "Both"], horizontal=True)
        mask = {CLEAN_SET: is_clean, SEEN_SET: ~is_clean, "Both": np.ones_like(is_clean)}[choice]
        reference = st.selectbox("Compare against", list(predictions), index=0)

        comparison = backtest.compare_models(
            {name: probs[mask] for name, probs in predictions.items()}, outcomes[mask], reference=reference
        )
        st.caption(f"{int(mask.sum()):,} matches.")
        diff_ll, diff_br = f"Log-loss vs {reference}", f"Brier vs {reference}"
        st.dataframe(
            comparison.rename(
                columns={
                    "model": "Model",
                    "log_loss": "Log-loss",
                    "brier": "Brier",
                    "ll_diff": diff_ll,
                    "ll_se": "± std. error",
                    "brier_diff": diff_br,
                    "brier_se": "± std. error ",
                }
            ),
            hide_index=True,
            column_config={
                col: st.column_config.NumberColumn(format="%.4f")
                for col in ["Log-loss", "Brier", diff_ll, "± std. error", diff_br, "± std. error "]
            },
        )
        st.caption(
            "Lower is better for both scores (always saying 1/3 gives 1.099 and 0.667). "
            "A negative difference means better than the comparison model. A difference smaller than "
            "about two standard errors is indistinguishable from luck, and fewer matches mean larger errors."
        )

        left, right = st.columns(2)
        with left:
            by_season = backtest.log_loss_by_group(predictions, outcomes, test_matches["season"])
            long = by_season.reset_index(names="Season").melt(
                id_vars="Season", var_name="Model", value_name="Log-loss"
            )
            fig = px.line(long, x="Season", y="Log-loss", color="Model", markers=True,
                          title="Log-loss by season (the last one is the clean test)")
            st.plotly_chart(fig)
        with right:
            outcome_label = st.radio("Calibration of", ["Home win", "Draw", "Away win"], horizontal=True)
            n_bins = st.slider("Bins", 5, 15, 8)
            outcome = ["Home win", "Draw", "Away win"].index(outcome_label)
            calibration = pd.concat(
                [
                    backtest.calibration_table(probs[mask], outcomes[mask], outcome, n_bins).assign(Model=name)
                    for name, probs in predictions.items()
                ]
            )
            fig = px.line(
                calibration, x="mean_predicted", y="observed_frequency", color="Model",
                markers=True, hover_data=["count"],
                labels={"mean_predicted": "Predicted probability", "observed_frequency": "Observed frequency"},
                title=f"Calibration: {outcome_label.lower()}",
            )
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot", "color": "grey"})
            st.plotly_chart(fig)
        st.caption(
            "Calibration: among matches where a model says X%, did the outcome happen about X% of the time? "
            "Points on the dotted diagonal are perfectly calibrated. Hover to see how many matches are in "
            "each bin; sparse bins are noisy."
        )
