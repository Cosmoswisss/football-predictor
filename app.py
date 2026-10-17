"""Streamlit dashboard: season simulation and single-match predictor.

This file only displays. All the logic lives in src/.
Run with:  python -m streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from src import data, poisson, simulate

st.set_page_config(page_title="Football Predictor", layout="wide")


@st.cache_data
def load_matches() -> pd.DataFrame:
    """Read the cleaned matches built by src.data.build_dataset."""
    return pd.read_csv(data.PROCESSED_PATH, parse_dates=["date"])


@st.cache_resource
def get_model(season: str) -> poisson.PoissonModel:
    """Poisson model fitted only on matches before the season starts."""
    return simulate.fit_before_season(load_matches(), season)


@st.cache_data
def run_simulation(season: str, n_sims: int, seed: int) -> simulate.SeasonSimulation:
    """Simulate the chosen season n_sims times."""
    fixtures = simulate.season_fixtures(load_matches(), season)
    return simulate.simulate_season(get_model(season), fixtures, n_sims, seed)


st.title("Football Predictor")
st.caption(
    "Poisson model + Monte Carlo simulation of the Premier League. "
    "A season is replayed: the model only sees matches played before it started."
)

if not data.PROCESSED_PATH.exists():
    st.error(
        f"{data.PROCESSED_PATH} not found. Build it first with:\n\n"
        "python -c \"from src import data; data.build_dataset(range(2015, 2025))\""
    )
    st.stop()

matches = load_matches()
seasons = sorted(matches["season"].unique())

# The first season has no earlier matches to learn from, so it cannot be replayed.
st.sidebar.header("Settings")
season = st.sidebar.selectbox("Season", seasons[1:], index=len(seasons) - 2)
n_sims = st.sidebar.select_slider("Simulations", [1_000, 5_000, 10_000, 20_000], value=10_000)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

tab_season, tab_match = st.tabs(["Season simulation", "Match predictor"])

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
        use_container_width=True,
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
        st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)
