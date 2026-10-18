"""Smoke test: the dashboard runs without errors on the real processed data."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import data

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")

pytestmark = pytest.mark.skipif(
    not data.PROCESSED_PATH.exists(), reason="data/processed/matches.csv not built"
)


def test_app_runs_without_exception_and_shows_both_tabs():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Season simulation", "Match predictor", "Team strength", "Model evaluation"
    ]
    assert len(app.dataframe) >= 2  # simulated-vs-real table and the model comparison


def test_match_predictor_shows_three_probabilities():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert [m.label for m in app.metric][-1].endswith("win")
    assert len(app.metric) == 3


def test_evaluation_tab_switches_between_clean_and_seen_sets_without_errors():
    app = AppTest.from_file(APP_PATH, default_timeout=120).run()

    labels = [r.label for r in app.radio]
    assert "Evaluation set" in labels
    evaluation_set = next(r for r in app.radio if r.label == "Evaluation set")
    for option in evaluation_set.options:
        evaluation_set.set_value(option).run()
        assert not app.exception
