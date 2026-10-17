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
    assert len(app.tabs) == 2
    assert len(app.dataframe) == 1  # the simulated-vs-real table


def test_match_predictor_shows_three_probabilities():
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()

    assert [m.label for m in app.metric][-1].endswith("win")
    assert len(app.metric) == 3
