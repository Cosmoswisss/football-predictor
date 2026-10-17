# CLAUDE.md

## Project overview
`football-predictor` is a personal learning project: a Poisson-based football match prediction model with Monte Carlo season simulation and backtesting against bookmaker odds, presented in a Streamlit dashboard.

I am a first-year EPFL student using this project to learn Python, data science, Git and Claude Code. **Understanding the code matters more than speed.**

## Tech stack
- Python 3.11+
- pandas, numpy, scipy, statsmodels (data and modeling)
- Streamlit (dashboard), matplotlib or plotly (charts)
- pytest (tests)
- Dependencies listed in `requirements.txt`

## Data sources
- Historical results and bookmaker odds: CSV files from football-data.co.uk (start with the English Premier League)
- Upcoming fixtures: football-data.org API (API key stored in a `.env` file, never committed)
- Raw downloaded files go in `data/raw/`, cleaned files in `data/processed/`. Both folders are git-ignored.

## Project structure
```
football-predictor/
├── app.py              # Streamlit dashboard
├── src/
│   ├── data.py         # downloading and cleaning data
│   ├── poisson.py      # Poisson model (fit + match probabilities)
│   ├── elo.py          # Elo baseline
│   ├── simulate.py     # Monte Carlo season simulation
│   └── backtest.py     # evaluation vs bookmaker odds
├── tests/              # pytest tests, one file per module
├── data/               # git-ignored
├── requirements.txt
└── README.md
```

## Modeling rules (critical)
- **No data leakage**: a prediction for a match may only use matches played strictly before that match's date. Backtesting must be walk-forward.
- Bookmaker implied probabilities must have the margin removed (normalize so they sum to 1) before comparison.
- Evaluate with log-loss and Brier score, always compared against the Elo baseline and bookmaker odds.
- Fix random seeds in simulations so results are reproducible.

## Roadmap
1. Data loading and cleaning
2. Poisson model + match outcome probabilities
3. Season simulation + first Streamlit dashboard (**MVP**)
4. Elo baseline + backtest against bookmakers
5. Charts (team strength over time, calibration plot) + deployment on Streamlit Community Cloud
6. README with screenshots, demo link and model explanation

## Coding conventions
- Small, focused functions with type hints and short docstrings
- Write or update pytest tests for every function in `src/`
- Keep the Streamlit app thin: logic lives in `src/`, `app.py` only displays
- Code, comments, commit messages and README in English

## How to work with me
- Plan before coding and work in small steps; one feature at a time.
- After each change, briefly explain what you did and why, especially the math and statistics.
- Ask before adding a new dependency.
- Do not commit or push: I handle Git myself.
- If something I ask for would cause data leakage or bad practice, tell me.
