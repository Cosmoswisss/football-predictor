"""Downloading and cleaning match data from football-data.co.uk."""

import re
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import pandas as pd
import requests
import truststore

BASE_URL = "https://www.football-data.co.uk/mmz4281"
# Use the operating system's certificate store: Python's bundled certifi list
# rejected the certificates on this machine (e.g. antivirus/proxy inspection).
truststore.inject_into_ssl()

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/matches.csv")

# Columns every season file must have, and odds columns that may be missing.
REQUIRED_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLUMNS = ["B365H", "B365D", "B365A"]


def season_code(start_year: int) -> str:
    """Return the football-data.co.uk season code, e.g. 2023 -> '2324'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download_season(
    start_year: int,
    dest_dir: Path = RAW_DIR,
    league: str = "E0",
    force: bool = False,
) -> Path:
    """Download one season's CSV into dest_dir and return its path.

    start_year is the year the season starts (2023 -> 2023/24). If the file
    already exists it is reused, unless force=True.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{league}_{season_code(start_year)}.csv"

    if path.exists() and not force:
        return path

    url = f"{BASE_URL}/{season_code(start_year)}/{league}.csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # raises on 404, 500, ... instead of saving an error page
    path.write_bytes(response.content)
    return path


def load_raw_season(path: Path) -> pd.DataFrame:
    """Read one raw season CSV, keeping only the columns we use.

    Handles non-UTF-8 files, drops fully blank rows, and adds the Bet365 odds
    columns filled with NaN if the file lacks them. Raises ValueError if a
    required column is missing. No parsing or renaming happens here.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig also strips a BOM
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")  # some old files are not UTF-8

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    for col in ODDS_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    df = df[REQUIRED_COLUMNS + ODDS_COLUMNS]
    return df.dropna(how="all").reset_index(drop=True)


COLUMN_NAMES = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "B365H": "odds_home",
    "B365D": "odds_draw",
    "B365A": "odds_away",
}


def parse_dates(dates: pd.Series) -> pd.Series:
    """Parse dd/mm/yyyy dates, falling back to dd/mm/yy; unparseable become NaT."""
    parsed = pd.to_datetime(dates, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(dates, format="%d/%m/%y", errors="coerce")
    return parsed.fillna(short)


def clean_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw season table into a clean one, sorted by date.

    Renames columns, parses dates, strips team names, drops rows with a
    missing date or score, casts goals to int, and sorts chronologically
    (needed later for walk-forward backtesting without leakage).
    """
    out = df.rename(columns=COLUMN_NAMES).copy()
    out["date"] = parse_dates(out["date"])
    out["home_team"] = out["home_team"].str.strip()
    out["away_team"] = out["away_team"].str.strip()

    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    out = out.astype({"home_goals": int, "away_goals": int})

    # sort_values is not stable by default; mergesort keeps same-day matches in file order.
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def find_suspicious_team_names(df: pd.DataFrame, threshold: float = 0.8) -> list[tuple[str, str]]:
    """Return pairs of team names that might be two spellings of one club.

    df needs 'season', 'home_team' and 'away_team' columns. Two different
    clubs can play in the same season, but two spellings of the same club
    never do. So we only compare names that never share a season, and flag
    pairs that are near-identical (same letters ignoring case/punctuation, or
    string similarity >= threshold). The result is a report; nothing is merged.
    """
    seasons_of: dict[str, set] = {}
    for col in ("home_team", "away_team"):
        for season, team in zip(df["season"], df[col]):
            seasons_of.setdefault(team, set()).add(season)

    def key(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    pairs = []
    for a, b in combinations(sorted(seasons_of), 2):
        if seasons_of[a] & seasons_of[b]:
            continue  # played in the same season, so they are different clubs
        similar = SequenceMatcher(None, key(a), key(b)).ratio() >= threshold
        if key(a) == key(b) or similar:
            pairs.append((a, b))
    return pairs


def season_label(start_year: int) -> str:
    """Return a readable season label, e.g. 2023 -> '2023/24'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"


def build_dataset(
    start_years: list[int],
    raw_dir: Path = RAW_DIR,
    out_path: Path = PROCESSED_PATH,
    league: str = "E0",
) -> pd.DataFrame:
    """Download, load and clean several seasons into one table saved as CSV.

    Adds a 'season' column, sorts all matches chronologically and writes the
    result to out_path. Returns the combined DataFrame.
    """
    frames = []
    for year in start_years:
        raw_path = download_season(year, dest_dir=raw_dir, league=league)
        season_df = clean_matches(load_raw_season(raw_path))
        season_df["season"] = season_label(year)
        frames.append(season_df)

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values("date", kind="mergesort").reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    return full
