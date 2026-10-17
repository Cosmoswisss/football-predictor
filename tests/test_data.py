import pandas as pd
import pytest
import requests

from src import data


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, content: bytes = b"a,b\n1,2\n", status: int = 200):
        self.content = content
        self.status = status

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise requests.HTTPError(f"status {self.status}")


def test_season_code():
    assert data.season_code(2023) == "2324"
    assert data.season_code(1999) == "9900"  # century wrap-around
    assert data.season_code(2009) == "0910"


def test_download_season_writes_file_and_uses_right_url(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(data.requests, "get", fake_get)
    path = data.download_season(2023, dest_dir=tmp_path)

    assert path == tmp_path / "E0_2324.csv"
    assert path.read_bytes() == b"a,b\n1,2\n"
    assert calls == ["https://www.football-data.co.uk/mmz4281/2324/E0.csv"]


def test_download_season_skips_existing_file(tmp_path, monkeypatch):
    (tmp_path / "E0_2324.csv").write_bytes(b"cached")

    def fail_get(url, timeout):
        raise AssertionError("network should not be used")

    monkeypatch.setattr(data.requests, "get", fail_get)
    path = data.download_season(2023, dest_dir=tmp_path)

    assert path.read_bytes() == b"cached"


def test_download_season_force_redownloads(tmp_path, monkeypatch):
    (tmp_path / "E0_2324.csv").write_bytes(b"old")
    monkeypatch.setattr(data.requests, "get", lambda url, timeout: FakeResponse(b"new"))

    path = data.download_season(2023, dest_dir=tmp_path, force=True)

    assert path.read_bytes() == b"new"


def test_download_season_http_error_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data.requests, "get", lambda url, timeout: FakeResponse(status=404)
    )

    with pytest.raises(requests.HTTPError):
        data.download_season(2023, dest_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


FULL_CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,Extra\n"
    "E0,12/08/2023,Burnley,Man City,0,3,A,7.5,5.0,1.4,x\n"
    "E0,12/08/2023,Arsenal,Nott'm Forest,2,1,H,1.3,6.0,10.0,x\n"
    ",,,,,,,,,,\n"
)


def test_load_raw_season_keeps_only_needed_columns_and_drops_blank_rows(tmp_path):
    path = tmp_path / "season.csv"
    path.write_text(FULL_CSV, encoding="utf-8")

    df = data.load_raw_season(path)

    assert list(df.columns) == data.REQUIRED_COLUMNS + data.ODDS_COLUMNS
    assert len(df) == 2
    assert df.loc[0, "HomeTeam"] == "Burnley"
    assert df.loc[1, "B365H"] == 1.3


def test_load_raw_season_adds_nan_odds_when_missing(tmp_path):
    path = tmp_path / "season.csv"
    path.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/01/2000,A,B,1,0,H\n",
        encoding="utf-8",
    )

    df = data.load_raw_season(path)

    assert list(df.columns) == data.REQUIRED_COLUMNS + data.ODDS_COLUMNS
    assert df[data.ODDS_COLUMNS].isna().all().all()


def test_load_raw_season_falls_back_to_latin1(tmp_path):
    path = tmp_path / "season.csv"
    path.write_bytes(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/01/2000,Café,B,1,0,H\n".encode(
            "latin-1"
        )
    )

    df = data.load_raw_season(path)

    assert df.loc[0, "HomeTeam"] == "Café"


def test_load_raw_season_missing_required_column_raises(tmp_path):
    path = tmp_path / "season.csv"
    path.write_text("Date,HomeTeam\n01/01/2000,A\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        data.load_raw_season(path)


def raw_frame(rows):
    cols = data.REQUIRED_COLUMNS + data.ODDS_COLUMNS
    return pd.DataFrame(rows, columns=cols)


def test_parse_dates_handles_four_and_two_digit_years():
    out = data.parse_dates(pd.Series(["12/08/2023", "05/01/99", "bad"]))

    assert out[0] == pd.Timestamp("2023-08-12")
    assert out[1] == pd.Timestamp("1999-01-05")  # dd/mm/yy, day first
    assert pd.isna(out[2])


def test_clean_matches_renames_sorts_and_casts():
    raw = raw_frame(
        [
            ["20/08/2023", " Arsenal ", "Chelsea", 2.0, 1.0, "H", 2.0, 3.4, 3.5],
            ["12/08/2023", "Burnley", "Man City", 0.0, 3.0, "A", 7.5, 5.0, 1.4],
        ]
    )

    df = data.clean_matches(raw)

    assert list(df.columns) == list(data.COLUMN_NAMES.values())
    assert df["date"].is_monotonic_increasing
    assert df.loc[0, "home_team"] == "Burnley"
    assert df.loc[1, "home_team"] == "Arsenal"  # whitespace stripped
    assert df["home_goals"].dtype == "int64"
    assert df.loc[0, "away_goals"] == 3


def test_clean_matches_drops_rows_missing_score_or_date():
    nan = float("nan")
    raw = raw_frame(
        [
            ["12/08/2023", "A", "B", 1.0, 0.0, "H", 2.0, 3.0, 4.0],
            ["13/08/2023", "C", "D", nan, nan, None, 2.0, 3.0, 4.0],  # unplayed
            ["not a date", "E", "F", 1.0, 1.0, "D", 2.0, 3.0, 4.0],
        ]
    )

    df = data.clean_matches(raw)

    assert len(df) == 1
    assert df.loc[0, "home_team"] == "A"


def test_clean_matches_keeps_file_order_within_a_day():
    raw = raw_frame(
        [
            ["12/08/2023", "A", "B", 1.0, 0.0, "H", 2.0, 3.0, 4.0],
            ["12/08/2023", "C", "D", 0.0, 0.0, "D", 2.0, 3.0, 4.0],
            ["12/08/2023", "E", "F", 2.0, 2.0, "D", 2.0, 3.0, 4.0],
        ]
    )

    df = data.clean_matches(raw)

    assert list(df["home_team"]) == ["A", "C", "E"]


def teams_frame(rows):
    return pd.DataFrame(rows, columns=["season", "home_team", "away_team"])


def test_find_suspicious_team_names_flags_spelling_variants():
    df = teams_frame(
        [
            [2022, "Man United", "Arsenal"],
            [2023, "Man Utd", "Arsenal"],  # same club, new spelling
        ]
    )

    assert data.find_suspicious_team_names(df) == [("Man United", "Man Utd")]


def test_find_suspicious_team_names_flags_case_and_punctuation_only():
    df = teams_frame([[2022, "Nott'm Forest", "Arsenal"], [2023, "Nottm Forest", "Arsenal"]])

    assert data.find_suspicious_team_names(df) == [("Nott'm Forest", "Nottm Forest")]


def test_find_suspicious_team_names_ignores_similar_clubs_in_same_season():
    df = teams_frame([[2023, "Sheffield United", "Sheffield Weds"]])

    assert data.find_suspicious_team_names(df) == []


def test_find_suspicious_team_names_clean_data_gives_empty_list():
    df = teams_frame([[2022, "Arsenal", "Chelsea"], [2023, "Arsenal", "Chelsea"]])

    assert data.find_suspicious_team_names(df) == []


def fake_download(raw_csvs):
    """Build a download_season stand-in that writes a prepared CSV per season."""

    def _download(start_year, dest_dir, league="E0", force=False):
        path = dest_dir / f"{league}_{data.season_code(start_year)}.csv"
        path.write_text(raw_csvs[start_year], encoding="utf-8")
        return path

    return _download


def test_season_label():
    assert data.season_label(2023) == "2023/24"
    assert data.season_label(1999) == "1999/00"


def test_build_dataset_combines_sorts_and_saves(tmp_path, monkeypatch):
    header = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
    csvs = {
        2023: header + "12/08/2023,Burnley,Man City,0,3,A,7.5,5.0,1.4\n",
        2022: header + "05/08/2022,Arsenal,Chelsea,2,1,H,2.0,3.4,3.5\n",
    }
    monkeypatch.setattr(data, "download_season", fake_download(csvs))
    out_path = tmp_path / "processed" / "matches.csv"

    df = data.build_dataset([2023, 2022], raw_dir=tmp_path, out_path=out_path)

    assert list(df["season"]) == ["2022/23", "2023/24"]  # sorted by date, not input order
    assert out_path.exists()
    saved = pd.read_csv(out_path, parse_dates=["date"])
    assert len(saved) == 2
    assert saved["date"].is_monotonic_increasing
