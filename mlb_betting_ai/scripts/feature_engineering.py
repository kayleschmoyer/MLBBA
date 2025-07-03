"""Utilities for engineering features from historical game data."""

from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime


# File paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "historical_games.csv"
FEATURES_FILE = BASE_DIR / "data" / "games_with_features.csv"

TEAM_NAME_FIX = {
    "D'Backs": "Diamondbacks",
    "Red Sox": "Red Sox",
    "White Sox": "White Sox",
}

def clean_team_name(name: str) -> str:
    return TEAM_NAME_FIX.get(name, name)

def load_games() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_win_loss(home_score: int, away_score: int) -> tuple[bool, bool]:
    home_win = home_score > away_score
    away_win = away_score > home_score
    return home_win, away_win

def add_win_loss_columns(df: pd.DataFrame) -> pd.DataFrame:
    wins = df.apply(lambda row: calculate_win_loss(row["home_score"], row["away_score"]), axis=1)
    df[["home_win", "away_win"]] = pd.DataFrame(wins.tolist(), index=df.index)
    return df

def _rolling_pct(results: list[int], n: int | None = None) -> float | None:
    sub = results[-n:] if n else results
    return sum(sub) / len(sub) if sub else None

def _rolling_mean(values: list[int], n: int | None = None) -> float | None:
    sub = values[-n:] if n else values
    return sum(sub) / len(sub) if sub else None

def _run_diff(scored: list[int], allowed: list[int], n: int | None = 10) -> float | None:
    scored_sub = scored[-n:] if n else scored
    allowed_sub = allowed[-n:] if n else allowed
    return sum(scored_sub) - sum(allowed_sub) if scored_sub else None

def add_rolling_win_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home_last5, home_last10, home_season = [], [], []
    away_last5, away_last10, away_season = [], [], []

    overall_results, season_results = {}, {}

    for row in df.itertuples(index=False):
        year = row.date.year
        for team in [row.home_team, row.away_team]:
            overall_results.setdefault(team, [])
            season_results.setdefault(team, {}).setdefault(year, [])

        h_overall, a_overall = overall_results[row.home_team], overall_results[row.away_team]
        h_season, a_season = season_results[row.home_team][year], season_results[row.away_team][year]

        home_last5.append(_rolling_pct(h_overall, 5))
        home_last10.append(_rolling_pct(h_overall, 10))
        home_season.append(_rolling_pct(h_season))
        away_last5.append(_rolling_pct(a_overall, 5))
        away_last10.append(_rolling_pct(a_overall, 10))
        away_season.append(_rolling_pct(a_season))

        h_result, a_result = int(row.home_win), int(row.away_win)
        h_overall.append(h_result)
        a_overall.append(a_result)
        h_season.append(h_result)
        a_season.append(a_result)

    df["home_win_pct_last5"] = home_last5
    df["home_win_pct_last10"] = home_last10
    df["home_win_pct_season"] = home_season
    df["away_win_pct_last5"] = away_last5
    df["away_win_pct_last10"] = away_last10
    df["away_win_pct_season"] = away_season

    return df

def add_home_away_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home_team_home_win_pct, away_team_away_win_pct = [], []
    home_records, away_records = {}, {}

    for row in df.itertuples(index=False):
        h_record = home_records.setdefault(row.home_team, [])
        a_record = away_records.setdefault(row.away_team, [])

        home_team_home_win_pct.append(_rolling_pct(h_record))
        away_team_away_win_pct.append(_rolling_pct(a_record))

        h_record.append(int(row.home_win))
        a_record.append(int(row.away_win))

    df["home_team_home_win_pct"] = home_team_home_win_pct
    df["away_team_away_win_pct"] = away_team_away_win_pct

    return df

def add_head_to_head_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    head_to_head_pct = []
    matchup_results = {}

    for row in df.itertuples(index=False):
        cutoff = row.date - pd.DateOffset(years=5)
        matchup = tuple(sorted([row.home_team, row.away_team]))
        results = matchup_results.get(matchup, [])
        recent = [(d, w) for d, w in results if d >= cutoff]

        wins = sum(1 for d, w in recent if w == row.home_team)
        pct = wins / len(recent) if recent else None
        head_to_head_pct.append(pct)

        winner = row.home_team if row.home_win else row.away_team
        results.append((row.date, winner))
        matchup_results[matchup] = [r for r in results if r[0] >= cutoff]

    df["head_to_head_win_pct"] = head_to_head_pct
    return df

def add_run_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scored, allowed = {}, {}

    home_scored_last5, home_scored_last10 = [], []
    home_allowed_last5, home_allowed_last10 = [], []
    home_run_diff_last10 = []
    away_scored_last5, away_scored_last10 = [], []
    away_allowed_last5, away_allowed_last10 = [], []
    away_run_diff_last10 = []

    for row in df.itertuples(index=False):
        for team in [row.home_team, row.away_team]:
            scored.setdefault(team, [])
            allowed.setdefault(team, [])

        h_scored, h_allowed = scored[row.home_team], allowed[row.home_team]
        a_scored, a_allowed = scored[row.away_team], allowed[row.away_team]

        home_scored_last5.append(_rolling_mean(h_scored, 5))
        home_scored_last10.append(_rolling_mean(h_scored, 10))
        home_allowed_last5.append(_rolling_mean(h_allowed, 5))
        home_allowed_last10.append(_rolling_mean(h_allowed, 10))
        home_run_diff_last10.append(_run_diff(h_scored, h_allowed, 10))

        away_scored_last5.append(_rolling_mean(a_scored, 5))
        away_scored_last10.append(_rolling_mean(a_scored, 10))
        away_allowed_last5.append(_rolling_mean(a_allowed, 5))
        away_allowed_last10.append(_rolling_mean(a_allowed, 10))
        away_run_diff_last10.append(_run_diff(a_scored, a_allowed, 10))

        h_scored.append(row.home_score)
        h_allowed.append(row.away_score)
        a_scored.append(row.away_score)
        a_allowed.append(row.home_score)

    df["home_avg_runs_scored_last5"] = home_scored_last5
    df["home_avg_runs_scored_last10"] = home_scored_last10
    df["home_avg_runs_allowed_last5"] = home_allowed_last5
    df["home_avg_runs_allowed_last10"] = home_allowed_last10
    df["home_run_diff_last10"] = home_run_diff_last10
    df["away_avg_runs_scored_last5"] = away_scored_last5
    df["away_avg_runs_scored_last10"] = away_scored_last10
    df["away_avg_runs_allowed_last5"] = away_allowed_last5
    df["away_avg_runs_allowed_last10"] = away_allowed_last10
    df["away_run_diff_last10"] = away_run_diff_last10

    return df

def fetch_run_differentials():
    import pandas as pd
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"

    batting_path = DATA_DIR / "team_batting_2025.csv"
    pitching_path = DATA_DIR / "team_pitching_2025.csv"

    print("📄 Loading local batting and pitching CSVs...")
    bat_df = pd.read_csv(batting_path)
    pit_df = pd.read_csv(pitching_path)

    # Ensure "Tm" is a string
    bat_df["Tm"] = bat_df["Tm"].astype(str)
    pit_df["Tm"] = pit_df["Tm"].astype(str)

    # Remove headers and league average rows
    bat_df = bat_df[~bat_df["Tm"].str.contains("LgAvg|Tm", na=False)]
    pit_df = pit_df[~pit_df["Tm"].str.contains("LgAvg|Tm", na=False)]

    # Clean team names
    bat_df["Team"] = bat_df["Tm"].str.replace("*", "", regex=False)
    pit_df["Team"] = pit_df["Tm"].str.replace("*", "", regex=False)

    # Fix: pitching table uses "R" for runs allowed — rename to RA
    pit_df = pit_df.rename(columns={"R": "RA"})

    # Merge and compute differential
    merged = pd.merge(
        bat_df[["Team", "R"]],
        pit_df[["Team", "RA"]],
        on="Team",
        how="inner"
    )
    merged["run_diff"] = merged["R"] - merged["RA"]

    print("✅ Run differential calculated for", len(merged), "teams")
    return merged.set_index("Team")["run_diff"].to_dict()

def estimate_rest_days(df):
    rest_days = {}
    last_played = {}

    for idx, row in df.sort_values("date").iterrows():
        date = pd.to_datetime(row["date"])
        for team in [row["home_team"], row["away_team"]]:
            if team not in last_played:
                rest_days[(idx, team)] = 3
            else:
                rest_days[(idx, team)] = min((date - last_played[team]).days, 7)
            last_played[team] = date

    df["home_rest_days"] = df.apply(lambda row: rest_days.get((row.name, row["home_team"]), 3), axis=1)
    df["away_rest_days"] = df.apply(lambda row: rest_days.get((row.name, row["away_team"]), 3), axis=1)
    return df

def add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    print("📊 Fetching run differentials...")
    run_diff = fetch_run_differentials()
    df["home_run_diff_season"] = df["home_team"].apply(lambda x: run_diff.get(clean_team_name(x), 0))
    df["away_run_diff_season"] = df["away_team"].apply(lambda x: run_diff.get(clean_team_name(x), 0))

    print("⏱ Estimating rest days...")
    df = estimate_rest_days(df)

    # ✅ New block: Inject real pitcher ERA
    print("📊 Loading pitcher ERA data...")
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_DIR = BASE_DIR / "data"

    era_df = pd.read_csv(DATA_DIR / "starting_pitcher_era_2025.csv")
    era_map = era_df.set_index("Team")["ERA"].to_dict()

    df["home_starting_pitcher_era"] = df["home_team"].apply(lambda x: era_map.get(clean_team_name(x), 4.00))
    df["away_starting_pitcher_era"] = df["away_team"].apply(lambda x: era_map.get(clean_team_name(x), 4.00))

    return df
if __name__ == "__main__":
    games = load_games()
    games = add_win_loss_columns(games)
    games = add_rolling_win_features(games)
    games = add_home_away_win_pct(games)
    games = add_head_to_head_win_pct(games)
    games = add_run_stats(games)
    games = add_new_features(games)
    games.to_csv(FEATURES_FILE, index=False)
    print(f"✅ Saved features to {FEATURES_FILE}")
    print("Feature engineering complete!")
    print(f"Total games processed: {len(games)}")
