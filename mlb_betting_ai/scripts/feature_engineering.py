"""Utilities for engineering features from historical game data."""

from pathlib import Path

import pandas as pd

# Path to the CSV containing scraped historical games
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "historical_games.csv"


def load_games() -> pd.DataFrame:
    """Load historical games into a DataFrame sorted by date."""
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def calculate_win_loss(home_score: int, away_score: int) -> tuple[bool, bool]:
    """Return a tuple indicating if the home and away teams won."""
    home_win = home_score > away_score
    away_win = away_score > home_score
    return home_win, away_win


def add_win_loss_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add win/loss columns for each team and return the DataFrame."""
    wins = df.apply(
        lambda row: calculate_win_loss(row["home_score"], row["away_score"]), axis=1
    )
    df[["home_win", "away_win"]] = pd.DataFrame(wins.tolist(), index=df.index)
    return df


def _rolling_pct(results: list[int], n: int | None = None) -> float | None:
    """Return win percentage over the last n results or entire list."""
    if n is None:
        sub = results
    else:
        sub = results[-n:]
    if not sub:
        return None
    return sum(sub) / len(sub)


def add_rolling_win_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling win percentage features for each team's prior games."""
    df = df.copy()

    home_last5: list[float | None] = []
    home_last10: list[float | None] = []
    home_season: list[float | None] = []
    away_last5: list[float | None] = []
    away_last10: list[float | None] = []
    away_season: list[float | None] = []

    overall_results: dict[str, list[int]] = {}
    season_results: dict[str, dict[int, list[int]]] = {}

    for row in df.itertuples(index=False):
        year = row.date.year

        for team in [row.home_team, row.away_team]:
            overall_results.setdefault(team, [])
            season_results.setdefault(team, {})
            season_results[team].setdefault(year, [])

        h_overall = overall_results[row.home_team]
        a_overall = overall_results[row.away_team]
        h_season = season_results[row.home_team][year]
        a_season = season_results[row.away_team][year]

        home_last5.append(_rolling_pct(h_overall, 5))
        home_last10.append(_rolling_pct(h_overall, 10))
        home_season.append(_rolling_pct(h_season))
        away_last5.append(_rolling_pct(a_overall, 5))
        away_last10.append(_rolling_pct(a_overall, 10))
        away_season.append(_rolling_pct(a_season))

        h_result = 1 if row.home_win else 0
        a_result = 1 if row.away_win else 0

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


if __name__ == "__main__":
    games = load_games()
    games = add_win_loss_columns(games)
    games = add_rolling_win_features(games)
    print(games.head())
