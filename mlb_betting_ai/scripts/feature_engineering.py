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


if __name__ == "__main__":
    games = load_games()
    games = add_win_loss_columns(games)
    print(games.head())
