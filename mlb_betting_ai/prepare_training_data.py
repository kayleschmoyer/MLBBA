"""Prepare MLB training data by adding 'winner' column based on scores."""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[0]
SOURCE_FILE = BASE_DIR / "data" / "games_with_features.csv"
OUTPUT_FILE = BASE_DIR / "data" / "games_with_winner.csv"

def add_winner_column():
    df = pd.read_csv(SOURCE_FILE)

    # Add winner: 1 if home_score > away_score, else 0
    df["winner"] = (df["home_score"] > df["away_score"]).astype(int)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved updated file with winner column → {OUTPUT_FILE}")

if __name__ == "__main__":
    add_winner_column()
