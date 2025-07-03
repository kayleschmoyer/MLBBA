"""Predict today's MLB home team wins using the trained XGBoost model."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier

# File paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "games_with_features.csv"
MODEL_FILE = BASE_DIR / "models" / "mlb_model.pkl"
FEATURE_FILE = BASE_DIR / "models" / "feature_names.txt"

# ESPN schedule URL
today_str = datetime.today().strftime("%Y%m%d")
ESPN_URL = f"https://www.espn.com/mlb/schedule/_/date/{today_str}"

def load_model() -> XGBClassifier:
    return joblib.load(MODEL_FILE)

def load_features() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE)
    df = df.dropna().copy()
    return df

def load_feature_names() -> list[str]:
    with open(FEATURE_FILE, "r") as f:
        return [line.strip() for line in f.readlines()]

def scrape_today_games() -> list[dict]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(ESPN_URL, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch ESPN schedule: {e}")
        return []

    soup = BeautifulSoup(response.text, "lxml")
    schedule_blocks = soup.find_all("div", class_="ResponsiveTable")
    games = []

    today = datetime.today()
    day_str = str(today.day)
    today_date = today.strftime(f"%A, %B {day_str}, %Y")

    for block in schedule_blocks:
        title = block.find("div", class_="Table__Title")
        if not title or today_date not in title.get_text(strip=True):
            continue

        team_links = block.select('a.AnchorLink[href^="/mlb/team/"]')
        team_names = [a.get_text(strip=True) for a in team_links if a.get_text(strip=True)]

        for i in range(0, len(team_names), 2):
            try:
                away = team_names[i]
                home = team_names[i + 1]
                games.append({"home_team": home, "away_team": away})
            except IndexError:
                continue
        break

    if not games:
        print("⚠️ No games found for today.")
    else:
        print(f"✅ Parsed {len(games)} games for today")
    return games

def map_team_name(raw_name: str, reference_df: pd.DataFrame) -> str:
    matches = [team for team in reference_df["home_team"].unique() if raw_name.lower() in team.lower()]
    return matches[0] if matches else raw_name

def get_latest_team_features(df: pd.DataFrame, team: str, is_home: bool) -> dict:
    side = "home" if is_home else "away"
    team_col = "home_team" if is_home else "away_team"
    team_games = df[df[team_col] == team]
    if team_games.empty:
        return {}
    latest_game = team_games.sort_values("date", ascending=False).iloc[0]
    return {col: latest_game[col] for col in df.columns if col.startswith(f"{side}_")}

def summarize_reasons(features: dict, feature_names: list[str], importance: list[float]) -> list[str]:
    top_indices = np.argsort(importance)[::-1][:5]
    top_features = [feature_names[i] for i in top_indices if feature_names[i] in features]
    return [f"{f}: {features[f]:.2f}" for f in top_features]

def predict_games(games: list[dict], df: pd.DataFrame, model: XGBClassifier, feature_names: list[str]) -> None:
    for game in games:
        home_team = map_team_name(game["home_team"], df)
        away_team = map_team_name(game["away_team"], df)

        home_feats = get_latest_team_features(df, home_team, is_home=True)
        away_feats = get_latest_team_features(df, away_team, is_home=False)
        features = {**home_feats, **away_feats}

        try:
            X_input = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names).astype(float)
        except KeyError as e:
            print(f"❌ Missing feature {e} for {away_team} @ {home_team}")
            continue
        except ValueError as e:
            print(f"❌ Value error for {away_team} @ {home_team}: {e}")
            continue

        prob = model.predict_proba(X_input)[0][1]
        reasons = summarize_reasons(features, feature_names, model.feature_importances_)

        print("\n🏆 Prediction:")
        print(f"{away_team} @ {home_team}")
        print(f"📈 Home Win Confidence: {prob * 100:.1f}%")
        print("🧠 Top Factors:")
        for reason in reasons:
            print(f"• {reason}")

def main() -> None:
    df = load_features()
    model = load_model()
    feature_names = load_feature_names()
    games = scrape_today_games()

    if not games:
        print("No games found today.")
        return

    predict_games(games, df, model, feature_names)

if __name__ == "__main__":
    main()
