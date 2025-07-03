"""Log today's predictions to CSV for accuracy tracking."""

import pandas as pd
from predict_today import (
    load_model, load_features, load_feature_names,
    scrape_today_games, map_team_name, get_latest_team_features
)
from datetime import datetime
from pathlib import Path
import numpy as np

LOG_FILE = Path("logs") / f"predictions_{datetime.today().date()}.csv"
LOG_FILE.parent.mkdir(exist_ok=True)

def log_predictions():
    df = load_features()
    model = load_model()
    feature_names = load_feature_names()
    games = scrape_today_games()

    rows = []

    for game in games:
        home_team = map_team_name(game["home_team"], df)
        away_team = map_team_name(game["away_team"], df)

        home_feats = get_latest_team_features(df, home_team, is_home=True)
        away_feats = get_latest_team_features(df, away_team, is_home=False)
        features = {**home_feats, **away_feats}

        try:
            X_input = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names).astype(float)
        except:
            continue

        prob = model.predict_proba(X_input)[0][1]
        rows.append({
            "date": datetime.today().date(),
            "home_team": home_team,
            "away_team": away_team,
            "confidence_home_win": round(prob, 4),
            "predicted_winner": home_team if prob >= 0.5 else away_team
        })

    pd.DataFrame(rows).to_csv(LOG_FILE, index=False)
    print(f"✅ Saved predictions to {LOG_FILE}")

if __name__ == "__main__":
    log_predictions()
