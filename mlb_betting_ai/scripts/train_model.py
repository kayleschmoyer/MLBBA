"""Train an XGBoost model to predict MLB home team wins."""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "games_with_features.csv"
MODEL_FILE = Path(__file__).resolve().parents[1] / "models" / "mlb_model.pkl"

def load_dataset() -> pd.DataFrame:
    """Load engineered game features."""
    return pd.read_csv(DATA_FILE)

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return feature matrix, target vector, and list of feature names."""
    df = df.dropna().copy()
    df["home_win"] = df["home_win"].astype(int)

    # Columns to exclude from modeling
    excluded_cols = {
        "home_win", "away_win", "home_score", "away_score",
        "date", "boxscore_url", "home_team", "away_team"
    }

    feature_cols: list[str] = []
    for col in df.columns:
        if col in excluded_cols:
            continue
        if col.startswith("home_") or col.startswith("away_"):
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

    X = df[feature_cols]
    y = df["home_win"]

    print(f"Using {len(feature_cols)} features:\n{feature_cols}")
    return X, y, feature_cols

def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    """Fit and return an XGBoost classifier."""
    model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X, y)
    return model

def evaluate_model(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> None:
    """Print accuracy, precision, and recall for the model."""
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")

def display_importances(model: XGBClassifier, feature_names: list[str]) -> None:
    """Print the top 10 most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    print("Top 10 Features:")
    for rank, idx in enumerate(indices, start=1):
        name = feature_names[idx]
        score = importances[idx]
        print(f"{rank}. {name}: {score:.4f}")

def main() -> None:
    df = load_dataset()
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    display_importances(model, feature_names)
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
