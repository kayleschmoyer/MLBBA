"""Utilities for engineering features from historical game data."""

from pathlib import Path

import pandas as pd

# Path to the CSV containing scraped historical games
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "historical_games.csv"
# Path to store the engineered features
FEATURES_FILE = Path(__file__).resolve().parents[1] / "data" / "games_with_features.csv"


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


def _rolling_mean(values: list[int], n: int | None = None) -> float | None:
    """Return average of last n values or the entire list."""
    if n is None:
        sub = values
    else:
        sub = values[-n:]
    if not sub:
        return None
    return sum(sub) / len(sub)


def _run_diff(scored: list[int], allowed: list[int], n: int | None = 10) -> float | None:
    """Return run differential over the last n games."""
    if n is None:
        scored_sub = scored
        allowed_sub = allowed
    else:
        scored_sub = scored[-n:]
        allowed_sub = allowed[-n:]
    if not scored_sub:
        return None
    return sum(scored_sub) - sum(allowed_sub)


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


def add_home_away_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Add win percentage for teams when playing at home or on the road."""
    df = df.copy()

    home_team_home_win_pct: list[float | None] = []
    away_team_away_win_pct: list[float | None] = []

    home_records: dict[str, list[int]] = {}
    away_records: dict[str, list[int]] = {}

    for row in df.itertuples(index=False):
        h_record = home_records.setdefault(row.home_team, [])
        a_record = away_records.setdefault(row.away_team, [])

        home_team_home_win_pct.append(_rolling_pct(h_record))
        away_team_away_win_pct.append(_rolling_pct(a_record))

        h_result = 1 if row.home_win else 0
        a_result = 1 if row.away_win else 0

        h_record.append(h_result)
        a_record.append(a_result)

    df["home_team_home_win_pct"] = home_team_home_win_pct
    df["away_team_away_win_pct"] = away_team_away_win_pct

    return df


def add_head_to_head_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Add win percentage for the home team against the away team over the last
    five years."""

    df = df.copy()

    head_to_head_pct: list[float | None] = []
    matchup_results: dict[tuple[str, str], list[tuple[pd.Timestamp, str]]] = {}

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
    """Add rolling run averages and differential for each team's prior games."""
    df = df.copy()

    home_scored_last5: list[float | None] = []
    home_scored_last10: list[float | None] = []
    home_allowed_last5: list[float | None] = []
    home_allowed_last10: list[float | None] = []
    home_run_diff_last10: list[float | None] = []
    away_scored_last5: list[float | None] = []
    away_scored_last10: list[float | None] = []
    away_allowed_last5: list[float | None] = []
    away_allowed_last10: list[float | None] = []
    away_run_diff_last10: list[float | None] = []

    scored: dict[str, list[int]] = {}
    allowed: dict[str, list[int]] = {}

    for row in df.itertuples(index=False):
        for team in [row.home_team, row.away_team]:
            scored.setdefault(team, [])
            allowed.setdefault(team, [])

        h_scored = scored[row.home_team]
        h_allowed = allowed[row.home_team]
        a_scored = scored[row.away_team]
        a_allowed = allowed[row.away_team]

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


if __name__ == "__main__":
    games = load_games()
    games = add_win_loss_columns(games)
    games = add_rolling_win_features(games)
    games = add_home_away_win_pct(games)
    games = add_head_to_head_win_pct(games)
    games = add_run_stats(games)
    games.to_csv(FEATURES_FILE, index=False, encoding="utf-8", sep=",")
    print(f"Saved features to {FEATURES_FILE}")
