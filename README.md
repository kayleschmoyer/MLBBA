# MLB Betting AI

This project aims to build an open-source MLB betting prediction system. The
code is organized under the `mlb_betting_ai` directory.

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Baseball-Reference scraper to gather historical game results:
   ```bash
   python mlb_betting_ai/scripts/scrape_reference.py
   ```
   This will create `mlb_betting_ai/data/historical_games.csv` with results from
   2010 through 2024.

More scripts will be added for feature engineering, model training, and
daily prediction generation.
