import csv
import re
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2010
END_YEAR = 2024
OUTPUT_FILE = Path(__file__).resolve().parents[1] / "data" / "historical_games.csv"


def parse_schedule(html: str, year: int):
    """Parse schedule HTML and yield game dicts."""
    soup = BeautifulSoup(html, "lxml")
    label_span = soup.find("span", {"data-label": "MLB Schedule"})
    if not label_span:
        raise ValueError(f"Could not find schedule section for {year}")
    section = label_span.find_parent("div", class_="section_wrapper")
    content = section.find("div", class_="section_content")

    for day_div in content.find_all("div", recursive=False):
        h3 = day_div.find("h3")
        if not h3:
            continue
        date_text = h3.get_text(strip=True)
        try:
            date = datetime.strptime(date_text.split(", ", 1)[1], "%B %d, %Y").strftime("%Y-%m-%d")
        except ValueError:
            # Skip malformed dates
            continue
        for p in day_div.find_all("p", class_="game"):
            text = p.get_text(" ", strip=True)
            if "Boxscore" not in text:
                # ignore games without results
                continue
            if "@" not in text:
                continue
            try:
                away_part, home_part = text.split("@")
                away_score = int(re.search(r"\((\d+)\)", away_part).group(1))
                home_score = int(re.search(r"\((\d+)\)", home_part).group(1))
                away_team = re.sub(r"\s*\(\d+\)", "", away_part).strip()
                home_team = re.sub(r"\s*\(\d+\)", "", home_part.split("Boxscore")[0]).strip()
            except Exception:
                continue
            link_tag = p.find("a", string="Boxscore")
            link = BASE_URL + link_tag["href"] if link_tag else ""
            yield {
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "boxscore_url": link,
            }


def scrape_year(year: int):
    url = f"{BASE_URL}/leagues/majors/{year}-schedule.shtml"
    resp = requests.get(url)
    resp.raise_for_status()
    return list(parse_schedule(resp.text, year))


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    all_games = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Scraping {year}...")
        try:
            games = scrape_year(year)
            all_games.extend(games)
        except Exception as e:
            print(f"Failed to scrape {year}: {e}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "boxscore_url",
            ],
        )
        writer.writeheader()
        for row in all_games:
            writer.writerow(row)
    print(f"Saved {len(all_games)} games to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
