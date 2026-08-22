import os
import time
import pandas as pd
import statsapi


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]


# --------------------------------------------------
# DOWNLOAD REGULAR-SEASON GAMES
# --------------------------------------------------

def get_regular_season_games(start_date, end_date):

    schedule = statsapi.schedule(
        start_date=start_date,
        end_date=end_date
    )

    rows = []

    for game in schedule:

        # Regular season only
        if game.get("game_type") != "R":
            continue

        # Must have scores
        if (
            game.get("away_score") is None
            or game.get("home_score") is None
        ):
            continue

        # Completed MLB regular-season games cannot tie
        if game["away_score"] == game["home_score"]:
            continue

        rows.append({
            "date": game["game_date"],
            "game_id": game["game_id"],
            "away_team": game["away_name"],
            "home_team": game["home_name"],
            "away_score": game["away_score"],
            "home_score": game["home_score"],
            "home_win":
                1
                if game["home_score"] > game["away_score"]
                else 0
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# BUILD ONE SEASON
# --------------------------------------------------

def build_season(year):

    print()
    print("=" * 50)
    print(f"BUILDING {year} SEASON")
    print("=" * 50)

    # MLB regular seasons fit safely inside
    # March through early October.
    date_ranges = [
        (f"{year}-03-01", f"{year}-03-31"),
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-15")
    ]

    all_games = []

    for start_date, end_date in date_ranges:

        print(
            f"Downloading {start_date} to {end_date}..."
        )

        games = get_regular_season_games(
            start_date,
            end_date
        )

        all_games.append(games)

        # Be polite to MLB's server
        time.sleep(0.5)

    season = pd.concat(
        all_games,
        ignore_index=True
    )

    # Remove duplicate game IDs
    season = season.drop_duplicates(
        subset="game_id"
    )

    # Chronological order
    season = season.sort_values(
        by=["date", "game_id"]
    )

    # Reset row numbers
    season = season.reset_index(
        drop=True
    )

    return season


# --------------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------------

os.makedirs(
    "data/raw",
    exist_ok=True
)


# --------------------------------------------------
# BUILD ALL SEASONS
# --------------------------------------------------

for year in YEARS:

    season = build_season(year)

    output_path = (
        f"data/raw/games_{year}.csv"
    )

    season.to_csv(
        output_path,
        index=False
    )

    print()
    print(f"{year} SAVED")
    print("Games:", len(season))
    print(
        "Location:",
        os.path.abspath(output_path)
    )

    # Count games played by every team
    team_counts = pd.concat([
        season["home_team"],
        season["away_team"]
    ]).value_counts().sort_values()

    print()
    print("Lowest team game counts:")
    print(team_counts.head())

    print()