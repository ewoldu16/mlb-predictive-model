import os
import time
import pandas as pd
import statsapi


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]


# --------------------------------------------------
# GET STARTERS FOR ONE GAME
# --------------------------------------------------

def get_starting_pitchers(game_id):

    try:
        # Get the full MLB boxscore for this game
        boxscore = statsapi.boxscore_data(game_id)

        # MLB boxscore data separates the two teams
        away_team = boxscore["away"]
        home_team = boxscore["home"]

        # Pitchers are listed in order of appearance.
        # The first pitcher should therefore be the starter.
        away_pitcher_ids = away_team["pitchers"]
        home_pitcher_ids = home_team["pitchers"]

        if (
            len(away_pitcher_ids) == 0
            or len(home_pitcher_ids) == 0
        ):
            return None

        away_starter_id = away_pitcher_ids[0]
        home_starter_id = home_pitcher_ids[0]

        # Player information is stored by ID
        away_player_key = (
            "ID" + str(away_starter_id)
        )

        home_player_key = (
            "ID" + str(home_starter_id)
        )

        away_starter_name = (
            away_team["players"]
            [away_player_key]
            ["person"]
            ["fullName"]
        )

        home_starter_name = (
            home_team["players"]
            [home_player_key]
            ["person"]
            ["fullName"]
        )

        return {
            "game_id": game_id,

            "away_starter_id":
                away_starter_id,

            "away_starter_name":
                away_starter_name,

            "home_starter_id":
                home_starter_id,

            "home_starter_name":
                home_starter_name
        }

    except Exception as error:

        print(
            f"ERROR on game {game_id}: "
            f"{error}"
        )

        return None


# --------------------------------------------------
# PROCESS ONE SEASON
# --------------------------------------------------

def build_starting_pitchers(year):

    print()
    print("=" * 60)
    print(
        f"STARTING PITCHERS - {year}"
    )
    print("=" * 60)

    games_path = (
        f"data/raw/games_{year}.csv"
    )

    games = pd.read_csv(
        games_path
    )

    print(
        "Games loaded:",
        len(games)
    )

    rows = []

    total_games = len(games)

    for index, game in games.iterrows():

        game_id = int(
            game["game_id"]
        )

        starter_data = (
            get_starting_pitchers(
                game_id
            )
        )

        if starter_data is not None:
            rows.append(
                starter_data
            )

        if (index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{index + 1} / "
                f"{total_games}"
            )

        # Avoid hammering MLB's API
        time.sleep(0.05)


    starters = pd.DataFrame(
        rows
    )


    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    os.makedirs(
        "data/raw",
        exist_ok=True
    )

    output_path = (
        f"data/raw/"
        f"starting_pitchers_{year}.csv"
    )

    starters.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print()
    print(
        f"{year} STARTERS COMPLETE"
    )

    print(
        "Games expected:",
        len(games)
    )

    print(
        "Games with starters:",
        len(starters)
    )

    print(
        "Missing games:",
        len(games) - len(starters)
    )

    print(
        "Saved to:",
        os.path.abspath(
            output_path
        )
    )

    print()

    if len(starters) > 0:

        print(
            starters.head(10)
        )


# --------------------------------------------------
# BUILD ALL SEASONS
# --------------------------------------------------

for year in YEARS:

    build_starting_pitchers(
        year
    )


print()
print("=" * 60)
print(
    "ALL STARTING PITCHER DATA COMPLETE"
)
print("=" * 60)