import os
import pandas as pd
import numpy as np


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]


TEAM_MAP = {
    "Arizona Diamondbacks": "AZ",
    "Athletics": "ATH",
    "Oakland Athletics": "OAK",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH"
}


def normalise_team(team):

    if team == "OAK":
        return "ATH"

    return team


# --------------------------------------------------
# BUILD ONE SEASON
# --------------------------------------------------

def build_platoon_features(year):

    print()
    print("=" * 70)
    print(f"BUILDING PLATOON FEATURES FOR {year}")
    print("=" * 70)


    # --------------------------------------------------
    # LOAD EXISTING ADVANCED DATA
    # --------------------------------------------------

    games_path = (
        f"data/processed/"
        f"games_{year}_advanced_features.csv"
    )

    pitch_path = (
        f"data/raw/pitching/"
        f"statcast_pitches_{year}.csv"
    )


    print("Loading data...")

    games = pd.read_csv(
        games_path
    )

    pitches = pd.read_csv(
        pitch_path
    )


    games["date"] = pd.to_datetime(
        games["date"]
    )

    pitches["game_date"] = pd.to_datetime(
        pitches["game_date"]
    )


    print(
        "Games:",
        len(games)
    )

    print(
        "Pitch rows:",
        len(pitches)
    )


    # --------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # --------------------------------------------------

    required = [
        "events",
        "stand",
        "p_throws",
        "woba_value",
        "woba_denom",
        "home_team",
        "away_team",
        "inning_topbot",
        "game_date",
        "game_pk",
        "pitcher"
    ]


    missing = [
        column
        for column in required
        if column not in pitches.columns
    ]


    if missing:

        print()
        print(
            "ERROR - REQUIRED COLUMNS MISSING:"
        )

        print(missing)

        print()
        print(
            "The current full-pitch files do not "
            "contain everything required for "
            "platoon features."
        )

        raise ValueError(
            "Missing required Statcast columns."
        )


    # --------------------------------------------------
    # ONLY PLATE-APPEARANCE ENDING ROWS
    # --------------------------------------------------

    pa = pitches[
        pitches["events"].notna()
    ].copy()


    # --------------------------------------------------
    # NORMALISE TEAMS
    # --------------------------------------------------

    pa["home_team"] = (
        pa["home_team"]
        .apply(normalise_team)
    )

    pa["away_team"] = (
        pa["away_team"]
        .apply(normalise_team)
    )


    # --------------------------------------------------
    # IDENTIFY BATTING TEAM
    # --------------------------------------------------

    pa["batting_team"] = np.where(
        pa["inning_topbot"] == "Top",
        pa["away_team"],
        pa["home_team"]
    )


    # --------------------------------------------------
    # TEAM CODES IN GAME DATA
    # --------------------------------------------------

    games["home_team_code_v3"] = (
        games["home_team"]
        .map(TEAM_MAP)
        .apply(normalise_team)
    )

    games["away_team_code_v3"] = (
        games["away_team"]
        .map(TEAM_MAP)
        .apply(normalise_team)
    )


    # --------------------------------------------------
    # FIND EACH STARTER'S THROWING HAND
    # --------------------------------------------------

    starter_hands = (
        pa[
            [
                "game_pk",
                "pitcher",
                "p_throws"
            ]
        ]
        .dropna()
        .drop_duplicates(
            subset=[
                "game_pk",
                "pitcher"
            ]
        )
    )


    # --------------------------------------------------
    # ADD HOME STARTER HAND
    # --------------------------------------------------

    home_hands = starter_hands.rename(
        columns={
            "game_pk": "game_id",
            "pitcher": "home_starter_id",
            "p_throws": "home_starter_hand"
        }
    )


    games = games.merge(
        home_hands[
            [
                "game_id",
                "home_starter_id",
                "home_starter_hand"
            ]
        ],
        on=[
            "game_id",
            "home_starter_id"
        ],
        how="left"
    )


    # --------------------------------------------------
    # ADD AWAY STARTER HAND
    # --------------------------------------------------

    away_hands = starter_hands.rename(
        columns={
            "game_pk": "game_id",
            "pitcher": "away_starter_id",
            "p_throws": "away_starter_hand"
        }
    )


    games = games.merge(
        away_hands[
            [
                "game_id",
                "away_starter_id",
                "away_starter_hand"
            ]
        ],
        on=[
            "game_id",
            "away_starter_id"
        ],
        how="left"
    )


    print()
    print(
        "Missing home starter hands:",
        games[
            "home_starter_hand"
        ].isna().sum()
    )

    print(
        "Missing away starter hands:",
        games[
            "away_starter_hand"
        ].isna().sum()
    )


    # --------------------------------------------------
    # BUILD TEAM / DATE / OPPOSING-PITCHER-HAND TABLE
    # --------------------------------------------------

    daily_platoon = (
        pa
        .groupby(
            [
                "batting_team",
                "game_date",
                "p_throws"
            ]
        )
        .agg(
            woba_value_sum=(
                "woba_value",
                "sum"
            ),

            woba_denom_sum=(
                "woba_denom",
                "sum"
            ),

            plate_appearances=(
                "events",
                "count"
            )
        )
        .reset_index()
    )


    print()
    print(
        "Team-date-hand rows:",
        len(daily_platoon)
    )


    # --------------------------------------------------
    # CALCULATE PRE-GAME PLATOON WOBA
    # --------------------------------------------------

    def calculate_platoon_woba(
        team,
        opponent_hand,
        game_date,
        days=None
    ):

        if pd.isna(opponent_hand):

            return {
                "woba": None,
                "pa": None
            }


        history = daily_platoon[
            (
                daily_platoon[
                    "batting_team"
                ] == team
            )
            &
            (
                daily_platoon[
                    "p_throws"
                ] == opponent_hand
            )
        ].copy()


        # ------------------------------------------
        # LEAKAGE PROTECTION
        # ------------------------------------------

        history = history[
            history["game_date"]
            < game_date
        ]


        # ------------------------------------------
        # OPTIONAL RECENT WINDOW
        # ------------------------------------------

        if days is not None:

            start_date = (
                game_date
                - pd.Timedelta(
                    days=days
                )
            )

            history = history[
                history["game_date"]
                >= start_date
            ]


        if len(history) == 0:

            return {
                "woba": None,
                "pa": None
            }


        numerator = (
            history[
                "woba_value_sum"
            ].sum()
        )

        denominator = (
            history[
                "woba_denom_sum"
            ].sum()
        )

        plate_appearances = (
            history[
                "plate_appearances"
            ].sum()
        )


        if denominator == 0:

            woba = None

        else:

            woba = (
                numerator
                / denominator
            )


        return {
            "woba": woba,
            "pa": plate_appearances
        }


    # --------------------------------------------------
    # BUILD GAME FEATURES
    # --------------------------------------------------

    print()
    print(
        "Calculating pre-game platoon features..."
    )


    rows = []

    total_games = len(games)


    for index, game in games.iterrows():

        date = game["date"]

        home_team = game[
            "home_team_code_v3"
        ]

        away_team = game[
            "away_team_code_v3"
        ]


        # Home hitters face the away starter.
        home_opponent_hand = game[
            "away_starter_hand"
        ]

        # Away hitters face the home starter.
        away_opponent_hand = game[
            "home_starter_hand"
        ]


        # ------------------------------------------
        # SEASON
        # ------------------------------------------

        home_season = (
            calculate_platoon_woba(
                home_team,
                home_opponent_hand,
                date
            )
        )

        away_season = (
            calculate_platoon_woba(
                away_team,
                away_opponent_hand,
                date
            )
        )


        # ------------------------------------------
        # LAST 30 DAYS
        # ------------------------------------------

        home_l30 = (
            calculate_platoon_woba(
                home_team,
                home_opponent_hand,
                date,
                days=30
            )
        )

        away_l30 = (
            calculate_platoon_woba(
                away_team,
                away_opponent_hand,
                date,
                days=30
            )
        )


        rows.append({

            "game_id":
                game["game_id"],

            "home_matchup_hand":
                home_opponent_hand,

            "away_matchup_hand":
                away_opponent_hand,


            # --------------------------------------
            # SEASON PLATOON
            # --------------------------------------

            "home_season_platoon_woba":
                home_season["woba"],

            "away_season_platoon_woba":
                away_season["woba"],

            "home_season_platoon_pa":
                home_season["pa"],

            "away_season_platoon_pa":
                away_season["pa"],


            # --------------------------------------
            # L30 PLATOON
            # --------------------------------------

            "home_l30_platoon_woba":
                home_l30["woba"],

            "away_l30_platoon_woba":
                away_l30["woba"],

            "home_l30_platoon_pa":
                home_l30["pa"],

            "away_l30_platoon_pa":
                away_l30["pa"]
        })


        if (
            (index + 1)
            % 250
            == 0
        ):

            print(
                f"Processed "
                f"{index + 1} / "
                f"{total_games}"
            )


    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    platoon_features = pd.DataFrame(
        rows
    )


    games = games.merge(
        platoon_features,
        on="game_id",
        how="left"
    )


    # --------------------------------------------------
    # DIFFERENTIAL FEATURES
    # --------------------------------------------------

    games[
        "season_platoon_woba_diff"
    ] = (
        games[
            "home_season_platoon_woba"
        ]
        -
        games[
            "away_season_platoon_woba"
        ]
    )


    games[
        "l30_platoon_woba_diff"
    ] = (
        games[
            "home_l30_platoon_woba"
        ]
        -
        games[
            "away_l30_platoon_woba"
        ]
    )


    # --------------------------------------------------
    # SAMPLE-SIZE FEATURES
    # --------------------------------------------------

    games[
        "home_season_platoon_pa_log"
    ] = np.log1p(
        games[
            "home_season_platoon_pa"
        ]
    )

    games[
        "away_season_platoon_pa_log"
    ] = np.log1p(
        games[
            "away_season_platoon_pa"
        ]
    )


    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    output_path = (
        f"data/processed/"
        f"games_{year}_platoon_features.csv"
    )


    games.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print()
    print(
        f"{year} PLATOON FEATURES COMPLETE"
    )

    print(
        "Games:",
        len(games)
    )

    print(
        "Saved to:",
        os.path.abspath(
            output_path
        )
    )


    print()
    print(
        "Starter hand counts:"
    )

    print(
        games[
            "home_starter_hand"
        ].value_counts(
            dropna=False
        )
    )


    print()
    print(
        "Missing platoon features:"
    )


    check_columns = [

        "home_season_platoon_woba",
        "away_season_platoon_woba",

        "home_l30_platoon_woba",
        "away_l30_platoon_woba",

        "season_platoon_woba_diff",
        "l30_platoon_woba_diff"
    ]


    print(
        games[
            check_columns
        ]
        .isna()
        .sum()
    )


# --------------------------------------------------
# BUILD ALL YEARS
# --------------------------------------------------

for year in YEARS:

    build_platoon_features(
        year
    )


print()
print("=" * 70)
print(
    "ALL PLATOON FEATURES COMPLETE"
)
print("=" * 70)