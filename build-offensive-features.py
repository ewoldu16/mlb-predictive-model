import os
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]


# --------------------------------------------------
# TEAM NAME MAPPING
# --------------------------------------------------

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


# --------------------------------------------------
# BUILD FEATURES FOR ONE SEASON
# --------------------------------------------------

def build_offensive_features(year):

    print()
    print("=" * 60)
    print(f"BUILDING OFFENSIVE FEATURES FOR {year}")
    print("=" * 60)

    games_path = f"data/raw/games_{year}.csv"
    statcast_path = f"data/raw/statcast_{year}.csv"

    print("Loading data...")

    games = pd.read_csv(games_path)
    statcast = pd.read_csv(statcast_path)

    games["date"] = pd.to_datetime(
        games["date"]
    )

    statcast["game_date"] = pd.to_datetime(
        statcast["game_date"]
    )

    print("Games loaded:", len(games))
    print(
        "Plate appearances loaded:",
        len(statcast)
    )


    # --------------------------------------------------
    # TEAM CODE MAPPING
    # --------------------------------------------------

    games["home_team_code"] = (
        games["home_team"].map(TEAM_MAP)
    )

    games["away_team_code"] = (
        games["away_team"].map(TEAM_MAP)
    )

    missing_home = games[
        games["home_team_code"].isna()
    ]["home_team"].unique()

    missing_away = games[
        games["away_team_code"].isna()
    ]["away_team"].unique()

    if (
        len(missing_home) > 0
        or len(missing_away) > 0
    ):

        print()
        print("MISSING TEAM MAPPINGS")

        print(
            "Home:",
            missing_home
        )

        print(
            "Away:",
            missing_away
        )

        return


    # --------------------------------------------------
    # NORMALISE OAKLAND / ATHLETICS CODE
    # --------------------------------------------------

    # 2021-2024 Statcast uses OAK
    # 2025 may use ATH
    statcast["batting_team"] = (
        statcast["batting_team"]
        .replace({
            "OAK": "ATH"
        })
    )

    games["home_team_code"] = (
        games["home_team_code"]
        .replace({
            "OAK": "ATH"
        })
    )

    games["away_team_code"] = (
        games["away_team_code"]
        .replace({
            "OAK": "ATH"
        })
    )


    # --------------------------------------------------
    # DAILY TEAM OFFENSE
    # --------------------------------------------------

    print()
    print(
        "Building daily offensive table..."
    )

    daily_offense = (
        statcast
        .groupby(
            [
                "batting_team",
                "game_date"
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
            )
        )
        .reset_index()
    )

    print(
        "Team-days:",
        len(daily_offense)
    )


    # --------------------------------------------------
    # CALCULATE PRE-GAME WOBA
    # --------------------------------------------------

    def calculate_team_woba(
        team_code,
        game_date,
        days=None
    ):

        team_data = daily_offense[
            daily_offense[
                "batting_team"
            ] == team_code
        ]

        # Only use games BEFORE prediction date
        team_data = team_data[
            team_data[
                "game_date"
            ] < game_date
        ]

        if days is not None:

            start_date = (
                game_date
                - pd.Timedelta(
                    days=days
                )
            )

            team_data = team_data[
                team_data[
                    "game_date"
                ] >= start_date
            ]

        if len(team_data) == 0:
            return None

        numerator = (
            team_data[
                "woba_value_sum"
            ].sum()
        )

        denominator = (
            team_data[
                "woba_denom_sum"
            ].sum()
        )

        if denominator == 0:
            return None

        return (
            numerator
            / denominator
        )


    # --------------------------------------------------
    # FEATURE CREATION
    # --------------------------------------------------

    print()
    print(
        "Calculating pre-game features..."
    )

    feature_rows = []

    total_games = len(games)

    for index, game in games.iterrows():

        game_date = game["date"]

        home_code = game[
            "home_team_code"
        ]

        away_code = game[
            "away_team_code"
        ]


        home_season_woba = (
            calculate_team_woba(
                home_code,
                game_date
            )
        )

        away_season_woba = (
            calculate_team_woba(
                away_code,
                game_date
            )
        )


        home_l30_woba = (
            calculate_team_woba(
                home_code,
                game_date,
                days=30
            )
        )

        away_l30_woba = (
            calculate_team_woba(
                away_code,
                game_date,
                days=30
            )
        )


        feature_rows.append({

            "game_id":
                game["game_id"],

            "home_season_woba":
                home_season_woba,

            "away_season_woba":
                away_season_woba,

            "home_l30_woba":
                home_l30_woba,

            "away_l30_woba":
                away_l30_woba
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

    features = pd.DataFrame(
        feature_rows
    )

    games_with_features = (
        games.merge(
            features,
            on="game_id",
            how="left"
        )
    )


    # --------------------------------------------------
    # DIFFERENTIAL FEATURES
    # --------------------------------------------------

    games_with_features[
        "season_woba_diff"
    ] = (
        games_with_features[
            "home_season_woba"
        ]
        -
        games_with_features[
            "away_season_woba"
        ]
    )

    games_with_features[
        "l30_woba_diff"
    ] = (
        games_with_features[
            "home_l30_woba"
        ]
        -
        games_with_features[
            "away_l30_woba"
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
        f"games_{year}_offensive_features.csv"
    )

    games_with_features.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    print()
    print(
        f"{year} COMPLETE"
    )

    print(
        "Games:",
        len(games_with_features)
    )

    print(
        "Saved to:",
        os.path.abspath(
            output_path
        )
    )

    print()
    print(
        "Missing values:"
    )

    print(
        games_with_features[
            [
                "home_season_woba",
                "away_season_woba",
                "home_l30_woba",
                "away_l30_woba"
            ]
        ]
        .isna()
        .sum()
    )


# --------------------------------------------------
# BUILD ALL SEASONS
# --------------------------------------------------

for year in YEARS:

    build_offensive_features(
        year
    )


print()
print("=" * 60)
print(
    "ALL OFFENSIVE FEATURE DATASETS COMPLETE"
)
print("=" * 60)