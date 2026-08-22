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


def normalise_team(team):
    if team == "OAK":
        return "ATH"

    return team


# --------------------------------------------------
# BUILD ONE SEASON
# --------------------------------------------------

def build_bullpen_features(year):

    print()
    print("=" * 60)
    print(f"BUILDING BULLPEN FEATURES FOR {year}")
    print("=" * 60)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    games_path = (
        f"data/processed/"
        f"games_{year}_offense_pitching.csv"
    )

    statcast_path = (
        f"data/raw/"
        f"statcast_{year}.csv"
    )

    print("Loading data...")

    games = pd.read_csv(games_path)
    statcast = pd.read_csv(statcast_path)

    games["date"] = pd.to_datetime(
        games["date"]
    )

    statcast["game_date"] = pd.to_datetime(
        statcast["game_date"]
    )

    print("Games:", len(games))
    print("Plate appearances:", len(statcast))


    # --------------------------------------------------
    # NORMALISE TEAM CODES
    # --------------------------------------------------

    games["home_team_code"] = (
        games["home_team"]
        .map(TEAM_MAP)
        .apply(normalise_team)
    )

    games["away_team_code"] = (
        games["away_team"]
        .map(TEAM_MAP)
        .apply(normalise_team)
    )

    statcast["home_team"] = (
        statcast["home_team"]
        .apply(normalise_team)
    )

    statcast["away_team"] = (
        statcast["away_team"]
        .apply(normalise_team)
    )

    statcast["batting_team"] = (
        statcast["batting_team"]
        .apply(normalise_team)
    )


    # --------------------------------------------------
    # IDENTIFY PITCHING TEAM
    # --------------------------------------------------

    # If the away team is batting,
    # the home team is pitching.
    # If the home team is batting,
    # the away team is pitching.

    statcast["pitching_team"] = statcast.apply(
        lambda row:
            row["home_team"]
            if row["batting_team"] == row["away_team"]
            else row["away_team"],
        axis=1
    )


    # --------------------------------------------------
    # MAP STARTER TO EACH GAME + TEAM
    # --------------------------------------------------

    starter_rows = []

    for _, game in games.iterrows():

        starter_rows.append({
            "game_id": game["game_id"],
            "pitching_team": game["home_team_code"],
            "starter_id": game["home_starter_id"]
        })

        starter_rows.append({
            "game_id": game["game_id"],
            "pitching_team": game["away_team_code"],
            "starter_id": game["away_starter_id"]
        })

    starter_map = pd.DataFrame(
        starter_rows
    )


    # --------------------------------------------------
    # ADD STARTER ID TO STATCAST DATA
    # --------------------------------------------------

    statcast = statcast.merge(
        starter_map,
        left_on=[
            "game_pk",
            "pitching_team"
        ],
        right_on=[
            "game_id",
            "pitching_team"
        ],
        how="left"
    )


    # --------------------------------------------------
    # REMOVE STARTING PITCHER
    # --------------------------------------------------

    bullpen = statcast[
        statcast["pitcher"]
        != statcast["starter_id"]
    ].copy()

    print()
    print(
        "Bullpen plate appearances:",
        len(bullpen)
    )


    # --------------------------------------------------
    # EVENT FLAGS
    # --------------------------------------------------

    bullpen["is_k"] = (
        bullpen["events"]
        .isin([
            "strikeout",
            "strikeout_double_play"
        ])
        .astype(int)
    )

    bullpen["is_bb"] = (
        bullpen["events"]
        .isin([
            "walk",
            "intent_walk"
        ])
        .astype(int)
    )

    bullpen["is_hr"] = (
        bullpen["events"]
        .eq("home_run")
        .astype(int)
    )


    # --------------------------------------------------
    # TEAM-DAY BULLPEN TABLE
    # --------------------------------------------------

    daily_bullpen = (
        bullpen
        .groupby([
            "pitching_team",
            "game_date"
        ])
        .agg(
            batters_faced=(
                "events",
                "count"
            ),

            strikeouts=(
                "is_k",
                "sum"
            ),

            walks=(
                "is_bb",
                "sum"
            ),

            home_runs=(
                "is_hr",
                "sum"
            ),

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
        "Bullpen team-days:",
        len(daily_bullpen)
    )


    # --------------------------------------------------
    # CALCULATE HISTORICAL BULLPEN STATS
    # --------------------------------------------------

    def calculate_bullpen_stats(
        team,
        game_date,
        days=None
    ):

        history = daily_bullpen[
            daily_bullpen[
                "pitching_team"
            ] == team
        ].copy()

        # CRITICAL:
        # never include today's game
        history = history[
            history["game_date"]
            < game_date
        ]

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
                "k_pct": None,
                "bb_pct": None,
                "hr_pct": None,
                "woba_allowed": None,
                "bf": None
            }

        bf = history[
            "batters_faced"
        ].sum()

        strikeouts = history[
            "strikeouts"
        ].sum()

        walks = history[
            "walks"
        ].sum()

        home_runs = history[
            "home_runs"
        ].sum()

        woba_sum = history[
            "woba_value_sum"
        ].sum()

        woba_denom = history[
            "woba_denom_sum"
        ].sum()

        if bf == 0:

            k_pct = None
            bb_pct = None
            hr_pct = None

        else:

            k_pct = strikeouts / bf
            bb_pct = walks / bf
            hr_pct = home_runs / bf

        if woba_denom == 0:

            woba_allowed = None

        else:

            woba_allowed = (
                woba_sum
                / woba_denom
            )

        return {
            "k_pct": k_pct,
            "bb_pct": bb_pct,
            "hr_pct": hr_pct,
            "woba_allowed": woba_allowed,
            "bf": bf
        }


    # --------------------------------------------------
    # BUILD GAME FEATURES
    # --------------------------------------------------

    print()
    print(
        "Calculating pre-game bullpen features..."
    )

    rows = []

    total_games = len(games)

    for index, game in games.iterrows():

        date = game["date"]

        home_team = game[
            "home_team_code"
        ]

        away_team = game[
            "away_team_code"
        ]


        # ------------------------------------------
        # SEASON TO DATE
        # ------------------------------------------

        home_season = (
            calculate_bullpen_stats(
                home_team,
                date
            )
        )

        away_season = (
            calculate_bullpen_stats(
                away_team,
                date
            )
        )


        # ------------------------------------------
        # LAST 30 DAYS
        # ------------------------------------------

        home_l30 = (
            calculate_bullpen_stats(
                home_team,
                date,
                days=30
            )
        )

        away_l30 = (
            calculate_bullpen_stats(
                away_team,
                date,
                days=30
            )
        )


        # ------------------------------------------
        # LAST 7 DAYS WORKLOAD
        # ------------------------------------------

        home_l7 = (
            calculate_bullpen_stats(
                home_team,
                date,
                days=7
            )
        )

        away_l7 = (
            calculate_bullpen_stats(
                away_team,
                date,
                days=7
            )
        )


        rows.append({

            "game_id":
                game["game_id"],


            # HOME SEASON
            "home_bp_season_k_pct":
                home_season["k_pct"],

            "home_bp_season_bb_pct":
                home_season["bb_pct"],

            "home_bp_season_hr_pct":
                home_season["hr_pct"],

            "home_bp_season_woba_allowed":
                home_season[
                    "woba_allowed"
                ],


            # AWAY SEASON
            "away_bp_season_k_pct":
                away_season["k_pct"],

            "away_bp_season_bb_pct":
                away_season["bb_pct"],

            "away_bp_season_hr_pct":
                away_season["hr_pct"],

            "away_bp_season_woba_allowed":
                away_season[
                    "woba_allowed"
                ],


            # HOME L30
            "home_bp_l30_k_pct":
                home_l30["k_pct"],

            "home_bp_l30_bb_pct":
                home_l30["bb_pct"],

            "home_bp_l30_woba_allowed":
                home_l30[
                    "woba_allowed"
                ],


            # AWAY L30
            "away_bp_l30_k_pct":
                away_l30["k_pct"],

            "away_bp_l30_bb_pct":
                away_l30["bb_pct"],

            "away_bp_l30_woba_allowed":
                away_l30[
                    "woba_allowed"
                ],


            # RECENT WORKLOAD
            "home_bp_l7_bf":
                home_l7["bf"],

            "away_bp_l7_bf":
                away_l7["bf"]
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

    bullpen_features = pd.DataFrame(
        rows
    )

    games = games.merge(
        bullpen_features,
        on="game_id",
        how="left"
    )


    # --------------------------------------------------
    # DIFFERENTIAL FEATURES
    # --------------------------------------------------

    # Higher K% = better
    games[
        "bp_season_k_pct_diff"
    ] = (
        games[
            "home_bp_season_k_pct"
        ]
        -
        games[
            "away_bp_season_k_pct"
        ]
    )


    # Lower BB% = better
    games[
        "bp_season_bb_pct_diff"
    ] = (
        games[
            "away_bp_season_bb_pct"
        ]
        -
        games[
            "home_bp_season_bb_pct"
        ]
    )


    # Lower wOBA allowed = better
    games[
        "bp_season_woba_allowed_diff"
    ] = (
        games[
            "away_bp_season_woba_allowed"
        ]
        -
        games[
            "home_bp_season_woba_allowed"
        ]
    )


    games[
        "bp_l30_k_pct_diff"
    ] = (
        games[
            "home_bp_l30_k_pct"
        ]
        -
        games[
            "away_bp_l30_k_pct"
        ]
    )


    games[
        "bp_l30_bb_pct_diff"
    ] = (
        games[
            "away_bp_l30_bb_pct"
        ]
        -
        games[
            "home_bp_l30_bb_pct"
        ]
    )


    games[
        "bp_l30_woba_allowed_diff"
    ] = (
        games[
            "away_bp_l30_woba_allowed"
        ]
        -
        games[
            "home_bp_l30_woba_allowed"
        ]
    )


    # Positive = away bullpen has worked more recently
    games[
        "bp_l7_bf_diff"
    ] = (
        games[
            "away_bp_l7_bf"
        ]
        -
        games[
            "home_bp_l7_bf"
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
        f"games_{year}_full_features.csv"
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
        f"{year} BULLPEN FEATURES COMPLETE"
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
        "Missing bullpen values:"
    )

    check_columns = [

        "home_bp_season_k_pct",
        "away_bp_season_k_pct",

        "home_bp_season_woba_allowed",
        "away_bp_season_woba_allowed",

        "home_bp_l30_k_pct",
        "away_bp_l30_k_pct",

        "home_bp_l7_bf",
        "away_bp_l7_bf"
    ]

    print(
        games[
            check_columns
        ]
        .isna()
        .sum()
    )


    print()
    print("Example rows:")

    print(
        games[
            [
                "date",
                "away_team",
                "home_team",

                "away_bp_season_woba_allowed",
                "home_bp_season_woba_allowed",

                "away_bp_l30_woba_allowed",
                "home_bp_l30_woba_allowed",

                "away_bp_l7_bf",
                "home_bp_l7_bf",

                "bp_season_woba_allowed_diff",

                "home_win"
            ]
        ]
        .tail(10)
    )


# --------------------------------------------------
# BUILD ALL YEARS
# --------------------------------------------------

for year in YEARS:

    build_bullpen_features(
        year
    )


print()
print("=" * 60)
print(
    "ALL BULLPEN FEATURES COMPLETE"
)
print("=" * 60)