import os
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]


# --------------------------------------------------
# BUILD PITCHING FEATURES FOR ONE SEASON
# --------------------------------------------------

def build_pitching_features(year):

    print()
    print("=" * 60)
    print(f"BUILDING STARTING PITCHER FEATURES FOR {year}")
    print("=" * 60)

    # ----------------------------------------------
    # LOAD DATA
    # ----------------------------------------------

    games_path = (
        f"data/processed/"
        f"games_{year}_offensive_features.csv"
    )

    starters_path = (
        f"data/raw/"
        f"starting_pitchers_{year}.csv"
    )

    statcast_path = (
        f"data/raw/"
        f"statcast_{year}.csv"
    )

    print("Loading data...")

    games = pd.read_csv(games_path)

    starters = pd.read_csv(
        starters_path
    )

    statcast = pd.read_csv(
        statcast_path
    )

    games["date"] = pd.to_datetime(
        games["date"]
    )

    statcast["game_date"] = pd.to_datetime(
        statcast["game_date"]
    )

    print("Games:", len(games))
    print("Starter records:", len(starters))
    print("Plate appearances:", len(statcast))


    # ----------------------------------------------
    # MERGE STARTER IDS INTO GAME TABLE
    # ----------------------------------------------

    games = games.merge(
        starters[
            [
                "game_id",
                "away_starter_id",
                "away_starter_name",
                "home_starter_id",
                "home_starter_name"
            ]
        ],
        on="game_id",
        how="left"
    )


    # ----------------------------------------------
    # PREPARE PITCHER PLATE-APPEARANCE DATA
    # ----------------------------------------------

    pitching_data = statcast[
        [
            "game_date",
            "game_pk",
            "pitcher",
            "events",
            "woba_value",
            "woba_denom"
        ]
    ].copy()


    # ----------------------------------------------
    # EVENT FLAGS
    # ----------------------------------------------

    # Strikeout
    pitching_data["is_k"] = (
        pitching_data["events"]
        .isin([
            "strikeout",
            "strikeout_double_play"
        ])
        .astype(int)
    )

    # Walk
    pitching_data["is_bb"] = (
        pitching_data["events"]
        .isin([
            "walk",
            "intent_walk"
        ])
        .astype(int)
    )

    # Home run
    pitching_data["is_hr"] = (
        pitching_data["events"]
        .eq("home_run")
        .astype(int)
    )


    # ----------------------------------------------
    # GROUP BY PITCHER + DATE
    # ----------------------------------------------

    daily_pitching = (
        pitching_data
        .groupby(
            [
                "pitcher",
                "game_date"
            ]
        )
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

            woba_allowed_sum=(
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

    print()
    print(
        "Pitcher-days created:",
        len(daily_pitching)
    )


    # ----------------------------------------------
    # CALCULATE PREGAME PITCHER FEATURES
    # ----------------------------------------------

    def calculate_pitcher_stats(
        pitcher_id,
        game_date,
        days=None
    ):

        pitcher_history = (
            daily_pitching[
                daily_pitching[
                    "pitcher"
                ] == pitcher_id
            ]
            .copy()
        )

        # ------------------------------------------
        # DATA LEAKAGE PROTECTION
        # ------------------------------------------

        pitcher_history = (
            pitcher_history[
                pitcher_history[
                    "game_date"
                ] < game_date
            ]
        )


        # ------------------------------------------
        # OPTIONAL ROLLING WINDOW
        # ------------------------------------------

        if days is not None:

            start_date = (
                game_date
                - pd.Timedelta(
                    days=days
                )
            )

            pitcher_history = (
                pitcher_history[
                    pitcher_history[
                        "game_date"
                    ] >= start_date
                ]
            )


        # No previous MLB data
        if len(pitcher_history) == 0:
            return {
                "k_pct": None,
                "bb_pct": None,
                "hr_pct": None,
                "woba_allowed": None
            }


        batters_faced = (
            pitcher_history[
                "batters_faced"
            ].sum()
        )

        strikeouts = (
            pitcher_history[
                "strikeouts"
            ].sum()
        )

        walks = (
            pitcher_history[
                "walks"
            ].sum()
        )

        home_runs = (
            pitcher_history[
                "home_runs"
            ].sum()
        )

        woba_sum = (
            pitcher_history[
                "woba_allowed_sum"
            ].sum()
        )

        woba_denom = (
            pitcher_history[
                "woba_denom_sum"
            ].sum()
        )


        if batters_faced == 0:

            k_pct = None
            bb_pct = None
            hr_pct = None

        else:

            k_pct = (
                strikeouts
                / batters_faced
            )

            bb_pct = (
                walks
                / batters_faced
            )

            hr_pct = (
                home_runs
                / batters_faced
            )


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
            "woba_allowed":
                woba_allowed
        }


    # ----------------------------------------------
    # BUILD FEATURES GAME BY GAME
    # ----------------------------------------------

    print()
    print(
        "Calculating pre-game starter features..."
    )

    feature_rows = []

    total_games = len(games)


    for index, game in games.iterrows():

        game_date = game["date"]


        home_id = game[
            "home_starter_id"
        ]

        away_id = game[
            "away_starter_id"
        ]


        # ------------------------------------------
        # SEASON-TO-DATE
        # ------------------------------------------

        home_season = (
            calculate_pitcher_stats(
                home_id,
                game_date
            )
        )

        away_season = (
            calculate_pitcher_stats(
                away_id,
                game_date
            )
        )


        # ------------------------------------------
        # LAST 30 DAYS
        # ------------------------------------------

        home_l30 = (
            calculate_pitcher_stats(
                home_id,
                game_date,
                days=30
            )
        )

        away_l30 = (
            calculate_pitcher_stats(
                away_id,
                game_date,
                days=30
            )
        )


        # ------------------------------------------
        # SAVE FEATURE ROW
        # ------------------------------------------

        feature_rows.append({

            "game_id":
                game["game_id"],

            # SEASON
            "home_sp_season_k_pct":
                home_season["k_pct"],

            "away_sp_season_k_pct":
                away_season["k_pct"],

            "home_sp_season_bb_pct":
                home_season["bb_pct"],

            "away_sp_season_bb_pct":
                away_season["bb_pct"],

            "home_sp_season_hr_pct":
                home_season["hr_pct"],

            "away_sp_season_hr_pct":
                away_season["hr_pct"],

            "home_sp_season_woba_allowed":
                home_season[
                    "woba_allowed"
                ],

            "away_sp_season_woba_allowed":
                away_season[
                    "woba_allowed"
                ],


            # LAST 30
            "home_sp_l30_k_pct":
                home_l30["k_pct"],

            "away_sp_l30_k_pct":
                away_l30["k_pct"],

            "home_sp_l30_bb_pct":
                home_l30["bb_pct"],

            "away_sp_l30_bb_pct":
                away_l30["bb_pct"],

            "home_sp_l30_hr_pct":
                home_l30["hr_pct"],

            "away_sp_l30_hr_pct":
                away_l30["hr_pct"],

            "home_sp_l30_woba_allowed":
                home_l30[
                    "woba_allowed"
                ],

            "away_sp_l30_woba_allowed":
                away_l30[
                    "woba_allowed"
                ]
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


    # ----------------------------------------------
    # CREATE FEATURE DATAFRAME
    # ----------------------------------------------

    pitcher_features = (
        pd.DataFrame(
            feature_rows
        )
    )


    # ----------------------------------------------
    # MERGE WITH EXISTING FEATURES
    # ----------------------------------------------

    games = games.merge(
        pitcher_features,
        on="game_id",
        how="left"
    )


    # ----------------------------------------------
    # CREATE DIFFERENTIAL FEATURES
    # ----------------------------------------------

    games[
        "sp_season_k_pct_diff"
    ] = (
        games[
            "home_sp_season_k_pct"
        ]
        -
        games[
            "away_sp_season_k_pct"
        ]
    )


    # Lower BB% is better,
    # so away - home makes positive = home advantage
    games[
        "sp_season_bb_pct_diff"
    ] = (
        games[
            "away_sp_season_bb_pct"
        ]
        -
        games[
            "home_sp_season_bb_pct"
        ]
    )


    games[
        "sp_season_woba_allowed_diff"
    ] = (
        games[
            "away_sp_season_woba_allowed"
        ]
        -
        games[
            "home_sp_season_woba_allowed"
        ]
    )


    games[
        "sp_l30_k_pct_diff"
    ] = (
        games[
            "home_sp_l30_k_pct"
        ]
        -
        games[
            "away_sp_l30_k_pct"
        ]
    )


    games[
        "sp_l30_bb_pct_diff"
    ] = (
        games[
            "away_sp_l30_bb_pct"
        ]
        -
        games[
            "home_sp_l30_bb_pct"
        ]
    )


    games[
        "sp_l30_woba_allowed_diff"
    ] = (
        games[
            "away_sp_l30_woba_allowed"
        ]
        -
        games[
            "home_sp_l30_woba_allowed"
        ]
    )


    # ----------------------------------------------
    # SAVE
    # ----------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    output_path = (
        f"data/processed/"
        f"games_{year}_offense_pitching.csv"
    )

    games.to_csv(
        output_path,
        index=False
    )


    # ----------------------------------------------
    # OUTPUT CHECKS
    # ----------------------------------------------

    print()
    print(
        f"{year} PITCHING FEATURES COMPLETE"
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
        "Missing starter feature values:"
    )

    columns_to_check = [
        "home_sp_season_k_pct",
        "away_sp_season_k_pct",
        "home_sp_season_woba_allowed",
        "away_sp_season_woba_allowed",
        "home_sp_l30_k_pct",
        "away_sp_l30_k_pct"
    ]

    print(
        games[
            columns_to_check
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
                "away_starter_name",
                "home_starter_name",
                "away_sp_season_k_pct",
                "home_sp_season_k_pct",
                "away_sp_season_woba_allowed",
                "home_sp_season_woba_allowed",
                "sp_season_k_pct_diff",
                "sp_season_woba_allowed_diff",
                "home_win"
            ]
        ]
        .tail(10)
    )


# --------------------------------------------------
# BUILD ALL SEASONS
# --------------------------------------------------

for year in YEARS:

    build_pitching_features(
        year
    )


print()
print("=" * 60)
print(
    "ALL STARTING PITCHER FEATURES COMPLETE"
)
print("=" * 60)