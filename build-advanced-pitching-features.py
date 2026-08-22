import os
import pandas as pd
import numpy as np


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]

FASTBALL_TYPES = [
    "FF",  # four-seam
    "SI",  # sinker
    "FC"   # cutter
]


# --------------------------------------------------
# BUILD ADVANCED STARTER FEATURES FOR ONE SEASON
# --------------------------------------------------

def build_advanced_pitching_features(year):

    print()
    print("=" * 70)
    print(f"BUILDING ADVANCED STARTER FEATURES FOR {year}")
    print("=" * 70)

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    games_path = (
        f"data/processed/"
        f"games_{year}_full_features.csv"
    )

    pitch_path = (
        f"data/raw/pitching/"
        f"statcast_pitches_{year}.csv"
    )

    print("Loading data...")

    games = pd.read_csv(games_path)
    pitches = pd.read_csv(pitch_path)

    games["date"] = pd.to_datetime(
        games["date"]
    )

    pitches["game_date"] = pd.to_datetime(
        pitches["game_date"]
    )

    print("Games:", len(games))
    print("Pitch rows:", len(pitches))


    # --------------------------------------------------
    # BASIC PITCH FLAGS
    # --------------------------------------------------

    # Whiff = batter swung and missed
    pitches["is_whiff"] = (
        pitches["description"]
        .isin([
            "swinging_strike",
            "swinging_strike_blocked",
            "missed_bunt"
        ])
        .astype(int)
    )

    # Swing = anything where batter offered at the pitch
    pitches["is_swing"] = (
        pitches["description"]
        .isin([
            "swinging_strike",
            "swinging_strike_blocked",
            "foul",
            "foul_tip",
            "hit_into_play",
            "hit_into_play_no_out",
            "hit_into_play_score",
            "missed_bunt",
            "foul_bunt"
        ])
        .astype(int)
    )

    # Fastball-family pitch
    pitches["is_fastball"] = (
        pitches["pitch_type"]
        .isin(FASTBALL_TYPES)
        .astype(int)
    )


    # --------------------------------------------------
    # CREATE ONE ROW PER PITCHER / GAME
    # --------------------------------------------------

    print()
    print("Building pitcher-game summaries...")

    pitcher_games = (
        pitches
        .groupby(
            [
                "pitcher",
                "game_pk",
                "game_date"
            ]
        )
        .agg(
            pitch_count=(
                "pitcher",
                "size"
            ),

            swings=(
                "is_swing",
                "sum"
            ),

            whiffs=(
                "is_whiff",
                "sum"
            ),

            fastballs=(
                "is_fastball",
                "sum"
            ),

            avg_fastball_velocity=(
                "release_speed",
                lambda x:
                    x[
                        pitches.loc[
                            x.index,
                            "is_fastball"
                        ] == 1
                    ].mean()
            ),

            xwoba_sum=(
                "estimated_woba_using_speedangle",
                "sum"
            ),

            xwoba_count=(
                "estimated_woba_using_speedangle",
                "count"
            )
        )
        .reset_index()
    )

    print(
        "Pitcher-game rows:",
        len(pitcher_games)
    )


    # --------------------------------------------------
    # ADD STARTER FLAG
    # --------------------------------------------------

    starter_rows = []

    for _, game in games.iterrows():

        starter_rows.append({
            "game_id":
                game["game_id"],

            "pitcher":
                game["home_starter_id"]
        })

        starter_rows.append({
            "game_id":
                game["game_id"],

            "pitcher":
                game["away_starter_id"]
        })

    starters = pd.DataFrame(
        starter_rows
    )

    starters["pitcher"] = pd.to_numeric(
        starters["pitcher"],
        errors="coerce"
    )

    pitcher_games["pitcher"] = pd.to_numeric(
        pitcher_games["pitcher"],
        errors="coerce"
    )

    pitcher_games = pitcher_games.merge(
        starters,
        left_on=[
            "game_pk",
            "pitcher"
        ],
        right_on=[
            "game_id",
            "pitcher"
        ],
        how="inner"
    )


    print(
        "Starter-game rows:",
        len(pitcher_games)
    )


    # --------------------------------------------------
    # DERIVED GAME-LEVEL METRICS
    # --------------------------------------------------

    pitcher_games["whiff_rate"] = np.where(
        pitcher_games["swings"] > 0,
        pitcher_games["whiffs"]
        / pitcher_games["swings"],
        np.nan
    )

    pitcher_games["xwoba_allowed"] = np.where(
        pitcher_games["xwoba_count"] > 0,
        pitcher_games["xwoba_sum"]
        / pitcher_games["xwoba_count"],
        np.nan
    )


    # --------------------------------------------------
    # HISTORICAL STARTER STATS FUNCTION
    # --------------------------------------------------

    def starter_history(
        pitcher_id,
        game_date,
        current_game_id
    ):

        history = pitcher_games[
            pitcher_games["pitcher"]
            == pitcher_id
        ].copy()

        # Strict leakage protection
        history = history[
            (
                history["game_date"]
                < game_date
            )
            |
            (
                (
                    history["game_date"]
                    == game_date
                )
                &
                (
                    history["game_pk"]
                    != current_game_id
                )
            )
        ]

        history = history.sort_values(
            "game_date"
        )

        if len(history) == 0:

            return {
                "days_rest": None,
                "prev_pitch_count": None,
                "season_avg_pitch_count": None,
                "l30_avg_pitch_count": None,
                "season_fastball_velocity": None,
                "l30_fastball_velocity": None,
                "season_whiff_rate": None,
                "l30_whiff_rate": None,
                "season_xwoba_allowed": None,
                "l30_xwoba_allowed": None
            }


        # --------------------------------------------------
        # DAYS REST
        # --------------------------------------------------

        previous_start_date = (
            history[
                "game_date"
            ].max()
        )

        days_rest = (
            game_date
            - previous_start_date
        ).days


        # --------------------------------------------------
        # PREVIOUS START
        # --------------------------------------------------

        previous_start = (
            history
            .sort_values(
                "game_date"
            )
            .iloc[-1]
        )

        prev_pitch_count = (
            previous_start[
                "pitch_count"
            ]
        )


        # --------------------------------------------------
        # SEASON-TO-DATE
        # --------------------------------------------------

        season_avg_pitch_count = (
            history[
                "pitch_count"
            ].mean()
        )

        season_fastball_velocity = (
            history[
                "avg_fastball_velocity"
            ].mean()
        )


        total_swings = (
            history[
                "swings"
            ].sum()
        )

        total_whiffs = (
            history[
                "whiffs"
            ].sum()
        )

        season_whiff_rate = (
            total_whiffs
            / total_swings
            if total_swings > 0
            else None
        )


        season_xwoba_numerator = (
            history[
                "xwoba_sum"
            ].sum()
        )

        season_xwoba_denominator = (
            history[
                "xwoba_count"
            ].sum()
        )

        season_xwoba_allowed = (
            season_xwoba_numerator
            / season_xwoba_denominator
            if season_xwoba_denominator > 0
            else None
        )


        # --------------------------------------------------
        # LAST 30 DAYS
        # --------------------------------------------------

        l30_start = (
            game_date
            - pd.Timedelta(
                days=30
            )
        )

        l30 = history[
            history["game_date"]
            >= l30_start
        ]

        if len(l30) == 0:

            l30_avg_pitch_count = None
            l30_fastball_velocity = None
            l30_whiff_rate = None
            l30_xwoba_allowed = None

        else:

            l30_avg_pitch_count = (
                l30[
                    "pitch_count"
                ].mean()
            )

            l30_fastball_velocity = (
                l30[
                    "avg_fastball_velocity"
                ].mean()
            )


            l30_swings = (
                l30[
                    "swings"
                ].sum()
            )

            l30_whiffs = (
                l30[
                    "whiffs"
                ].sum()
            )

            l30_whiff_rate = (
                l30_whiffs
                / l30_swings
                if l30_swings > 0
                else None
            )


            l30_xwoba_num = (
                l30[
                    "xwoba_sum"
                ].sum()
            )

            l30_xwoba_den = (
                l30[
                    "xwoba_count"
                ].sum()
            )

            l30_xwoba_allowed = (
                l30_xwoba_num
                / l30_xwoba_den
                if l30_xwoba_den > 0
                else None
            )


        return {

            "days_rest":
                days_rest,

            "prev_pitch_count":
                prev_pitch_count,

            "season_avg_pitch_count":
                season_avg_pitch_count,

            "l30_avg_pitch_count":
                l30_avg_pitch_count,

            "season_fastball_velocity":
                season_fastball_velocity,

            "l30_fastball_velocity":
                l30_fastball_velocity,

            "season_whiff_rate":
                season_whiff_rate,

            "l30_whiff_rate":
                l30_whiff_rate,

            "season_xwoba_allowed":
                season_xwoba_allowed,

            "l30_xwoba_allowed":
                l30_xwoba_allowed
        }


    # --------------------------------------------------
    # BUILD FEATURES FOR EVERY GAME
    # --------------------------------------------------

    print()
    print(
        "Calculating advanced starter features..."
    )

    rows = []

    total_games = len(games)

    for index, game in games.iterrows():

        date = game["date"]
        game_id = game["game_id"]

        home_id = game[
            "home_starter_id"
        ]

        away_id = game[
            "away_starter_id"
        ]


        home = starter_history(
            home_id,
            date,
            game_id
        )

        away = starter_history(
            away_id,
            date,
            game_id
        )


        rows.append({

            "game_id":
                game_id,


            # --------------------------------------
            # HOME
            # --------------------------------------

            "home_sp_days_rest":
                home["days_rest"],

            "home_sp_prev_pitch_count":
                home["prev_pitch_count"],

            "home_sp_season_avg_pitch_count":
                home[
                    "season_avg_pitch_count"
                ],

            "home_sp_l30_avg_pitch_count":
                home[
                    "l30_avg_pitch_count"
                ],

            "home_sp_season_fastball_velocity":
                home[
                    "season_fastball_velocity"
                ],

            "home_sp_l30_fastball_velocity":
                home[
                    "l30_fastball_velocity"
                ],

            "home_sp_season_whiff_rate":
                home[
                    "season_whiff_rate"
                ],

            "home_sp_l30_whiff_rate":
                home[
                    "l30_whiff_rate"
                ],

            "home_sp_season_xwoba_allowed":
                home[
                    "season_xwoba_allowed"
                ],

            "home_sp_l30_xwoba_allowed":
                home[
                    "l30_xwoba_allowed"
                ],


            # --------------------------------------
            # AWAY
            # --------------------------------------

            "away_sp_days_rest":
                away["days_rest"],

            "away_sp_prev_pitch_count":
                away["prev_pitch_count"],

            "away_sp_season_avg_pitch_count":
                away[
                    "season_avg_pitch_count"
                ],

            "away_sp_l30_avg_pitch_count":
                away[
                    "l30_avg_pitch_count"
                ],

            "away_sp_season_fastball_velocity":
                away[
                    "season_fastball_velocity"
                ],

            "away_sp_l30_fastball_velocity":
                away[
                    "l30_fastball_velocity"
                ],

            "away_sp_season_whiff_rate":
                away[
                    "season_whiff_rate"
                ],

            "away_sp_l30_whiff_rate":
                away[
                    "l30_whiff_rate"
                ],

            "away_sp_season_xwoba_allowed":
                away[
                    "season_xwoba_allowed"
                ],

            "away_sp_l30_xwoba_allowed":
                away[
                    "l30_xwoba_allowed"
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


    # --------------------------------------------------
    # MERGE
    # --------------------------------------------------

    advanced = pd.DataFrame(
        rows
    )

    games = games.merge(
        advanced,
        on="game_id",
        how="left"
    )


    # --------------------------------------------------
    # DIFFERENCE FEATURES
    # --------------------------------------------------

    # Positive = home starter has more rest
    games[
        "sp_days_rest_diff"
    ] = (
        games[
            "home_sp_days_rest"
        ]
        -
        games[
            "away_sp_days_rest"
        ]
    )


    # Positive = home starter threw more pitches last start
    games[
        "sp_prev_pitch_count_diff"
    ] = (
        games[
            "home_sp_prev_pitch_count"
        ]
        -
        games[
            "away_sp_prev_pitch_count"
        ]
    )


    # Positive = home velocity advantage
    games[
        "sp_season_velocity_diff"
    ] = (
        games[
            "home_sp_season_fastball_velocity"
        ]
        -
        games[
            "away_sp_season_fastball_velocity"
        ]
    )


    games[
        "sp_l30_velocity_diff"
    ] = (
        games[
            "home_sp_l30_fastball_velocity"
        ]
        -
        games[
            "away_sp_l30_fastball_velocity"
        ]
    )


    # Higher whiff rate = better
    games[
        "sp_season_whiff_diff"
    ] = (
        games[
            "home_sp_season_whiff_rate"
        ]
        -
        games[
            "away_sp_season_whiff_rate"
        ]
    )


    games[
        "sp_l30_whiff_diff"
    ] = (
        games[
            "home_sp_l30_whiff_rate"
        ]
        -
        games[
            "away_sp_l30_whiff_rate"
        ]
    )


    # Lower xwOBA allowed = better,
    # so away - home gives positive home advantage.
    games[
        "sp_season_xwoba_allowed_diff"
    ] = (
        games[
            "away_sp_season_xwoba_allowed"
        ]
        -
        games[
            "home_sp_season_xwoba_allowed"
        ]
    )


    games[
        "sp_l30_xwoba_allowed_diff"
    ] = (
        games[
            "away_sp_l30_xwoba_allowed"
        ]
        -
        games[
            "home_sp_l30_xwoba_allowed"
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
        f"games_{year}_advanced_features.csv"
    )

    games.to_csv(
        output_path,
        index=False
    )


    # --------------------------------------------------
    # CHECKS
    # --------------------------------------------------

    print()
    print(
        f"{year} ADVANCED STARTER FEATURES COMPLETE"
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


    check_columns = [

        "home_sp_days_rest",
        "away_sp_days_rest",

        "home_sp_prev_pitch_count",
        "away_sp_prev_pitch_count",

        "home_sp_season_fastball_velocity",
        "away_sp_season_fastball_velocity",

        "home_sp_season_whiff_rate",
        "away_sp_season_whiff_rate",

        "home_sp_season_xwoba_allowed",
        "away_sp_season_xwoba_allowed"
    ]


    print()
    print(
        "Missing advanced values:"
    )

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

    build_advanced_pitching_features(
        year
    )


print()
print("=" * 70)
print(
    "ALL ADVANCED STARTER FEATURES COMPLETE"
)
print("=" * 70)