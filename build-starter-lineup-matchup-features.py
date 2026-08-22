import os
import numpy as np
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]

ORDER_WEIGHTS = {
    1: 1.00,
    2: 0.97,
    3: 0.94,
    4: 0.91,
    5: 0.88,
    6: 0.85,
    7: 0.82,
    8: 0.79,
    9: 0.76
}

HANDS = ["L", "R"]
WINDOWS = ["season", "l30"]
METRICS = [
    "xwoba_allowed",
    "k_pct",
    "bb_pct",
    "whiff_pct"
]

SWING_DESCRIPTIONS = [
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "missed_bunt",
    "foul_bunt"
]

WHIFF_DESCRIPTIONS = [
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt"
]

STRIKEOUT_EVENTS = [
    "strikeout",
    "strikeout_double_play"
]

WALK_EVENTS = [
    "walk",
    "intent_walk"
]


# --------------------------------------------------
# LOAD AND PREPARE FULL PITCH DATA
# --------------------------------------------------

def load_pitch_data(year):

    path = (
        f"data/raw/pitching/"
        f"statcast_pitches_{year}.csv"
    )

    pitches = pd.read_csv(path)

    required = [
        "game_date",
        "game_pk",
        "pitcher",
        "batter",
        "stand",
        "description",
        "events",
        "estimated_woba_using_speedangle"
    ]

    missing = [
        column
        for column in required
        if column not in pitches.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Statcast columns for {year}: {missing}"
        )

    pitches["game_date"] = pd.to_datetime(
        pitches["game_date"]
    )

    pitches["pitcher"] = pd.to_numeric(
        pitches["pitcher"],
        errors="coerce"
    )

    pitches["batter"] = pd.to_numeric(
        pitches["batter"],
        errors="coerce"
    )

    pitches["is_swing"] = pitches[
        "description"
    ].isin(SWING_DESCRIPTIONS).astype(int)

    pitches["is_whiff"] = pitches[
        "description"
    ].isin(WHIFF_DESCRIPTIONS).astype(int)

    pitches["is_pa"] = pitches[
        "events"
    ].notna().astype(int)

    pitches["is_k"] = pitches[
        "events"
    ].isin(STRIKEOUT_EVENTS).astype(int)

    pitches["is_bb"] = pitches[
        "events"
    ].isin(WALK_EVENTS).astype(int)

    return pitches


# --------------------------------------------------
# BUILD CUMULATIVE PITCHER/HAND HISTORIES
# --------------------------------------------------

def build_pitcher_hand_histories(pitches):

    valid = pitches[
        pitches["stand"].isin(HANDS)
        & pitches["pitcher"].notna()
    ].copy()

    valid["xwoba_value"] = valid[
        "estimated_woba_using_speedangle"
    ].fillna(0)

    valid["xwoba_count"] = valid[
        "estimated_woba_using_speedangle"
    ].notna().astype(int)

    daily = (
        valid
        .groupby(
            ["pitcher", "stand", "game_date"],
            as_index=False
        )
        .agg(
            xwoba_sum=("xwoba_value", "sum"),
            xwoba_count=("xwoba_count", "sum"),
            plate_appearances=("is_pa", "sum"),
            strikeouts=("is_k", "sum"),
            walks=("is_bb", "sum"),
            swings=("is_swing", "sum"),
            whiffs=("is_whiff", "sum")
        )
        .sort_values(["pitcher", "stand", "game_date"])
    )

    total_columns = [
        "xwoba_sum",
        "xwoba_count",
        "plate_appearances",
        "strikeouts",
        "walks",
        "swings",
        "whiffs"
    ]

    histories = {}

    for (pitcher_id, hand), rows in daily.groupby(
        ["pitcher", "stand"]
    ):

        histories[(int(pitcher_id), hand)] = {
            "dates": rows["game_date"].to_numpy(
                dtype="datetime64[ns]"
            ),
            "cumulative": {
                column: rows[column].cumsum().to_numpy()
                for column in total_columns
            }
        }

    return histories


# --------------------------------------------------
# GET PRE-GAME SPLIT METRICS
# --------------------------------------------------

def metrics_before_game(
    pitcher_id,
    batter_hand,
    game_date,
    histories,
    days=None
):

    empty = {
        metric: np.nan
        for metric in METRICS
    }

    if pd.isna(pitcher_id):
        return empty

    history = histories.get(
        (int(pitcher_id), batter_hand)
    )

    if history is None:
        return empty

    game_date = np.datetime64(game_date, "ns")
    dates = history["dates"]

    # Strictly exclude the current date. This also conservatively
    # excludes earlier doubleheader games on the same date.
    right = np.searchsorted(
        dates,
        game_date,
        side="left"
    )

    if right == 0:
        return empty

    left = 0

    if days is not None:
        start_date = (
            game_date
            - np.timedelta64(days, "D")
        )
        left = np.searchsorted(
            dates,
            start_date,
            side="left"
        )

    totals = {}

    for column, cumulative in history[
        "cumulative"
    ].items():

        total = cumulative[right - 1]

        if left > 0:
            total -= cumulative[left - 1]

        totals[column] = total

    xwoba_allowed = np.nan
    k_pct = np.nan
    bb_pct = np.nan
    whiff_pct = np.nan

    if totals["xwoba_count"] > 0:
        xwoba_allowed = (
            totals["xwoba_sum"]
            / totals["xwoba_count"]
        )

    if totals["plate_appearances"] > 0:
        k_pct = (
            totals["strikeouts"]
            / totals["plate_appearances"]
        )
        bb_pct = (
            totals["walks"]
            / totals["plate_appearances"]
        )

    if totals["swings"] > 0:
        whiff_pct = (
            totals["whiffs"]
            / totals["swings"]
        )

    return {
        "xwoba_allowed": xwoba_allowed,
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "whiff_pct": whiff_pct
    }


# --------------------------------------------------
# MAP LINEUP HITTERS TO THEIR SIDE VS THE STARTER
# --------------------------------------------------

def build_game_stand_lookup(pitches, games):

    starter_rows = []

    for _, game in games.iterrows():

        if pd.notna(game["home_starter_id"]):
            starter_rows.append({
                "game_pk": int(game["game_id"]),
                "pitcher": int(game["home_starter_id"])
            })

        if pd.notna(game["away_starter_id"]):
            starter_rows.append({
                "game_pk": int(game["game_id"]),
                "pitcher": int(game["away_starter_id"])
            })

    starters = pd.DataFrame(starter_rows).drop_duplicates()

    starter_pitches = pitches.merge(
        starters,
        on=["game_pk", "pitcher"],
        how="inner"
    )

    starter_pitches = starter_pitches[
        starter_pitches["stand"].isin(HANDS)
        & starter_pitches["batter"].notna()
    ]

    stand_counts = (
        starter_pitches
        .groupby(
            ["game_pk", "pitcher", "batter", "stand"]
        )
        .size()
        .reset_index(name="pitch_count")
        .sort_values(
            [
                "game_pk",
                "pitcher",
                "batter",
                "pitch_count",
                "stand"
            ],
            ascending=[True, True, True, False, True]
        )
        .drop_duplicates(
            ["game_pk", "pitcher", "batter"]
        )
    )

    return {
        (
            int(row.game_pk),
            int(row.pitcher),
            int(row.batter)
        ): row.stand
        for row in stand_counts.itertuples()
    }


# --------------------------------------------------
# WEIGHT SPLIT METRICS FOR THE ACTUAL LINEUP
# --------------------------------------------------

def matchup_metrics(
    lineup,
    split_metrics
):

    result = {
        metric: np.nan
        for metric in METRICS
    }

    if (
        len(lineup) != 9
        or lineup["stand"].isna().any()
    ):
        return result

    weights = lineup["batting_order"].map(
        ORDER_WEIGHTS
    ).to_numpy()

    for metric in METRICS:

        values = lineup["stand"].map({
            hand: split_metrics[hand][metric]
            for hand in HANDS
        })

        if values.isna().any():
            continue

        result[metric] = np.average(
            values.to_numpy(),
            weights=weights
        )

    return result


# --------------------------------------------------
# BUILD ONE SEASON
# --------------------------------------------------

def build_season(year):

    print()
    print("=" * 76)
    print(f"BUILDING STARTER/LINEUP MATCHUP FEATURES FOR {year}")
    print("=" * 76)

    games_path = (
        f"data/processed/"
        f"games_{year}_lineup_features.csv"
    )

    lineups_path = (
        f"data/raw/lineups/"
        f"starting_lineups_{year}.csv"
    )

    games = pd.read_csv(games_path)
    lineups = pd.read_csv(lineups_path)
    pitches = load_pitch_data(year)

    games["date"] = pd.to_datetime(games["date"])

    for column in ["home_starter_id", "away_starter_id"]:
        games[column] = pd.to_numeric(
            games[column],
            errors="coerce"
        )

    lineups["player_id"] = pd.to_numeric(
        lineups["player_id"],
        errors="coerce"
    )

    histories = build_pitcher_hand_histories(
        pitches
    )

    stand_lookup = build_game_stand_lookup(
        pitches,
        games
    )

    lineups_by_game_side = {
        (int(game_id), team_side): rows.sort_values(
            "batting_order"
        ).copy()
        for (game_id, team_side), rows in lineups.groupby(
            ["game_id", "team_side"]
        )
    }

    feature_rows = []
    matched_games = 0
    missing_hitter_handedness = 0

    for index, game in games.iterrows():

        game_id = int(game["game_id"])
        game_date = game["date"]

        home_lineup = lineups_by_game_side.get(
            (game_id, "home")
        )
        away_lineup = lineups_by_game_side.get(
            (game_id, "away")
        )

        if (
            home_lineup is not None
            and away_lineup is not None
            and len(home_lineup) == 9
            and len(away_lineup) == 9
        ):
            matched_games += 1

        row = {"game_id": game_id}

        side_inputs = {
            "home": (
                game["home_starter_id"],
                away_lineup
            ),
            "away": (
                game["away_starter_id"],
                home_lineup
            )
        }

        for side, (starter_id, opposing_lineup) in (
            side_inputs.items()
        ):

            if opposing_lineup is None:
                opposing_lineup = pd.DataFrame(
                    columns=[
                        "batting_order",
                        "player_id"
                    ]
                )
            else:
                opposing_lineup = opposing_lineup.copy()

            if pd.isna(starter_id):
                opposing_lineup["stand"] = np.nan
            else:
                opposing_lineup["stand"] = (
                    opposing_lineup["player_id"].map(
                        lambda player_id: stand_lookup.get(
                            (
                                game_id,
                                int(starter_id),
                                int(player_id)
                            )
                        )
                        if pd.notna(player_id)
                        else None
                    )
                )

            missing_hitter_handedness += (
                opposing_lineup["stand"].isna().sum()
            )

            lineup_weights = opposing_lineup[
                "batting_order"
            ].map(ORDER_WEIGHTS)

            total_weight = lineup_weights.sum()

            for hand in HANDS:
                hand_name = "lhb" if hand == "L" else "rhb"
                hand_weight = lineup_weights[
                    opposing_lineup["stand"] == hand
                ].sum()
                row[
                    f"{side}_sp_opposing_lineup_{hand_name}_weight"
                ] = (
                    hand_weight / total_weight
                    if total_weight > 0
                    else np.nan
                )

            for window in WINDOWS:

                days = 30 if window == "l30" else None

                split_metrics = {
                    hand: metrics_before_game(
                        starter_id,
                        hand,
                        game_date,
                        histories,
                        days=days
                    )
                    for hand in HANDS
                }

                for hand in HANDS:

                    hand_name = (
                        "lhb" if hand == "L" else "rhb"
                    )

                    for metric in METRICS:
                        row[
                            f"{side}_sp_{window}_{metric}_vs_{hand_name}"
                        ] = split_metrics[hand][metric]

                weighted = matchup_metrics(
                    opposing_lineup,
                    split_metrics
                )

                for metric in METRICS:
                    row[
                        f"{side}_sp_matchup_{window}_{metric}"
                    ] = weighted[metric]

        feature_rows.append(row)

        if (index + 1) % 250 == 0:
            print(
                f"Processed {index + 1} / {len(games)}"
            )

    features = pd.DataFrame(feature_rows)

    # Positive differentials consistently represent home advantage.
    higher_is_better = {"k_pct", "whiff_pct"}

    for window in WINDOWS:
        for metric in METRICS:

            home_column = (
                f"home_sp_matchup_{window}_{metric}"
            )
            away_column = (
                f"away_sp_matchup_{window}_{metric}"
            )
            diff_column = (
                f"sp_matchup_{window}_{metric}_diff"
            )

            if metric in higher_is_better:
                features[diff_column] = (
                    features[home_column]
                    - features[away_column]
                )
            else:
                features[diff_column] = (
                    features[away_column]
                    - features[home_column]
                )

    output = games.merge(
        features,
        on="game_id",
        how="left",
        validate="one_to_one"
    )

    output_path = (
        f"data/processed/"
        f"games_{year}_starter_lineup_matchup_features.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    matchup_columns = [
        column
        for column in features.columns
        if "sp_matchup_" in column
    ]

    missing_starter_ids = (
        games[
            ["home_starter_id", "away_starter_id"]
        ].isna().sum().sum()
    )

    print()
    print(f"{year} STARTER/LINEUP MATCHUPS COMPLETE")
    print("Games processed:", len(output))
    print("Games matched to actual lineups:", matched_games)
    print("Missing starter IDs:", missing_starter_ids)
    print("Missing hitter handedness:", missing_hitter_handedness)
    print("Missing matchup features:")
    print(output[matchup_columns].isna().sum())
    print("Duplicate game IDs:", output["game_id"].duplicated().sum())
    print("Saved to:", os.path.abspath(output_path))


# --------------------------------------------------
# BUILD ALL SEASONS
# --------------------------------------------------

for year in YEARS:
    build_season(year)


print()
print("=" * 76)
print("ALL STARTER/LINEUP MATCHUP FEATURE DATASETS COMPLETE")
print("=" * 76)
