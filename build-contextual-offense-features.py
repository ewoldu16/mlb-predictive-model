import argparse
import os

import numpy as np
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]
RECENT_DAYS = 30
MIN_RECENT_SPLIT_PA = 75
MIN_RECENT_COMBINED_PA = 50

STRIKEOUT_EVENTS = [
    "strikeout",
    "strikeout_double_play"
]

WALK_EVENTS = [
    "walk",
    "intent_walk"
]

TOTAL_COLUMNS = [
    "woba_value_sum",
    "woba_denom_sum",
    "plate_appearances",
    "strikeouts",
    "walks"
]

METRICS = ["woba", "k_pct", "bb_pct"]
CONTEXTS = ["venue", "hand", "combined"]
WINDOWS = ["season", "l30"]


# --------------------------------------------------
# LOAD KNOWN GAMES AND PREGAME STARTER CONTEXT
# --------------------------------------------------

def load_games(year):

    games = pd.read_csv(
        f"data/raw/games_{year}.csv"
    )

    starter_context = pd.read_csv(
        f"data/processed/"
        f"games_{year}_starter_lineup_matchup_features.csv",
        usecols=[
            "game_id",
            "home_starter_id",
            "away_starter_id",
            "home_starter_hand",
            "away_starter_hand"
        ]
    )

    games["date"] = pd.to_datetime(games["date"])
    games["game_id"] = pd.to_numeric(
        games["game_id"],
        errors="raise"
    ).astype("int64")

    if games["game_id"].duplicated().any():
        raise ValueError(
            f"Duplicate game IDs in the {year} game universe."
        )

    if starter_context["game_id"].duplicated().any():
        raise ValueError(
            f"Duplicate game IDs in {year} starter context."
        )

    missing_game_ids = (
        set(games["game_id"])
        - set(starter_context["game_id"])
    )

    extra_game_ids = (
        set(starter_context["game_id"])
        - set(games["game_id"])
    )

    if missing_game_ids or extra_game_ids:
        raise ValueError(
            f"{year} starter-context game coverage mismatch. "
            f"Missing: {len(missing_game_ids)}; "
            f"extra: {len(extra_game_ids)}"
        )

    games = games.merge(
        starter_context,
        on="game_id",
        how="left",
        validate="one_to_one"
    )

    return games


# --------------------------------------------------
# LOAD REGULAR-SEASON PLATE APPEARANCES
# --------------------------------------------------

def load_plate_appearances(year, games):

    pitches = pd.read_csv(
        f"data/raw/statcast_enriched_{year}.csv",
        usecols=[
            "game_date",
            "game_pk",
            "inning_topbot",
            "pitcher",
            "p_throws",
            "events",
            "woba_value",
            "woba_denom"
        ]
    )

    pitches["game_date"] = pd.to_datetime(
        pitches["game_date"]
    )

    known_games = games[
        ["game_id", "home_team", "away_team"]
    ].rename(columns={"game_id": "game_pk"})

    pitches = pitches.merge(
        known_games,
        on="game_pk",
        how="inner",
        validate="many_to_one"
    )

    covered_games = set(
        pitches["game_pk"].astype("int64").unique()
    )
    # A pregame target row is metadata, not historical Statcast source data.
    completed = games[games["away_score"].notna() & games["home_score"].notna()]
    expected_games = set(completed["game_id"])

    if covered_games != expected_games:
        raise ValueError(
            f"{year} regular-season Statcast coverage mismatch. "
            f"Missing: {len(expected_games - covered_games)}; "
            f"extra: {len(covered_games - expected_games)}"
        )

    plate_appearances = pitches[
        pitches["events"].notna()
    ].copy()

    plate_appearances["batting_team"] = np.where(
        plate_appearances["inning_topbot"] == "Top",
        plate_appearances["away_team"],
        plate_appearances["home_team"]
    )

    plate_appearances["venue"] = np.where(
        plate_appearances["inning_topbot"] == "Top",
        "away",
        "home"
    )

    plate_appearances["is_k"] = plate_appearances[
        "events"
    ].isin(STRIKEOUT_EVENTS).astype(int)

    plate_appearances["is_bb"] = plate_appearances[
        "events"
    ].isin(WALK_EVENTS).astype(int)

    return plate_appearances, len(pitches), len(covered_games)


# --------------------------------------------------
# DAILY CONTEXT TABLES
# --------------------------------------------------

def aggregate_daily(data, group_columns):

    return (
        data
        .groupby(
            group_columns + ["game_date"],
            as_index=False
        )
        .agg(
            woba_value_sum=("woba_value", "sum"),
            woba_denom_sum=("woba_denom", "sum"),
            plate_appearances=("events", "count"),
            strikeouts=("is_k", "sum"),
            walks=("is_bb", "sum")
        )
        .sort_values(group_columns + ["game_date"])
    )


def build_daily_contexts(plate_appearances):

    valid_hand = plate_appearances[
        plate_appearances["p_throws"].isin(["L", "R"])
    ].copy()

    return {
        "venue": aggregate_daily(
            plate_appearances,
            ["batting_team", "venue"]
        ),
        "hand": aggregate_daily(
            valid_hand,
            ["batting_team", "p_throws"]
        ),
        "combined": aggregate_daily(
            valid_hand,
            ["batting_team", "venue", "p_throws"]
        )
    }


# --------------------------------------------------
# CUMULATIVE CONTEXT HISTORIES
# --------------------------------------------------

def build_histories(daily, key_columns):

    histories = {}

    group_key = (
        key_columns[0]
        if len(key_columns) == 1
        else key_columns
    )

    for key, rows in daily.groupby(group_key):

        if not isinstance(key, tuple):
            key = (key,)

        histories[key] = {
            "dates": rows["game_date"].to_numpy(
                dtype="datetime64[ns]"
            ),
            "cumulative": {
                column: rows[column].cumsum().to_numpy()
                for column in TOTAL_COLUMNS
            }
        }

    return histories


def build_all_histories(daily_contexts):

    return {
        "venue": build_histories(
            daily_contexts["venue"],
            ["batting_team", "venue"]
        ),
        "hand": build_histories(
            daily_contexts["hand"],
            ["batting_team", "p_throws"]
        ),
        "combined": build_histories(
            daily_contexts["combined"],
            ["batting_team", "venue", "p_throws"]
        )
    }


# --------------------------------------------------
# STRICTLY PREGAME LOOKUP
# --------------------------------------------------

def totals_before_date(
    history,
    game_date,
    days=None
):

    if history is None:
        return None

    dates = history["dates"]
    game_date = np.datetime64(game_date, "ns")

    # Strictly exclude the target date and all current-game PAs.
    right = np.searchsorted(
        dates,
        game_date,
        side="left"
    )

    if right == 0:
        return None

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

        value = cumulative[right - 1]

        if left > 0:
            value -= cumulative[left - 1]

        totals[column] = value

    return totals


def context_rates(totals, minimum_pa=1):

    empty = {
        "woba": np.nan,
        "k_pct": np.nan,
        "bb_pct": np.nan,
        "pa": np.nan,
        "woba_denom": np.nan
    }

    if totals is None:
        return empty

    result = empty.copy()
    result["pa"] = totals["plate_appearances"]
    result["woba_denom"] = totals["woba_denom_sum"]

    if totals["plate_appearances"] < minimum_pa:
        return result

    if totals["woba_denom_sum"] > 0:
        result["woba"] = (
            totals["woba_value_sum"]
            / totals["woba_denom_sum"]
        )

    result["k_pct"] = (
        totals["strikeouts"]
        / totals["plate_appearances"]
    )

    result["bb_pct"] = (
        totals["walks"]
        / totals["plate_appearances"]
    )

    return result


# --------------------------------------------------
# BUILD ONE SEASON
# --------------------------------------------------

def build_season(year):

    print()
    print("=" * 76)
    print(f"BUILDING V8 CONTEXTUAL OFFENSE - {year}")
    print("=" * 76)

    games = load_games(year)

    plate_appearances, pitch_rows, covered_games = (
        load_plate_appearances(year, games)
    )

    daily_contexts = build_daily_contexts(
        plate_appearances
    )

    histories = build_all_histories(daily_contexts)
    feature_rows = []

    for index, game in games.iterrows():

        game_id = int(game["game_id"])

        row = {"game_id": game_id}

        side_context = {
            "home": {
                "team": game["home_team"],
                "venue": "home",
                "starter_hand": game["away_starter_hand"]
            },
            "away": {
                "team": game["away_team"],
                "venue": "away",
                "starter_hand": game["home_starter_hand"]
            }
        }

        for side, values in side_context.items():

            team = values["team"]
            venue = values["venue"]
            starter_hand = values["starter_hand"]

            context_keys = {
                "venue": (team, venue),
                "hand": (
                    (team, starter_hand)
                    if starter_hand in ["L", "R"]
                    else None
                ),
                "combined": (
                    (team, venue, starter_hand)
                    if starter_hand in ["L", "R"]
                    else None
                )
            }

            for context, key in context_keys.items():

                history = (
                    histories[context].get(key)
                    if key is not None
                    else None
                )

                for window in WINDOWS:

                    recent = window == "l30"

                    totals = totals_before_date(
                        history,
                        game["date"],
                        days=(RECENT_DAYS if recent else None)
                    )

                    minimum_pa = 1

                    if recent:
                        minimum_pa = (
                            MIN_RECENT_COMBINED_PA
                            if context == "combined"
                            else MIN_RECENT_SPLIT_PA
                        )

                    rates = context_rates(
                        totals,
                        minimum_pa=minimum_pa
                    )

                    prefix = (
                        f"{side}_ctx_{context}_{window}"
                    )

                    for metric in METRICS:
                        row[f"{prefix}_{metric}"] = rates[
                            metric
                        ]

                    row[f"{prefix}_pa"] = rates["pa"]
                    row[
                        f"{prefix}_woba_denom"
                    ] = rates["woba_denom"]

        feature_rows.append(row)

        if (index + 1) % 250 == 0:
            print(
                f"Processed {index + 1} / {len(games)}"
            )

    features = pd.DataFrame(feature_rows)

    for context in CONTEXTS:
        for window in WINDOWS:
            for metric in METRICS:

                features[
                    f"ctx_{context}_{window}_{metric}_diff"
                ] = (
                    features[
                        f"home_ctx_{context}_{window}_{metric}"
                    ]
                    - features[
                        f"away_ctx_{context}_{window}_{metric}"
                    ]
                )

    if len(features) != len(games):
        raise ValueError(
            f"{year} output row count changed."
        )

    if features["game_id"].duplicated().any():
        raise ValueError(
            f"{year} output contains duplicate game IDs."
        )

    numeric = features.select_dtypes(include=[np.number])
    infinite_count = int(
        np.isinf(numeric.to_numpy()).sum()
    )

    if infinite_count:
        raise ValueError(
            f"{year} output contains {infinite_count} "
            "infinite values."
        )

    rate_columns = [
        column
        for column in features.columns
        if any(
            column.endswith(suffix)
            for suffix in ["_woba", "_k_pct", "_bb_pct"]
        )
    ]

    invalid_rate_columns = [
        column
        for column in rate_columns
        if (
            features[column].dropna().lt(0).any()
            or features[column].dropna().gt(1).any()
        )
    ]

    if invalid_rate_columns:
        raise ValueError(
            f"{year} rate columns outside [0, 1]: "
            f"{invalid_rate_columns}"
        )

    os.makedirs("data/processed", exist_ok=True)

    output_path = (
        f"data/processed/"
        f"features_v8_contextual_offense_{year}.csv"
    )

    features.to_csv(output_path, index=False)

    missing_home_starter_ids = int(
        games["home_starter_id"].isna().sum()
    )
    missing_away_starter_ids = int(
        games["away_starter_id"].isna().sum()
    )
    missing_home_starter_hands = int(
        games["home_starter_hand"].isna().sum()
    )
    missing_away_starter_hands = int(
        games["away_starter_hand"].isna().sum()
    )

    sample_columns = [
        column
        for column in features.columns
        if column.endswith("_pa")
        or column.endswith("_woba_denom")
    ]

    print()
    print(f"{year} V8 CONTEXTUAL OFFENSE COMPLETE")
    print("Expected games:", len(games))
    print("Output games:", len(features))
    print("Duplicate game IDs:", features["game_id"].duplicated().sum())
    print("Regular-season Statcast games:", covered_games)
    print("Regular-season pitch rows:", pitch_rows)
    print("Missing home starter IDs:", missing_home_starter_ids)
    print("Missing away starter IDs:", missing_away_starter_ids)
    print("Missing home starter hands:", missing_home_starter_hands)
    print("Missing away starter hands:", missing_away_starter_hands)
    print("Infinite values:", infinite_count)
    print()
    print("Feature missingness:")
    print(features.drop(columns="game_id").isna().sum())
    print()
    print("Feature coverage:")
    print(
        features.drop(columns="game_id")
        .notna()
        .mean()
        .sort_values()
    )
    print()
    print("Rate min/max:")
    print(
        features[rate_columns]
        .agg(["min", "max"])
        .transpose()
    )
    print()
    print("Sample-size distributions:")
    print(
        features[sample_columns]
        .describe()
        .transpose()[
            ["count", "mean", "min", "25%", "50%", "75%", "max"]
        ]
    )
    print("Saved to:", os.path.abspath(output_path))


# --------------------------------------------------
# COMMAND-LINE ENTRY POINT
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Build leakage-safe V8 contextual offense."
    )

    parser.add_argument(
        "--year",
        type=int,
        choices=YEARS,
        help="Build one validation season; omit for all seasons."
    )

    return parser.parse_args()


def main():

    args = parse_args()
    years = [args.year] if args.year else YEARS

    for year in years:
        build_season(year)

    print()
    print("=" * 76)
    print("REQUESTED V8 CONTEXTUAL-OFFENSE DATASETS COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()
