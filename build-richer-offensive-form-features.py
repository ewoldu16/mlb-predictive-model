import argparse
import os

import numpy as np
import pandas as pd


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]
WINDOWS = [7, 15, 30]
HARD_HIT_MPH = 95.0
MIN_RECENT_HR_FB_FLY_BALLS = 20

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
    "walks",
    "hard_hits",
    "qualifying_batted_balls",
    "home_runs_on_fly_balls",
    "fly_balls"
]


# --------------------------------------------------
# LOAD KNOWN REGULAR-SEASON GAMES
# --------------------------------------------------

def load_games(year):

    path = f"data/raw/games_{year}.csv"

    games = pd.read_csv(path)

    games["date"] = pd.to_datetime(
        games["date"]
    )

    games["game_id"] = pd.to_numeric(
        games["game_id"],
        errors="raise"
    ).astype("int64")

    if games["game_id"].duplicated().any():
        raise ValueError(
            f"Duplicate game IDs in {year} game universe."
        )

    return games


# --------------------------------------------------
# BUILD DAILY REGULAR-SEASON TEAM TOTALS
# --------------------------------------------------

def build_daily_team_offense(year, games):

    path = f"data/raw/statcast_enriched_{year}.csv"

    columns = [
        "game_date",
        "game_pk",
        "inning_topbot",
        "events",
        "woba_value",
        "woba_denom",
        "launch_speed",
        "bb_type"
    ]

    pitches = pd.read_csv(
        path,
        usecols=columns
    )

    pitches["game_date"] = pd.to_datetime(
        pitches["game_date"]
    )

    pitches["game_pk"] = pd.to_numeric(
        pitches["game_pk"],
        errors="coerce"
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

    # Scheduled target games legitimately have no Statcast rows yet. Coverage
    # validation applies to completed historical source games only.
    completed = games[games["away_score"].notna() & games["home_score"].notna()]
    expected_games = set(completed["game_id"])

    if covered_games != expected_games:
        raise ValueError(
            f"{year} enriched Statcast game coverage mismatch. "
            f"Missing: {len(expected_games - covered_games)}; "
            f"extra: {len(covered_games - expected_games)}"
        )

    pitches["batting_team"] = np.where(
        pitches["inning_topbot"] == "Top",
        pitches["away_team"],
        pitches["home_team"]
    )

    pitches["is_pa"] = pitches[
        "events"
    ].notna().astype(int)

    pitches["is_k"] = pitches[
        "events"
    ].isin(STRIKEOUT_EVENTS).astype(int)

    pitches["is_bb"] = pitches[
        "events"
    ].isin(WALK_EVENTS).astype(int)

    pitches["pa_woba_value"] = pitches[
        "woba_value"
    ].where(pitches["is_pa"] == 1)

    pitches["pa_woba_denom"] = pitches[
        "woba_denom"
    ].where(pitches["is_pa"] == 1)

    qualifying_batted_ball = (
        (pitches["is_pa"] == 1)
        & pitches["launch_speed"].notna()
    )

    # Missing launch speed remains missing. Only measured batted
    # balls on PA-ending rows enter the HardHit% calculation.
    pitches["hard_hit"] = np.where(
        qualifying_batted_ball,
        (
            pitches["launch_speed"]
            >= HARD_HIT_MPH
        ).astype(int),
        np.nan
    )

    pitches["qualifying_batted_ball"] = pitches[
        "hard_hit"
    ].notna().astype(int)

    pitches["is_fly_ball"] = (
        (pitches["is_pa"] == 1)
        & (pitches["bb_type"] == "fly_ball")
    ).astype(int)

    pitches["is_home_run_on_fly_ball"] = (
        (pitches["is_pa"] == 1)
        & (pitches["events"] == "home_run")
        & (pitches["bb_type"] == "fly_ball")
    ).astype(int)

    daily = (
        pitches
        .groupby(
            ["batting_team", "game_date"],
            as_index=False
        )
        .agg(
            woba_value_sum=("pa_woba_value", "sum"),
            woba_denom_sum=("pa_woba_denom", "sum"),
            plate_appearances=("is_pa", "sum"),
            strikeouts=("is_k", "sum"),
            walks=("is_bb", "sum"),
            hard_hits=("hard_hit", "sum"),
            qualifying_batted_balls=(
                "qualifying_batted_ball",
                "sum"
            ),
            home_runs_on_fly_balls=(
                "is_home_run_on_fly_ball",
                "sum"
            ),
            fly_balls=("is_fly_ball", "sum")
        )
        .sort_values(["batting_team", "game_date"])
    )

    return daily, len(pitches), len(covered_games)


# --------------------------------------------------
# BUILD FAST CUMULATIVE TEAM LOOKUPS
# --------------------------------------------------

def build_team_histories(daily):

    histories = {}

    for team, rows in daily.groupby("batting_team"):

        histories[team] = {
            "dates": rows["game_date"].to_numpy(
                dtype="datetime64[ns]"
            ),
            "cumulative": {
                column: rows[column].cumsum().to_numpy()
                for column in TOTAL_COLUMNS
            }
        }

    return histories


# --------------------------------------------------
# TOTALS STRICTLY BEFORE A TARGET DATE
# --------------------------------------------------

def totals_before_date(
    team,
    game_date,
    histories,
    days=None
):

    history = histories.get(team)

    if history is None:
        return None

    dates = history["dates"]
    game_date = np.datetime64(game_date, "ns")

    # side="left" excludes all activity on the target date,
    # including the current game and same-day doubleheaders.
    right = np.searchsorted(
        dates,
        game_date,
        side="left"
    )

    if right == 0:
        return None

    left = 0

    if days is not None:
        window_start = (
            game_date
            - np.timedelta64(days, "D")
        )

        left = np.searchsorted(
            dates,
            window_start,
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

    return totals


# --------------------------------------------------
# DERIVE RATES WITHOUT FILLING MISSING HISTORY
# --------------------------------------------------

def rates_from_totals(totals, recent=False):

    result = {
        "woba": np.nan,
        "k_pct": np.nan,
        "bb_pct": np.nan,
        "hardhit_pct": np.nan,
        "hr_fb": np.nan,
        "woba_denom": np.nan,
        "pa": np.nan,
        "bbe": np.nan,
        "fly_balls": np.nan
    }

    if totals is None:
        return result

    result["woba_denom"] = totals[
        "woba_denom_sum"
    ]
    result["pa"] = totals["plate_appearances"]
    result["bbe"] = totals[
        "qualifying_batted_balls"
    ]
    result["fly_balls"] = totals["fly_balls"]

    if totals["woba_denom_sum"] > 0:
        result["woba"] = (
            totals["woba_value_sum"]
            / totals["woba_denom_sum"]
        )

    if totals["plate_appearances"] > 0:
        result["k_pct"] = (
            totals["strikeouts"]
            / totals["plate_appearances"]
        )
        result["bb_pct"] = (
            totals["walks"]
            / totals["plate_appearances"]
        )

    if totals["qualifying_batted_balls"] > 0:
        result["hardhit_pct"] = (
            totals["hard_hits"]
            / totals["qualifying_batted_balls"]
        )

    enough_fly_balls = totals["fly_balls"] > 0

    if recent:
        enough_fly_balls = (
            totals["fly_balls"]
            >= MIN_RECENT_HR_FB_FLY_BALLS
        )

    if enough_fly_balls:
        result["hr_fb"] = (
            totals["home_runs_on_fly_balls"]
            / totals["fly_balls"]
        )

    return result


# --------------------------------------------------
# BUILD ONE SEASON
# --------------------------------------------------

def build_season(year):

    print()
    print("=" * 72)
    print(f"BUILDING V7 RICHER OFFENSIVE FORM - {year}")
    print("=" * 72)

    games = load_games(year)

    daily, pitch_rows, statcast_games = (
        build_daily_team_offense(year, games)
    )

    histories = build_team_histories(daily)
    feature_rows = []

    for index, game in games.iterrows():

        row = {"game_id": int(game["game_id"])}

        side_teams = {
            "home": game["home_team"],
            "away": game["away_team"]
        }

        for side, team in side_teams.items():

            season_totals = totals_before_date(
                team,
                game["date"],
                histories
            )

            season_rates = rates_from_totals(
                season_totals
            )

            row[
                f"{side}_off_season_hardhit_pct"
            ] = season_rates["hardhit_pct"]

            row[
                f"{side}_off_season_hr_fb"
            ] = season_rates["hr_fb"]

            row[
                f"{side}_off_season_bbe"
            ] = season_rates["bbe"]

            row[
                f"{side}_off_season_fly_balls"
            ] = season_rates["fly_balls"]

            for days in WINDOWS:

                totals = totals_before_date(
                    team,
                    game["date"],
                    histories,
                    days=days
                )

                rates = rates_from_totals(
                    totals,
                    recent=(days == 30)
                )

                prefix = f"{side}_off_l{days}"

                row[f"{prefix}_woba"] = rates["woba"]
                row[f"{prefix}_k_pct"] = rates["k_pct"]
                row[f"{prefix}_bb_pct"] = rates["bb_pct"]
                row[f"{prefix}_hardhit_pct"] = rates[
                    "hardhit_pct"
                ]
                row[f"{prefix}_woba_denom"] = rates[
                    "woba_denom"
                ]
                row[f"{prefix}_pa"] = rates["pa"]
                row[f"{prefix}_bbe"] = rates["bbe"]
                row[f"{prefix}_fly_balls"] = rates[
                    "fly_balls"
                ]

                if days == 30:
                    row[f"{prefix}_hr_fb"] = rates["hr_fb"]

        feature_rows.append(row)

        if (index + 1) % 250 == 0:
            print(
                f"Processed {index + 1} / {len(games)}"
            )

    features = pd.DataFrame(feature_rows)

    rate_names = [
        "season_hardhit_pct",
        "season_hr_fb"
    ]

    for days in WINDOWS:
        rate_names.extend([
            f"l{days}_woba",
            f"l{days}_k_pct",
            f"l{days}_bb_pct",
            f"l{days}_hardhit_pct"
        ])

    rate_names.append("l30_hr_fb")

    for rate_name in rate_names:

        features[
            f"off_{rate_name}_diff"
        ] = (
            features[f"home_off_{rate_name}"]
            - features[f"away_off_{rate_name}"]
        )

    if len(features) != len(games):
        raise ValueError(
            f"{year} output row count does not match game count."
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
            token in column
            for token in [
                "_woba",
                "_k_pct",
                "_bb_pct",
                "_hardhit_pct",
                "_hr_fb"
            ]
        )
        and not column.endswith("_denom")
        and not column.endswith("_diff")
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

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    output_path = (
        f"data/processed/"
        f"features_v7_offensive_form_{year}.csv"
    )

    features.to_csv(
        output_path,
        index=False
    )

    sample_columns = [
        column
        for column in features.columns
        if any(
            column.endswith(suffix)
            for suffix in [
                "_woba_denom",
                "_pa",
                "_bbe",
                "_fly_balls"
            ]
        )
    ]

    print()
    print(f"{year} V7 OFFENSIVE FORM COMPLETE")
    print("Expected games:", len(games))
    print("Output games:", len(features))
    print("Duplicate game IDs:", features["game_id"].duplicated().sum())
    print("Regular-season Statcast games:", statcast_games)
    print("Regular-season pitch rows:", pitch_rows)
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
    print("Sample-size summaries:")
    print(
        features[sample_columns]
        .describe()
        .transpose()[
            ["count", "mean", "min", "50%", "max"]
        ]
    )
    print("Saved to:", os.path.abspath(output_path))


# --------------------------------------------------
# COMMAND-LINE ENTRY POINT
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Build leakage-safe V7 offensive-form features."
    )

    parser.add_argument(
        "--year",
        type=int,
        choices=YEARS,
        help="Build one season for validation; omit for all seasons."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    years = [args.year] if args.year else YEARS

    for year in years:
        build_season(year)

    print()
    print("=" * 72)
    print("REQUESTED V7 OFFENSIVE-FORM DATASETS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
