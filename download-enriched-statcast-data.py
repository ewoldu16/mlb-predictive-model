import argparse
import os
import time

import pandas as pd
import pybaseball
from pybaseball import statcast


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]
CHUNK_DAYS = 7
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5


# --------------------------------------------------
# ENRICHED DATA CONTRACT
# --------------------------------------------------

COLUMNS = [
    # Pitch identity and participants
    "game_date",
    "game_pk",
    "game_type",
    "at_bat_number",
    "pitch_number",
    "pitcher",
    "batter",
    "stand",
    "p_throws",

    # Pitch characteristics
    "pitch_type",
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "release_extension",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "zone",
    "description",

    # Plate-appearance and contact results
    "events",
    "bb_type",
    "launch_speed",
    "launch_angle",
    "launch_speed_angle",
    "estimated_woba_using_speedangle",
    "woba_value",
    "woba_denom",

    # Count and game state
    "balls",
    "strikes",
    "outs_when_up",
    "inning",
    "inning_topbot",
    "home_team",
    "away_team",
    "bat_score",
    "fld_score",
    "on_1b",
    "on_2b",
    "on_3b",

    # Run and win-probability context
    "delta_run_exp",
    "delta_home_win_exp"
]


PITCH_ID_COLUMNS = [
    "game_pk",
    "at_bat_number",
    "pitch_number"
]


IMPORTANT_MISSINGNESS_COLUMNS = [
    "pitch_type",
    "release_speed",
    "release_spin_rate",
    "release_extension",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "estimated_woba_using_speedangle",
    "delta_run_exp",
    "delta_home_win_exp"
]


# --------------------------------------------------
# ENABLE PYBASEBALL CACHE
# --------------------------------------------------

pybaseball.cache.enable()


# --------------------------------------------------
# LOAD KNOWN REGULAR-SEASON GAME UNIVERSE
# --------------------------------------------------

def load_regular_season_games(year):

    path = f"data/raw/games_{year}.csv"

    games = pd.read_csv(
        path,
        usecols=["date", "game_id"]
    )

    games["date"] = pd.to_datetime(
        games["date"]
    )

    games["game_id"] = pd.to_numeric(
        games["game_id"],
        errors="raise"
    ).astype("int64")

    if games["game_id"].duplicated().any():
        raise ValueError(
            f"Duplicate game IDs in known {year} game universe."
        )

    return games


# --------------------------------------------------
# VALIDATE AND NORMALISE A CHUNK
# --------------------------------------------------

def validate_chunk(
    data,
    regular_game_ids,
    label
):

    missing_columns = [
        column
        for column in COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{label} missing required columns: "
            f"{missing_columns}"
        )

    chunk = data[COLUMNS].copy()

    chunk["game_pk"] = pd.to_numeric(
        chunk["game_pk"],
        errors="coerce"
    )

    chunk = chunk[
        chunk["game_pk"].isin(regular_game_ids)
    ].copy()

    if len(chunk) == 0:
        return chunk

    if chunk[PITCH_ID_COLUMNS].isna().any().any():
        missing_ids = int(
            chunk[PITCH_ID_COLUMNS]
            .isna()
            .any(axis=1)
            .sum()
        )

        raise ValueError(
            f"{label} contains {missing_ids} rows "
            "without a complete pitch identity."
        )

    chunk["game_pk"] = chunk["game_pk"].astype("int64")

    duplicate_identity = chunk.duplicated(
        subset=PITCH_ID_COLUMNS,
        keep=False
    )

    if duplicate_identity.any():

        duplicate_rows = chunk[
            duplicate_identity
        ]

        conflicting = (
            duplicate_rows
            .drop_duplicates()
            .duplicated(
                subset=PITCH_ID_COLUMNS,
                keep=False
            )
        )

        if conflicting.any():
            raise ValueError(
                f"{label} contains conflicting rows with "
                "the same pitch identity."
            )

        chunk = chunk.drop_duplicates(
            subset=PITCH_ID_COLUMNS,
            keep="first"
        )

    invalid_games = set(
        chunk["game_pk"].unique()
    ) - regular_game_ids

    if invalid_games:
        raise ValueError(
            f"{label} contains games outside the known "
            f"regular-season universe: {sorted(invalid_games)[:10]}"
        )

    return chunk


# --------------------------------------------------
# DOWNLOAD ONE VALIDATED CHUNK
# --------------------------------------------------

def download_chunk(
    start_date,
    end_date,
    regular_game_ids,
    label
):

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:

            print(
                f"Downloading {start_date} through {end_date} "
                f"(attempt {attempt}/{MAX_ATTEMPTS})"
            )

            data = statcast(
                start_dt=start_date,
                end_dt=end_date
            )

            chunk = validate_chunk(
                data,
                regular_game_ids,
                label
            )

            print(
                f"Validated regular-season pitch rows: "
                f"{len(chunk)}"
            )

            return chunk


        except Exception as error:

            print(
                f"Attempt {attempt} failed for "
                f"{start_date} through {end_date}: {error}"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print()
                print(
                    "ENRICHED STATCAST DOWNLOAD STOPPED - "
                    f"FAILED RANGE: {start_date} through {end_date}"
                )
                raise RuntimeError(
                    "Enriched Statcast chunk failed after "
                    f"{MAX_ATTEMPTS} attempts: "
                    f"{start_date} through {end_date}"
                ) from error


# --------------------------------------------------
# ATOMIC CSV SAVE
# --------------------------------------------------

def save_atomic(data, output_path):

    temporary_path = output_path + ".tmp"

    data.to_csv(
        temporary_path,
        index=False
    )

    os.replace(
        temporary_path,
        output_path
    )


# --------------------------------------------------
# SMALL SCHEMA TEST
# --------------------------------------------------

def run_small_test(year):

    print()
    print("=" * 70)
    print(f"SMALL ENRICHED STATCAST TEST - {year}")
    print("=" * 70)

    games = load_regular_season_games(year)

    test_date = games["date"].min().strftime(
        "%Y-%m-%d"
    )

    regular_game_ids = set(
        games["game_id"]
    )

    test_data = download_chunk(
        test_date,
        test_date,
        regular_game_ids,
        f"test {test_date}"
    )

    if len(test_data) == 0:
        raise ValueError(
            f"Small test returned no known regular-season "
            f"pitches for {test_date}."
        )

    print()
    print("SMALL TEST PASSED")
    print("Date:", test_date)
    print("Rows:", len(test_data))
    print("Games:", test_data["game_pk"].nunique())
    print("Required columns returned:", len(COLUMNS))
    print("Important-field missingness:")
    print(
        test_data[
            IMPORTANT_MISSINGNESS_COLUMNS
        ].isna().sum()
    )


# --------------------------------------------------
# DOWNLOAD ONE COMPLETE SEASON
# --------------------------------------------------

def download_season(year):

    print()
    print("=" * 70)
    print(f"ENRICHED REGULAR-SEASON STATCAST - {year}")
    print("=" * 70)

    games = load_regular_season_games(year)
    regular_game_ids = set(games["game_id"])

    cache_folder = (
        f"data/raw/statcast_enriched_chunks_{year}"
    )

    os.makedirs(
        cache_folder,
        exist_ok=True
    )

    season_start = pd.Timestamp(f"{year}-03-01")
    season_end = pd.Timestamp(f"{year}-10-15")

    chunk_paths = []
    chunk_start = season_start

    while chunk_start <= season_end:

        chunk_end = min(
            chunk_start
            + pd.Timedelta(days=CHUNK_DAYS - 1),
            season_end
        )

        start_date = chunk_start.strftime("%Y-%m-%d")
        end_date = chunk_end.strftime("%Y-%m-%d")

        chunk_path = os.path.join(
            cache_folder,
            f"statcast_enriched_{start_date}_{end_date}.csv"
        )

        label = f"{year} chunk {start_date} through {end_date}"

        expected_chunk_game_ids = set(
            games.loc[
                games["date"].between(
                    chunk_start,
                    chunk_end
                ),
                "game_id"
            ]
        )

        if not expected_chunk_game_ids:

            empty_chunk = pd.DataFrame(
                columns=COLUMNS
            )

            save_atomic(
                empty_chunk,
                chunk_path
            )

            print(
                f"No known regular-season games: "
                f"{start_date} through {end_date}; "
                "saved validated empty chunk."
            )

            chunk_paths.append(chunk_path)

            chunk_start = (
                chunk_end
                + pd.Timedelta(days=1)
            )

            continue

        use_cached_chunk = False

        if os.path.exists(chunk_path):

            try:
                cached = pd.read_csv(chunk_path)
                validate_chunk(
                    cached,
                    regular_game_ids,
                    label
                )
                use_cached_chunk = True
                print(
                    f"Using validated cached chunk: "
                    f"{start_date} through {end_date}"
                )
            except Exception as error:
                print(
                    f"Cached chunk is invalid and will be "
                    f"re-downloaded: {error}"
                )

        if not use_cached_chunk:

            chunk = download_chunk(
                start_date,
                end_date,
                regular_game_ids,
                label
            )

            save_atomic(
                chunk,
                chunk_path
            )

        chunk_paths.append(chunk_path)

        chunk_start = (
            chunk_end
            + pd.Timedelta(days=1)
        )


    chunks = []

    for chunk_path in chunk_paths:

        chunk = pd.read_csv(chunk_path)

        chunks.append(
            validate_chunk(
                chunk,
                regular_game_ids,
                chunk_path
            )
        )

    season = pd.concat(
        chunks,
        ignore_index=True
    )

    rows_before_deduplication = len(season)

    duplicate_mask = season.duplicated(
        subset=PITCH_ID_COLUMNS,
        keep=False
    )

    duplicate_count = int(
        season.duplicated(
            subset=PITCH_ID_COLUMNS,
            keep="first"
        ).sum()
    )

    if duplicate_mask.any():

        duplicates = season[duplicate_mask]

        conflicting = (
            duplicates
            .drop_duplicates()
            .duplicated(
                subset=PITCH_ID_COLUMNS,
                keep=False
            )
        )

        if conflicting.any():
            raise ValueError(
                f"{year} contains conflicting duplicate pitch IDs."
            )

        season = season.drop_duplicates(
            subset=PITCH_ID_COLUMNS,
            keep="first"
        )

    season = (
        season
        .sort_values(
            [
                "game_date",
                "game_pk",
                "at_bat_number",
                "pitch_number"
            ],
            kind="stable"
        )
        .reset_index(drop=True)
    )

    downloaded_game_ids = set(
        season["game_pk"].astype("int64").unique()
    )

    missing_game_ids = regular_game_ids - downloaded_game_ids
    extra_game_ids = downloaded_game_ids - regular_game_ids

    missing_required_columns = [
        column
        for column in COLUMNS
        if column not in season.columns
    ]

    if missing_required_columns:
        raise ValueError(
            f"{year} final dataset is missing required columns: "
            f"{missing_required_columns}"
        )

    if missing_game_ids or extra_game_ids:
        raise ValueError(
            f"{year} game coverage failed. "
            f"Missing games: {sorted(missing_game_ids)[:20]}; "
            f"extra games: {sorted(extra_game_ids)[:20]}"
        )

    output_path = (
        f"data/raw/statcast_enriched_{year}.csv"
    )

    save_atomic(
        season,
        output_path
    )

    print()
    print(f"{year} ENRICHED STATCAST COMPLETE")
    print("Pitch rows before deduplication:", rows_before_deduplication)
    print("Pitch rows saved:", len(season))
    print("Duplicate pitch rows detected:", duplicate_count)
    print("Expected regular-season games:", len(regular_game_ids))
    print("Covered regular-season games:", len(downloaded_game_ids))
    print("Missing regular-season games:", len(missing_game_ids))
    print("Extra games:", len(extra_game_ids))
    print("Missing required columns:", missing_required_columns)
    print("Important-field missingness:")
    print(
        season[
            IMPORTANT_MISSINGNESS_COLUMNS
        ].isna().sum()
    )
    print("Saved to:", os.path.abspath(output_path))

    return {
        "year": year,
        "pitch_rows": len(season),
        "games_expected": len(regular_game_ids),
        "games_covered": len(downloaded_game_ids),
        "duplicate_rows": duplicate_count,
        "missing_required_columns": missing_required_columns,
        "output_path": os.path.abspath(output_path)
    }


# --------------------------------------------------
# COMMAND-LINE ENTRY POINT
# --------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Download versioned, resumable enriched Statcast data."
        )
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a one-day schema/coverage test without saving a season."
    )

    parser.add_argument(
        "--test-year",
        type=int,
        default=2025,
        choices=YEARS
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if args.test:
        run_small_test(args.test_year)
        return

    summaries = [
        download_season(year)
        for year in YEARS
    ]

    print()
    print("=" * 90)
    print("ALL ENRICHED REGULAR-SEASON STATCAST DATA COMPLETE")
    print("=" * 90)
    print(
        pd.DataFrame(summaries).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
