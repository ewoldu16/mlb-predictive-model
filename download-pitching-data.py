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
# ENABLE CACHE
# --------------------------------------------------

pybaseball.cache.enable()


# --------------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------------

os.makedirs(
    "data/raw/pitching",
    exist_ok=True
)


# --------------------------------------------------
# DOWNLOAD FULL PITCH-LEVEL SEASON
# --------------------------------------------------

def download_pitching_season(year):

    output_path = (
        f"data/raw/pitching/"
        f"statcast_pitches_{year}.csv"
    )

    chunk_folder = (
        f"data/raw/pitching/"
        f"statcast_chunks_{year}"
    )

    os.makedirs(
        chunk_folder,
        exist_ok=True
    )

    print()
    print("=" * 60)
    print(
        f"DOWNLOADING FULL PITCH DATA FOR {year}"
    )
    print("=" * 60)


    # --------------------------------------------------
    # KEEP PITCHING COLUMNS WE ACTUALLY NEED
    # --------------------------------------------------

    columns = [

        # Game / pitcher identity
        "game_date",
        "game_pk",
        "pitcher",
        "batter",
        "stand",
        "p_throws",

        # Pitch information
        "pitch_type",
        "release_speed",
        "description",

        # Plate appearance result
        "events",

        # Count information
        "balls",
        "strikes",

        # Batted-ball / expected results
        "estimated_woba_using_speedangle",
        "woba_value",
        "woba_denom",

        # Game context
        "home_team",
        "away_team",
        "inning_topbot",

        # Outs / runners useful for workload
        "outs_when_up"
    ]


    # --------------------------------------------------
    # DOWNLOAD RECOVERABLE DATE CHUNKS
    # --------------------------------------------------

    season_start = pd.Timestamp(
        f"{year}-03-01"
    )

    season_end = pd.Timestamp(
        f"{year}-10-15"
    )

    chunk_paths = []
    chunk_start = season_start


    while chunk_start <= season_end:

        chunk_end = min(
            chunk_start
            + pd.Timedelta(
                days=CHUNK_DAYS - 1
            ),
            season_end
        )

        start_date = chunk_start.strftime(
            "%Y-%m-%d"
        )

        end_date = chunk_end.strftime(
            "%Y-%m-%d"
        )

        chunk_path = os.path.join(
            chunk_folder,
            f"statcast_{start_date}_{end_date}.csv"
        )

        chunk_paths.append(
            chunk_path
        )


        if os.path.exists(chunk_path):

            print(
                f"Using completed chunk: "
                f"{start_date} through {end_date}"
            )

            chunk_start = (
                chunk_end
                + pd.Timedelta(days=1)
            )

            continue


        for attempt in range(
            1,
            MAX_ATTEMPTS + 1
        ):

            try:

                print(
                    f"Downloading {start_date} "
                    f"through {end_date} "
                    f"(attempt {attempt}/{MAX_ATTEMPTS})"
                )

                data = statcast(
                    start_dt=start_date,
                    end_dt=end_date
                )

                pitching_chunk = data.reindex(
                    columns=columns
                ).copy()

                pitching_chunk.to_csv(
                    chunk_path,
                    index=False
                )

                print(
                    "Chunk rows saved:",
                    len(pitching_chunk)
                )

                break


            except Exception as error:

                print(
                    f"Chunk attempt {attempt} failed "
                    f"for {start_date} through "
                    f"{end_date}: {error}"
                )

                if attempt < MAX_ATTEMPTS:

                    time.sleep(
                        RETRY_DELAY_SECONDS
                    )

                else:

                    print()
                    print(
                        "DOWNLOAD STOPPED - FAILED DATE "
                        f"RANGE: {start_date} through "
                        f"{end_date}"
                    )

                    raise RuntimeError(
                        "Statcast chunk failed after "
                        f"{MAX_ATTEMPTS} attempts: "
                        f"{start_date} through {end_date}"
                    ) from error


        chunk_start = (
            chunk_end
            + pd.Timedelta(days=1)
        )


    # --------------------------------------------------
    # COMBINE COMPLETED CHUNKS
    # --------------------------------------------------

    chunks = [
        pd.read_csv(chunk_path)
        for chunk_path in chunk_paths
    ]

    pitching_data = pd.concat(
        chunks,
        ignore_index=True
    )

    rows_before_deduplication = len(
        pitching_data
    )

    pitching_data = (
        pitching_data
        .drop_duplicates()
        .sort_values(
            [
                "game_date",
                "game_pk",
                "pitcher",
                "batter"
            ],
            kind="stable"
        )
        .reset_index(drop=True)
    )


    print()
    print(
        "Pitch rows combined:",
        rows_before_deduplication
    )

    print(
        "Duplicate rows removed:",
        rows_before_deduplication
        - len(pitching_data)
    )


    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    pitching_data.to_csv(
        output_path,
        index=False
    )


    print()
    print(
        f"{year} FULL PITCH DATA SAVED"
    )

    print(
        "Rows:",
        len(pitching_data)
    )

    print(
        "Columns:",
        len(pitching_data.columns)
    )

    print(
        "Saved to:",
        os.path.abspath(
            output_path
        )
    )


# --------------------------------------------------
# DOWNLOAD ALL SEASONS
# --------------------------------------------------

for year in YEARS:

    download_pitching_season(
        year
    )


print()
print("=" * 60)
print(
    "ALL FULL PITCH DATA COMPLETE"
)
print("=" * 60)
