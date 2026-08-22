import os
import pandas as pd
import pybaseball
from pybaseball import statcast


# --------------------------------------------------
# ENABLE PYBASEBALL CACHE
# --------------------------------------------------

pybaseball.cache.enable()


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]


# --------------------------------------------------
# OUTPUT FOLDER
# --------------------------------------------------

os.makedirs(
    "data/raw",
    exist_ok=True
)


# --------------------------------------------------
# DOWNLOAD ONE SEASON
# --------------------------------------------------

def download_statcast_season(year):

    output_path = (
        f"data/raw/statcast_{year}.csv"
    )

    # Don't download again if we already have it
    if os.path.exists(output_path):

        print()
        print(
            f"{year} already exists."
        )

        print(
            "Skipping download:",
            output_path
        )

        return


    print()
    print("=" * 50)

    print(
        f"DOWNLOADING STATCAST {year}"
    )

    print("=" * 50)


    start_date = f"{year}-03-01"
    end_date = f"{year}-10-15"


    data = statcast(
        start_dt=start_date,
        end_dt=end_date
    )


    print()
    print(
        "Raw pitch rows downloaded:",
        len(data)
    )


    # ----------------------------------------------
    # KEEP COMPLETED PLATE APPEARANCES
    # ----------------------------------------------

    plate_appearances = data[
        data["events"].notna()
    ].copy()


    # ----------------------------------------------
    # KEEP COLUMNS WE NEED
    # ----------------------------------------------

    plate_appearances = (
        plate_appearances[
            [
                "game_date",
                "game_pk",
                "batter",
                "pitcher",
                "events",
                "home_team",
                "away_team",
                "inning_topbot",
                "woba_value",
                "woba_denom"
            ]
        ]
    )


    # ----------------------------------------------
    # IDENTIFY BATTING TEAM
    # ----------------------------------------------

    plate_appearances[
        "batting_team"
    ] = plate_appearances.apply(

        lambda row:
            row["away_team"]
            if row["inning_topbot"] == "Top"
            else row["home_team"],

        axis=1
    )


    # ----------------------------------------------
    # CLEAN DATE
    # ----------------------------------------------

    plate_appearances[
        "game_date"
    ] = pd.to_datetime(

        plate_appearances[
            "game_date"
        ]
    )


    # ----------------------------------------------
    # REMOVE ROWS WITHOUT WOBA DENOMINATOR
    # ----------------------------------------------

    plate_appearances = (
        plate_appearances[
            plate_appearances[
                "woba_denom"
            ].notna()
        ]
        .copy()
    )


    # ----------------------------------------------
    # SAVE
    # ----------------------------------------------

    plate_appearances.to_csv(
        output_path,
        index=False
    )


    print()
    print(
        f"{year} STATCAST DATA SAVED"
    )

    print(
        "Plate appearances:",
        len(plate_appearances)
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

    download_statcast_season(
        year
    )


print()
print(
    "ALL REQUESTED STATCAST SEASONS COMPLETE"
)