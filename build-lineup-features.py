import os
import time
import numpy as np
import pandas as pd
import statsapi


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

YEARS = [2021, 2022, 2023, 2024, 2025]
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3
REQUEST_DELAY_SECONDS = 0.05

# Static expected-PA weights by batting-order slot.
# They are normalised again when a lineup is aggregated.
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

LINEUP_COLUMNS = [
    "game_id",
    "team_side",
    "batting_order",
    "player_id",
    "player_name",
    "position"
]


# --------------------------------------------------
# EXTRACT STARTING LINEUP FROM ONE MLB BOXSCORE
# --------------------------------------------------

def extract_starting_lineup(game_id, boxscore):

    rows = []

    for team_side in ["away", "home"]:

        team = boxscore["teams"][team_side]
        starters_by_order = {}

        for player_id in team["batters"]:

            player = team["players"].get(
                "ID" + str(player_id),
                {}
            )

            raw_order = player.get("battingOrder")

            if raw_order in [None, ""]:
                continue

            raw_order = int(raw_order)

            # MLB encodes starters as 100, 200, ... 900.
            # Substitutes use values such as 101 or 201.
            if raw_order % 100 != 0:
                continue

            batting_order = raw_order // 100

            starters_by_order[batting_order] = {
                "game_id": int(game_id),
                "team_side": team_side,
                "batting_order": batting_order,
                "player_id": int(player_id),
                "player_name": player[
                    "person"
                ]["fullName"],
                "position": player.get(
                    "position",
                    {}
                ).get("abbreviation")
            }


        expected_orders = set(range(1, 10))

        if set(starters_by_order) != expected_orders:

            raise ValueError(
                f"Expected batting orders 1-9 for {team_side}; "
                f"found {sorted(starters_by_order)}"
            )

        rows.extend(
            starters_by_order[order]
            for order in range(1, 10)
        )


    return pd.DataFrame(
        rows,
        columns=LINEUP_COLUMNS
    )


# --------------------------------------------------
# DOWNLOAD AND CACHE ONE GAME
# --------------------------------------------------

def download_game_lineup(game_id, cache_path):

    for attempt in range(1, MAX_ATTEMPTS + 1):

        try:

            boxscore = statsapi.get(
                "game_boxscore",
                {
                    "gamePk": int(game_id),
                    "fields": (
                        "teams,away,home,batters,players,id,"
                        "person,fullName,battingOrder,position,"
                        "abbreviation"
                    )
                }
            )

            lineup = extract_starting_lineup(
                game_id,
                boxscore
            )

            temporary_path = cache_path + ".tmp"

            lineup.to_csv(
                temporary_path,
                index=False
            )

            os.replace(
                temporary_path,
                cache_path
            )

            return lineup


        except Exception as error:

            print(
                f"Game {game_id} attempt "
                f"{attempt}/{MAX_ATTEMPTS} failed: {error}"
            )

            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print()
                print(
                    "LINEUP DOWNLOAD STOPPED - FAILED GAME: ",
                    game_id
                )
                raise RuntimeError(
                    f"Could not download lineup for game {game_id}"
                ) from error


# --------------------------------------------------
# DOWNLOAD/COLLECT ONE SEASON OF LINEUPS
# --------------------------------------------------

def collect_lineups(year, games):

    cache_folder = (
        f"data/raw/lineups/{year}"
    )

    os.makedirs(
        cache_folder,
        exist_ok=True
    )

    lineup_frames = []
    downloaded = 0

    for index, game_id in enumerate(
        games["game_id"],
        start=1
    ):

        game_id = int(game_id)

        cache_path = os.path.join(
            cache_folder,
            f"lineup_{game_id}.csv"
        )

        if os.path.exists(cache_path):
            lineup = pd.read_csv(cache_path)
        else:
            lineup = download_game_lineup(
                game_id,
                cache_path
            )
            downloaded += 1
            time.sleep(REQUEST_DELAY_SECONDS)

        counts = lineup.groupby(
            "team_side"
        )["batting_order"].nunique()

        if (
            len(lineup) != 18
            or counts.to_dict() != {"away": 9, "home": 9}
        ):
            raise ValueError(
                f"Invalid cached lineup for game {game_id}: "
                f"{cache_path}"
            )

        lineup_frames.append(lineup)

        if index % 100 == 0:
            print(
                f"Lineups processed: {index} / {len(games)} "
                f"({downloaded} newly downloaded)"
            )


    lineups = pd.concat(
        lineup_frames,
        ignore_index=True
    )

    output_path = (
        f"data/raw/lineups/"
        f"starting_lineups_{year}.csv"
    )

    lineups.to_csv(
        output_path,
        index=False
    )

    print(
        f"{year} lineups: {len(lineups)} hitter rows; "
        f"{downloaded} new game requests"
    )

    return lineups


# --------------------------------------------------
# BUILD PLAYER HISTORY LOOKUPS
# --------------------------------------------------

def build_player_histories(year):

    statcast_path = (
        f"data/raw/statcast_{year}.csv"
    )

    plate_appearances = pd.read_csv(
        statcast_path,
        usecols=[
            "game_date",
            "batter",
            "woba_value",
            "woba_denom"
        ]
    )

    plate_appearances["game_date"] = pd.to_datetime(
        plate_appearances["game_date"]
    )

    daily = (
        plate_appearances
        .groupby(
            ["batter", "game_date"],
            as_index=False
        )
        .agg(
            woba_value_sum=("woba_value", "sum"),
            woba_denom_sum=("woba_denom", "sum")
        )
        .sort_values(["batter", "game_date"])
    )

    histories = {}

    for player_id, player_days in daily.groupby("batter"):

        histories[int(player_id)] = {
            "dates": player_days["game_date"].to_numpy(
                dtype="datetime64[ns]"
            ),
            "woba_value_cumsum": player_days[
                "woba_value_sum"
            ].cumsum().to_numpy(),
            "woba_denom_cumsum": player_days[
                "woba_denom_sum"
            ].cumsum().to_numpy()
        }

    return histories


# --------------------------------------------------
# LEAKAGE-SAFE PLAYER WOBA BEFORE A GAME DATE
# --------------------------------------------------

def player_woba_before_game(
    player_id,
    game_date,
    histories,
    days=None
):

    history = histories.get(int(player_id))

    if history is None:
        return np.nan

    dates = history["dates"]
    values = history["woba_value_cumsum"]
    denoms = history["woba_denom_cumsum"]

    game_date = np.datetime64(game_date, "ns")

    # side="left" strictly excludes all PAs on this game date.
    right = np.searchsorted(
        dates,
        game_date,
        side="left"
    )

    if right == 0:
        return np.nan

    value_sum = values[right - 1]
    denom_sum = denoms[right - 1]

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

        if left > 0:
            value_sum -= values[left - 1]
            denom_sum -= denoms[left - 1]

    if denom_sum <= 0:
        return np.nan

    return value_sum / denom_sum


# --------------------------------------------------
# AGGREGATE NINE STARTERS INTO ONE LINEUP VALUE
# --------------------------------------------------

def weighted_lineup_woba(lineup, value_column):

    values = lineup[value_column]

    # Leave incomplete lineups missing. A later model pipeline can
    # learn any fallback from training data without inventing stats.
    if len(lineup) != 9 or values.isna().any():
        return np.nan

    weights = lineup["batting_order"].map(
        ORDER_WEIGHTS
    )

    return np.average(
        values,
        weights=weights
    )


# --------------------------------------------------
# BUILD ONE SEASON OF LINEUP FEATURES
# --------------------------------------------------

def build_lineup_features(year):

    print()
    print("=" * 70)
    print(f"BUILDING ACTUAL LINEUP FEATURES FOR {year}")
    print("=" * 70)

    games_path = (
        f"data/processed/"
        f"games_{year}_platoon_features.csv"
    )

    games = pd.read_csv(games_path)
    games["date"] = pd.to_datetime(games["date"])

    lineups = collect_lineups(
        year,
        games
    )

    histories = build_player_histories(year)

    game_dates = games.set_index(
        "game_id"
    )["date"]

    lineups["game_date"] = lineups["game_id"].map(
        game_dates
    )

    if lineups["game_date"].isna().any():
        raise ValueError("Lineup rows could not be mapped to game dates.")

    print("Calculating pre-game hitter quality...")

    lineups["season_woba_before_game"] = lineups.apply(
        lambda row: player_woba_before_game(
            row["player_id"],
            row["game_date"],
            histories
        ),
        axis=1
    )

    lineups["l30_woba_before_game"] = lineups.apply(
        lambda row: player_woba_before_game(
            row["player_id"],
            row["game_date"],
            histories,
            days=30
        ),
        axis=1
    )

    feature_rows = []

    for game_id, game_lineups in lineups.groupby("game_id"):

        row = {"game_id": game_id}

        for team_side in ["home", "away"]:

            lineup = game_lineups[
                game_lineups["team_side"] == team_side
            ].sort_values("batting_order")

            row[f"{team_side}_lineup_season_woba"] = (
                weighted_lineup_woba(
                    lineup,
                    "season_woba_before_game"
                )
            )

            row[f"{team_side}_lineup_l30_woba"] = (
                weighted_lineup_woba(
                    lineup,
                    "l30_woba_before_game"
                )
            )

            row[f"{team_side}_lineup_season_known_hitters"] = (
                lineup["season_woba_before_game"].notna().sum()
            )

            row[f"{team_side}_lineup_l30_known_hitters"] = (
                lineup["l30_woba_before_game"].notna().sum()
            )

        feature_rows.append(row)


    features = pd.DataFrame(feature_rows)

    features["lineup_season_woba_diff"] = (
        features["home_lineup_season_woba"]
        - features["away_lineup_season_woba"]
    )

    features["lineup_l30_woba_diff"] = (
        features["home_lineup_l30_woba"]
        - features["away_lineup_l30_woba"]
    )

    output = games.merge(
        features,
        on="game_id",
        how="left",
        validate="one_to_one"
    )

    if len(output) != len(games):
        raise ValueError("Lineup merge changed the game row count.")

    output_path = (
        f"data/processed/"
        f"games_{year}_lineup_features.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    check_columns = [
        "home_lineup_season_woba",
        "away_lineup_season_woba",
        "home_lineup_l30_woba",
        "away_lineup_l30_woba",
        "lineup_season_woba_diff",
        "lineup_l30_woba_diff"
    ]

    print()
    print(f"{year} LINEUP FEATURES COMPLETE")
    print("Games:", len(output))
    print("Starting hitters:", len(lineups))
    print("Expected starting hitters:", len(games) * 18)
    print("Duplicate game IDs:", output["game_id"].duplicated().sum())
    print("Missing feature values:")
    print(output[check_columns].isna().sum())
    print("Saved to:", os.path.abspath(output_path))


# --------------------------------------------------
# BUILD ALL SEASONS
# --------------------------------------------------

for year in YEARS:
    build_lineup_features(year)


print()
print("=" * 70)
print("ALL ACTUAL LINEUP FEATURE DATASETS COMPLETE")
print("=" * 70)
