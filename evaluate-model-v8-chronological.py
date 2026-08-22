import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

TARGET = "home_win"
HIGH_CORRELATION_THRESHOLD = 0.80
COEFFICIENT_REPORT_THRESHOLD = 0.05

CHRONOLOGICAL_FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024)
]


# --------------------------------------------------
# EXACT V5 FEATURE SET
# --------------------------------------------------

V1_FEATURES = [
    "season_woba_diff",
    "l30_woba_diff",
    "sp_season_k_pct_diff",
    "sp_season_bb_pct_diff",
    "sp_season_woba_allowed_diff",
    "sp_l30_k_pct_diff",
    "sp_l30_bb_pct_diff",
    "sp_l30_woba_allowed_diff",
    "bp_season_k_pct_diff",
    "bp_season_bb_pct_diff",
    "bp_season_woba_allowed_diff",
    "bp_l30_k_pct_diff",
    "bp_l30_bb_pct_diff",
    "bp_l30_woba_allowed_diff",
    "bp_l7_bf_diff"
]


ADVANCED_FEATURES = [
    "sp_days_rest_diff",
    "sp_prev_pitch_count_diff",
    "sp_season_velocity_diff",
    "sp_l30_velocity_diff",
    "sp_season_whiff_diff",
    "sp_l30_whiff_diff",
    "sp_season_xwoba_allowed_diff",
    "sp_l30_xwoba_allowed_diff"
]


PLATOON_FEATURES = [
    "season_platoon_woba_diff",
    "l30_platoon_woba_diff"
]


MATCHUP_FEATURES = [
    "sp_matchup_season_xwoba_allowed_diff",
    "sp_matchup_season_k_pct_diff",
    "sp_matchup_season_bb_pct_diff",
    "sp_matchup_season_whiff_pct_diff",
    "sp_matchup_l30_xwoba_allowed_diff",
    "sp_matchup_l30_k_pct_diff",
    "sp_matchup_l30_bb_pct_diff",
    "sp_matchup_l30_whiff_pct_diff"
]


V5_FEATURES = (
    V1_FEATURES
    + ADVANCED_FEATURES
    + PLATOON_FEATURES
    + MATCHUP_FEATURES
)


# --------------------------------------------------
# V8 CONTEXTUAL DIFFERENTIAL FEATURES
# --------------------------------------------------

V8_FEATURES = [
    "ctx_venue_season_woba_diff",
    "ctx_venue_season_k_pct_diff",
    "ctx_venue_season_bb_pct_diff",
    "ctx_venue_l30_woba_diff",
    "ctx_venue_l30_k_pct_diff",
    "ctx_venue_l30_bb_pct_diff",
    "ctx_hand_season_woba_diff",
    "ctx_hand_season_k_pct_diff",
    "ctx_hand_season_bb_pct_diff",
    "ctx_hand_l30_woba_diff",
    "ctx_hand_l30_k_pct_diff",
    "ctx_hand_l30_bb_pct_diff",
    "ctx_combined_season_woba_diff",
    "ctx_combined_season_k_pct_diff",
    "ctx_combined_season_bb_pct_diff",
    "ctx_combined_l30_woba_diff",
    "ctx_combined_l30_k_pct_diff",
    "ctx_combined_l30_bb_pct_diff"
]


V5_V8_FEATURES = V5_FEATURES + V8_FEATURES


# --------------------------------------------------
# LOAD AND MERGE DEVELOPMENT SEASONS ONLY
# --------------------------------------------------

def load_season(year):

    v5 = pd.read_csv(
        f"data/processed/"
        f"games_{year}_starter_lineup_matchup_features.csv"
    )

    v8 = pd.read_csv(
        f"data/processed/"
        f"features_v8_contextual_offense_{year}.csv"
    )

    if v5["game_id"].duplicated().any():
        raise ValueError(
            f"Duplicate V5 game IDs for {year}."
        )

    if v8["game_id"].duplicated().any():
        raise ValueError(
            f"Duplicate V8 game IDs for {year}."
        )

    missing_columns = [
        column
        for column in V8_FEATURES
        if column not in v8.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing V8 columns for {year}: {missing_columns}"
        )

    missing_game_ids = set(v5["game_id"]) - set(v8["game_id"])
    extra_game_ids = set(v8["game_id"]) - set(v5["game_id"])

    if missing_game_ids or extra_game_ids:
        raise ValueError(
            f"V8 game coverage mismatch for {year}. "
            f"Missing: {len(missing_game_ids)}; "
            f"extra: {len(extra_game_ids)}"
        )

    data = v5.merge(
        v8,
        on="game_id",
        how="left",
        validate="one_to_one"
    )

    if len(data) != len(v5):
        raise ValueError(
            f"V8 merge changed the {year} game count."
        )

    return data


print()
print("Loading 2021-2024 development datasets only...")

SEASONS = {
    year: load_season(year)
    for year in [2021, 2022, 2023, 2024]
}

for year, data in SEASONS.items():
    print(year, "games:", len(data))


def combine_years(years):

    return pd.concat(
        [SEASONS[year] for year in years],
        ignore_index=True
    )


# --------------------------------------------------
# DEVELOPMENT-ONLY CORRELATION DIAGNOSTICS
# --------------------------------------------------

development = combine_years([2021, 2022, 2023, 2024])
correlation = development[V8_FEATURES].corr()

correlation_pairs = []

for first_index, first_feature in enumerate(V8_FEATURES):
    for second_feature in V8_FEATURES[first_index + 1:]:

        value = correlation.loc[first_feature, second_feature]

        correlation_pairs.append({
            "feature_1": first_feature,
            "feature_2": second_feature,
            "correlation": value,
            "abs_correlation": abs(value)
        })


correlation_pairs = (
    pd.DataFrame(correlation_pairs)
    .sort_values("abs_correlation", ascending=False)
)

high_correlation_pairs = correlation_pairs[
    correlation_pairs["abs_correlation"]
    > HIGH_CORRELATION_THRESHOLD
]


print()
print("=" * 90)
print("HIGHLY CORRELATED V8 DEVELOPMENT FEATURES")
print("=" * 90)

if len(high_correlation_pairs) == 0:
    print("None")
else:
    print(
        high_correlation_pairs[
            ["feature_1", "feature_2", "correlation"]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )


# --------------------------------------------------
# UNCHANGED LOGISTIC PIPELINE
# --------------------------------------------------

def create_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(max_iter=3000)
        )
    ])


def evaluate_fold(
    name,
    features,
    training_data,
    validation_data,
    validation_year
):

    model = create_model()

    model.fit(
        training_data[features],
        training_data[TARGET]
    )

    probabilities = model.predict_proba(
        validation_data[features]
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    result = {
        "model": name,
        "validation_year": validation_year,
        "accuracy": accuracy_score(
            validation_data[TARGET],
            predictions
        ),
        "log_loss": log_loss(
            validation_data[TARGET],
            probabilities
        ),
        "brier_score": brier_score_loss(
            validation_data[TARGET],
            probabilities
        ),
        "roc_auc": roc_auc_score(
            validation_data[TARGET],
            probabilities
        )
    }

    return result, model


# --------------------------------------------------
# CHRONOLOGICAL EVALUATION
# --------------------------------------------------

results = []
coefficient_rows = []

for training_years, validation_year in CHRONOLOGICAL_FOLDS:

    training_data = combine_years(training_years)
    validation_data = SEASONS[validation_year]

    print()
    print("=" * 90)
    print(
        f"TRAIN {training_years} -> VALIDATE {validation_year}"
    )
    print("=" * 90)

    v5_result, _ = evaluate_fold(
        "V5",
        V5_FEATURES,
        training_data,
        validation_data,
        validation_year
    )

    v8_result, v8_model = evaluate_fold(
        "V5 + V8",
        V5_V8_FEATURES,
        training_data,
        validation_data,
        validation_year
    )

    results.extend([v5_result, v8_result])

    for result in [v5_result, v8_result]:
        print()
        print(result["model"])
        print(f"Accuracy: {result['accuracy']:.6f}")
        print(f"Log Loss: {result['log_loss']:.6f}")
        print(f"Brier Score: {result['brier_score']:.6f}")
        print(f"ROC AUC: {result['roc_auc']:.6f}")

    coefficients = pd.DataFrame({
        "feature": V5_V8_FEATURES,
        "coefficient": v8_model[
            "model"
        ].coef_[0]
    })

    coefficients = coefficients[
        coefficients["feature"].isin(V8_FEATURES)
    ].copy()

    coefficients["abs_coefficient"] = coefficients[
        "coefficient"
    ].abs()

    coefficients = coefficients.sort_values(
        "abs_coefficient",
        ascending=False
    )

    above_threshold = coefficients[
        coefficients["abs_coefficient"]
        > COEFFICIENT_REPORT_THRESHOLD
    ]

    print()
    print(
        "V8 STANDARDISED COEFFICIENTS WITH "
        f"ABSOLUTE VALUE > {COEFFICIENT_REPORT_THRESHOLD:.2f}"
    )

    if len(above_threshold) == 0:
        print("None")
    else:
        print(
            above_threshold[
                ["feature", "coefficient"]
            ].to_string(index=False)
        )

    for coefficient in coefficients.itertuples():
        coefficient_rows.append({
            "validation_year": validation_year,
            "feature": coefficient.feature,
            "coefficient": coefficient.coefficient,
            "abs_coefficient": coefficient.abs_coefficient
        })


# --------------------------------------------------
# MEANS, STANDARD DEVIATIONS, AND CHANGES
# --------------------------------------------------

results = pd.DataFrame(results)

summary = (
    results
    .groupby("model", as_index=False)
    .agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        mean_log_loss=("log_loss", "mean"),
        std_log_loss=("log_loss", "std"),
        mean_brier_score=("brier_score", "mean"),
        std_brier_score=("brier_score", "std"),
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std")
    )
)

v5_mean = summary[summary["model"] == "V5"].iloc[0]
v8_mean = summary[summary["model"] == "V5 + V8"].iloc[0]


print()
print("=" * 90)
print("CHRONOLOGICAL DEVELOPMENT SUMMARY - 2022 TO 2024")
print("=" * 90)
print(
    summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

print()
print("AVERAGE V5+V8 MINUS V5 CHANGE")
print(
    "Accuracy:",
    f"{v8_mean['mean_accuracy'] - v5_mean['mean_accuracy']:+.6f}"
)
print(
    "Log Loss:",
    f"{v8_mean['mean_log_loss'] - v5_mean['mean_log_loss']:+.6f}"
)
print(
    "Brier Score:",
    f"{v8_mean['mean_brier_score'] - v5_mean['mean_brier_score']:+.6f}"
)
print(
    "ROC AUC:",
    f"{v8_mean['mean_roc_auc'] - v5_mean['mean_roc_auc']:+.6f}"
)

print()
print("2025 was not loaded or evaluated.")
