import os
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

TRAIN_YEARS = [2021, 2022, 2023, 2024]
VALIDATION_YEAR = 2025
TARGET = "home_win"
HIGH_CORRELATION_THRESHOLD = 0.80


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
# V7 OFFENSIVE-FORM DIFFERENTIALS
# --------------------------------------------------

V7_FEATURES = [
    "off_l7_woba_diff",
    "off_l15_woba_diff",
    "off_l30_woba_diff",
    "off_l7_k_pct_diff",
    "off_l15_k_pct_diff",
    "off_l30_k_pct_diff",
    "off_l7_bb_pct_diff",
    "off_l15_bb_pct_diff",
    "off_l30_bb_pct_diff",
    "off_l7_hardhit_pct_diff",
    "off_l15_hardhit_pct_diff",
    "off_l30_hardhit_pct_diff",
    "off_season_hardhit_pct_diff",
    "off_season_hr_fb_diff",
    "off_l30_hr_fb_diff"
]


MODEL_V7_FEATURES = V5_FEATURES + V7_FEATURES


# --------------------------------------------------
# LOAD AND MERGE ONE SEASON
# --------------------------------------------------

def load_season(year):

    v5_path = (
        f"data/processed/"
        f"games_{year}_starter_lineup_matchup_features.csv"
    )

    v7_path = (
        f"data/processed/"
        f"features_v7_offensive_form_{year}.csv"
    )

    base = pd.read_csv(v5_path)
    offensive_form = pd.read_csv(v7_path)

    if base["game_id"].duplicated().any():
        raise ValueError(
            f"Duplicate game IDs in V5 dataset for {year}."
        )

    if offensive_form["game_id"].duplicated().any():
        raise ValueError(
            f"Duplicate game IDs in V7 dataset for {year}."
        )

    missing_v7_columns = [
        column
        for column in V7_FEATURES
        if column not in offensive_form.columns
    ]

    if missing_v7_columns:
        raise ValueError(
            f"Missing V7 columns for {year}: "
            f"{missing_v7_columns}"
        )

    missing_game_ids = (
        set(base["game_id"])
        - set(offensive_form["game_id"])
    )

    extra_game_ids = (
        set(offensive_form["game_id"])
        - set(base["game_id"])
    )

    if missing_game_ids or extra_game_ids:
        raise ValueError(
            f"V7 game coverage mismatch for {year}. "
            f"Missing IDs: {len(missing_game_ids)}; "
            f"extra IDs: {len(extra_game_ids)}"
        )

    data = base.merge(
        offensive_form,
        on="game_id",
        how="left",
        validate="one_to_one"
    )

    if len(data) != len(base):
        raise ValueError(
            f"V7 merge changed the {year} game count."
        )

    data = pd.concat(
        [
            data,
            pd.Series(
                year,
                index=data.index,
                name="season"
            )
        ],
        axis=1
    )

    return data


# --------------------------------------------------
# LOAD TRAINING AND VALIDATION DATA
# --------------------------------------------------

print()
print("Loading V5 and V7 datasets...")

training_frames = []

for year in TRAIN_YEARS:

    data = load_season(year)
    training_frames.append(data)

    print(year, "games:", len(data))


train = pd.concat(
    training_frames,
    ignore_index=True
)

validation = load_season(
    VALIDATION_YEAR
)


print()
print("Training games:", len(train))
print("Validation games:", len(validation))
print("V5 features:", len(V5_FEATURES))
print("New V7 features:", len(V7_FEATURES))


# --------------------------------------------------
# TRAINING-ONLY V7 CORRELATION DIAGNOSTICS
# --------------------------------------------------

correlation = train[V7_FEATURES].corr()

correlation_pairs = []

for first_index, first_feature in enumerate(V7_FEATURES):
    for second_feature in V7_FEATURES[first_index + 1:]:

        value = correlation.loc[
            first_feature,
            second_feature
        ]

        correlation_pairs.append({
            "feature_1": first_feature,
            "feature_2": second_feature,
            "correlation": value,
            "abs_correlation": abs(value)
        })


correlation_pairs = pd.DataFrame(
    correlation_pairs
).sort_values(
    "abs_correlation",
    ascending=False
)


high_correlation_pairs = correlation_pairs[
    correlation_pairs["abs_correlation"]
    > HIGH_CORRELATION_THRESHOLD
]


print()
print("=" * 76)
print("V7 TRAINING-SEASON CORRELATION MATRIX")
print("=" * 76)
print(
    correlation.to_string(
        float_format=lambda value: f"{value:.3f}"
    )
)


print()
print(
    "V7 PAIRS WITH ABSOLUTE CORRELATION "
    f"> {HIGH_CORRELATION_THRESHOLD:.2f}"
)

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
# MODEL PIPELINE
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


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_model(name, features):

    X_train = train[features]
    y_train = train[TARGET]

    X_validation = validation[features]
    y_validation = validation[TARGET]

    model = create_model()

    print()
    print(f"Training {name}...")

    # The full preprocessing/model pipeline is fitted only
    # on the 2021-2024 training seasons.
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    result = {
        "name": name,
        "accuracy": accuracy_score(
            y_validation,
            predictions
        ),
        "log_loss": log_loss(
            y_validation,
            probabilities
        ),
        "brier": brier_score_loss(
            y_validation,
            probabilities
        ),
        "auc": roc_auc_score(
            y_validation,
            probabilities
        ),
        "model": model,
        "probabilities": probabilities
    }

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Log Loss: {result['log_loss']:.4f}")
    print(f"Brier Score: {result['brier']:.4f}")
    print(f"ROC AUC: {result['auc']:.4f}")

    return result


v5 = evaluate_model(
    "V5 STARTER/LINEUP MATCHUPS",
    V5_FEATURES
)

v7 = evaluate_model(
    "V7 RICHER OFFENSIVE FORM",
    MODEL_V7_FEATURES
)


# --------------------------------------------------
# V5 VS V7
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "model": result["name"],
        "accuracy": result["accuracy"],
        "log_loss": result["log_loss"],
        "brier_score": result["brier"],
        "roc_auc": result["auc"]
    }
    for result in [v5, v7]
])


print()
print("=" * 70)
print("V5 VS V7")
print("=" * 70)
print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)


print()
print("V5 TO V7 EXACT CHANGES")
print(
    "Accuracy:",
    f"{v7['accuracy'] - v5['accuracy']:+.6f}"
)
print(
    "Log Loss:",
    f"{v7['log_loss'] - v5['log_loss']:+.6f}"
)
print(
    "Brier Score:",
    f"{v7['brier'] - v5['brier']:+.6f}"
)
print(
    "ROC AUC:",
    f"{v7['auc'] - v5['auc']:+.6f}"
)


# --------------------------------------------------
# SAVE V7 PREDICTIONS SEPARATELY
# --------------------------------------------------

prediction_output = validation.copy()

prediction_output[
    "v7_home_win_probability"
] = v7["probabilities"]

prediction_output[
    "v7_away_win_probability"
] = 1 - prediction_output[
    "v7_home_win_probability"
]

os.makedirs("results", exist_ok=True)

output_path = (
    "results/"
    "predictions_2025_v7_offensive_form.csv"
)

prediction_output.to_csv(
    output_path,
    index=False
)

print()
print(
    "V7 predictions saved to:",
    os.path.abspath(output_path)
)


# --------------------------------------------------
# STANDARDISED COEFFICIENTS
# --------------------------------------------------

coefficients = pd.DataFrame({
    "feature": MODEL_V7_FEATURES,
    "coefficient": v7[
        "model"
    ][
        "model"
    ].coef_[0]
})

coefficients["abs_coefficient"] = coefficients[
    "coefficient"
].abs()

coefficients = coefficients.sort_values(
    "abs_coefficient",
    ascending=False
)


print()
print("=" * 70)
print("V7 STANDARDISED COEFFICIENTS")
print("=" * 70)
print(
    coefficients[
        ["feature", "coefficient"]
    ].to_string(index=False)
)


print()
print("=" * 70)
print("NEW V7 OFFENSIVE-FORM COEFFICIENTS")
print("=" * 70)
print(
    coefficients[
        coefficients["feature"].isin(V7_FEATURES)
    ][
        ["feature", "coefficient"]
    ].to_string(index=False)
)
