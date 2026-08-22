import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
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


# --------------------------------------------------
# V2 / V3 / V5 FEATURES
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


V2_FEATURES = V1_FEATURES + ADVANCED_FEATURES
V3_FEATURES = V2_FEATURES + PLATOON_FEATURES
V5_FEATURES = V3_FEATURES + MATCHUP_FEATURES


# --------------------------------------------------
# LOAD SEASON
# --------------------------------------------------

def load_season(year):

    path = (
        f"data/processed/"
        f"games_{year}_starter_lineup_matchup_features.csv"
    )

    data = pd.read_csv(path)
    data["season"] = year

    return data


# --------------------------------------------------
# LOAD TRAINING AND VALIDATION DATA
# --------------------------------------------------

print()
print("Loading training data...")

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


# --------------------------------------------------
# MODEL PIPELINE
# --------------------------------------------------

def create_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000
            )
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

    # Imputation, scaling, and regression are fitted only
    # with the 2021-2024 training seasons.
    model.fit(
        X_train,
        y_train
    )

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


v2 = evaluate_model(
    "V2 ADVANCED STARTER",
    V2_FEATURES
)

v3 = evaluate_model(
    "V3 PLATOON",
    V3_FEATURES
)

v5 = evaluate_model(
    "V5 STARTER/LINEUP MATCHUPS",
    V5_FEATURES
)


# --------------------------------------------------
# V2 VS V3 VS V5
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "model": result["name"],
        "accuracy": result["accuracy"],
        "log_loss": result["log_loss"],
        "brier_score": result["brier"],
        "roc_auc": result["auc"]
    }
    for result in [v2, v3, v5]
])


print()
print("=" * 60)
print("V2 VS V3 VS V5")
print("=" * 60)
print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


print()
print("V3 TO V5 CHANGES")
print(
    "Accuracy:",
    f"{v5['accuracy'] - v3['accuracy']:+.4f}"
)
print(
    "Log Loss:",
    f"{v5['log_loss'] - v3['log_loss']:+.4f}"
)
print(
    "Brier Score:",
    f"{v5['brier'] - v3['brier']:+.4f}"
)
print(
    "ROC AUC:",
    f"{v5['auc'] - v3['auc']:+.4f}"
)


# --------------------------------------------------
# SAVE V5 PREDICTIONS SEPARATELY
# --------------------------------------------------

validation[
    "v5_home_win_probability"
] = v5["probabilities"]

validation[
    "v5_away_win_probability"
] = (
    1
    - validation["v5_home_win_probability"]
)

os.makedirs(
    "results",
    exist_ok=True
)

output_path = (
    "results/"
    "predictions_2025_v5_starter_lineup_matchups.csv"
)

validation.to_csv(
    output_path,
    index=False
)

print()
print(
    "V5 predictions saved to:",
    os.path.abspath(output_path)
)


# --------------------------------------------------
# STANDARDISED V5 COEFFICIENTS
# --------------------------------------------------

coefficients = pd.DataFrame({
    "feature": V5_FEATURES,
    "coefficient": v5[
        "model"
    ][
        "model"
    ].coef_[0]
})

coefficients["abs_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "abs_coefficient",
    ascending=False
)


print()
print("=" * 60)
print("V5 STANDARDISED COEFFICIENTS")
print("=" * 60)
print(
    coefficients[
        ["feature", "coefficient"]
    ].to_string(index=False)
)


print()
print("=" * 60)
print("V5 MATCHUP FEATURE COEFFICIENTS")
print("=" * 60)
print(
    coefficients[
        coefficients["feature"].isin(
            MATCHUP_FEATURES
        )
    ][
        ["feature", "coefficient"]
    ].to_string(index=False)
)
