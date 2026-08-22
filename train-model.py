import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

TRAIN_YEARS = [
    2021,
    2022,
    2023,
    2024
]

VALIDATION_YEAR = 2025


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

FEATURES = [

    # ----------------------------------------------
    # OFFENSE
    # ----------------------------------------------

    "season_woba_diff",
    "l30_woba_diff",


    # ----------------------------------------------
    # STARTING PITCHING
    # ----------------------------------------------

    "sp_season_k_pct_diff",
    "sp_season_bb_pct_diff",
    "sp_season_woba_allowed_diff",

    "sp_l30_k_pct_diff",
    "sp_l30_bb_pct_diff",
    "sp_l30_woba_allowed_diff",


    # ----------------------------------------------
    # BULLPEN
    # ----------------------------------------------

    "bp_season_k_pct_diff",
    "bp_season_bb_pct_diff",
    "bp_season_woba_allowed_diff",

    "bp_l30_k_pct_diff",
    "bp_l30_bb_pct_diff",
    "bp_l30_woba_allowed_diff",

    "bp_l7_bf_diff"
]


TARGET = "home_win"


# --------------------------------------------------
# LOAD SEASON
# --------------------------------------------------

def load_season(year):

    path = (
        f"data/processed/"
        f"games_{year}_full_features.csv"
    )

    data = pd.read_csv(path)

    data["season"] = year

    return data


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

print()
print("Loading training seasons...")

training_frames = []

for year in TRAIN_YEARS:

    season = load_season(year)

    training_frames.append(
        season
    )

    print(
        year,
        "games:",
        len(season)
    )


train = pd.concat(
    training_frames,
    ignore_index=True
)


# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------

validation = load_season(
    VALIDATION_YEAR
)


print()
print(
    "Total training games:",
    len(train)
)

print(
    "Validation games:",
    len(validation)
)


# --------------------------------------------------
# PREPARE X / Y
# --------------------------------------------------

X_train = train[
    FEATURES
]

y_train = train[
    TARGET
]


X_validation = validation[
    FEATURES
]

y_validation = validation[
    TARGET
]


# --------------------------------------------------
# MODEL PIPELINE
# --------------------------------------------------

model = Pipeline([

    # Missing early-season / rookie data
    # gets replaced by training-set median.
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000
        )
    )
])


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print()
print(
    "Training logistic regression..."
)

model.fit(
    X_train,
    y_train
)


# --------------------------------------------------
# PREDICT 2025
# --------------------------------------------------

home_win_probability = (
    model.predict_proba(
        X_validation
    )[:, 1]
)


predictions = (
    home_win_probability
    >= 0.50
).astype(int)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

accuracy = accuracy_score(
    y_validation,
    predictions
)

logloss = log_loss(
    y_validation,
    home_win_probability
)

brier = brier_score_loss(
    y_validation,
    home_win_probability
)

auc = roc_auc_score(
    y_validation,
    home_win_probability
)


# --------------------------------------------------
# NAIVE HOME-TEAM BASELINE
# --------------------------------------------------

baseline_predictions = [
    1
    for _ in range(
        len(validation)
    )
]

baseline_accuracy = (
    accuracy_score(
        y_validation,
        baseline_predictions
    )
)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("=" * 60)
print("2025 VALIDATION RESULTS")
print("=" * 60)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Log Loss: {logloss:.4f}"
)

print(
    f"Brier Score: {brier:.4f}"
)

print(
    f"ROC AUC: {auc:.4f}"
)

print()

print(
    "Always-pick-home accuracy:",
    f"{baseline_accuracy:.4f}"
)


# --------------------------------------------------
# ADD PREDICTIONS TO VALIDATION DATA
# --------------------------------------------------

validation[
    "model_home_win_probability"
] = home_win_probability

validation[
    "model_away_win_probability"
] = (
    1
    - home_win_probability
)

validation[
    "model_pick"
] = predictions


# --------------------------------------------------
# CONFIDENCE
# --------------------------------------------------

validation[
    "model_confidence"
] = validation[
    [
        "model_home_win_probability",
        "model_away_win_probability"
    ]
].max(
    axis=1
)


# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

output_path = (
    "results/"
    "predictions_2025_baseline.csv"
)

validation.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# COEFFICIENTS
# --------------------------------------------------

coefficients = pd.DataFrame({

    "feature": FEATURES,

    "coefficient":
        model[
            "model"
        ].coef_[0]
})


coefficients[
    "absolute_coefficient"
] = (
    coefficients[
        "coefficient"
    ].abs()
)


coefficients = (
    coefficients
    .sort_values(
        "absolute_coefficient",
        ascending=False
    )
)


print()
print("=" * 60)
print("MODEL COEFFICIENTS")
print("=" * 60)

print(
    coefficients[
        [
            "feature",
            "coefficient"
        ]
    ]
    .to_string(
        index=False
    )
)


# --------------------------------------------------
# MOST CONFIDENT PREDICTIONS
# --------------------------------------------------

print()
print("=" * 60)
print(
    "MOST CONFIDENT 2025 PREDICTIONS"
)
print("=" * 60)


display_columns = [

    "date",

    "away_team",
    "home_team",

    "model_away_win_probability",
    "model_home_win_probability",

    "home_win"
]


most_confident = (
    validation[
        display_columns
        + ["model_confidence"]
    ]
    .sort_values(
        "model_confidence",
        ascending=False
    )
    .head(20)
)


print(
    most_confident[
        display_columns
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "Predictions saved to:",
    os.path.abspath(
        output_path
    )
)