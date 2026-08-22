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

TRAIN_YEARS = [
    2021,
    2022,
    2023,
    2024
]

VALIDATION_YEAR = 2025


# --------------------------------------------------
# V2 FEATURES
# --------------------------------------------------

V1_FEATURES = [

    # OFFENSE
    "season_woba_diff",
    "l30_woba_diff",

    # STARTER
    "sp_season_k_pct_diff",
    "sp_season_bb_pct_diff",
    "sp_season_woba_allowed_diff",

    "sp_l30_k_pct_diff",
    "sp_l30_bb_pct_diff",
    "sp_l30_woba_allowed_diff",

    # BULLPEN
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


V2_FEATURES = (
    V1_FEATURES
    + ADVANCED_FEATURES
)


# --------------------------------------------------
# NEW V3 PLATOON FEATURES
# --------------------------------------------------

PLATOON_FEATURES = [
    "season_platoon_woba_diff",
    "l30_platoon_woba_diff"
]


V3_FEATURES = (
    V2_FEATURES
    + PLATOON_FEATURES
)


TARGET = "home_win"


# --------------------------------------------------
# LOAD SEASON
# --------------------------------------------------

def load_season(year):

    path = (
        f"data/processed/"
        f"games_{year}_platoon_features.csv"
    )

    data = pd.read_csv(path)

    data["season"] = year

    return data


# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------

print()
print("Loading training data...")

training_frames = []

for year in TRAIN_YEARS:

    data = load_season(year)

    training_frames.append(
        data
    )

    print(
        year,
        "games:",
        len(data)
    )


train = pd.concat(
    training_frames,
    ignore_index=True
)


validation = load_season(
    VALIDATION_YEAR
)


print()
print(
    "Training games:",
    len(train)
)

print(
    "Validation games:",
    len(validation)
)


# --------------------------------------------------
# MODEL BUILDER
# --------------------------------------------------

def create_model():

    return Pipeline([

        # The imputer is fitted on 2021-2024 only,
        # so validation values cannot affect medians.
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
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(
    name,
    features
):

    X_train = train[
        features
    ]

    y_train = train[
        TARGET
    ]

    X_validation = validation[
        features
    ]

    y_validation = validation[
        TARGET
    ]


    model = create_model()


    print()
    print(
        f"Training {name}..."
    )


    model.fit(
        X_train,
        y_train
    )


    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )


    predictions = (
        probabilities >= 0.50
    ).astype(int)


    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    logloss = log_loss(
        y_validation,
        probabilities
    )

    brier = brier_score_loss(
        y_validation,
        probabilities
    )

    auc = roc_auc_score(
        y_validation,
        probabilities
    )


    print()
    print("=" * 60)
    print(name)
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


    return {
        "name": name,
        "accuracy": accuracy,
        "log_loss": logloss,
        "brier": brier,
        "auc": auc,
        "model": model,
        "probabilities": probabilities
    }


# --------------------------------------------------
# RUN V2 AND V3
# --------------------------------------------------

v2 = evaluate_model(
    "V2 ADVANCED STARTER",
    V2_FEATURES
)


v3 = evaluate_model(
    "V3 PLATOON",
    V3_FEATURES
)


# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

print()
print("=" * 60)
print("V2 VS V3")
print("=" * 60)

print()

print(
    "Accuracy change:",
    f"{v3['accuracy'] - v2['accuracy']:+.4f}"
)

print(
    "Log Loss change:",
    f"{v3['log_loss'] - v2['log_loss']:+.4f}"
)

print(
    "Brier Score change:",
    f"{v3['brier'] - v2['brier']:+.4f}"
)

print(
    "ROC AUC change:",
    f"{v3['auc'] - v2['auc']:+.4f}"
)


# --------------------------------------------------
# SAVE V3 PREDICTIONS
# --------------------------------------------------

validation[
    "v3_home_win_probability"
] = v3[
    "probabilities"
]

validation[
    "v3_away_win_probability"
] = (
    1
    - validation[
        "v3_home_win_probability"
    ]
)


os.makedirs(
    "results",
    exist_ok=True
)


output_path = (
    "results/"
    "predictions_2025_v3_platoon.csv"
)


validation.to_csv(
    output_path,
    index=False
)


print()
print(
    "V3 predictions saved to:",
    os.path.abspath(
        output_path
    )
)


# --------------------------------------------------
# STANDARDISED V3 COEFFICIENTS
# --------------------------------------------------

coefficients = pd.DataFrame({

    "feature":
        V3_FEATURES,

    "coefficient":
        v3[
            "model"
        ][
            "model"
        ].coef_[0]
})


coefficients[
    "abs_coefficient"
] = (
    coefficients[
        "coefficient"
    ].abs()
)


coefficients = (
    coefficients
    .sort_values(
        "abs_coefficient",
        ascending=False
    )
)


print()
print("=" * 60)
print(
    "V3 PLATOON STANDARDISED COEFFICIENTS"
)
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
