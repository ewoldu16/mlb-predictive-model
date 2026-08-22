import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier
)
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
RANDOM_SEED = 42


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
# LOAD DATA
# --------------------------------------------------

def load_season(year):

    path = (
        f"data/processed/"
        f"games_{year}_starter_lineup_matchup_features.csv"
    )

    data = pd.read_csv(path)
    data["season"] = year

    return data


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


X_train = train[V5_FEATURES]
y_train = train[TARGET]

X_validation = validation[V5_FEATURES]
y_validation = validation[TARGET]


print()
print("Training games:", len(train))
print("Validation games:", len(validation))
print("Features used by every model:", len(V5_FEATURES))


# --------------------------------------------------
# MODELS
# --------------------------------------------------

# Logistic regression exactly matches train-model-v5.py.
logistic_regression = Pipeline([
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


# Conservative first-pass forest. Parameters are fixed before
# evaluating the untouched 2025 validation season.
random_forest = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
    )
])


# Conservative low-learning-rate gradient boosting benchmark.
hist_gradient_boosting = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "model",
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=RANDOM_SEED
        )
    )
])


MODELS = [
    ("V5 Logistic Regression", logistic_regression),
    ("Random Forest", random_forest),
    ("HistGradientBoosting", hist_gradient_boosting)
]


# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------

results = []
probability_distributions = []

for name, model in MODELS:

    print()
    print(f"Training {name}...")

    # Every preprocessing step and estimator is fitted only
    # on the 2021-2024 training matrix.
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
        "model": name,
        "accuracy": accuracy_score(
            y_validation,
            predictions
        ),
        "log_loss": log_loss(
            y_validation,
            probabilities
        ),
        "brier_score": brier_score_loss(
            y_validation,
            probabilities
        ),
        "roc_auc": roc_auc_score(
            y_validation,
            probabilities
        )
    }

    results.append(result)

    probability_distributions.append({
        "model": name,
        "min": np.min(probabilities),
        "p05": np.percentile(probabilities, 5),
        "median": np.median(probabilities),
        "p95": np.percentile(probabilities, 95),
        "max": np.max(probabilities)
    })

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Log Loss: {result['log_loss']:.4f}")
    print(f"Brier Score: {result['brier_score']:.4f}")
    print(f"ROC AUC: {result['roc_auc']:.4f}")


# --------------------------------------------------
# COMPARISON SORTED BY LOG LOSS
# --------------------------------------------------

comparison = (
    pd.DataFrame(results)
    .sort_values("log_loss")
    .reset_index(drop=True)
)


print()
print("=" * 76)
print("V5 NONLINEAR BENCHMARK - SORTED BY LOG LOSS")
print("=" * 76)
print(
    comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# --------------------------------------------------
# PROBABILITY DISTRIBUTIONS
# --------------------------------------------------

distribution = pd.DataFrame(
    probability_distributions
)


print()
print("=" * 76)
print("PREDICTED HOME-WIN PROBABILITY DISTRIBUTIONS")
print("=" * 76)
print(
    distribution.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)
