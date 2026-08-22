import os
import warnings

import numpy as np
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
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
FINAL_HOLDOUT_YEAR = 2025
TARGET = "home_win"
RANDOM_SEED = 42

C_GRID = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1,
    3,
    10,
    30,
    100
]

L1_RATIOS = [0.25, 0.5, 0.75]

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
# LOAD SEASONS
# --------------------------------------------------

def load_season(year):

    path = (
        f"data/processed/"
        f"games_{year}_starter_lineup_matchup_features.csv"
    )

    data = pd.read_csv(path)

    missing = [
        column
        for column in V5_FEATURES + [TARGET]
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing V5 columns for {year}: {missing}"
        )

    data["season"] = year

    return data


print()
print("Loading V5 datasets...")

SEASONS = {
    year: load_season(year)
    for year in TRAIN_YEARS + [FINAL_HOLDOUT_YEAR]
}

for year, data in SEASONS.items():
    print(year, "games:", len(data))


# --------------------------------------------------
# MODEL CONFIGURATIONS
# --------------------------------------------------

CONFIGURATIONS = []

for c_value in C_GRID:
    CONFIGURATIONS.append({
        "penalty": "l2",
        "C": c_value,
        "l1_ratio": None
    })

for c_value in C_GRID:
    CONFIGURATIONS.append({
        "penalty": "l1",
        "C": c_value,
        "l1_ratio": None
    })

for l1_ratio in L1_RATIOS:
    for c_value in C_GRID:
        CONFIGURATIONS.append({
            "penalty": "elasticnet",
            "C": c_value,
            "l1_ratio": l1_ratio
        })


def configuration_name(configuration):

    name = (
        f"{configuration['penalty'].upper()} "
        f"C={configuration['C']:g}"
    )

    if configuration["l1_ratio"] is not None:
        name += (
            f" l1_ratio={configuration['l1_ratio']:g}"
        )

    return name


# --------------------------------------------------
# PIPELINES
# --------------------------------------------------

def create_candidate_model(configuration):

    penalty = configuration["penalty"]

    model_args = {
        "penalty": penalty,
        "C": configuration["C"],
        "max_iter": 10000
    }

    if penalty == "l2":
        model_args["solver"] = "lbfgs"
    else:
        model_args["solver"] = "saga"
        model_args["random_state"] = RANDOM_SEED

    if penalty == "elasticnet":
        model_args["l1_ratio"] = configuration[
            "l1_ratio"
        ]

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
            LogisticRegression(**model_args)
        )
    ])


def create_original_v5_model():

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
# EVALUATION HELPERS
# --------------------------------------------------

def combine_years(years):

    return pd.concat(
        [SEASONS[year] for year in years],
        ignore_index=True
    )


def score_probabilities(y_true, probabilities):

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    return {
        "log_loss": log_loss(y_true, probabilities),
        "brier_score": brier_score_loss(
            y_true,
            probabilities
        ),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions)
    }


def fit_and_score(model, training_data, validation_data):

    model.fit(
        training_data[V5_FEATURES],
        training_data[TARGET]
    )

    probabilities = model.predict_proba(
        validation_data[V5_FEATURES]
    )[:, 1]

    return (
        score_probabilities(
            validation_data[TARGET],
            probabilities
        ),
        probabilities
    )


# --------------------------------------------------
# CHRONOLOGICAL CONFIGURATION SEARCH
# --------------------------------------------------

print()
print(
    f"Evaluating {len(CONFIGURATIONS)} configurations "
    "across 3 chronological folds..."
)

fold_results = []

warnings.filterwarnings(
    "error",
    category=ConvergenceWarning
)

for config_index, configuration in enumerate(
    CONFIGURATIONS,
    start=1
):

    name = configuration_name(configuration)

    for training_years, validation_year in (
        CHRONOLOGICAL_FOLDS
    ):

        training_data = combine_years(training_years)
        validation_data = SEASONS[validation_year]

        model = create_candidate_model(configuration)

        try:
            scores, _ = fit_and_score(
                model,
                training_data,
                validation_data
            )
        except ConvergenceWarning as warning:
            raise RuntimeError(
                f"Model failed to converge: {name}, "
                f"validation year {validation_year}"
            ) from warning

        fold_results.append({
            "configuration": name,
            "penalty": configuration["penalty"],
            "C": configuration["C"],
            "l1_ratio": configuration["l1_ratio"],
            "training_years": ",".join(
                str(year) for year in training_years
            ),
            "validation_year": validation_year,
            **scores
        })

    if config_index % 10 == 0:
        print(
            f"Configurations completed: "
            f"{config_index} / {len(CONFIGURATIONS)}"
        )


fold_results = pd.DataFrame(fold_results)


# --------------------------------------------------
# RANK CONFIGURATIONS WITHOUT USING 2025
# --------------------------------------------------

ranking = (
    fold_results
    .groupby(
        ["configuration", "penalty", "C", "l1_ratio"],
        dropna=False,
        as_index=False
    )
    .agg(
        mean_log_loss=("log_loss", "mean"),
        std_log_loss=("log_loss", "std"),
        mean_brier_score=("brier_score", "mean"),
        std_brier_score=("brier_score", "std"),
        mean_roc_auc=("roc_auc", "mean"),
        std_roc_auc=("roc_auc", "std"),
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std")
    )
    .sort_values(
        [
            "mean_log_loss",
            "mean_brier_score",
            "mean_roc_auc"
        ],
        ascending=[True, True, False]
    )
    .reset_index(drop=True)
)


winner_row = ranking.iloc[0]

winner = {
    "penalty": winner_row["penalty"],
    "C": float(winner_row["C"]),
    "l1_ratio": (
        None
        if pd.isna(winner_row["l1_ratio"])
        else float(winner_row["l1_ratio"])
    )
}

winner_name = configuration_name(winner)


os.makedirs("results", exist_ok=True)

tuning_results_path = (
    "results/"
    "v5_logistic_chronological_tuning.csv"
)

fold_results.to_csv(
    tuning_results_path,
    index=False
)

ranking_path = (
    "results/"
    "v5_logistic_chronological_ranking.csv"
)

ranking.to_csv(
    ranking_path,
    index=False
)


print()
print("=" * 90)
print("CONFIGURATION RANKING - 2022 TO 2024 ONLY")
print("=" * 90)
print(
    ranking.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)


# --------------------------------------------------
# ORIGINAL V5 ON THE SAME CHRONOLOGICAL FOLDS
# --------------------------------------------------

original_fold_rows = []

for training_years, validation_year in CHRONOLOGICAL_FOLDS:

    scores, _ = fit_and_score(
        create_original_v5_model(),
        combine_years(training_years),
        SEASONS[validation_year]
    )

    original_fold_rows.append({
        "validation_year": validation_year,
        **scores
    })


original_folds = pd.DataFrame(original_fold_rows)

winner_folds = fold_results[
    fold_results["configuration"] == winner_name
].sort_values("validation_year")


print()
print("=" * 90)
print("WINNING CONFIGURATION")
print("=" * 90)
print("Penalty:", winner["penalty"])
print("C:", winner["C"])
print("l1_ratio:", winner["l1_ratio"])

print()
print("Winning chronological fold performance:")
print(
    winner_folds[
        [
            "validation_year",
            "log_loss",
            "brier_score",
            "roc_auc",
            "accuracy"
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

print()
print("Winning mean and variability across years:")
print(
    ranking.iloc[[0]][
        [
            "mean_log_loss",
            "std_log_loss",
            "mean_brier_score",
            "std_brier_score",
            "mean_roc_auc",
            "std_roc_auc",
            "mean_accuracy",
            "std_accuracy"
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

print()
print("Original V5 on identical chronological folds:")
print(
    original_folds.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

print()
print("Chronological mean comparison:")
print(
    pd.DataFrame([
        {
            "model": "Original V5",
            "log_loss": original_folds["log_loss"].mean(),
            "brier_score": original_folds[
                "brier_score"
            ].mean(),
            "roc_auc": original_folds["roc_auc"].mean(),
            "accuracy": original_folds["accuracy"].mean()
        },
        {
            "model": "Tuned V5",
            "log_loss": winner_folds["log_loss"].mean(),
            "brier_score": winner_folds[
                "brier_score"
            ].mean(),
            "roc_auc": winner_folds["roc_auc"].mean(),
            "accuracy": winner_folds["accuracy"].mean()
        }
    ]).to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)


# --------------------------------------------------
# ONE FINAL FIT AND ONE UNTOUCHED 2025 EVALUATION
# --------------------------------------------------

all_training = combine_years(TRAIN_YEARS)
holdout = SEASONS[FINAL_HOLDOUT_YEAR]

tuned_model = create_candidate_model(winner)
original_model = create_original_v5_model()

tuned_2025, tuned_probabilities = fit_and_score(
    tuned_model,
    all_training,
    holdout
)

original_2025, original_probabilities = fit_and_score(
    original_model,
    all_training,
    holdout
)


holdout_comparison = pd.DataFrame([
    {"model": "Original V5", **original_2025},
    {"model": "Tuned V5", **tuned_2025}
])


print()
print("=" * 90)
print("UNTOUCHED 2025 HOLDOUT - ORIGINAL V5 VS TUNED V5")
print("=" * 90)
print(
    holdout_comparison.to_string(
        index=False,
        float_format=lambda value: f"{value:.6f}"
    )
)

print()
print("Tuned minus original 2025 changes:")
for metric in [
    "log_loss",
    "brier_score",
    "roc_auc",
    "accuracy"
]:
    print(
        f"{metric}:",
        f"{tuned_2025[metric] - original_2025[metric]:+.6f}"
    )


# --------------------------------------------------
# SAVE TUNED PREDICTIONS SEPARATELY
# --------------------------------------------------

prediction_output = holdout.copy()

prediction_output[
    "v5_tuned_home_win_probability"
] = tuned_probabilities

prediction_output[
    "v5_tuned_away_win_probability"
] = 1 - tuned_probabilities

prediction_output[
    "v5_original_home_win_probability"
] = original_probabilities

predictions_path = (
    "results/"
    "predictions_2025_v5_logistic_tuned.csv"
)

prediction_output.to_csv(
    predictions_path,
    index=False
)


# --------------------------------------------------
# FINAL STANDARDISED COEFFICIENTS
# --------------------------------------------------

final_coefficients = tuned_model[
    "model"
].coef_[0]

coefficients = pd.DataFrame({
    "feature": V5_FEATURES,
    "coefficient": final_coefficients
})

coefficients["abs_coefficient"] = coefficients[
    "coefficient"
].abs()

coefficients = coefficients.sort_values(
    "abs_coefficient",
    ascending=False
)

zero_coefficients = int(
    np.isclose(
        final_coefficients,
        0.0,
        atol=1e-12
    ).sum()
)


print()
print("=" * 90)
print("FINAL TUNED V5 STANDARDISED COEFFICIENTS")
print("=" * 90)
print(
    coefficients[
        ["feature", "coefficient"]
    ].to_string(index=False)
)

if winner["penalty"] in ["l1", "elasticnet"]:
    print()
    print(
        "Coefficients reduced to zero:",
        zero_coefficients,
        "/",
        len(V5_FEATURES)
    )

print()
print("Fold results saved to:", os.path.abspath(tuning_results_path))
print("Ranking saved to:", os.path.abspath(ranking_path))
print("Tuned predictions saved to:", os.path.abspath(predictions_path))
