"""One-time frozen holdout evaluation of V5 plus underlying team strength."""
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_YEARS = [2021, 2022, 2023, 2024]
HOLDOUT_YEAR = 2025
TARGET = "home_win"

V5_FEATURES = [
    "season_woba_diff", "l30_woba_diff",
    "sp_season_k_pct_diff", "sp_season_bb_pct_diff", "sp_season_woba_allowed_diff",
    "sp_l30_k_pct_diff", "sp_l30_bb_pct_diff", "sp_l30_woba_allowed_diff",
    "bp_season_k_pct_diff", "bp_season_bb_pct_diff", "bp_season_woba_allowed_diff",
    "bp_l30_k_pct_diff", "bp_l30_bb_pct_diff", "bp_l30_woba_allowed_diff", "bp_l7_bf_diff",
    "sp_days_rest_diff", "sp_prev_pitch_count_diff", "sp_season_velocity_diff",
    "sp_l30_velocity_diff", "sp_season_whiff_diff", "sp_l30_whiff_diff",
    "sp_season_xwoba_allowed_diff", "sp_l30_xwoba_allowed_diff",
    "season_platoon_woba_diff", "l30_platoon_woba_diff",
    "sp_matchup_season_xwoba_allowed_diff", "sp_matchup_season_k_pct_diff",
    "sp_matchup_season_bb_pct_diff", "sp_matchup_season_whiff_pct_diff",
    "sp_matchup_l30_xwoba_allowed_diff", "sp_matchup_l30_k_pct_diff",
    "sp_matchup_l30_bb_pct_diff", "sp_matchup_l30_whiff_pct_diff",
]

# Frozen before the holdout evaluation from the 2021-2024 chronological study.
UNDERLYING_TEAM_STRENGTH_FEATURES = [
    "sit_run_diff_diff",
    "sit_run_diff_per_game_diff",
    "sit_pythagorean_win_pct_diff",
    "sit_actual_minus_pythagorean_diff",
]


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000)),
    ])


def load_season(year):
    base = pd.read_csv(
        f"data/processed/games_{year}_starter_lineup_matchup_features.csv"
    )
    situational = pd.read_csv(
        f"data/processed/features_situational_{year}.csv",
        usecols=["game_id"] + UNDERLYING_TEAM_STRENGTH_FEATURES,
    )
    if base["game_id"].duplicated().any() or situational["game_id"].duplicated().any():
        raise ValueError(f"Duplicate game_id in {year}")
    if set(base["game_id"]) != set(situational["game_id"]):
        raise ValueError(f"Situational coverage does not match V5 for {year}")
    return base.merge(situational, on="game_id", validate="one_to_one")


def evaluate(name, features, train, holdout):
    model = make_pipeline()
    model.fit(train[features], train[TARGET])
    probabilities = model.predict_proba(holdout[features])[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(holdout[TARGET], predictions),
        "log_loss": log_loss(holdout[TARGET], probabilities),
        "brier_score": brier_score_loss(holdout[TARGET], probabilities),
        "roc_auc": roc_auc_score(holdout[TARGET], probabilities),
    }
    coefficients = pd.DataFrame({
        "model": name,
        "feature": features,
        "standardized_coefficient": model.named_steps["model"].coef_[0],
    })
    coefficients["abs_coefficient"] = coefficients["standardized_coefficient"].abs()
    return metrics, probabilities, predictions, coefficients


def calibration(name, probabilities, actual):
    # Fixed bins selected before examining holdout outcomes.
    edges = np.linspace(0.0, 1.0, 11)
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(10)]
    frame = pd.DataFrame({
        "model": name,
        "probability": probabilities,
        "actual_home_win": actual.to_numpy(),
        "probability_bin": pd.cut(
            probabilities, bins=edges, labels=labels, include_lowest=True, right=True
        ),
    })
    result = frame.groupby(
        ["model", "probability_bin"], observed=False, as_index=False
    ).agg(
        games=("actual_home_win", "size"),
        mean_predicted_probability=("probability", "mean"),
        observed_home_win_rate=("actual_home_win", "mean"),
    )
    result["calibration_gap"] = (
        result["observed_home_win_rate"] - result["mean_predicted_probability"]
    )
    return result


def main():
    print("Frozen specification: no feature, preprocessing, or hyperparameter changes.")
    training_frames = []
    for year in TRAIN_YEARS:
        frame = load_season(year)
        training_frames.append(frame)
        print(f"Training {year}: {len(frame)} games")
    train = pd.concat(training_frames, ignore_index=True)
    holdout = load_season(HOLDOUT_YEAR)
    print(f"Holdout {HOLDOUT_YEAR}: {len(holdout)} games")

    enhanced_features = V5_FEATURES + UNDERLYING_TEAM_STRENGTH_FEATURES
    v5_metrics, v5_prob, v5_pred, v5_coef = evaluate(
        "Original V5", V5_FEATURES, train, holdout
    )
    strength_metrics, strength_prob, strength_pred, strength_coef = evaluate(
        "V5 + underlying team strength", enhanced_features, train, holdout
    )

    metrics = pd.DataFrame([v5_metrics, strength_metrics])
    v5_row = metrics.loc[metrics["model"].eq("Original V5")].iloc[0]
    for metric in ["accuracy", "log_loss", "brier_score", "roc_auc"]:
        metrics[f"delta_vs_v5_{metric}"] = metrics[metric] - v5_row[metric]

    calibration_table = pd.concat([
        calibration("Original V5", v5_prob, holdout[TARGET]),
        calibration("V5 + underlying team strength", strength_prob, holdout[TARGET]),
    ], ignore_index=True)
    coefficients = pd.concat([v5_coef, strength_coef], ignore_index=True)
    winner_differs = v5_pred != strength_pred

    print("\nHOLDOUT METRICS AND CHANGES")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nPredicted winner differs from V5 in {winner_differs.sum()} of {len(holdout)} games.")
    print("\nPROBABILITY CALIBRATION (fixed 0.10-wide bins)")
    print(calibration_table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSTANDARDIZED COEFFICIENTS")
    for name in metrics["model"]:
        print(f"\n{name}")
        subset = coefficients[coefficients["model"].eq(name)].sort_values(
            "abs_coefficient", ascending=False
        )
        print(subset[["feature", "standardized_coefficient"]].to_string(
            index=False, float_format=lambda x: f"{x:.6f}"
        ))

    output = holdout[[
        column for column in [
            "date", "game_id", "away_team", "home_team", "away_score", "home_score", TARGET
        ] if column in holdout.columns
    ]].copy()
    output["v5_home_win_probability"] = v5_prob
    output["v5_predicted_home_win"] = v5_pred
    output["v5_strength_home_win_probability"] = strength_prob
    output["v5_strength_predicted_home_win"] = strength_pred
    output["predicted_winner_differs"] = winner_differs

    os.makedirs("results", exist_ok=True)
    output.to_csv("results/predictions_2025_v5_underlying_team_strength.csv", index=False)
    metrics.to_csv("results/v5_underlying_team_strength_2025_metrics.csv", index=False)
    calibration_table.to_csv("results/v5_underlying_team_strength_2025_calibration.csv", index=False)
    coefficients.to_csv("results/v5_underlying_team_strength_2025_coefficients.csv", index=False)
    print("\nEvaluation complete. No follow-up experiment was run.")


if __name__ == "__main__":
    main()
