"""Chronological OOS backtest for the frozen V5 + team-strength champion."""
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024),
         ([2021, 2022, 2023, 2024], 2025)]
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
TEAM_STRENGTH_FEATURES = [
    "sit_run_diff_diff", "sit_run_diff_per_game_diff",
    "sit_pythagorean_win_pct_diff", "sit_actual_minus_pythagorean_diff",
]
FEATURES = V5_FEATURES + TEAM_STRENGTH_FEATURES
BUCKET_EDGES = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 1.0000001]
BUCKET_LABELS = ["50-52%", "52-54%", "54-56%", "56-58%", "58-60%",
                 "60-62%", "62-65%", "65-70%", "70%+"]
THRESHOLDS = [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70]


def pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000)),
    ])


def load_year(year):
    base = pd.read_csv(f"data/processed/games_{year}_starter_lineup_matchup_features.csv")
    situational = pd.read_csv(
        f"data/processed/features_situational_{year}.csv",
        usecols=["game_id"] + TEAM_STRENGTH_FEATURES,
    )
    if base.game_id.duplicated().any() or situational.game_id.duplicated().any():
        raise ValueError(f"Duplicate game IDs in {year}")
    if set(base.game_id) != set(situational.game_id):
        raise ValueError(f"Coverage mismatch in {year}")
    merged = base.merge(situational, on="game_id", validate="one_to_one")
    missing = [column for column in FEATURES + [TARGET] if column not in merged]
    if missing:
        raise ValueError(f"Missing columns in {year}: {missing}")
    merged["season"] = year
    return merged


def metrics(frame, scope):
    actual = frame[TARGET]
    probability = frame["predicted_home_probability"]
    predicted = (probability >= 0.5).astype(int)
    return {
        "scope": scope, "games": len(frame),
        "accuracy": accuracy_score(actual, predicted),
        "log_loss": log_loss(actual, probability),
        "brier_score": brier_score_loss(actual, probability),
        "roc_auc": roc_auc_score(actual, probability),
        "mean_predicted_home_probability": probability.mean(),
        "actual_home_win_rate": actual.mean(),
    }


def bucket_rows(frame, scope, probability_type, probability, outcome):
    work = pd.DataFrame({"probability": probability, "outcome": outcome})
    work = work[work.probability >= 0.50].copy()
    work["bucket"] = pd.cut(work.probability, BUCKET_EDGES, labels=BUCKET_LABELS,
                            right=False, include_lowest=True)
    rows = []
    for label in BUCKET_LABELS:
        group = work[work.bucket == label]
        count = len(group)
        predicted_average = group.probability.mean() if count else np.nan
        actual_rate = group.outcome.mean() if count else np.nan
        correct = int(group.outcome.sum()) if count else 0
        rows.append({
            "scope": scope, "probability_type": probability_type, "bucket": label,
            "games": count, "average_predicted_probability": predicted_average,
            "actual_win_percentage": actual_rate,
            "calibration_error": actual_rate - predicted_average,
            "correct": correct, "incorrect": count - correct,
        })
    return rows


def threshold_rows(frame, scope, probability_type, probability, outcome):
    rows = []
    for threshold in THRESHOLDS:
        selected = probability >= threshold
        count = int(selected.sum())
        rows.append({
            "scope": scope, "probability_type": probability_type,
            "threshold": threshold, "games": count,
            "average_predicted_probability": probability[selected].mean() if count else np.nan,
            "actual_win_percentage": outcome[selected].mean() if count else np.nan,
        })
    return rows


def calibration_statistics(scope, probability_type, probability, outcome):
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome, dtype=int)
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000)
    calibration_model.fit(logits, outcome)
    equal_edges = np.linspace(0, 1, 11)
    bin_ids = np.clip(np.digitize(probability, equal_edges, right=True) - 1, 0, 9)
    ece = 0.0
    for bin_id in range(10):
        selected = bin_ids == bin_id
        if selected.any():
            ece += selected.mean() * abs(outcome[selected].mean() - probability[selected].mean())
    return {
        "scope": scope, "probability_type": probability_type, "games": len(outcome),
        "mean_predicted_probability": probability.mean(), "actual_win_rate": outcome.mean(),
        "calibration_intercept": calibration_model.intercept_[0],
        "calibration_slope": calibration_model.coef_[0, 0],
        "expected_calibration_error": ece,
    }


def main():
    print("Frozen V5 + four team-strength features; no tuning or specification changes.")
    years = {year for train_years, validation_year in FOLDS for year in train_years + [validation_year]}
    data = {year: load_year(year) for year in sorted(years)}
    prediction_frames = []
    for training_years, validation_year in FOLDS:
        train = pd.concat([data[year] for year in training_years], ignore_index=True)
        validation = data[validation_year]
        model = pipeline()
        model.fit(train[FEATURES], train[TARGET])
        home_probability = model.predict_proba(validation[FEATURES])[:, 1]
        predicted_home = home_probability >= 0.50
        selected_team = np.where(predicted_home, validation.home_team, validation.away_team)
        selected_probability = np.maximum(home_probability, 1 - home_probability)
        selected_won = np.where(predicted_home, validation[TARGET].eq(1), validation[TARGET].eq(0))
        actual_winner = np.where(validation[TARGET].eq(1), validation.home_team, validation.away_team)
        frame = pd.DataFrame({
            "game_id": validation.game_id.to_numpy(),
            "date": validation.date.to_numpy(),
            "season": validation_year,
            "home_team": validation.home_team.to_numpy(),
            "away_team": validation.away_team.to_numpy(),
            TARGET: validation[TARGET].to_numpy(),
            "actual_winner": actual_winner,
            "predicted_home_probability": home_probability,
            "predicted_away_probability": 1 - home_probability,
            "model_selected_team": selected_team,
            "model_selected_probability": selected_probability,
            "model_selection_won": selected_won.astype(int),
        })
        prediction_frames.append(frame)
        print(f"Train {training_years} -> predict {validation_year}: {len(frame)} games")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs in combined OOS predictions")

    scopes = [(str(year), predictions[predictions.season.eq(year)]) for year in sorted(predictions.season.unique())]
    scopes.append(("combined_2022_2025", predictions))
    season_metrics, buckets, thresholds, calibration = [], [], [], []
    for scope, frame in scopes:
        season_metrics.append(metrics(frame, scope))
        probability_sets = [
            ("home_team", frame.predicted_home_probability, frame[TARGET]),
            ("model_selected_winner", frame.model_selected_probability, frame.model_selection_won),
        ]
        for probability_type, probability, outcome in probability_sets:
            buckets.extend(bucket_rows(frame, scope, probability_type, probability, outcome))
            thresholds.extend(threshold_rows(frame, scope, probability_type, probability, outcome))
            calibration.append(calibration_statistics(scope, probability_type, probability, outcome))

    season_metrics = pd.DataFrame(season_metrics)
    buckets = pd.DataFrame(buckets)
    thresholds = pd.DataFrame(thresholds)
    calibration = pd.DataFrame(calibration)
    combined_buckets = buckets[buckets.scope.eq("combined_2022_2025")]
    combined_thresholds = thresholds[thresholds.scope.eq("combined_2022_2025")]
    combined_calibration = calibration[calibration.scope.eq("combined_2022_2025")]

    print("\nSEASON AND COMBINED METRICS")
    print(season_metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCOMBINED PROBABILITY BUCKETS")
    print(combined_buckets.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCOMBINED CONFIDENCE THRESHOLDS")
    print(combined_thresholds.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCALIBRATION RESULTS")
    print(calibration.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    os.makedirs("results", exist_ok=True)
    predictions.to_csv("results/v5_team_strength_oos_predictions_2022_2025.csv", index=False)
    buckets.to_csv("results/v5_team_strength_oos_probability_buckets.csv", index=False)
    thresholds.to_csv("results/v5_team_strength_oos_confidence_thresholds.csv", index=False)
    season_metrics.to_csv("results/v5_team_strength_oos_season_metrics.csv", index=False)
    calibration.to_csv("results/v5_team_strength_oos_calibration.csv", index=False)
    print("\nDiagnostic backtest complete. No model changes or additional experiments were run.")


if __name__ == "__main__":
    main()
