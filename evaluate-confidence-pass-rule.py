"""Diagnostic confidence-only PASS rules using frozen chronological OOS predictions."""
import os

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss


INPUT = "results/v5_team_strength_oos_predictions_2022_2025.csv"
MAIN_THRESHOLD = 0.55
THRESHOLDS = [0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
              0.58, 0.60, 0.62, 0.65, 0.70]
YEARS = [2022, 2023, 2024, 2025]


def selection_summary(frame, scope, threshold, include_scores=False):
    selected = frame[frame["model_selected_probability"] >= threshold]
    total = len(frame)
    wins = int(selected["model_selection_won"].sum())
    count = len(selected)
    row = {
        "scope": scope,
        "threshold": threshold,
        "total_games": total,
        "games_selected": count,
        "games_passed": total - count,
        "percentage_selected": count / total if total else float("nan"),
        "wins": wins,
        "losses": count - wins,
        "win_rate": wins / count if count else float("nan"),
        "average_model_confidence": selected["model_selected_probability"].mean(),
    }
    if include_scores:
        row["selected_log_loss"] = log_loss(
            selected["model_selection_won"], selected["model_selected_probability"],
            labels=[0, 1],
        ) if count else float("nan")
        row["selected_brier_score"] = brier_score_loss(
            selected["model_selection_won"], selected["model_selected_probability"]
        ) if count else float("nan")
    return row


def main():
    predictions = pd.read_csv(INPUT)
    required = {"season", "model_selected_probability", "model_selection_won"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing OOS prediction columns: {missing}")
    if sorted(predictions["season"].unique().tolist()) != YEARS:
        raise ValueError("Unexpected seasons in OOS predictions")
    if predictions["model_selected_probability"].lt(0.5).any():
        raise ValueError("Selected-winner confidence below 0.50 detected")

    scopes = [(str(year), predictions[predictions["season"].eq(year)]) for year in YEARS]
    scopes.append(("combined_2022_2025", predictions))
    main_rows = [
        selection_summary(frame, scope, MAIN_THRESHOLD, include_scores=True)
        for scope, frame in scopes
    ]
    comparison_rows = [
        selection_summary(frame, scope, threshold)
        for scope, frame in scopes
        for threshold in THRESHOLDS
    ]
    main_results = pd.DataFrame(main_rows)
    comparisons = pd.DataFrame(comparison_rows)

    print("55% CONFIDENCE PASS RULE")
    print(main_results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nFIXED THRESHOLD COMPARISONS")
    print(comparisons.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nDiagnostic only: no threshold selection, retraining, model changes, or profit analysis performed.")

    os.makedirs("results", exist_ok=True)
    main_results.to_csv("results/v5_team_strength_confidence_pass_55pct.csv", index=False)
    comparisons.to_csv("results/v5_team_strength_confidence_threshold_comparison.csv", index=False)


if __name__ == "__main__":
    main()
