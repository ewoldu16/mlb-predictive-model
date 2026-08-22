"""Chronological ablation of the pregame situational feature family.

Development seasons only: training transformations and models are independently
fit inside each chronological fold.
"""
import os

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


YEARS = [2021, 2022, 2023, 2024]
FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
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

SUBFAMILIES = {
    "underlying team strength": [
        "sit_run_diff_diff", "sit_run_diff_per_game_diff",
        "sit_pythagorean_win_pct_diff", "sit_actual_minus_pythagorean_diff",
    ],
    "record/context": ["sit_win_pct_diff", "sit_venue_win_pct_diff"],
    "recent form": [
        "sit_l10_win_pct_diff", "sit_l10_run_diff_diff",
        "sit_l10_run_diff_per_game_diff",
    ],
    "momentum/state": ["sit_streak_diff", "sit_previous_game_win_diff"],
}
FULL_SITUATIONAL = [feature for group in SUBFAMILIES.values() for feature in group]


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000)),
    ])


def concatenate(store, years):
    return pd.concat([store[year] for year in years], ignore_index=True)


def score(model_name, experiment_type, subfamily, added_features, train, validation, year):
    features = V5_FEATURES + added_features
    model = make_pipeline()
    model.fit(train[features], train[TARGET])
    probabilities = model.predict_proba(validation[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "model": model_name,
        "experiment_type": experiment_type,
        "subfamily": subfamily,
        "validation_year": year,
        "features_added": len(added_features),
        "log_loss": log_loss(validation[TARGET], probabilities),
        "brier": brier_score_loss(validation[TARGET], probabilities),
        "auc": roc_auc_score(validation[TARGET], probabilities),
        "accuracy": accuracy_score(validation[TARGET], predictions),
    }


def build_experiments():
    experiments = [
        ("Baseline V5", "baseline", "none", []),
        ("V5 + full situational", "full", "all", FULL_SITUATIONAL),
    ]
    for group, group_features in SUBFAMILIES.items():
        retained = [feature for feature in FULL_SITUATIONAL if feature not in group_features]
        experiments.append((f"Full minus {group}", "leave_one_out", group, retained))
    for group, group_features in SUBFAMILIES.items():
        experiments.append((f"V5 + {group} only", "subfamily_alone", group, group_features))
    return experiments


def main():
    print("Loading development seasons only; holdout-season paths are never constructed.")
    data = {}
    required = V5_FEATURES + FULL_SITUATIONAL + [TARGET]
    for year in YEARS:
        base_path = f"data/processed/games_{year}_starter_lineup_matchup_features.csv"
        situational_path = f"data/processed/features_situational_{year}.csv"
        base = pd.read_csv(base_path)
        situational = pd.read_csv(situational_path)
        if base["game_id"].duplicated().any() or situational["game_id"].duplicated().any():
            raise ValueError(f"Duplicate game_id in {year} inputs")
        if set(base["game_id"]) != set(situational["game_id"]):
            raise ValueError(f"Situational coverage differs from V5 in {year}")
        merged = base.merge(
            situational[["game_id"] + FULL_SITUATIONAL], on="game_id", validate="one_to_one"
        )
        missing = [column for column in required if column not in merged.columns]
        if missing:
            raise ValueError(f"Missing required columns in {year}: {missing}")
        data[year] = merged
        print(f"{year}: {len(merged)} games")

    experiments = build_experiments()
    rows = []
    for train_years, validation_year in FOLDS:
        train = concatenate(data, train_years)
        validation = data[validation_year]
        print(f"\nTrain {train_years} -> validate {validation_year}")
        for model_name, experiment_type, subfamily, features in experiments:
            result = score(
                model_name, experiment_type, subfamily, features,
                train, validation, validation_year,
            )
            rows.append(result)
            print(
                f"{model_name}: Log Loss={result['log_loss']:.6f} "
                f"Brier={result['brier']:.6f} AUC={result['auc']:.6f} "
                f"Accuracy={result['accuracy']:.6f}"
            )

    fold_results = pd.DataFrame(rows)
    baseline = fold_results[fold_results["model"].eq("Baseline V5")].set_index("validation_year")
    full = fold_results[fold_results["model"].eq("V5 + full situational")].set_index("validation_year")
    for metric in ["log_loss", "brier", "auc", "accuracy"]:
        fold_results[f"delta_vs_v5_{metric}"] = fold_results.apply(
            lambda row: row[metric] - baseline.loc[row["validation_year"], metric], axis=1
        )
        fold_results[f"delta_vs_full_{metric}"] = fold_results.apply(
            lambda row: row[metric] - full.loc[row["validation_year"], metric], axis=1
        )

    summary = fold_results.groupby(
        ["model", "experiment_type", "subfamily"], as_index=False
    ).agg(
        features_added=("features_added", "first"),
        mean_log_loss=("log_loss", "mean"), std_log_loss=("log_loss", "std"),
        mean_brier=("brier", "mean"), std_brier=("brier", "std"),
        mean_auc=("auc", "mean"), std_auc=("auc", "std"),
        mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
        mean_delta_vs_v5_log_loss=("delta_vs_v5_log_loss", "mean"),
        mean_delta_vs_full_log_loss=("delta_vs_full_log_loss", "mean"),
    )
    improved = fold_results.assign(
        improved_vs_v5=fold_results["delta_vs_v5_log_loss"] < 0
    ).groupby("model")["improved_vs_v5"].sum()
    summary["years_logloss_improved_vs_v5"] = summary["model"].map(improved).astype(int)
    summary = summary.sort_values("mean_log_loss").reset_index(drop=True)

    full_mean = float(summary.loc[summary["model"].eq("V5 + full situational"), "mean_log_loss"].iloc[0])
    simpler = summary[
        summary["experiment_type"].eq("subfamily_alone")
        & summary["mean_log_loss"].le(full_mean)
        & summary["years_logloss_improved_vs_v5"].ge(2)
    ].sort_values("mean_log_loss")
    if simpler.empty:
        recommendation = (
            "Retain the complete situational family: no individual situational subfamily "
            "matched or beat its mean Log Loss while improving over V5 in at least 2/3 years."
        )
    else:
        winner = simpler.iloc[0]
        recommendation = (
            f"Recommend the simpler '{winner['subfamily']}' representation: its mean Log Loss "
            f"({winner['mean_log_loss']:.6f}) matched or beat the full situational model "
            f"({full_mean:.6f}) and improved over V5 in "
            f"{int(winner['years_logloss_improved_vs_v5'])}/3 years."
        )

    print("\nFOLD-LEVEL RESULTS")
    print(fold_results.sort_values(["validation_year", "log_loss"]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nMEAN AND STANDARD DEVIATION SUMMARY (ranked by mean Log Loss)")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nRECOMMENDATION")
    print(recommendation)
    print("Holdout data was not loaded, inspected, or evaluated.")

    os.makedirs("results", exist_ok=True)
    fold_results.to_csv("results/situational_feature_importance_fold_results.csv", index=False)
    summary.to_csv("results/situational_feature_importance_summary.csv", index=False)
    with open("results/situational_feature_importance_recommendation.txt", "w", encoding="utf-8") as handle:
        handle.write(recommendation + "\nHoldout data was not loaded, inspected, or evaluated.\n")


if __name__ == "__main__":
    main()
