"""Chronological evaluation of market-only and market-aware V9 models."""
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


FOLDS = [([2022], 2023), ([2022, 2023], 2024), ([2022, 2023, 2024], 2025)]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
TARGET = "home_win"
V5 = [
    "season_woba_diff", "l30_woba_diff", "sp_season_k_pct_diff", "sp_season_bb_pct_diff",
    "sp_season_woba_allowed_diff", "sp_l30_k_pct_diff", "sp_l30_bb_pct_diff",
    "sp_l30_woba_allowed_diff", "bp_season_k_pct_diff", "bp_season_bb_pct_diff",
    "bp_season_woba_allowed_diff", "bp_l30_k_pct_diff", "bp_l30_bb_pct_diff",
    "bp_l30_woba_allowed_diff", "bp_l7_bf_diff", "sp_days_rest_diff",
    "sp_prev_pitch_count_diff", "sp_season_velocity_diff", "sp_l30_velocity_diff",
    "sp_season_whiff_diff", "sp_l30_whiff_diff", "sp_season_xwoba_allowed_diff",
    "sp_l30_xwoba_allowed_diff", "season_platoon_woba_diff", "l30_platoon_woba_diff",
    "sp_matchup_season_xwoba_allowed_diff", "sp_matchup_season_k_pct_diff",
    "sp_matchup_season_bb_pct_diff", "sp_matchup_season_whiff_pct_diff",
    "sp_matchup_l30_xwoba_allowed_diff", "sp_matchup_l30_k_pct_diff",
    "sp_matchup_l30_bb_pct_diff", "sp_matchup_l30_whiff_pct_diff",
]
STRENGTH = ["sit_run_diff_diff", "sit_run_diff_per_game_diff",
            "sit_pythagorean_win_pct_diff", "sit_actual_minus_pythagorean_diff"]
SPECS = {
    "Market + V5": ("ordinary", V5),
    "Market + V5 + team strength": ("ordinary", V5 + STRENGTH),
    "Market-offset + V5": ("offset", V5),
    "Market-offset + V5 + team strength": ("offset", V5 + STRENGTH),
}
ADJUSTMENT_EDGES = [0, .01, .02, .03, .05, .075, .10, np.inf]
ADJUSTMENT_LABELS = ["<1pp", "1-2pp", "2-3pp", "3-5pp", "5-7.5pp", "7.5-10pp", "10pp+"]


def american_to_decimal(series):
    series = pd.to_numeric(series, errors="coerce")
    valid = series.notna() & series.ne(0) & series.abs().ge(100)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    positive, negative = valid & series.gt(0), valid & series.lt(0)
    result.loc[positive] = 1 + series.loc[positive] / 100
    result.loc[negative] = 1 + 100 / series.loc[negative].abs()
    return result


def logit(probability):
    p = np.clip(np.asarray(probability, float), 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def load_data():
    predictions = pd.read_csv("results/v5_team_strength_oos_predictions_2022_2025.csv")
    odds = pd.read_csv("data/processed/historical_mlb_moneylines_2022_2025.csv")
    odds = odds[odds.sportsbook.eq("bet365") & odds.match_status.eq("matched")].copy()
    if odds.game_id.duplicated().any():
        raise ValueError("Duplicate matched Bet365 game")
    for snapshot in ["opening", "current"]:
        for side in ["home", "away"]:
            odds[f"{snapshot}_{side}_decimal"] = american_to_decimal(odds[f"{snapshot}_{side}_odds"])
        raw_home = 1 / odds[f"{snapshot}_home_decimal"]
        raw_away = 1 / odds[f"{snapshot}_away_decimal"]
        odds[f"{snapshot}_home_no_vig"] = raw_home / (raw_home + raw_away)
    odds = odds.dropna(subset=["opening_home_no_vig", "current_home_no_vig"])
    market_columns = ["game_id", "opening_home_odds", "opening_away_odds", "current_home_odds",
                      "current_away_odds", "opening_home_no_vig", "current_home_no_vig"]
    frames = []
    for year in range(2022, 2026):
        base = pd.read_csv(f"data/processed/games_{year}_starter_lineup_matchup_features.csv")
        situational = pd.read_csv(f"data/processed/features_situational_{year}.csv",
                                  usecols=["game_id"] + STRENGTH)
        frame = base.merge(situational, on="game_id", validate="one_to_one")
        frame = frame.merge(predictions[["game_id"]], on="game_id", validate="one_to_one")
        frame = frame.merge(odds[market_columns], on="game_id", validate="one_to_one")
        frame["season"] = year
        frame["market_logit"] = logit(frame.opening_home_no_vig)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def preprocess(train, validation, features):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_imp = imputer.fit_transform(train[features])
    validation_imp = imputer.transform(validation[features])
    return scaler.fit_transform(train_imp), scaler.transform(validation_imp), imputer, scaler


def fit_ordinary(train, validation, features, c):
    train_z, validation_z, imputer, scaler = preprocess(train, validation, features)
    x_train = np.column_stack([train.market_logit.to_numpy(), train_z])
    x_validation = np.column_stack([validation.market_logit.to_numpy(), validation_z])
    model = LogisticRegression(C=c, max_iter=5000)
    model.fit(x_train, train[TARGET])
    return model.predict_proba(x_validation)[:, 1], model.coef_[0], model.intercept_[0], imputer, scaler


def fit_offset(train, validation, features, c):
    train_z, validation_z, imputer, scaler = preprocess(train, validation, features)
    y = train[TARGET].to_numpy(float)
    baseline = train.market_logit.to_numpy()
    design = np.column_stack([np.ones(len(train_z)), train_z])

    def objective(parameters):
        eta = baseline + design @ parameters
        loss = np.logaddexp(0, eta).sum() - y @ eta
        penalty = .5 / c * np.dot(parameters[1:], parameters[1:])
        probability = expit(eta)
        gradient = design.T @ (probability - y)
        gradient[1:] += parameters[1:] / c
        return loss + penalty, gradient

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Offset optimization failed: {result.message}")
    validation_design = np.column_stack([np.ones(len(validation_z)), validation_z])
    probability = expit(validation.market_logit.to_numpy() + validation_design @ result.x)
    return probability, result.x[1:], result.x[0], imputer, scaler


def metric_values(actual, probability):
    prediction = np.asarray(probability) >= .5
    return {"games": len(actual), "log_loss": log_loss(actual, probability),
            "brier": brier_score_loss(actual, probability), "auc": roc_auc_score(actual, probability),
            "accuracy": accuracy_score(actual, prediction)}


def calibration(actual, probability):
    x = logit(probability).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=3000).fit(x, actual)
    return model.intercept_[0], model.coef_[0, 0]


def model_probability(kind, train, validation, features, c):
    return fit_ordinary(train, validation, features, c) if kind == "ordinary" else fit_offset(train, validation, features, c)


def main():
    data = load_data()
    print("Matched games by season:", data.groupby("season").size().to_dict())
    tuning_rows = []
    for name, (kind, features) in SPECS.items():
        for c in C_GRID:
            for train_years, validation_year in FOLDS:
                train = data[data.season.isin(train_years)]
                validation = data[data.season.eq(validation_year)]
                probability, *_ = model_probability(kind, train, validation, features, c)
                values = metric_values(validation[TARGET], probability)
                tuning_rows.append({"model": name, "C": c, "validation_year": validation_year, **values})
    tuning = pd.DataFrame(tuning_rows)
    tuning_summary = tuning.groupby(["model", "C"], as_index=False).agg(
        mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"),
        mean_auc=("auc", "mean"), mean_accuracy=("accuracy", "mean"))
    winners = (tuning_summary.sort_values(["model", "mean_log_loss", "mean_brier"])
               .groupby("model", as_index=False).first().set_index("model"))
    print("Selected regularization from predefined chronological grid:\n", winners.to_string())

    fold_rows, predictions, coefficient_rows = [], [], []
    for train_years, validation_year in FOLDS:
        train = data[data.season.isin(train_years)]
        validation = data[data.season.eq(validation_year)]
        y = validation[TARGET]
        raw_probability = validation.opening_home_no_vig.to_numpy()
        intercept, slope = calibration(y, raw_probability)
        fold_rows.append({"model": "Raw opening market", "validation_year": validation_year,
                          **metric_values(y, raw_probability), "calibration_intercept": intercept,
                          "calibration_slope": slope})
        predictions.append(pd.DataFrame({"game_id": validation.game_id, "date": validation.date,
                                         "season": validation_year, TARGET: y,
                                         "model": "Raw opening market", "probability": raw_probability}))
        recal = LogisticRegression(C=1e6, max_iter=3000).fit(train[["market_logit"]], train[TARGET])
        recal_probability = recal.predict_proba(validation[["market_logit"]])[:, 1]
        intercept, slope = calibration(y, recal_probability)
        fold_rows.append({"model": "Recalibrated market only", "validation_year": validation_year,
                          **metric_values(y, recal_probability), "calibration_intercept": intercept,
                          "calibration_slope": slope})
        predictions.append(pd.DataFrame({"game_id": validation.game_id, "date": validation.date,
                                         "season": validation_year, TARGET: y,
                                         "model": "Recalibrated market only", "probability": recal_probability}))
        coefficient_rows.append({"model": "Recalibrated market only", "validation_year": validation_year,
                                 "feature": "market_logit", "coefficient": recal.coef_[0, 0]})
        coefficient_rows.append({"model": "Recalibrated market only", "validation_year": validation_year,
                                 "feature": "intercept", "coefficient": recal.intercept_[0]})
        for name, (kind, features) in SPECS.items():
            c = winners.loc[name, "C"]
            probability, coefficients, model_intercept, _, _ = model_probability(
                kind, train, validation, features, c)
            intercept, slope = calibration(y, probability)
            fold_rows.append({"model": name, "validation_year": validation_year,
                              **metric_values(y, probability), "calibration_intercept": intercept,
                              "calibration_slope": slope})
            predictions.append(pd.DataFrame({"game_id": validation.game_id, "date": validation.date,
                                             "season": validation_year, TARGET: y,
                                             "model": name, "probability": probability}))
            names = (["market_logit"] + features) if kind == "ordinary" else features
            for feature, value in zip(names, coefficients):
                coefficient_rows.append({"model": name, "validation_year": validation_year,
                                         "feature": feature, "coefficient": value, "C": c})
            coefficient_rows.append({"model": name, "validation_year": validation_year,
                                     "feature": "intercept", "coefficient": model_intercept, "C": c})
            if kind == "offset":
                coefficient_rows.append({"model": name, "validation_year": validation_year,
                                         "feature": "market_logit_fixed_offset", "coefficient": 1.0, "C": c})

    fold = pd.DataFrame(fold_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    combined_rows = []
    for name, group in prediction_frame.groupby("model"):
        values = metric_values(group[TARGET], group.probability)
        intercept, slope = calibration(group[TARGET], group.probability)
        combined_rows.append({"model": name, "validation_year": "combined", **values,
                              "calibration_intercept": intercept, "calibration_slope": slope})
    all_metrics = pd.concat([fold, pd.DataFrame(combined_rows)], ignore_index=True)
    raw = all_metrics[all_metrics.model.eq("Raw opening market")].set_index("validation_year")
    for metric in ["log_loss", "brier", "auc"]:
        all_metrics[f"delta_{metric}_vs_raw_market"] = all_metrics.apply(
            lambda row: row[metric] - raw.loc[row.validation_year, metric], axis=1)

    baseball_models = list(SPECS)
    combined = all_metrics[all_metrics.validation_year.eq("combined") & all_metrics.model.isin(baseball_models)]
    best_name = combined.sort_values(["log_loss", "brier"]).iloc[0].model
    best_predictions = prediction_frame[prediction_frame.model.eq(best_name)].copy()
    market_lookup = data[["game_id", "opening_home_no_vig", "current_home_no_vig"]]
    best_predictions = best_predictions.merge(market_lookup, on="game_id", validate="one_to_one")
    best_predictions["market_adjustment"] = best_predictions.probability - best_predictions.opening_home_no_vig
    best_predictions["absolute_adjustment"] = best_predictions.market_adjustment.abs()
    best_predictions["adjustment_bucket"] = pd.cut(best_predictions.absolute_adjustment,
                                                    ADJUSTMENT_EDGES, labels=ADJUSTMENT_LABELS,
                                                    right=False, include_lowest=True)
    eps = 1e-15
    y = best_predictions[TARGET].to_numpy()
    p_v9 = np.clip(best_predictions.probability.to_numpy(), eps, 1 - eps)
    p_market = np.clip(best_predictions.opening_home_no_vig.to_numpy(), eps, 1 - eps)
    best_predictions["v9_game_log_loss"] = -(y * np.log(p_v9) + (1 - y) * np.log(1 - p_v9))
    best_predictions["market_game_log_loss"] = -(y * np.log(p_market) + (1 - y) * np.log(1 - p_market))
    best_predictions["v9_game_brier"] = (y - p_v9) ** 2
    best_predictions["market_game_brier"] = (y - p_market) ** 2
    best_predictions["adjustment_direction_correct"] = np.where(
        best_predictions.market_adjustment.gt(0), best_predictions[TARGET].eq(1),
        np.where(best_predictions.market_adjustment.lt(0), best_predictions[TARGET].eq(0), np.nan))
    adjustment_rows = []
    for bucket in ADJUSTMENT_LABELS:
        group = best_predictions[best_predictions.adjustment_bucket.eq(bucket)]
        adjustment_rows.append({"bucket": bucket, "games": len(group),
                                "mean_absolute_adjustment": group.absolute_adjustment.mean(),
                                "mean_log_loss_change_v9_minus_market": (group.v9_game_log_loss - group.market_game_log_loss).mean(),
                                "mean_brier_change_v9_minus_market": (group.v9_game_brier - group.market_game_brier).mean(),
                                "pct_games_log_loss_improved": group.v9_game_log_loss.lt(group.market_game_log_loss).mean(),
                                "pct_adjustment_direction_correct": group.adjustment_direction_correct.mean()})
    adjustment_diagnostics = pd.DataFrame(adjustment_rows)
    direction_rows = []
    for direction, group in [("raises_home_probability", best_predictions[best_predictions.market_adjustment.gt(0)]),
                             ("lowers_home_probability", best_predictions[best_predictions.market_adjustment.lt(0)])]:
        direction_rows.append({"direction": direction, "games": len(group),
                               "market_mean_probability": group.opening_home_no_vig.mean(),
                               "v9_mean_probability": group.probability.mean(),
                               "actual_home_win_rate": group[TARGET].mean(),
                               "mean_log_loss_change_v9_minus_market": (group.v9_game_log_loss - group.market_game_log_loss).mean(),
                               "pct_adjustment_direction_correct": group.adjustment_direction_correct.mean()})
    direction_diagnostics = pd.DataFrame(direction_rows)

    best_predictions["current_market_movement"] = best_predictions.current_home_no_vig - best_predictions.opening_home_no_vig
    nonzero = best_predictions.market_adjustment.ne(0) & best_predictions.current_market_movement.ne(0)
    best_predictions["direction_agrees_with_market_movement"] = np.where(
        nonzero, np.sign(best_predictions.market_adjustment).eq(np.sign(best_predictions.current_market_movement)), np.nan)
    best_predictions["movement_in_v9_direction"] = np.sign(best_predictions.market_adjustment) * best_predictions.current_market_movement
    movement_rows = []
    for bucket, group in [("all", best_predictions)] + [
            (label, best_predictions[best_predictions.adjustment_bucket.eq(label)]) for label in ADJUSTMENT_LABELS]:
        agreement = group.direction_agrees_with_market_movement.dropna()
        agreed = group[group.direction_agrees_with_market_movement.eq(True)]
        movement_rows.append({"adjustment_bucket": bucket, "games": len(group),
                              "direction_comparable_games": len(agreement),
                              "pct_direction_agrees": agreement.mean(),
                              "average_movement_in_v9_direction": group.movement_in_v9_direction.mean(),
                              "average_market_movement_when_agree": agreed.movement_in_v9_direction.mean(),
                              "median_movement_in_v9_direction": group.movement_in_v9_direction.median()})
    movement = pd.DataFrame(movement_rows)

    os.makedirs("results", exist_ok=True)
    tuning.to_csv("results/v9_market_aware_regularization_folds.csv", index=False)
    tuning_summary.to_csv("results/v9_market_aware_regularization_summary.csv", index=False)
    winners.reset_index().to_csv("results/v9_market_aware_selected_regularization.csv", index=False)
    all_metrics.to_csv("results/v9_market_aware_fold_results.csv", index=False)
    prediction_frame.to_csv("results/v9_market_aware_oos_predictions.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv("results/v9_market_aware_coefficients.csv", index=False)
    best_predictions.to_csv("results/v9_best_specification_adjustment_predictions.csv", index=False)
    adjustment_diagnostics.to_csv("results/v9_market_adjustment_buckets.csv", index=False)
    direction_diagnostics.to_csv("results/v9_market_adjustment_direction.csv", index=False)
    movement.to_csv("results/v9_opening_to_current_diagnostics.csv", index=False)

    print("\nFOLD AND COMBINED RESULTS")
    print(all_metrics.sort_values(["validation_year", "log_loss"], key=lambda s: s.astype(str) if s.name == "validation_year" else s).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nBest development-selected baseball specification: {best_name}")
    print("\nMARKET ADJUSTMENT BUCKETS"); print(adjustment_diagnostics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nADJUSTMENT DIRECTION"); print(direction_diagnostics.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nOPENING-TO-CURRENT MARKET MOVEMENT (not verified closing-line movement)")
    print(movement.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nNo betting thresholds, ROI, current-line training, or additional feature families were used.")


if __name__ == "__main__":
    main()
