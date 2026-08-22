"""V10 COMPONENT ADVANTAGE SCORE: interpretable, baseball-only architecture."""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEV_FOLDS = [([2021], 2022), ([2021, 2022], 2023), ([2021, 2022, 2023], 2024)]
TARGET = "home_win"
WEIGHT_APPROACHES = {"equal_weights": None, "nonnegative_ridge_C_0.03": .03,
                     "nonnegative_ridge_C_0.1": .1, "nonnegative_ridge_C_0.3": .3,
                     "nonnegative_ridge_C_1": 1.0}
ROBUST_CAP = 3.0
TARGET_COMPONENT_SD_POINTS = 12.0
BUCKET_EDGES = [0, 5, 10, 15, 20, 25, 30, np.inf]
BUCKET_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]

# Each sign makes a larger oriented value a home-team advantage.
COMPONENTS = {
    "SP_SCORE": {
        "sp_season_k_pct_diff": 1, "sp_season_bb_pct_diff": -1,
        "sp_season_xwoba_allowed_diff": -1, "sp_l30_k_pct_diff": 1,
        "sp_l30_bb_pct_diff": -1, "sp_l30_xwoba_allowed_diff": -1,
        "sp_season_whiff_diff": 1, "sp_l30_whiff_diff": 1,
        "sp_days_rest_diff": 1, "sp_prev_pitch_count_diff": -1,
    },
    "OFFENSE_SCORE": {"season_woba_diff": 1, "l30_woba_diff": 1},
    "BULLPEN_SCORE": {
        "bp_season_k_pct_diff": 1, "bp_season_bb_pct_diff": -1,
        "bp_season_woba_allowed_diff": -1, "bp_l30_k_pct_diff": 1,
        "bp_l30_bb_pct_diff": -1, "bp_l30_woba_allowed_diff": -1,
        "bp_l7_bf_diff": -1,
    },
    "LINEUP_MATCHUP_SCORE": {
        "season_platoon_woba_diff": 1, "l30_platoon_woba_diff": 1,
        "sp_matchup_season_xwoba_allowed_diff": -1, "sp_matchup_season_k_pct_diff": 1,
        "sp_matchup_season_bb_pct_diff": -1, "sp_matchup_season_whiff_pct_diff": 1,
        "sp_matchup_l30_xwoba_allowed_diff": -1, "sp_matchup_l30_k_pct_diff": 1,
        "sp_matchup_l30_bb_pct_diff": -1, "sp_matchup_l30_whiff_pct_diff": 1,
    },
    "TEAM_STRENGTH_SCORE": {
        "sit_run_diff_per_game_diff": 1,
        "sit_actual_minus_pythagorean_diff": 1,
    },
}
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


def load_year(year):
    base = pd.read_csv(f"data/processed/games_{year}_starter_lineup_matchup_features.csv")
    strength = pd.read_csv(f"data/processed/features_situational_{year}.csv",
                           usecols=["game_id", "sit_run_diff_per_game_diff",
                                    "sit_actual_minus_pythagorean_diff"])
    if base.game_id.duplicated().any() or strength.game_id.duplicated().any():
        raise ValueError(f"Duplicate game ID in {year}")
    frame = base.merge(strength, on="game_id", validate="one_to_one")
    frame["season"] = year
    return frame


def robust_components(train, validation):
    train_scores, validation_scores, definitions = {}, {}, []
    for component, mapping in COMPONENTS.items():
        train_terms, validation_terms = [], []
        for feature, direction in mapping.items():
            train_values = pd.to_numeric(train[feature], errors="coerce")
            validation_values = pd.to_numeric(validation[feature], errors="coerce")
            median = train_values.median()
            q1, q3 = train_values.quantile([.25, .75])
            scale = (q3 - q1) / 1.349
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = (train_values - median).abs().median() * 1.4826
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            train_z = ((train_values.fillna(median) - median) / scale).clip(-ROBUST_CAP, ROBUST_CAP) * direction
            validation_z = ((validation_values.fillna(median) - median) / scale).clip(-ROBUST_CAP, ROBUST_CAP) * direction
            train_terms.append(train_z.to_numpy()); validation_terms.append(validation_z.to_numpy())
            definitions.append({"component": component, "feature": feature, "direction": direction,
                                "training_median": median, "training_robust_scale": scale,
                                "winsorization_cap": ROBUST_CAP})
        train_scores[component] = np.mean(train_terms, axis=0)
        validation_scores[component] = np.mean(validation_terms, axis=0)
    names = list(COMPONENTS)
    return (np.column_stack([train_scores[name] for name in names]),
            np.column_stack([validation_scores[name] for name in names]), names,
            pd.DataFrame(definitions))


def learn_weights(components, outcome, c):
    if c is None:
        return np.ones(components.shape[1])
    y = np.asarray(outcome, float)
    design = np.column_stack([np.ones(len(components)), components])

    def objective(parameters):
        eta = design @ parameters
        loss = np.logaddexp(0, eta).sum() - y @ eta
        gradient = design.T @ (expit(eta) - y)
        loss += .5 / c * np.dot(parameters[1:], parameters[1:])
        gradient[1:] += parameters[1:] / c
        return loss, gradient

    bounds = [(None, None)] + [(0, None)] * components.shape[1]
    result = minimize(objective, np.r_[0.0, np.ones(components.shape[1]) * .1],
                      jac=True, bounds=bounds, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x[1:]


def fit_calibrator(scores, outcome, method):
    if method == "logistic":
        model = LogisticRegression(C=1e6, max_iter=3000).fit(scores.reshape(-1, 1), outcome)
        return lambda values: model.predict_proba(np.asarray(values).reshape(-1, 1))[:, 1], model
    model = IsotonicRegression(out_of_bounds="clip").fit(scores, outcome)
    return lambda values: model.predict(np.asarray(values)), model


def metrics(outcome, probability):
    return {"games": len(outcome), "log_loss": log_loss(outcome, probability),
            "brier": brier_score_loss(outcome, probability),
            "auc": roc_auc_score(outcome, probability),
            "accuracy": accuracy_score(outcome, np.asarray(probability) >= .5)}


def calibration_stats(outcome, probability):
    p = np.clip(np.asarray(probability), 1e-8, 1 - 1e-8)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=3000).fit(x, outcome)
    return model.intercept_[0], model.coef_[0, 0]


def points_scale(raw_training_score):
    scale = np.std(raw_training_score, ddof=0)
    return TARGET_COMPONENT_SD_POINTS / scale if scale > 1e-12 else 1.0


def score_frame(validation, validation_components, names, weights, factor, probability, fold):
    result = validation[["game_id", "date", "season", "away_team", "home_team", TARGET]].copy()
    for index, name in enumerate(names):
        result[name] = validation_components[:, index] * weights[index] * factor
    result["TOTAL_SCORE"] = result[names].sum(axis=1)
    result["v10_home_win_probability"] = probability
    result["score_favoured_team"] = np.where(result.TOTAL_SCORE.ge(0), result.home_team, result.away_team)
    result["score_favoured_won"] = np.where(result.TOTAL_SCORE.ge(0), result[TARGET].eq(1), result[TARGET].eq(0)).astype(int)
    result["fold"] = fold
    return result


def bucket_summary(scores, scope):
    work = scores.copy()
    work["score_bucket"] = pd.cut(work.TOTAL_SCORE.abs(), BUCKET_EDGES, labels=BUCKET_LABELS,
                                  right=False, include_lowest=True)
    rows = []
    for bucket in BUCKET_LABELS:
        group = work[work.score_bucket.eq(bucket)]
        n = len(group); rate = group.score_favoured_won.mean() if n else np.nan
        se = np.sqrt(rate * (1 - rate) / n) if n else np.nan
        rows.append({"scope": scope, "score_bucket": bucket, "games": n,
                     "wins": int(group.score_favoured_won.sum()), "losses": n - int(group.score_favoured_won.sum()),
                     "win_rate": rate, "average_absolute_score": group.TOTAL_SCORE.abs().mean(),
                     "standard_error": se, "ci95_low": max(0, rate - 1.96 * se) if n else np.nan,
                     "ci95_high": min(1, rate + 1.96 * se) if n else np.nan})
    return rows


def agreement_summary(scores, scope):
    components = list(COMPONENTS)
    signs = np.sign(scores[components])
    total_sign = np.sign(scores.TOTAL_SCORE)
    abs_components = scores[components].abs()
    conditions = {
        "all_major_components_agree": (signs.eq(signs.iloc[:, 0], axis=0).all(axis=1) & signs.ne(0).all(axis=1)),
        "starter_and_offense_agree": signs.SP_SCORE.eq(signs.OFFENSE_SCORE) & signs.SP_SCORE.ne(0),
        "starter_strongly_disagrees_with_offense": signs.SP_SCORE.ne(signs.OFFENSE_SCORE)
            & scores.SP_SCORE.abs().ge(5) & scores.OFFENSE_SCORE.abs().ge(5),
        "bullpen_is_deciding_component": np.sign(scores.TOTAL_SCORE - scores.BULLPEN_SCORE).ne(total_sign)
            & np.sign(scores.BULLPEN_SCORE).eq(total_sign),
        "one_component_over_half_absolute_contribution": abs_components.max(axis=1).div(abs_components.sum(axis=1)).gt(.5),
    }
    rows = []
    for label, condition in conditions.items():
        group = scores[condition]; n = len(group)
        rows.append({"scope": scope, "diagnostic": label, "games": n,
                     "favoured_team_wins": int(group.score_favoured_won.sum()),
                     "favoured_team_win_rate": group.score_favoured_won.mean() if n else np.nan,
                     "average_absolute_score": group.TOTAL_SCORE.abs().mean() if n else np.nan})
    return rows


def main():
    audit = [
        ("starting pitcher quality/recent/workload", "included", "Selected nonredundant rate, contact-quality, rest and workload measures"),
        ("offense", "included", "Frozen season and L30 wOBA"),
        ("bullpen quality/form", "included", "Frozen V5 bullpen rates and recent workload"),
        ("platoon and starter-lineup matchup", "included", "Frozen V5 pregame matchup differentials"),
        ("underlying team strength", "included", "Run differential per game plus actual-minus-Pythagorean residual"),
        ("raw run differential", "excluded_redundant", "Duplicates scale-adjusted run differential per game"),
        ("starter wOBA allowed", "excluded_redundant", "xwOBA allowed retained instead"),
        ("starter velocity", "excluded", "Not directionally equivalent to pitcher quality"),
        ("actual lineup quality", "excluded_prior_validation", "V4 did not improve chronological/OOS performance"),
        ("bullpen availability", "excluded_prior_validation", "Independent family evaluation rejected V6"),
        ("richer offensive form", "excluded_prior_validation", "V7 rejected"),
        ("contextual offense", "excluded", "Not part of frozen champion and not consistently validated"),
        ("richer starter", "excluded_prior_validation", "Independent family evaluation rejected it"),
        ("opponent-quality offense", "excluded_prior_validation", "Independent family evaluation rejected it"),
        ("arsenal-lineup matchup", "excluded_prior_validation", "Independent family evaluation rejected it"),
    ]
    audit = pd.DataFrame(audit, columns=["information_family", "decision", "reason"])
    dev = {year: load_year(year) for year in range(2021, 2025)}
    approach_rows, fold_cache, weight_rows = [], {}, []
    for approach, c in WEIGHT_APPROACHES.items():
        for train_years, validation_year in DEV_FOLDS:
            train = pd.concat([dev[year] for year in train_years], ignore_index=True)
            validation = dev[validation_year]
            train_c, validation_c, names, definitions = robust_components(train, validation)
            weights = learn_weights(train_c, train[TARGET], c)
            raw_train, raw_validation = train_c @ weights, validation_c @ weights
            factor = points_scale(raw_train)
            for name, weight in zip(names, weights):
                weight_rows.append({"approach": approach, "validation_year": validation_year,
                                    "component": name, "weight": weight})
            for calibration_method in ["logistic", "isotonic"]:
                predict, _ = fit_calibrator(raw_train, train[TARGET], calibration_method)
                probability = np.clip(predict(raw_validation), 1e-6, 1 - 1e-6)
                approach_rows.append({"approach": approach, "calibration_method": calibration_method,
                                      "validation_year": validation_year, **metrics(validation[TARGET], probability)})
                fold_cache[(approach, calibration_method, validation_year)] = (
                    validation, validation_c, names, weights, factor, probability)
    approach_results = pd.DataFrame(approach_rows)
    logistic_summary = (approach_results[approach_results.calibration_method.eq("logistic")]
                        .groupby("approach", as_index=False).agg(mean_log_loss=("log_loss", "mean"),
                        mean_brier=("brier", "mean"), mean_auc=("auc", "mean")))
    chosen_approach = logistic_summary.sort_values(["mean_log_loss", "mean_brier"]).iloc[0].approach
    calibration_summary = (approach_results[approach_results.approach.eq(chosen_approach)]
                           .groupby("calibration_method", as_index=False).agg(
                               mean_log_loss=("log_loss", "mean"), mean_brier=("brier", "mean"),
                               mean_auc=("auc", "mean"), mean_accuracy=("accuracy", "mean")))
    chosen_calibration = calibration_summary.sort_values(["mean_log_loss", "mean_brier"]).iloc[0].calibration_method
    print("Frozen from 2021-2024 development:", chosen_approach, "+", chosen_calibration)

    dev_scores = []
    for _, validation_year in DEV_FOLDS:
        cached = fold_cache[(chosen_approach, chosen_calibration, validation_year)]
        dev_scores.append(score_frame(*cached, fold=f"validate_{validation_year}"))
    dev_scores = pd.concat(dev_scores, ignore_index=True)
    dev_fold_results = approach_results[(approach_results.approach.eq(chosen_approach))
                                        & (approach_results.calibration_method.eq(chosen_calibration))].copy()
    dev_fold_results["calibration_intercept"] = np.nan
    dev_fold_results["calibration_slope"] = np.nan
    for validation_year in [2022, 2023, 2024]:
        fold_scores = dev_scores[dev_scores.season.eq(validation_year)]
        fold_intercept, fold_slope = calibration_stats(
            fold_scores[TARGET], fold_scores.v10_home_win_probability)
        dev_fold_results.loc[
            dev_fold_results.validation_year.eq(validation_year),
            ["calibration_intercept", "calibration_slope"],
        ] = [fold_intercept, fold_slope]
    combined_probability = dev_scores.v10_home_win_probability
    combined_metrics = metrics(dev_scores[TARGET], combined_probability)
    ci, cs = calibration_stats(dev_scores[TARGET], combined_probability)
    combined_metrics.update({"approach": chosen_approach, "calibration_method": chosen_calibration,
                             "validation_year": "combined_2022_2024", "calibration_intercept": ci,
                             "calibration_slope": cs})

    bucket_rows, agreement_rows = [], []
    for year in [2022, 2023, 2024]:
        subset = dev_scores[dev_scores.season.eq(year)]
        bucket_rows += bucket_summary(subset, str(year)); agreement_rows += agreement_summary(subset, str(year))
    bucket_rows += bucket_summary(dev_scores, "combined_2022_2024")
    agreement_rows += agreement_summary(dev_scores, "combined_2022_2024")

    # Architecture is now frozen. Load and evaluate the untouched 2025 season once.
    train = pd.concat([dev[year] for year in range(2021, 2025)], ignore_index=True)
    validation_2025 = load_year(2025)
    train_c, validation_c, names, definitions = robust_components(train, validation_2025)
    chosen_c = WEIGHT_APPROACHES[chosen_approach]
    final_weights = learn_weights(train_c, train[TARGET], chosen_c)
    raw_train, raw_2025 = train_c @ final_weights, validation_c @ final_weights
    factor = points_scale(raw_train)
    predict, calibrator = fit_calibrator(raw_train, train[TARGET], chosen_calibration)
    probability_2025 = np.clip(predict(raw_2025), 1e-6, 1 - 1e-6)
    scores_2025 = score_frame(validation_2025, validation_c, names, final_weights, factor,
                              probability_2025, "untouched_2025")
    v10_2025 = metrics(validation_2025[TARGET], probability_2025)
    ci, cs = calibration_stats(validation_2025[TARGET], probability_2025)
    v10_2025.update({"model": "V10 COMPONENT ADVANTAGE SCORE", "calibration_intercept": ci,
                     "calibration_slope": cs})
    v5_model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler()),
                         ("model", LogisticRegression(max_iter=3000))])
    v5_model.fit(train[V5], train[TARGET]); v5_probability = v5_model.predict_proba(validation_2025[V5])[:, 1]
    v5_2025 = metrics(validation_2025[TARGET], v5_probability)
    ci, cs = calibration_stats(validation_2025[TARGET], v5_probability)
    v5_2025.update({"model": "Frozen V5", "calibration_intercept": ci, "calibration_slope": cs})
    untouched_results = pd.DataFrame([v5_2025, v10_2025])
    bucket_2025 = pd.DataFrame(bucket_summary(scores_2025, "untouched_2025"))
    agreement_2025 = pd.DataFrame(agreement_summary(scores_2025, "untouched_2025"))
    final_weight_rows = []
    for name, weight in zip(names, final_weights):
        final_weight_rows.append({"component": name, "raw_weight": weight,
                                  "point_multiplier": weight * factor})
    final_weights_frame = pd.DataFrame(final_weight_rows)
    weight_stability = (pd.DataFrame(weight_rows).query("approach == @chosen_approach")
                        .groupby("component", as_index=False).agg(mean_weight=("weight", "mean"),
                        std_weight=("weight", "std"), min_weight=("weight", "min"),
                        max_weight=("weight", "max")))

    os.makedirs("results", exist_ok=True)
    audit.to_csv("results/v10_information_audit.csv", index=False)
    definitions.to_csv("results/v10_component_definitions.csv", index=False)
    approach_results.to_csv("results/v10_weighting_and_calibration_folds.csv", index=False)
    logistic_summary.to_csv("results/v10_weighting_approach_summary.csv", index=False)
    calibration_summary.to_csv("results/v10_calibration_method_summary.csv", index=False)
    pd.DataFrame(weight_rows).to_csv("results/v10_component_weights_by_fold.csv", index=False)
    final_weights_frame.to_csv("results/v10_final_component_weights.csv", index=False)
    weight_stability.to_csv("results/v10_component_weight_stability.csv", index=False)
    pd.concat([dev_fold_results, pd.DataFrame([combined_metrics])], ignore_index=True).to_csv(
        "results/v10_chronological_fold_results.csv", index=False)
    dev_scores.to_csv("results/v10_development_oos_game_scores.csv", index=False)
    pd.DataFrame(bucket_rows).to_csv("results/v10_development_score_buckets.csv", index=False)
    pd.DataFrame(agreement_rows).to_csv("results/v10_development_component_agreement.csv", index=False)
    scores_2025.to_csv("results/v10_2025_component_breakdown.csv", index=False)
    untouched_results.to_csv("results/v10_untouched_2025_results.csv", index=False)
    bucket_2025.to_csv("results/v10_untouched_2025_score_buckets.csv", index=False)
    agreement_2025.to_csv("results/v10_untouched_2025_component_agreement.csv", index=False)
    with open("results/v10_frozen_specification.json", "w", encoding="utf-8") as handle:
        json.dump({"weighting_approach": chosen_approach, "calibration_method": chosen_calibration,
                   "robust_cap": ROBUST_CAP, "target_score_sd": TARGET_COMPONENT_SD_POINTS,
                   "components": COMPONENTS}, handle, indent=2)

    print("\nWEIGHTING APPROACHES"); print(logistic_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCALIBRATION METHODS"); print(calibration_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nFINAL COMPONENT WEIGHTS"); print(final_weights_frame.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nWEIGHT STABILITY"); print(weight_stability.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nDEVELOPMENT OOS BUCKETS"); print(pd.DataFrame(bucket_rows).query("scope == 'combined_2022_2024'").to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nDEVELOPMENT AGREEMENT"); print(pd.DataFrame(agreement_rows).query("scope == 'combined_2022_2024'").to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nUNTOUCHED 2025 RESULTS"); print(untouched_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nUNTOUCHED 2025 BUCKETS"); print(bucket_2025.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nV10 was not modified after the 2025 evaluation. No sportsbook data or ROI was used.")


if __name__ == "__main__":
    main()
