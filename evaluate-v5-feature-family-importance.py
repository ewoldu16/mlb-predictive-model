import os

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "home_win"
YEARS = [2021, 2022, 2023, 2024]
FOLDS = [
    ([2021], 2022),
    ([2021, 2022], 2023),
    ([2021, 2022, 2023], 2024),
]

FAMILIES = {
    "general offense": [
        "season_woba_diff",
        "l30_woba_diff",
    ],
    "platoon offense": [
        "season_platoon_woba_diff",
        "l30_platoon_woba_diff",
    ],
    "starting-pitcher season quality": [
        "sp_season_k_pct_diff",
        "sp_season_bb_pct_diff",
        "sp_season_woba_allowed_diff",
        "sp_season_velocity_diff",
        "sp_season_whiff_diff",
        "sp_season_xwoba_allowed_diff",
    ],
    "starting-pitcher recent quality": [
        "sp_l30_k_pct_diff",
        "sp_l30_bb_pct_diff",
        "sp_l30_woba_allowed_diff",
        "sp_l30_velocity_diff",
        "sp_l30_whiff_diff",
        "sp_l30_xwoba_allowed_diff",
    ],
    "starter x actual-lineup matchup": [
        "sp_matchup_season_xwoba_allowed_diff",
        "sp_matchup_season_k_pct_diff",
        "sp_matchup_season_bb_pct_diff",
        "sp_matchup_season_whiff_pct_diff",
        "sp_matchup_l30_xwoba_allowed_diff",
        "sp_matchup_l30_k_pct_diff",
        "sp_matchup_l30_bb_pct_diff",
        "sp_matchup_l30_whiff_pct_diff",
    ],
    "bullpen quality/form": [
        "bp_season_k_pct_diff",
        "bp_season_bb_pct_diff",
        "bp_season_woba_allowed_diff",
        "bp_l30_k_pct_diff",
        "bp_l30_bb_pct_diff",
        "bp_l30_woba_allowed_diff",
        "bp_l7_bf_diff",
    ],
    "pitcher workload/rest": [
        "sp_days_rest_diff",
        "sp_prev_pitch_count_diff",
    ],
}

SP_SEASON = "starting-pitcher season quality"
SP_RECENT = "starting-pitcher recent quality"
SP_MATCHUP = "starter x actual-lineup matchup"
SP_FAMILIES = [SP_SEASON, SP_RECENT, SP_MATCHUP]

# Orient each standardized input so a larger value means better home-starter
# quality relative to the away starter.
PITCHER_SIGNS = {
    "sp_season_k_pct_diff": 1,
    "sp_season_bb_pct_diff": -1,
    "sp_season_woba_allowed_diff": -1,
    "sp_season_velocity_diff": 1,
    "sp_season_whiff_diff": 1,
    "sp_season_xwoba_allowed_diff": -1,
    "sp_l30_k_pct_diff": 1,
    "sp_l30_bb_pct_diff": -1,
    "sp_l30_woba_allowed_diff": -1,
    "sp_l30_velocity_diff": 1,
    "sp_l30_whiff_diff": 1,
    "sp_l30_xwoba_allowed_diff": -1,
    "sp_matchup_season_xwoba_allowed_diff": -1,
    "sp_matchup_season_k_pct_diff": 1,
    "sp_matchup_season_bb_pct_diff": -1,
    "sp_matchup_season_whiff_pct_diff": 1,
    "sp_matchup_l30_xwoba_allowed_diff": -1,
    "sp_matchup_l30_k_pct_diff": 1,
    "sp_matchup_l30_bb_pct_diff": -1,
    "sp_matchup_l30_whiff_pct_diff": 1,
}

COMPOSITE_SPECS = {
    "sp_season_quality_composite": FAMILIES[SP_SEASON],
    "sp_recent_quality_composite": FAMILIES[SP_RECENT],
    "sp_matchup_quality_composite": FAMILIES[SP_MATCHUP],
}

V5_FEATURES = [
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
    "bp_l7_bf_diff",
    "sp_days_rest_diff",
    "sp_prev_pitch_count_diff",
    "sp_season_velocity_diff",
    "sp_l30_velocity_diff",
    "sp_season_whiff_diff",
    "sp_l30_whiff_diff",
    "sp_season_xwoba_allowed_diff",
    "sp_l30_xwoba_allowed_diff",
    "season_platoon_woba_diff",
    "l30_platoon_woba_diff",
    "sp_matchup_season_xwoba_allowed_diff",
    "sp_matchup_season_k_pct_diff",
    "sp_matchup_season_bb_pct_diff",
    "sp_matchup_season_whiff_pct_diff",
    "sp_matchup_l30_xwoba_allowed_diff",
    "sp_matchup_l30_k_pct_diff",
    "sp_matchup_l30_bb_pct_diff",
    "sp_matchup_l30_whiff_pct_diff",
]


def validate_feature_groups():
    grouped = [feature for features in FAMILIES.values() for feature in features]
    if len(grouped) != len(set(grouped)):
        raise ValueError("A V5 feature was assigned to more than one family.")
    if set(grouped) != set(V5_FEATURES):
        raise ValueError(
            "Feature-family definitions do not exactly cover V5. "
            f"Missing={set(V5_FEATURES) - set(grouped)}; "
            f"extra={set(grouped) - set(V5_FEATURES)}"
        )


def load_development_data():
    seasons = {}
    print("Loading 2021-2024 development data only...")
    for year in YEARS:
        path = (
            f"data/processed/"
            f"games_{year}_starter_lineup_matchup_features.csv"
        )
        data = pd.read_csv(path)
        required = ["game_id", TARGET] + V5_FEATURES
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing columns for {year}: {missing}")
        if data["game_id"].duplicated().any():
            raise ValueError(f"Duplicate game IDs for {year}.")
        seasons[year] = data
        print(f"{year}: {len(data)} games")
    return seasons


def combine_years(seasons, years):
    return pd.concat([seasons[year] for year in years], ignore_index=True)


def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000)),
    ])


def score_model(name, experiment, features, train, validation, year):
    model = create_model()
    model.fit(train[features], train[TARGET])
    probabilities = model.predict_proba(validation[features])[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    return {
        "experiment": experiment,
        "model": name,
        "validation_year": year,
        "feature_count": len(features),
        "log_loss": log_loss(validation[TARGET], probabilities),
        "brier_score": brier_score_loss(validation[TARGET], probabilities),
        "roc_auc": roc_auc_score(validation[TARGET], probabilities),
        "accuracy": accuracy_score(validation[TARGET], predictions),
    }


def evaluate_feature_set(seasons, name, experiment, features):
    rows = []
    for training_years, validation_year in FOLDS:
        train = combine_years(seasons, training_years)
        validation = seasons[validation_year]
        rows.append(
            score_model(
                name, experiment, features, train, validation, validation_year
            )
        )
    return rows


def summarize(rows):
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["experiment", "model"], as_index=False)
        .agg(
            feature_count=("feature_count", "first"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_accuracy=("accuracy", "mean"),
        )
    )


def add_fold_safe_composites(train, validation):
    train = train.copy()
    validation = validation.copy()
    for composite, source_features in COMPOSITE_SPECS.items():
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_imputed = imputer.fit_transform(train[source_features])
        validation_imputed = imputer.transform(validation[source_features])
        train_scaled = scaler.fit_transform(train_imputed)
        validation_scaled = scaler.transform(validation_imputed)
        signs = np.array([PITCHER_SIGNS[feature] for feature in source_features])
        train[composite] = (train_scaled * signs).mean(axis=1)
        validation[composite] = (validation_scaled * signs).mean(axis=1)
    return train, validation


def evaluate_composite_model(seasons, name, replace_individuals):
    rows = []
    individual_sp = [
        feature for family in SP_FAMILIES for feature in FAMILIES[family]
    ]
    composites = list(COMPOSITE_SPECS)
    features = (
        [feature for feature in V5_FEATURES if feature not in individual_sp]
        + composites
        if replace_individuals
        else V5_FEATURES + composites
    )
    for training_years, validation_year in FOLDS:
        train = combine_years(seasons, training_years)
        validation = seasons[validation_year]
        train, validation = add_fold_safe_composites(train, validation)
        rows.append(
            score_model(
                name,
                "adaptive pitcher composites",
                features,
                train,
                validation,
                validation_year,
            )
        )
    return rows


def metric_line(row):
    return (
        f"Log Loss={row['mean_log_loss']:.6f}  "
        f"Brier={row['mean_brier_score']:.6f}  "
        f"AUC={row['mean_roc_auc']:.6f}  "
        f"Accuracy={row['mean_accuracy']:.6f}"
    )


validate_feature_groups()
seasons = load_development_data()
all_rows = []

# Round 0: fixed baseline, leave-one-family-out ablations, family-alone models,
# and the combined-SP ablation required to choose Case A versus Case C.
all_rows += evaluate_feature_set(
    seasons, "original V5", "baseline", V5_FEATURES
)

for family, family_features in FAMILIES.items():
    retained = [feature for feature in V5_FEATURES if feature not in family_features]
    all_rows += evaluate_feature_set(
        seasons, family, "leave-one-family-out", retained
    )
    all_rows += evaluate_feature_set(
        seasons, family, "family alone", family_features
    )

all_sp_features = [
    feature for family in SP_FAMILIES for feature in FAMILIES[family]
]
without_all_sp = [
    feature for feature in V5_FEATURES if feature not in all_sp_features
]
all_rows += evaluate_feature_set(
    seasons,
    "all starting-pitcher quality families",
    "combined-SP ablation",
    without_all_sp,
)

initial_summary = summarize(all_rows)
baseline = initial_summary[
    (initial_summary["experiment"] == "baseline")
    & (initial_summary["model"] == "original V5")
].iloc[0]

ablation = initial_summary[
    initial_summary["experiment"] == "leave-one-family-out"
].copy()
ablation["log_loss_deterioration"] = (
    ablation["mean_log_loss"] - baseline["mean_log_loss"]
)
ablation["brier_deterioration"] = (
    ablation["mean_brier_score"] - baseline["mean_brier_score"]
)
ablation["auc_change"] = ablation["mean_roc_auc"] - baseline["mean_roc_auc"]
ablation["accuracy_change"] = (
    ablation["mean_accuracy"] - baseline["mean_accuracy"]
)
ablation = ablation.sort_values("log_loss_deterioration", ascending=False)

combined = initial_summary[
    initial_summary["experiment"] == "combined-SP ablation"
].iloc[0]
combined_ll_deterioration = (
    combined["mean_log_loss"] - baseline["mean_log_loss"]
)
combined_auc_drop = baseline["mean_roc_auc"] - combined["mean_roc_auc"]
pitching_important = (
    combined_ll_deterioration >= 0.002 or combined_auc_drop >= 0.005
)

sp_effects = ablation[ablation["model"].isin(SP_FAMILIES)].set_index("model")
season_effect = sp_effects.loc[SP_SEASON, "log_loss_deterioration"]
recent_effect = sp_effects.loc[SP_RECENT, "log_loss_deterioration"]
matchup_effect = sp_effects.loc[SP_MATCHUP, "log_loss_deterioration"]

# "Substantially larger" is fixed before adaptive testing as at least 0.001
# greater Log Loss deterioration than the generic season family.
recent_or_matchup_stronger = (
    max(recent_effect, matchup_effect) >= season_effect + 0.001
)

# Dominance requires a positive effect and at least twice both other SP effects.
positive_sp_effects = {
    family: max(0.0, sp_effects.loc[family, "log_loss_deterioration"])
    for family in SP_FAMILIES
}
dominant_family = max(positive_sp_effects, key=positive_sp_effects.get)
other_effects = [
    value for family, value in positive_sp_effects.items()
    if family != dominant_family
]
one_family_dominates = (
    positive_sp_effects[dominant_family] > 0
    and all(
        positive_sp_effects[dominant_family] >= 2 * value
        for value in other_effects
    )
)

adaptive_rows = []
decision_reason = ""

if not pitching_important:
    decision_case = "CASE C"
    decision_reason = (
        f"Removing all SP quality families changed mean Log Loss by "
        f"{combined_ll_deterioration:+.6f} and AUC by "
        f"{-combined_auc_drop:+.6f}; neither importance threshold was met."
    )
elif one_family_dominates:
    decision_case = "CASE D"
    decision_reason = (
        f"{dominant_family} had at least twice the non-negative Log Loss "
        "deterioration of each other SP family."
    )
    redundant_sp = [
        feature for family in SP_FAMILIES if family != dominant_family
        for feature in FAMILIES[family]
    ]
    dominant_only_features = [
        feature for feature in V5_FEATURES if feature not in redundant_sp
    ]
    adaptive_rows += evaluate_feature_set(
        seasons,
        f"V5 retaining only {dominant_family}",
        "adaptive dominant-family representation",
        dominant_only_features,
    )
elif recent_or_matchup_stronger:
    decision_case = "CASE B"
    decision_reason = (
        "Recent or matchup SP ablation deterioration exceeded the generic "
        "season SP effect by at least 0.001 Log Loss."
    )
    simplified_features = [
        feature for feature in V5_FEATURES
        if feature not in FAMILIES[SP_SEASON]
    ]
    adaptive_rows += evaluate_feature_set(
        seasons,
        "V5 without generic SP season family",
        "adaptive simplified pitcher representation",
        simplified_features,
    )
else:
    decision_case = "CASE A"
    decision_reason = (
        f"Removing all SP quality families worsened mean Log Loss by "
        f"{combined_ll_deterioration:.6f} and changed AUC by "
        f"{-combined_auc_drop:+.6f}, meeting the importance threshold."
    )
    adaptive_rows += evaluate_composite_model(
        seasons, "V5 + pitcher composites", replace_individuals=False
    )
    adaptive_rows += evaluate_composite_model(
        seasons, "V5 replacing individual SP quality with composites",
        replace_individuals=True,
    )

all_rows += adaptive_rows
all_results = pd.DataFrame(all_rows)
all_summary = summarize(all_rows)

print("\n" + "=" * 100)
print("ORIGINAL V5 CHRONOLOGICAL PERFORMANCE")
print("=" * 100)
baseline_folds = all_results[all_results["experiment"] == "baseline"]
print(
    baseline_folds[
        ["validation_year", "log_loss", "brier_score", "roc_auc", "accuracy"]
    ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
)
print("Mean:", metric_line(baseline))

print("\n" + "=" * 100)
print("FEATURE-FAMILY IMPORTANCE RANKING (REMOVAL MINUS V5)")
print("=" * 100)
print(
    ablation[
        [
            "model", "mean_log_loss", "log_loss_deterioration",
            "mean_brier_score", "brier_deterioration", "mean_roc_auc",
            "auc_change", "mean_accuracy", "accuracy_change",
        ]
    ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
)

print("\n" + "=" * 100)
print("EACH-FAMILY-ALONE MODELS")
print("=" * 100)
alone = initial_summary[initial_summary["experiment"] == "family alone"]
print(alone.sort_values("mean_log_loss").to_string(
    index=False, float_format=lambda value: f"{value:.6f}"
))

print("\nCOMBINED SP ABLATION")
print(metric_line(combined))
print("Log Loss deterioration:", f"{combined_ll_deterioration:+.6f}")
print("AUC change:", f"{-combined_auc_drop:+.6f}")

print("\n" + "=" * 100)
print("AUTOMATIC DECISION")
print("=" * 100)
print(decision_case)
print(decision_reason)

original_remains_champion = True
if adaptive_rows:
    adaptive_summary = all_summary[
        all_summary["experiment"].str.startswith("adaptive")
    ].copy()
    adaptive_summary["log_loss_change_vs_v5"] = (
        adaptive_summary["mean_log_loss"] - baseline["mean_log_loss"]
    )
    adaptive_summary["brier_change_vs_v5"] = (
        adaptive_summary["mean_brier_score"] - baseline["mean_brier_score"]
    )
    adaptive_summary["auc_change_vs_v5"] = (
        adaptive_summary["mean_roc_auc"] - baseline["mean_roc_auc"]
    )
    print("\nADAPTIVE EXPERIMENT RESULTS")
    print(adaptive_summary.to_string(
        index=False, float_format=lambda value: f"{value:.6f}"
    ))
    # Material deterioration guardrails are fixed at +0.001 Brier or -0.005 AUC.
    acceptable = adaptive_summary[
        (adaptive_summary["mean_log_loss"] < baseline["mean_log_loss"])
        & (adaptive_summary["brier_change_vs_v5"] <= 0.001)
        & (adaptive_summary["auc_change_vs_v5"] >= -0.005)
    ]
    original_remains_champion = acceptable.empty
    if original_remains_champion:
        recommendation = (
            "Keep original V5: no triggered representation improved mean Log "
            "Loss while satisfying the fixed Brier/AUC safeguards."
        )
    else:
        winner = acceptable.sort_values("mean_log_loss").iloc[0]
        recommendation = (
            f"Promote '{winner['model']}' for further development validation; "
            "it improved mean Log Loss without materially worsening Brier/AUC."
        )
else:
    recommendation = (
        "Do not force additional pitcher weighting. The current SP variables "
        "lack sufficient independent chronological signal; proceed to richer "
        "pitcher information such as arsenal x hitter matchup."
    )

print("\nFINAL RECOMMENDATION")
print(recommendation)
print("Original V5 remains champion:", "YES" if original_remains_champion else "NO")

os.makedirs("results", exist_ok=True)
fold_path = "results/v5_feature_family_importance_fold_results.csv"
summary_path = "results/v5_feature_family_importance_summary.csv"
decision_path = "results/v5_feature_family_importance_decision.txt"
all_results.to_csv(fold_path, index=False)
all_summary.to_csv(summary_path, index=False)
with open(decision_path, "w", encoding="utf-8") as output:
    output.write(f"{decision_case}\n{decision_reason}\n\n{recommendation}\n")
    output.write(
        "Original V5 remains champion: "
        + ("YES" if original_remains_champion else "NO")
        + "\n"
    )

print("\nSaved separate experiment results:")
print(os.path.abspath(fold_path))
print(os.path.abspath(summary_path))
print(os.path.abspath(decision_path))
print("\n2025 was not loaded, inspected, or evaluated.")
