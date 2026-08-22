"""Gate live deployment by reconstructing preserved chronological OOS forecasts."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

def load_v11():
    spec = importlib.util.spec_from_file_location("v11", ROOT / "evaluate-v11-unified-team-runs.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def model():
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", PoissonRegressor(alpha=10, max_iter=3000))])

def main():
    frozen = json.loads((RESULTS / "v11_2_compact_frozen_specification.json").read_text())
    features = frozen["features"]; v11 = load_v11()
    data = pd.concat([v11.long_year(year) for year in range(2021, 2026)], ignore_index=True)
    preserved = pd.read_csv(RESULTS / "v11_2_confidence_oos_game_predictions_2022_2025.csv")
    candidates = preserved.copy()
    selections = []; used = set()
    def take(label, ordered):
        row = ordered[~ordered.game_id.isin(used)].iloc[0]; used.add(int(row.game_id)); selections.append((label, row))
    take("ordinary", candidates.iloc[[len(candidates)//2]])
    take("high_confidence", candidates.sort_values("favorite_probability", ascending=False))
    take("low_confidence", candidates.assign(distance=(candidates.favorite_probability-.5).abs()).sort_values("distance"))
    take("early_season", candidates.sort_values("date"))
    take("late_season", candidates.sort_values("date", ascending=False))
    # Handedness examples are tied to actual matchup-source coverage rather than inferred names.
    matchup = data[data.game_id.isin(preserved.game_id)].dropna(subset=["opp_recent100_lhb_woba_allowed", "opp_recent100_rhb_woba_allowed"])
    matched = preserved[preserved.game_id.isin(matchup.game_id)]
    take("left_hand_matchup", matched.sort_values("game_id"))
    take("right_hand_matchup", matched.sort_values("game_id", ascending=False))
    rows = []
    fitted = {}
    for label, selected in selections:
        year = int(selected.season)
        if year not in fitted:
            train = data[data.season.lt(year)]; fitted[year] = model().fit(train[features], train.actual_team_runs)
        game = data[data.game_id.eq(selected.game_id)]
        prediction = dict(zip(game.team_side, np.clip(fitted[year].predict(game[features]), .05, None)))
        for side in ("away", "home"):
            expected = float(selected[f"predicted_runs_{side}"]); actual = float(prediction[side])
            rows.append({"case": label, "game_id": int(selected.game_id), "season": year, "team_side": side, "reconstructed": actual, "preserved": expected, "absolute_difference": abs(actual-expected), "feature_count": len(features), "feature_order_exact": list(game[features].columns)==features})
    audit = pd.DataFrame(rows); out = RESULTS / "v11_2_production_historical_reconstruction.csv"; audit.to_csv(out, index=False)
    maximum = audit.absolute_difference.max(); passed = bool(audit.feature_order_exact.all() and maximum <= 1e-10)
    summary = {"games": audit.game_id.nunique(), "team_predictions": len(audit), "maximum_absolute_difference": maximum, "tolerance": 1e-10, "passed": passed}
    (RESULTS / "v11_2_production_historical_reconstruction_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not passed: raise SystemExit("RECONSTRUCTION GATE FAILED: live prediction generation must remain disabled")

if __name__ == "__main__":
    main()
