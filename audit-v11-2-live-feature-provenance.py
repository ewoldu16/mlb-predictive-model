"""Create the machine-readable provenance contract for the frozen V11.2 inputs."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

BUILDERS = {
    "features_v7_offensive_form_YEAR.csv": ("build-richer-offensive-form-features.py", "enriched Statcast PA/contact rows", "game_date < target date"),
    "features_v8_contextual_offense_YEAR.csv": ("build-contextual-offense-features.py", "enriched Statcast plus known starters", "game_date < target date"),
    "features_statsimpl_offense_risp_YEAR.csv": ("build-statsimpl-offense-risp-features.py", "enriched Statcast base state and PA results", "game_date < target date"),
    "features_richer_starter_YEAR.csv": ("build-richer-starter-features.py", "enriched Statcast plus starting-pitcher IDs", "game_date < target date"),
    "features_official_starter_pitching_YEAR.csv": ("build-official-starter-pitching-features.py", "official MLB pitcher game lines", "appearance date < target date"),
    "features_official_bullpen_pitching_YEAR.csv": ("build-official-bullpen-pitching-features.py", "official MLB relief game lines", "appearance date < target date"),
    "features_v6_bullpen_availability_YEAR.csv": ("build-bullpen-availability-features.py", "enriched Statcast relief appearances", "appearance date < target date"),
    "features_statsimpl_starter_recent100_YEAR.csv": ("build-statsimpl-starter-recent100-features.py", "last 100 TBF from enriched Statcast", "PA date < target date"),
    "features_arsenal_lineup_matchup_YEAR.csv": ("build-arsenal-lineup-matchup-features.py", "enriched Statcast, starter pitch mix, actual lineup", "pitch/PA date < target date"),
    "features_opponent_quality_offense_YEAR.csv": ("build-opponent-quality-offense-features.py", "enriched Statcast with pregame opposing-starter quality", "game_date < target date"),
    "features_situational_YEAR.csv": ("build-situational-team-features.py", "official regular-season game results", "game date < target date"),
    "games_YEAR_starter_lineup_matchup_features.csv": ("build-offensive-features.py through build-starter-lineup-matchup-features.py", "regular-season games, full Statcast, actual lineups and starters", "all histories strictly before target date"),
}

def raw_contract(feature):
    name = feature.removeprefix("opp_")
    if name.startswith("ctx_"):
        return "game_pk, game_date, batting team, home/away, starter hand, events, woba_value, woba_denom"
    if name.startswith("sp_matchup_"):
        return "game_pk, game_date, pitcher, stand, events, description, xwOBA/wOBA fields"
    if name.startswith("recent100_"):
        return "game_pk, game_date, pitcher, stand, events, launch_speed, wOBA/xwOBA fields"
    if name.startswith("arsenal_"):
        return "pitcher, batter, pitch_type, stand, velocity/movement, events, contact quality, batting order"
    if name.startswith("bp_official_"):
        return "official pitcher ID/team/role, IP/outs, ER, hits, walks, strikeouts, batters faced"
    if name.startswith("bp_avail_"):
        return "reliever ID/team, prior appearance dates, pitch counts, prior-only role/quality"
    if name.startswith("bp_"):
        return "relief pitcher ID/team, game_date, BF, K/BB/events and contact results"
    if name.startswith("sp_official_"):
        return "official starter ID, game_date, starts, IP/outs, K/BB/HR"
    if name.startswith("sp_"):
        return "starter ID, game_date, pitch results, velocity, xwOBA/contact fields"
    if name.startswith("lineup_"):
        return "confirmed starting hitter IDs/order plus hitter PA wOBA history"
    if name.startswith("oq_"):
        return "team PA results joined to pregame opposing-starter continuous quality"
    if name.startswith("sit_"):
        return "pregame W/L and runs scored/allowed history"
    return "team PA events, wOBA/SLG/contact fields"

def fallback(feature):
    return "Preserve NaN in feature builder; frozen 2021-2024 SimpleImputer median is applied only at inference"

def main():
    spec = json.loads((RESULTS / "v11_2_compact_frozen_specification.json").read_text())
    inventory = pd.read_csv(RESULTS / "v11_2_feature_inventory.csv").set_index("feature")
    rows = []
    for position, feature in enumerate(spec["features"]):
        base = feature.removeprefix("opp_")
        source = inventory.loc[base, "source_dataset"] if base in inventory.index else "games_YEAR_starter_lineup_matchup_features.csv"
        builder, source_detail, cutoff = BUILDERS[source]
        window = inventory.loc[base, "window"] if base in inventory.index else ("L30" if "l30" in base else "season-to-date")
        rows.append({
            "feature_position": position,
            "feature": feature,
            "family": spec["family_mapping"][feature],
            "source_dataset": source,
            "source_script": builder,
            "source_detail": source_detail,
            "required_raw_columns": raw_contract(feature),
            "lookback": window,
            "aggregation": "exact implementation in source_script; no production approximation permitted",
            "orientation": "opponent-owned" if feature.startswith("opp_") else "batting-team-owned",
            "handedness_logic": "starter/batter hand used when encoded by ctx, matchup, recent100 or arsenal family" if any(x in base for x in ("ctx_", "matchup", "recent100", "arsenal")) else "not applicable",
            "lineup_logic": "confirmed nine-hitter order with historical fixed order weights" if "lineup" in base or "arsenal" in base else "not applicable",
            "missing_value_behavior": fallback(feature),
            "pregame_cutoff": cutoff,
            "target_game_excluded": True,
        })
    out = RESULTS / "v11_2_live_feature_provenance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    (RESULTS / "v11_2_live_feature_provenance.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved exact-order provenance for {len(rows)} features: {out}")

if __name__ == "__main__":
    main()
