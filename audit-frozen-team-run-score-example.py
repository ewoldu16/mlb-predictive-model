"""Reproduce one fixed-random 2025 frozen Run Score game in full detail."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

SEED = 20250819
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed"
RES = ROOT / "results"


def load_score_module():
    path = ROOT / "build-team-run-score-experiment.py"
    spec = importlib.util.spec_from_file_location("frozen_run_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def merged_wide(module, year):
    frame = module.wide(year)
    extra = pd.read_csv(OUT / f"features_statsimpl_starter_recent100_{year}.csv")
    return frame.merge(extra, on="game_id", validate="one_to_one")


def fmt(value):
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.9f}"


def side_diagnostics(game, side):
    opp = "home" if side == "away" else "away"
    pairs = [
        ("Offense", "season AVG", f"{side}_season_avg"),
        ("Offense", "season OBP", f"{side}_season_obp"),
        ("Offense", "season SLG", f"{side}_season_slg"),
        ("Offense", "L30 wOBA", f"{side}_off_l30_woba"),
        ("Offense", "actual lineup L30 wOBA", f"{side}_lineup_l30_woba"),
        ("RISP", "L30 RISP OPS", f"{side}_l30_risp_ops"),
        ("Starter hand", "vs-hand OPS", None),
        ("Location x hand", "combined OPS", None),
        ("Opposing starter", "season ERA", f"{opp}_sp_official_season_era"),
        ("Opposing starter", "season WHIP", f"{opp}_sp_official_season_whip"),
        ("Opposing starter", "L30 ERA", f"{opp}_sp_official_l30_era"),
        ("Opposing starter", "L30 WHIP", f"{opp}_sp_official_l30_whip"),
        ("Opposing starter", "previous-3-start ERA", f"{opp}_sp_official_recent3_starts_era"),
        ("Opposing starter", "previous-3-start WHIP", f"{opp}_sp_official_recent3_starts_whip"),
        ("Opposing starter", "season xwOBA allowed", f"{opp}_sp_season_xwoba_allowed"),
        ("Opposing starter", "season K%", f"{opp}_sp_official_season_k_pct"),
        ("Opposing starter", "season BB%", f"{opp}_sp_official_season_bb_pct"),
        ("Opposing starter", "season HR/9", f"{opp}_sp_official_season_hr9"),
        ("Opposing starter", "recent100 LHB wOBA allowed", f"{opp}_recent100_lhb_woba_allowed"),
        ("Opposing starter", "recent100 LHB xwOBA allowed", f"{opp}_recent100_lhb_xwoba_allowed"),
        ("Opposing starter", "recent100 LHB K%", f"{opp}_recent100_lhb_k_pct"),
        ("Opposing starter", "recent100 LHB BB%", f"{opp}_recent100_lhb_bb_pct"),
        ("Opposing starter", "recent100 RHB wOBA allowed", f"{opp}_recent100_rhb_woba_allowed"),
        ("Opposing starter", "recent100 RHB xwOBA allowed", f"{opp}_recent100_rhb_xwoba_allowed"),
        ("Opposing starter", "recent100 RHB K%", f"{opp}_recent100_rhb_k_pct"),
        ("Opposing starter", "recent100 RHB BB%", f"{opp}_recent100_rhb_bb_pct"),
        ("Opposing starter", f"season {opp} ERA", f"{opp}_sp_official_season_{opp}_era"),
        ("Opposing starter", f"season {opp} WHIP", f"{opp}_sp_official_season_{opp}_whip"),
        ("Bullpen", "season ERA", f"{opp}_bp_official_season_era"),
        ("Bullpen", "season WHIP", f"{opp}_bp_official_season_whip"),
        ("Bullpen", "L30 ERA", f"{opp}_bp_official_l30_era"),
        ("Bullpen", "L30 WHIP", f"{opp}_bp_official_l30_whip"),
        ("Bullpen", "season K%", f"{opp}_bp_official_season_k_pct"),
        ("Bullpen", "season BB%", f"{opp}_bp_official_season_bb_pct"),
        ("Bullpen", "season wOBA allowed", f"{opp}_bp_season_woba_allowed"),
        ("Bullpen", "season xwOBA allowed", None),
        ("Bullpen", "available-pool ERA", f"{opp}_bp_official_available_pool_era"),
        ("Bullpen", "available-pool WHIP", f"{opp}_bp_official_available_pool_whip"),
        ("Bullpen", "availability mean fatigue", f"{opp}_bp_avail_mean_fatigue"),
        ("Bullpen", "pitches previous 1 day", f"{opp}_bp_avail_w1_pitches"),
        ("Bullpen", "pitches previous 2 days", f"{opp}_bp_avail_w2_pitches"),
        ("Bullpen", "pitches previous 3 days", f"{opp}_bp_avail_w3_pitches"),
        ("Bullpen", "available pool size", f"{opp}_bp_avail_pool_size"),
        ("Bullpen", "rested relievers", f"{opp}_bp_avail_rested"),
        ("Bullpen", "fatigued relievers", f"{opp}_bp_avail_fatigued"),
        ("Bullpen", "used yesterday", f"{opp}_bp_avail_yesterday"),
        ("Bullpen", "heavy recent workload", f"{opp}_bp_avail_heavy"),
    ]
    return [(c, s, col, np.nan if col is None else getattr(game, col)) for c, s, col in pairs]


def main():
    module = load_score_module()
    history = pd.concat([module.long(y) for y in range(2021, 2025)], ignore_index=True)
    score = module.Score().fit(history)
    long25 = module.long(2025)
    wide25 = merged_wide(module, 2025)
    all_inputs = sorted({f for spec in module.COMP.values() for f in spec})
    complete_ids = long25.groupby("game_id").filter(
        lambda x: len(x) == 2 and x.both_starters_15ip_gate.all() and x[all_inputs].notna().all().all()
    ).game_id.unique()
    if not len(complete_ids):
        raise RuntimeError("No fully covered gated 2025 games")
    game_id = int(pd.Series(sorted(complete_ids)).sample(1, random_state=SEED).iloc[0])
    base = long25[long25.game_id.eq(game_id)].reset_index(drop=True)
    transformed = score.transform(base)
    stored = pd.read_csv(RES / "team_run_score_complete_team_games_2021_2025.csv")
    stored = stored[(stored.season.eq(2025)) & (stored.game_id.eq(game_id))].sort_values("team_side")
    wide_game = wide25[wide25.game_id.eq(game_id)].iloc[0]

    rows = []
    for i, teamrow in base.iterrows():
        other = base.iloc[1 - i]
        for component, inputs in module.COMP.items():
            for feature, subweight in inputs.items():
                raw = teamrow[feature]
                imputed = score.med[feature] if pd.isna(raw) else raw
                z = (imputed - score.mean[feature]) / score.sd[feature]
                clipped = np.clip(z, -module.CLIP, module.CLIP)
                variable_points = module.SCALE * module.TOP[component] * subweight * clipped / score.cs[component]
                rows.append({
                    "game_id": game_id, "team_side": teamrow.team_side, "team": teamrow.team,
                    "component": component, "statistic": feature, "raw_value": raw,
                    "imputed_value": imputed if pd.isna(raw) else np.nan,
                    "opponent_analog_value": other[feature], "training_median": score.med[feature],
                    "training_mean": score.mean[feature], "scale_sd": score.sd[feature],
                    "standardized_value": z, "clipped_value": clipped,
                    "direction": "+" if subweight > 0 else "-", "subweight": subweight,
                    "variable_point_term": variable_points,
                })
    detail = pd.DataFrame(rows)
    details_path = RES / "team_run_score_fixed_example_input_audit.csv"
    detail.to_csv(details_path, index=False)

    diagnostic_rows = []
    for side in ("away", "home"):
        team = wide_game[f"{side}_team"]
        for component, statistic, column, value in side_diagnostics(wide_game, side):
            diagnostic_rows.append({"game_id": game_id, "team_side": side, "team": team,
                                    "component": component, "statistic": statistic,
                                    "source_column": column, "raw_value": value,
                                    "used_in_frozen_score": False, "point_contribution": 0.0})
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics_path = RES / "team_run_score_fixed_example_unused_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)

    date = pd.Timestamp(base.date.iloc[0])
    print(f"FIXED SEED: {SEED}")
    print(f"SELECTED GAME: {game_id} | {date.date()} | {wide_game.away_team} at {wide_game.home_team}")
    print("Coverage: both starters >=15 IP and all 32 frozen inputs nonmissing for both teams")
    print("Scaling: training median imputation, then training mean/sample SD; IQR is NOT used; z clip +/-3")
    print(f"Transformation fit: 2021-2024 ({len(history):,} team-games)\n")

    for i, teamrow in base.iterrows():
        side = teamrow.team_side
        print("=" * 150)
        print(f"{teamrow.team} ({side.upper()}) scoring against {teamrow.opponent}")
        print("Component | Statistic | RAW | IMPUTED | Opponent analog | Train median | Train mean | Scale SD | z | clipped z | Dir | subweight | variable point term")
        for r in detail[detail.team_side.eq(side)].itertuples(index=False):
            print(f"{r.component} | {r.statistic} | {fmt(r.raw_value)} | {fmt(r.imputed_value)} | "
                  f"{fmt(r.opponent_analog_value)} | {fmt(r.training_median)} | {fmt(r.training_mean)} | "
                  f"{fmt(r.scale_sd)} | {fmt(r.standardized_value)} | {fmt(r.clipped_value)} | "
                  f"{r.direction} | {fmt(r.subweight)} | {fmt(r.variable_point_term)}")
        print("\nEXACT COMPONENT ARITHMETIC")
        for component in module.COMP:
            raw = transformed.loc[i, f"{component}_raw"]
            cs = score.cs[component]
            cm = score.cm[component]
            std_unclipped = (raw - cm) / cs
            std = transformed.loc[i, f"{component}_standardized"]
            points = transformed.loc[i, f"{component}_point_contribution"]
            terms = detail[(detail.team_side.eq(side)) & (detail.component.eq(component))]
            pieces = " + ".join(f"({r.subweight:.3f}*{r.clipped_value:.6f})" for r in terms.itertuples())
            print(f"{component}: raw={pieces}={raw:.9f}; component z=clip(({raw:.9f}-{cm:.9f})/{cs:.9f})="
                  f"clip({std_unclipped:.9f})={std:.9f}; points=15*{module.TOP[component]:.3f}*{std:.9f}={points:.9f}")
        point_values = [transformed.loc[i, f"{c}_point_contribution"] for c in module.COMP]
        print("Run Score = 50 + " + " + ".join(f"{x:.9f}" for x in point_values) +
              f" = {transformed.loc[i, 'run_score_unclipped']:.9f} -> clipped {transformed.loc[i, 'run_score']:.9f}")
        print("\nAVAILABLE BUT UNUSED DIAGNOSTICS (all contribution=0)")
        print(diagnostics[diagnostics.team_side.eq(side)][["component", "statistic", "source_column", "raw_value"]]
              .to_string(index=False, na_rep="NaN"))

    print("\nSOURCE DATE RANGES (target date excluded in every case)")
    print(f"Season metrics: 2025 season start through {date.date() - pd.Timedelta(days=1)}")
    print(f"L7 metrics: {(date - pd.Timedelta(days=7)).date()} through {date.date() - pd.Timedelta(days=1)}")
    print(f"L30 metrics: {(date - pd.Timedelta(days=30)).date()} through {date.date() - pd.Timedelta(days=1)}")
    print("Previous-3-start metrics: the pitcher's three latest starts with date strictly before the target date.")
    print("Recent100 splits: last 100 total pregame PA faced, then split LHB/RHB (not 100 PA per hand); current date excluded.")
    print("Bullpen workload W1/W2/W3: [date-N days, target date), target date excluded.")
    print("Season venue/hand/combined splits: 2025 season start through day before target; L30 analogues use the L30 range above.")

    # Reveal outcomes last, after all score inputs and arithmetic.
    print("\nACTUAL RESULT (revealed last)")
    print(f"{wide_game.away_team} {int(wide_game.away_score)}, {wide_game.home_team} {int(wide_game.home_score)}")
    print(f"Detailed input audit: {details_path.relative_to(ROOT)}")
    print(f"Unused diagnostic audit: {diagnostics_path.relative_to(ROOT)}")

    # Exact identity check against the frozen saved artifact.
    calc = transformed.assign(team_side=base.team_side).sort_values("team_side")
    if not np.allclose(calc.run_score.to_numpy(), stored.run_score.to_numpy(), atol=1e-12):
        raise RuntimeError("Reconstructed scores do not equal frozen saved scores")


if __name__ == "__main__":
    main()
