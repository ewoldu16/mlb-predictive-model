"""Kelly bankroll diagnostics on the fixed Bet365 >=55% backtest ledger."""
import os

import numpy as np
import pandas as pd


INPUT = "results/backtest_bet365_55pct_game_ledger.csv"
STARTING_BANKROLL = 100.0
SHRINKAGE_SLOPE = 0.7629
METHODS = {
    "full_kelly": (1.0, None),
    "half_kelly": (0.5, None),
    "quarter_kelly": (0.25, None),
    "eighth_kelly": (0.125, None),
    "tenth_kelly": (0.10, None),
    "quarter_kelly_capped_2pct": (0.25, 0.02),
    "eighth_kelly_capped_2pct": (0.125, 0.02),
    "tenth_kelly_capped_2pct": (0.10, 0.02),
}


def simulate(frame, probability_variant, probability_column, method, multiplier, cap, scope):
    frame = frame.sort_values(["date", "game_id"]).copy()
    bankroll = STARTING_BANKROLL
    maximum_bankroll = bankroll
    minimum_bankroll = bankroll
    peak = bankroll
    maximum_drawdown_pct = 0.0
    rows = []
    for game in frame.itertuples(index=False):
        probability = float(getattr(game, probability_column))
        odds = float(game.selected_current_decimal_odds)
        full_kelly = (odds * probability - 1) / (odds - 1)
        applied_fraction = max(0.0, full_kelly) * multiplier
        if cap is not None:
            applied_fraction = min(applied_fraction, cap)
        before = bankroll
        stake = before * applied_fraction
        profit = stake * (odds - 1) if game.result == 1 else -stake
        bankroll = before + profit
        peak = max(peak, bankroll)
        if peak > 0:
            maximum_drawdown_pct = max(maximum_drawdown_pct, (peak - bankroll) / peak)
        maximum_bankroll = max(maximum_bankroll, bankroll)
        minimum_bankroll = min(minimum_bankroll, bankroll)
        rows.append({
            "simulation_scope": scope,
            "probability_variant": probability_variant,
            "staking_method": method,
            "game_id": game.game_id,
            "date": game.date,
            "season": game.season,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "model_selected_team": game.model_selected_team,
            "bankroll_before": before,
            "model_probability_raw": game.model_probability,
            "probability_used": probability,
            "decimal_odds": odds,
            "full_kelly_fraction": full_kelly,
            "applied_stake_fraction": applied_fraction,
            "stake_units": stake,
            "result": game.result,
            "profit_units": profit,
            "bankroll_after": bankroll,
        })
    ledger = pd.DataFrame(rows)
    bets = ledger.applied_stake_fraction.gt(0)
    bet_fractions = ledger.loc[bets, "applied_stake_fraction"]
    summary = {
        "scope": scope,
        "probability_variant": probability_variant,
        "staking_method": method,
        "starting_bankroll": STARTING_BANKROLL,
        "ending_bankroll": bankroll,
        "total_return_pct": (bankroll / STARTING_BANKROLL - 1) * 100,
        "eligible_games": len(ledger),
        "number_of_bets": int(bets.sum()),
        "passes_nonpositive_kelly": int((ledger.full_kelly_fraction <= 0).sum()),
        "average_stake_pct_bankroll": bet_fractions.mean() * 100,
        "median_stake_pct_bankroll": bet_fractions.median() * 100,
        "maximum_stake_pct_bankroll": bet_fractions.max() * 100,
        "maximum_bankroll_drawdown_pct": maximum_drawdown_pct * 100,
        "minimum_bankroll": minimum_bankroll,
        "maximum_bankroll": maximum_bankroll,
    }
    return summary, ledger


def safe_filename(value):
    return value.replace("/", "_").replace(" ", "_")


def main():
    games = pd.read_csv(INPUT)
    required = {"game_id", "date", "season", "model_probability",
                "selected_current_decimal_odds", "result"}
    missing = sorted(required - set(games.columns))
    if missing:
        raise ValueError(f"Missing ledger columns: {missing}")
    games["date"] = pd.to_datetime(games.date)
    games["probability_raw"] = games.model_probability
    games["probability_shrunk"] = 0.5 + SHRINKAGE_SLOPE * (games.model_probability - 0.5)
    if not games[["probability_raw", "probability_shrunk"]].apply(lambda x: x.between(0, 1)).all().all():
        raise ValueError("Probability outside [0,1]")

    scopes = [(str(year), games[games.season.eq(year)]) for year in range(2022, 2026)]
    scopes.append(("combined_2022_2025", games))
    variants = [("raw_probability", "probability_raw"),
                ("shrunk_probability_slope_0_7629", "probability_shrunk")]
    summaries, ledgers = [], []
    for scope, scoped in scopes:
        for variant, probability_column in variants:
            for method, (multiplier, cap) in METHODS.items():
                summary, ledger = simulate(
                    scoped, variant, probability_column, method, multiplier, cap, scope
                )
                summaries.append(summary)
                ledgers.append(ledger)
    summary = pd.DataFrame(summaries)
    all_ledgers = pd.concat(ledgers, ignore_index=True)

    os.makedirs("results/kelly_ledgers", exist_ok=True)
    summary.to_csv("results/kelly_bankroll_simulation_summary.csv", index=False)
    all_ledgers.to_csv("results/kelly_bankroll_simulation_game_ledger.csv", index=False)
    for (variant, method), ledger in all_ledgers.groupby(["probability_variant", "staking_method"]):
        ledger.to_csv(
            f"results/kelly_ledgers/{safe_filename(variant)}__{safe_filename(method)}.csv",
            index=False,
        )

    combined = summary[summary.scope.eq("combined_2022_2025")]
    yearly = summary[~summary.scope.eq("combined_2022_2025")]
    print("RAW-PROBABILITY KELLY — COMBINED")
    print(combined[combined.probability_variant.eq("raw_probability")].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nSHRUNK-PROBABILITY KELLY — COMBINED")
    print(combined[combined.probability_variant.str.startswith("shrunk")].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nYEARLY SIMULATIONS — EACH STARTS AT 100 UNITS")
    print(yearly.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nProbability of ruin was not estimated from this single historical path.")
    print("No model, probability, shrinkage coefficient, or Kelly fraction was optimized.")


if __name__ == "__main__":
    main()
