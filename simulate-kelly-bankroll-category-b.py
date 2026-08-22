"""Exact Kelly simulation for predefined Category B selections only."""
import os

import numpy as np
import pandas as pd


INPUT = "results/backtest_bet365_55pct_game_ledger.csv"
STARTING_BANKROLL = 100.0
SHRINKAGE_SLOPE = 0.7629
CATEGORY_B_MIN_CONFIDENCE = 0.55
CATEGORY_B_MIN_EDGE = 0.04
METHODS = {
    "full_kelly": (1.0, None), "half_kelly": (0.5, None),
    "quarter_kelly": (0.25, None), "eighth_kelly": (0.125, None),
    "tenth_kelly": (0.10, None),
    "quarter_kelly_capped_2pct": (0.25, 0.02),
    "eighth_kelly_capped_2pct": (0.125, 0.02),
    "tenth_kelly_capped_2pct": (0.10, 0.02),
}


def max_drawdown_pct(path):
    values = np.r_[STARTING_BANKROLL, np.asarray(path, dtype=float)]
    peaks = np.maximum.accumulate(values)
    return float(np.max((peaks - values) / peaks) * 100)


def simulate(frame, variant, probability_column, method, multiplier, cap, scope):
    frame = frame.sort_values(["date", "game_id"])
    bankroll = STARTING_BANKROLL
    minimum = maximum = bankroll
    rows = []
    for game in frame.itertuples(index=False):
        probability = float(getattr(game, probability_column))
        odds = float(game.selected_current_decimal_odds)
        full_kelly = (odds * probability - 1) / (odds - 1)
        fraction = max(0.0, full_kelly) * multiplier
        if cap is not None:
            fraction = min(fraction, cap)
        before = bankroll
        stake = before * fraction
        profit = stake * (odds - 1) if game.result == 1 else -stake
        bankroll += profit
        minimum, maximum = min(minimum, bankroll), max(maximum, bankroll)
        rows.append({
            "simulation_scope": scope, "probability_variant": variant,
            "staking_method": method, "game_id": game.game_id, "date": game.date,
            "season": game.season, "home_team": game.home_team, "away_team": game.away_team,
            "model_selected_team": game.model_selected_team, "model_edge": game.model_edge,
            "bankroll_before": before, "model_probability_raw": game.model_probability,
            "probability_used": probability, "decimal_odds": odds,
            "full_kelly_fraction": full_kelly, "applied_stake_fraction": fraction,
            "stake_units": stake, "result": game.result, "profit_units": profit,
            "bankroll_after": bankroll,
        })
    ledger = pd.DataFrame(rows)
    bets = ledger.applied_stake_fraction.gt(0)
    fractions = ledger.loc[bets, "applied_stake_fraction"]
    return {
        "scope": scope, "probability_variant": variant, "staking_method": method,
        "starting_bankroll": STARTING_BANKROLL, "ending_bankroll": bankroll,
        "total_return_pct": (bankroll / STARTING_BANKROLL - 1) * 100,
        "category_b_games": len(ledger), "number_of_bets": int(bets.sum()),
        "passes_nonpositive_kelly": int((ledger.full_kelly_fraction <= 0).sum()),
        "average_stake_pct_bankroll": fractions.mean() * 100,
        "median_stake_pct_bankroll": fractions.median() * 100,
        "maximum_stake_pct_bankroll": fractions.max() * 100,
        "maximum_bankroll_drawdown_pct": max_drawdown_pct(ledger.bankroll_after),
        "minimum_bankroll": minimum, "maximum_bankroll": maximum,
    }, ledger


def flat_reference(frame, scope):
    profit = np.where(frame.result.eq(1), frame.selected_current_decimal_odds - 1, -1.0)
    cumulative = np.cumsum(profit)
    values = np.r_[0.0, cumulative]
    peaks = np.maximum.accumulate(values)
    wins = int(frame.result.sum())
    return {
        "scope": scope, "category_b_games": len(frame), "bets": len(frame),
        "wins": wins, "losses": len(frame) - wins,
        "win_rate": wins / len(frame) if len(frame) else np.nan,
        "total_units": profit.sum(), "roi": profit.mean() if len(frame) else np.nan,
        "maximum_drawdown_units": float((peaks - values).max()),
    }


def main():
    source = pd.read_csv(INPUT)
    source["date"] = pd.to_datetime(source.date)
    category_b = source[
        source.model_probability.ge(CATEGORY_B_MIN_CONFIDENCE)
        & source.model_edge.ge(CATEGORY_B_MIN_EDGE)
    ].copy()
    category_b["probability_raw"] = category_b.model_probability
    category_b["probability_shrunk"] = 0.5 + SHRINKAGE_SLOPE * (category_b.model_probability - 0.5)
    scopes = [(str(year), category_b[category_b.season.eq(year)]) for year in range(2022, 2026)]
    scopes.append(("combined_2022_2025", category_b))
    variants = [("raw_probability", "probability_raw"),
                ("shrunk_probability_slope_0_7629", "probability_shrunk")]
    summaries, ledgers, flat = [], [], []
    for scope, frame in scopes:
        flat.append(flat_reference(frame, scope))
        for variant, probability_column in variants:
            for method, (multiplier, cap) in METHODS.items():
                summary, ledger = simulate(
                    frame, variant, probability_column, method, multiplier, cap, scope
                )
                summaries.append(summary); ledgers.append(ledger)
    summary = pd.DataFrame(summaries)
    ledger = pd.concat(ledgers, ignore_index=True)
    flat = pd.DataFrame(flat)

    os.makedirs("results/kelly_ledgers_category_b", exist_ok=True)
    summary.to_csv("results/category_b_kelly_bankroll_simulation_summary.csv", index=False)
    flat.to_csv("results/category_b_flat_1u_reference.csv", index=False)
    ledger.to_csv("results/category_b_kelly_bankroll_game_ledger.csv", index=False)
    for (variant, method), group in ledger.groupby(["probability_variant", "staking_method"]):
        group.to_csv(f"results/kelly_ledgers_category_b/{variant}__{method}.csv", index=False)

    combined = summary[summary.scope.eq("combined_2022_2025")]
    print(f"Category B definition: probability >= {CATEGORY_B_MIN_CONFIDENCE:.2f} AND edge >= {CATEGORY_B_MIN_EDGE:.2f}")
    print("\nFLAT 1U REFERENCE"); print(flat.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nRAW-PROBABILITY KELLY — COMBINED")
    print(combined[combined.probability_variant.eq("raw_probability")].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSHRUNK-PROBABILITY KELLY — COMBINED")
    print(combined[combined.probability_variant.str.startswith("shrunk")].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nINDEPENDENT YEARLY RESULTS — EACH STARTS AT 100 UNITS")
    print(summary[~summary.scope.eq("combined_2022_2025")].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nNo parameter, threshold, sportsbook, or staking method was optimized.")


if __name__ == "__main__":
    main()
