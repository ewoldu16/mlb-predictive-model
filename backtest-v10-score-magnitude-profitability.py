"""Flat-stake profitability diagnostics for frozen OOS V10 score magnitudes."""
import os

import numpy as np
import pandas as pd


DEVELOPMENT_SCORES = "results/v10_development_oos_game_scores.csv"
HOLDOUT_SCORES = "results/v10_2025_component_breakdown.csv"
ODDS = "data/processed/historical_mlb_moneylines_2022_2025.csv"
PRIMARY_THRESHOLD = 15.0
THRESHOLDS = [5, 10, 15, 20, 25, 30]
BUCKET_EDGES = [0, 5, 10, 15, 20, 25, 30, np.inf]
BUCKET_LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]
BOOTSTRAPS = 10_000
RANDOM_SEED = 20251015


def american_to_decimal(series):
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna() & values.ne(0) & values.abs().ge(100)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    positive, negative = valid & values.gt(0), valid & values.lt(0)
    result.loc[positive] = 1 + values.loc[positive] / 100
    result.loc[negative] = 1 + 100 / values.loc[negative].abs()
    return result


def maximum_drawdown(profits):
    cumulative = np.asarray(profits, dtype=float).cumsum()
    path = np.r_[0.0, cumulative]
    peaks = np.maximum.accumulate(path)
    return float((peaks - path).max())


def summary(frame, scope, strategy, eligible=None):
    frame = frame.sort_values(["date", "game_id"])
    bets = len(frame); wins = int(frame.result.sum()) if bets else 0
    units = frame.profit_units.sum() if bets else 0.0
    return {
        "scope": scope, "strategy": strategy,
        "eligible_games": eligible if eligible is not None else np.nan,
        "bets": bets, "passes": eligible - bets if eligible is not None else np.nan,
        "wins": wins, "losses": bets - wins,
        "win_rate": wins / bets if bets else np.nan,
        "average_decimal_odds": frame.selected_decimal_odds.mean(),
        "average_raw_implied_probability": frame.selected_raw_implied_probability.mean(),
        "units": units, "roi": units / bets if bets else np.nan,
        "maximum_drawdown": maximum_drawdown(frame.profit_units) if bets else np.nan,
    }


def main():
    development = pd.read_csv(DEVELOPMENT_SCORES)
    holdout = pd.read_csv(HOLDOUT_SCORES)
    scores = pd.concat([development, holdout], ignore_index=True)
    if scores.game_id.duplicated().any():
        raise ValueError("Duplicate game IDs across frozen OOS V10 score files")
    scores["date"] = pd.to_datetime(scores.date)
    scores["absolute_score"] = scores.TOTAL_SCORE.abs()
    scores["selected_side"] = np.where(scores.TOTAL_SCORE.gt(0), "home",
                                       np.where(scores.TOTAL_SCORE.lt(0), "away", "none"))
    scores["selected_team"] = np.where(scores.TOTAL_SCORE.gt(0), scores.home_team,
                                       np.where(scores.TOTAL_SCORE.lt(0), scores.away_team, pd.NA))
    scores["result"] = np.where(scores.TOTAL_SCORE.gt(0), scores.home_win.eq(1),
                                np.where(scores.TOTAL_SCORE.lt(0), scores.home_win.eq(0), np.nan))

    odds = pd.read_csv(ODDS)
    odds = odds[odds.sportsbook.eq("bet365") & odds.match_status.eq("matched")].copy()
    if odds.game_id.duplicated().any():
        raise ValueError("Duplicate uniquely matched Bet365 game IDs")
    odds["current_home_decimal"] = american_to_decimal(odds.current_home_odds)
    odds["current_away_decimal"] = american_to_decimal(odds.current_away_odds)
    odds = odds.dropna(subset=["current_home_decimal", "current_away_decimal"])
    columns = ["game_id", "current_home_odds", "current_away_odds",
               "current_home_decimal", "current_away_decimal"]
    ledger = scores.merge(odds[columns], on="game_id", how="inner", validate="one_to_one")
    ledger["selected_american_odds"] = np.where(
        ledger.selected_side.eq("home"), ledger.current_home_odds,
        np.where(ledger.selected_side.eq("away"), ledger.current_away_odds, np.nan))
    ledger["selected_decimal_odds"] = np.where(
        ledger.selected_side.eq("home"), ledger.current_home_decimal,
        np.where(ledger.selected_side.eq("away"), ledger.current_away_decimal, np.nan))
    ledger["selected_raw_implied_probability"] = 1 / ledger.selected_decimal_odds
    ledger["profit_units"] = np.where(ledger.result.eq(1), ledger.selected_decimal_odds - 1,
                                      np.where(ledger.result.eq(0), -1.0, np.nan))
    ledger["score_bucket"] = pd.cut(ledger.absolute_score, BUCKET_EDGES,
                                    labels=BUCKET_LABELS, right=False, include_lowest=True)
    ledger["primary_ge_15_selection"] = ledger.absolute_score.ge(PRIMARY_THRESHOLD)
    ledger = ledger.sort_values(["date", "game_id"]).reset_index(drop=True)
    primary_mask = ledger.primary_ge_15_selection
    ledger["primary_ge_15_cumulative_units"] = np.nan
    ledger.loc[primary_mask, "primary_ge_15_cumulative_units"] = ledger.loc[
        primary_mask, "profit_units"].cumsum().to_numpy()

    scopes = [(str(year), ledger[ledger.season.eq(year)]) for year in range(2022, 2026)]
    scopes.append(("combined_2022_2025", ledger))
    primary_rows, bucket_rows, threshold_rows = [], [], []
    for scope, scoped in scopes:
        selectable = scoped[scoped.selected_side.ne("none")]
        primary_rows.append(summary(selectable[selectable.absolute_score.ge(PRIMARY_THRESHOLD)],
                                    scope, "absolute_score_ge_15", len(scoped)))
        for bucket in BUCKET_LABELS:
            bucket_rows.append(summary(selectable[selectable.score_bucket.eq(bucket)], scope, bucket))
        for threshold in THRESHOLDS:
            threshold_rows.append(summary(selectable[selectable.absolute_score.ge(threshold)],
                                          scope, f"absolute_score_ge_{threshold}", len(scoped)))
    primary = pd.DataFrame(primary_rows)
    buckets = pd.DataFrame(bucket_rows)
    thresholds = pd.DataFrame(threshold_rows)

    combined_primary = ledger[ledger.selected_side.ne("none") & ledger.absolute_score.ge(PRIMARY_THRESHOLD)]
    profits = combined_primary.profit_units.to_numpy(float)
    rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_roi = rng.choice(profits, size=(BOOTSTRAPS, len(profits)), replace=True).mean(axis=1)
    uncertainty = pd.DataFrame([{
        "strategy": "absolute_score_ge_15", "bets": len(profits), "observed_roi": profits.mean(),
        "standard_error_roi": profits.std(ddof=1) / np.sqrt(len(profits)),
        "bootstrap_95pct_low": np.quantile(bootstrap_roi, .025),
        "bootstrap_95pct_high": np.quantile(bootstrap_roi, .975),
        "bootstrap_samples": BOOTSTRAPS, "random_seed": RANDOM_SEED,
    }])

    os.makedirs("results", exist_ok=True)
    ledger.to_csv("results/backtest_v10_score_magnitude_game_ledger.csv", index=False)
    primary.to_csv("results/backtest_v10_score_ge15_summary.csv", index=False)
    buckets.to_csv("results/backtest_v10_score_buckets.csv", index=False)
    thresholds.to_csv("results/backtest_v10_score_thresholds.csv", index=False)
    uncertainty.to_csv("results/backtest_v10_score_ge15_bootstrap.csv", index=False)

    print(f"Frozen OOS V10 games with valid matched Bet365 currentLine: {len(ledger):,}")
    print("\nPREDEFINED ABSOLUTE SCORE >=15")
    print(primary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nNON-OVERLAPPING SCORE BUCKETS — COMBINED")
    print(buckets[buckets.scope.eq("combined_2022_2025")].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nCUMULATIVE THRESHOLDS — COMBINED")
    print(thresholds[thresholds.scope.eq("combined_2022_2025")].to_string(
        index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nBOOTSTRAP UNCERTAINTY")
    print(uncertainty.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nNo V10 changes, Kelly staking, threshold optimization, or retraining performed.")


if __name__ == "__main__":
    main()
