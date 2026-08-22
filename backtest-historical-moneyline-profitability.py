"""Predefined historical moneyline diagnostics for frozen OOS predictions."""
import os

import numpy as np
import pandas as pd


PREDICTIONS = "results/v5_team_strength_oos_predictions_2022_2025.csv"
ODDS = "data/processed/historical_mlb_moneylines_2022_2025.csv"
PRIMARY_BOOK = "bet365"
ROBUSTNESS_BOOKS = ["bet365", "draftkings", "fanduel", "caesars", "betmgm"]
CONFIDENCE_THRESHOLDS = [.55, .56, .58, .60, .62, .65, .70]
EDGE_THRESHOLDS = [("edge_gt_0pct", 0.0, False), ("edge_ge_1pct", .01, True),
                   ("edge_ge_2pct", .02, True), ("edge_ge_3pct", .03, True),
                   ("edge_ge_4pct", .04, True), ("edge_ge_5pct", .05, True),
                   ("edge_ge_7_5pct", .075, True), ("edge_ge_10pct", .10, True)]
EDGE_BUCKETS = ["edge_le_0pct", "edge_0_2pct", "edge_2_4pct", "edge_4_6pct",
                "edge_6_8pct", "edge_8_10pct", "edge_10pct_plus"]
EV_BUCKETS = ["ev_le_0pct", "ev_0_2pct", "ev_2_5pct", "ev_5_10pct",
              "ev_10_15pct", "ev_15pct_plus"]
SCOPES = ["2022", "2023", "2024", "2025", "combined"]
BOOTSTRAPS = 10_000
RANDOM_SEED = 20250816


def american_to_decimal(series):
    series = pd.to_numeric(series, errors="coerce")
    valid = series.notna() & series.ne(0) & series.abs().ge(100)
    result = pd.Series(np.nan, index=series.index, dtype=float)
    positive = valid & series.gt(0)
    negative = valid & series.lt(0)
    result.loc[positive] = 1 + series.loc[positive] / 100
    result.loc[negative] = 1 + 100 / series.loc[negative].abs()
    return result


def max_drawdown(profits):
    cumulative = np.asarray(profits, dtype=float).cumsum()
    path = np.r_[0.0, cumulative]
    peaks = np.maximum.accumulate(path)
    return float((peaks - path).max())


def scope_frames(frame):
    return [(str(year), frame[frame.season.eq(year)]) for year in range(2022, 2026)] + [("combined", frame)]


def performance(frame, scope, strategy, eligible_games=None):
    bets = len(frame)
    wins = int(frame.result.sum()) if bets else 0
    units = frame.profit_units.sum() if bets else 0.0
    return {
        "scope": scope, "strategy": strategy,
        "eligible_games": eligible_games if eligible_games is not None else np.nan,
        "bets": bets,
        "passes": eligible_games - bets if eligible_games is not None else np.nan,
        "wins": wins, "losses": bets - wins,
        "win_rate": wins / bets if bets else np.nan,
        "average_model_probability": frame.model_probability.mean(),
        "average_market_no_vig_probability": frame.market_no_vig_probability.mean(),
        "average_edge": frame.model_edge.mean(),
        "average_decimal_odds": frame.selected_current_decimal_odds.mean(),
        "median_decimal_odds": frame.selected_current_decimal_odds.median(),
        "units": units, "roi": units / bets if bets else np.nan,
        "maximum_drawdown": max_drawdown(frame.sort_values(["date", "game_id"]).profit_units) if bets else np.nan,
    }


def prepare_book(predictions, odds, sportsbook):
    book = odds[odds.sportsbook.eq(sportsbook) & odds.match_status.eq("matched")].copy()
    if book.game_id.duplicated().any():
        raise ValueError(f"Duplicate {sportsbook} game IDs")
    frame = predictions.merge(book, on="game_id", how="inner", suffixes=("", "_odds"), validate="one_to_one")
    for snapshot in ["opening", "current"]:
        for side in ["home", "away"]:
            frame[f"{snapshot}_{side}_decimal"] = american_to_decimal(frame[f"{snapshot}_{side}_odds"])
        raw_home = 1 / frame[f"{snapshot}_home_decimal"]
        raw_away = 1 / frame[f"{snapshot}_away_decimal"]
        total = raw_home + raw_away
        frame[f"{snapshot}_no_vig_home"] = raw_home / total
        frame[f"{snapshot}_no_vig_away"] = raw_away / total
    required = ["current_home_decimal", "current_away_decimal", "current_no_vig_home", "current_no_vig_away"]
    frame = frame.dropna(subset=required).copy()
    sums = frame.current_no_vig_home + frame.current_no_vig_away
    if not np.allclose(sums, 1.0, atol=1e-12):
        raise ValueError(f"{sportsbook}: no-vig probabilities do not sum to one")
    selected_home = frame.model_selected_team.map(lambda x: "Oakland Athletics" if x == "Athletics" else x).eq(frame.home_team_normalized)
    selected_away = frame.model_selected_team.map(lambda x: "Oakland Athletics" if x == "Athletics" else x).eq(frame.away_team_normalized)
    if not (selected_home ^ selected_away).all():
        raise ValueError(f"{sportsbook}: selected-team orientation failure")
    frame["selected_side"] = np.where(selected_home, "home", "away")
    frame["model_probability"] = frame.model_selected_probability
    frame["market_no_vig_probability"] = np.where(selected_home, frame.current_no_vig_home, frame.current_no_vig_away)
    frame["selected_current_decimal_odds"] = np.where(selected_home, frame.current_home_decimal, frame.current_away_decimal)
    frame["selected_opening_decimal_odds"] = np.where(selected_home, frame.opening_home_decimal, frame.opening_away_decimal)
    frame["opening_market_no_vig_probability"] = np.where(selected_home, frame.opening_no_vig_home, frame.opening_no_vig_away)
    frame["model_edge"] = frame.model_probability - frame.market_no_vig_probability
    frame["model_ev"] = frame.model_probability * frame.selected_current_decimal_odds - 1
    frame["result"] = frame.model_selection_won.astype(int)
    frame["profit_units"] = np.where(frame.result.eq(1), frame.selected_current_decimal_odds - 1, -1.0)
    return frame.sort_values(["date", "game_id"]).reset_index(drop=True)


def assign_edge_bucket(edge):
    return np.select([edge.le(0), edge.le(.02), edge.le(.04), edge.le(.06), edge.le(.08), edge.le(.10)],
                     EDGE_BUCKETS[:-1], default=EDGE_BUCKETS[-1])


def assign_ev_bucket(ev):
    return np.select([ev.le(0), ev.le(.02), ev.le(.05), ev.le(.10), ev.le(.15)],
                     EV_BUCKETS[:-1], default=EV_BUCKETS[-1])


def grouped_bucket_results(frame, bucket_column, ordered_labels):
    rows = []
    for scope, scoped in scope_frames(frame):
        for label in ordered_labels:
            rows.append(performance(scoped[scoped[bucket_column].eq(label)], scope, label))
    return pd.DataFrame(rows)


def uncertainty(strategies):
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for strategy, frame in strategies:
        profit = frame.profit_units.to_numpy(float)
        n = len(profit)
        if n:
            means = rng.choice(profit, size=(BOOTSTRAPS, n), replace=True).mean(axis=1)
            low, high = np.quantile(means, [.025, .975])
            se = profit.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        else:
            low = high = se = np.nan
        rows.append({"strategy": strategy, "bets": n, "roi": profit.mean() if n else np.nan,
                     "standard_error_roi": se, "bootstrap_95pct_low": low,
                     "bootstrap_95pct_high": high, "bootstrap_samples": BOOTSTRAPS,
                     "random_seed": RANDOM_SEED})
    return pd.DataFrame(rows)


def main():
    predictions = pd.read_csv(PREDICTIONS)
    predictions["date"] = pd.to_datetime(predictions.date)
    odds = pd.read_csv(ODDS)
    observed = pd.concat([odds[c] for c in ["opening_home_odds", "opening_away_odds", "current_home_odds", "current_away_odds"]])
    nonzero = observed.dropna().loc[lambda x: x.ne(0)]
    if not nonzero.abs().ge(100).all() or not (nonzero.gt(0).any() and nonzero.lt(0).any()):
        raise ValueError("Observed odds do not validate as American format")
    print("Odds format validated as American: all nonzero magnitudes >= 100; positive and negative prices present.")

    books = {book: prepare_book(predictions, odds, book) for book in ROBUSTNESS_BOOKS}
    primary = books[PRIMARY_BOOK]
    baseline = primary[primary.model_probability.ge(.55)].copy()
    baseline["cumulative_units"] = baseline.profit_units.cumsum()
    baseline["edge_bucket"] = assign_edge_bucket(baseline.model_edge)
    baseline["ev_bucket"] = assign_ev_bucket(baseline.model_ev)

    baseline_rows = [performance(scoped[scoped.model_probability.ge(.55)], scope, "confidence_ge_55pct", len(scoped))
                     for scope, scoped in scope_frames(primary)]
    baseline_results = pd.DataFrame(baseline_rows)

    confidence_rows = []
    for scope, scoped in scope_frames(primary):
        for threshold in CONFIDENCE_THRESHOLDS:
            confidence_rows.append(performance(scoped[scoped.model_probability.ge(threshold)], scope,
                                                f"confidence_ge_{threshold:.0%}", len(scoped)))
    confidence = pd.DataFrame(confidence_rows)

    edge_rows = []
    for scope, scoped in scope_frames(baseline):
        for name, threshold, inclusive in EDGE_THRESHOLDS:
            selected = scoped[scoped.model_edge.ge(threshold) if inclusive else scoped.model_edge.gt(threshold)]
            edge_rows.append(performance(selected, scope, name))
    edge_thresholds = pd.DataFrame(edge_rows)
    edge_buckets = grouped_bucket_results(baseline, "edge_bucket", EDGE_BUCKETS)
    ev_buckets = grouped_bucket_results(baseline, "ev_bucket", EV_BUCKETS)

    pickem_tolerance = .005
    baseline["market_class"] = np.where(
        baseline.market_no_vig_probability.gt(.5 + pickem_tolerance), "market_favourite",
        np.where(baseline.market_no_vig_probability.lt(.5 - pickem_tolerance), "market_underdog", "approximately_pickem"))
    baseline["favourite_price_band"] = pd.cut(
        baseline.selected_current_decimal_odds, [1.0, 1.50, 1.67, 1.83, 2.0, np.inf],
        labels=["fav_odds_le_1_50", "fav_odds_1_50_1_67", "fav_odds_1_67_1_83",
                "fav_odds_1_83_2_00", "fav_odds_gt_2_00"], include_lowest=True)
    favourite_rows = []
    for scope, scoped in scope_frames(baseline):
        for category in ["market_favourite", "market_underdog", "approximately_pickem"]:
            favourite_rows.append(performance(scoped[scoped.market_class.eq(category)], scope, category))
        favourites = scoped[scoped.market_class.eq("market_favourite")]
        for band in baseline.favourite_price_band.cat.categories:
            favourite_rows.append(performance(favourites[favourites.favourite_price_band.eq(band)], scope, str(band)))
    favourite = pd.DataFrame(favourite_rows)

    movement = baseline.dropna(subset=["opening_market_no_vig_probability", "market_no_vig_probability"]).copy()
    movement["opening_to_current_probability_movement"] = (
        movement.market_no_vig_probability - movement.opening_market_no_vig_probability)
    movement["moved_toward_model_side"] = movement.opening_to_current_probability_movement.gt(0)
    movement_rows = []
    for scope, scoped in scope_frames(movement):
        for bucket, group in [("all", scoped)] + [(label, scoped[scoped.edge_bucket.eq(label)]) for label in EDGE_BUCKETS]:
            movement_rows.append({"scope": scope, "edge_bucket": bucket, "selections": len(group),
                                  "pct_moved_toward_model_side": group.moved_toward_model_side.mean(),
                                  "average_probability_movement": group.opening_to_current_probability_movement.mean(),
                                  "median_probability_movement": group.opening_to_current_probability_movement.median()})
    movement_results = pd.DataFrame(movement_rows)

    robustness_rows = []
    robustness_rules = [("confidence_ge_55pct", lambda x: x.model_probability.ge(.55)),
                        ("confidence_ge_55pct_and_edge_gt_0pct", lambda x: x.model_probability.ge(.55) & x.model_edge.gt(0)),
                        ("confidence_ge_55pct_and_edge_ge_3pct", lambda x: x.model_probability.ge(.55) & x.model_edge.ge(.03)),
                        ("confidence_ge_55pct_and_edge_ge_5pct", lambda x: x.model_probability.ge(.55) & x.model_edge.ge(.05))]
    for book, frame in books.items():
        for scope, scoped in scope_frames(frame):
            for name, rule in robustness_rules:
                row = performance(scoped[rule(scoped)], scope, name, len(scoped)); row["sportsbook"] = book
                robustness_rows.append(row)
    robustness = pd.DataFrame(robustness_rows)

    yearly = pd.concat([
        baseline_results.assign(analysis="baseline"),
        confidence.assign(analysis="confidence_threshold"),
        edge_thresholds.assign(analysis="edge_threshold"),
    ], ignore_index=True)
    uncertainty_inputs = [(f"confidence_ge_{threshold:.0%}", primary[primary.model_probability.ge(threshold)])
                          for threshold in CONFIDENCE_THRESHOLDS]
    uncertainty_inputs += [(name, baseline[baseline.model_edge.ge(threshold) if inclusive else baseline.model_edge.gt(threshold)])
                           for name, threshold, inclusive in EDGE_THRESHOLDS]
    uncertainty_results = uncertainty(uncertainty_inputs)

    ledger_columns = ["game_id", "date", "season", "home_team", "away_team", "model_selected_team",
                      "model_probability", "opening_home_odds", "opening_away_odds", "current_home_odds",
                      "current_away_odds", "selected_opening_decimal_odds", "selected_current_decimal_odds",
                      "market_no_vig_probability", "model_edge", "model_ev", "result", "profit_units",
                      "cumulative_units"]
    os.makedirs("results", exist_ok=True)
    baseline_results.to_csv("results/backtest_bet365_55pct.csv", index=False)
    confidence.to_csv("results/backtest_bet365_confidence_thresholds.csv", index=False)
    edge_thresholds.to_csv("results/backtest_bet365_edge_thresholds.csv", index=False)
    edge_buckets.to_csv("results/backtest_bet365_edge_buckets.csv", index=False)
    ev_buckets.to_csv("results/backtest_bet365_ev_buckets.csv", index=False)
    favourite.to_csv("results/backtest_bet365_favourite_underdog.csv", index=False)
    yearly.to_csv("results/backtest_bet365_yearly.csv", index=False)
    movement_results.to_csv("results/backtest_opening_to_current_movement.csv", index=False)
    robustness.to_csv("results/backtest_sportsbook_robustness.csv", index=False)
    uncertainty_results.to_csv("results/backtest_profitability_uncertainty.csv", index=False)
    baseline[ledger_columns].to_csv("results/backtest_bet365_55pct_game_ledger.csv", index=False)

    combined = lambda df: df[df.scope.eq("combined")]
    print(f"Eligible uniquely matched Bet365 games: {len(primary):,}")
    print("\nBET365 >=55% BASELINE"); print(combined(baseline_results).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nCONFIDENCE THRESHOLDS"); print(combined(confidence).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nEDGE THRESHOLDS"); print(combined(edge_thresholds).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nEDGE BUCKETS"); print(combined(edge_buckets).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nEV BUCKETS"); print(combined(ev_buckets).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nFAVOURITE / UNDERDOG"); print(combined(favourite).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nYEARLY >=55% BASELINE"); print(baseline_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nOPENING-TO-CURRENT MARKET MOVEMENT"); print(combined(movement_results).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSPORTSBOOK ROBUSTNESS"); print(robustness[robustness.scope.eq("combined")].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nBOOTSTRAP UNCERTAINTY"); print(uncertainty_results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nNo strategy optimization, retraining, recalibration, or staking optimization performed.")


if __name__ == "__main__":
    main()
