"""Clean and deterministically match the public mlb-odds-scraper dataset."""
import json
import os

import numpy as np
import pandas as pd


RAW_PATH = "data/raw/historical_odds/mlb_odds_dataset.json"
PREDICTIONS_PATH = "results/v5_team_strength_oos_predictions_2022_2025.csv"
COVERAGE_END = pd.Timestamp("2025-08-16")
YEARS = [2022, 2023, 2024, 2025]
TEAM_ALIASES = {"Athletics": "Oakland Athletics"}


def normalize_team(value):
    value = str(value).strip()
    return TEAM_ALIASES.get(value, value)


def extract_odds():
    with open(RAW_PATH, encoding="utf-8") as handle:
        raw = json.load(handle)
    games, lines = [], []
    for source_date, day_games in raw.items():
        date = pd.Timestamp(source_date)
        if date.year not in YEARS:
            continue
        for source_index, game in enumerate(day_games):
            view = game.get("gameView", {})
            if view.get("gameType") != "R":
                continue
            key = f"{source_date}_{source_index:02d}"
            away = view.get("awayTeam") or {}
            home = view.get("homeTeam") or {}
            moneylines = (game.get("odds") or {}).get("moneyline") or []
            game_row = {
                "source_game_key": key, "source_date": source_date,
                "start_datetime": view.get("startDate"),
                "start_time": pd.to_datetime(view.get("startDate"), utc=True, errors="coerce"),
                "away_team_original": away.get("fullName"), "home_team_original": home.get("fullName"),
                "away_team_code_original": away.get("shortName"),
                "home_team_code_original": home.get("shortName"),
                "away_team_normalized": normalize_team(away.get("fullName")),
                "home_team_normalized": normalize_team(home.get("fullName")),
                "away_score": view.get("awayTeamScore"), "home_score": view.get("homeTeamScore"),
                "game_status": view.get("gameStatusText"), "game_type": view.get("gameType"),
                "venue": view.get("venueName"), "sportsbooks": len(moneylines),
            }
            games.append(game_row)
            for market_index, market in enumerate(moneylines):
                opening = market.get("openingLine") or {}
                current = market.get("currentLine") or {}
                lines.append({
                    **game_row, "moneyline_index": market_index,
                    "sportsbook": market.get("sportsbook"),
                    "opening_home_odds": opening.get("homeOdds"),
                    "opening_away_odds": opening.get("awayOdds"),
                    "current_home_odds": current.get("homeOdds"),
                    "current_away_odds": current.get("awayOdds"),
                })
    return pd.DataFrame(games), pd.DataFrame(lines), raw


def load_model_games():
    predictions = pd.read_csv(PREDICTIONS_PATH, usecols=["game_id", "season"])
    universes = []
    for year in YEARS:
        frame = pd.read_csv(f"data/raw/games_{year}.csv")
        frame["season"] = year
        universes.append(frame)
    universe = pd.concat(universes, ignore_index=True)
    model = predictions.merge(universe, on=["game_id", "season"], validate="one_to_one")
    if len(model) != len(predictions):
        raise ValueError("Not every OOS prediction matched the known game universe")
    model["date"] = pd.to_datetime(model["date"])
    model["home_team_normalized"] = model["home_team"].map(normalize_team)
    model["away_team_normalized"] = model["away_team"].map(normalize_team)
    return model


def match_games(model, odds_games):
    odds_games = odds_games.copy()
    odds_games["date"] = pd.to_datetime(odds_games["source_date"])
    key_columns = ["date", "home_team_normalized", "away_team_normalized"]
    matches = model.copy()
    matches["source_game_key"] = pd.NA
    matches["match_status"] = "unmatched"
    matches["candidate_count"] = 0
    matches["match_method"] = pd.NA
    outside = matches.date.gt(COVERAGE_END)
    matches.loc[outside, "match_status"] = "outside_advertised_coverage"
    odds_grouped = {key: group for key, group in odds_games.groupby(key_columns, dropna=False)}
    for key, model_group in matches.loc[~outside].groupby(key_columns, dropna=False):
        candidates = odds_grouped.get(key, pd.DataFrame())
        matches.loc[model_group.index, "candidate_count"] = len(candidates)
        if candidates.empty:
            continue
        if len(model_group) == 1 and len(candidates) == 1:
            index = model_group.index[0]
            matches.loc[index, ["source_game_key", "match_status", "match_method"]] = [
                candidates.iloc[0].source_game_key, "matched", "exact_date_teams"
            ]
            continue
        # Resolve doubleheaders only through unique exact score pairings. Each
        # source record is consumed at most once.
        pairs = []
        for model_index, game in model_group.iterrows():
            score_matches = candidates[
                pd.to_numeric(candidates.home_score, errors="coerce").eq(game.home_score)
                & pd.to_numeric(candidates.away_score, errors="coerce").eq(game.away_score)
            ]
            pairs.extend((model_index, source_key) for source_key in score_matches.source_game_key)
        model_counts = pd.Series([pair[0] for pair in pairs]).value_counts() if pairs else pd.Series(dtype=int)
        source_counts = pd.Series([pair[1] for pair in pairs]).value_counts() if pairs else pd.Series(dtype=int)
        used_sources = set()
        for model_index, source_key in pairs:
            if model_counts[model_index] == 1 and source_counts[source_key] == 1:
                matches.loc[model_index, ["source_game_key", "match_status", "match_method"]] = [
                    source_key, "matched", "exact_date_teams_scores"
                ]
                used_sources.add(source_key)
        unresolved = matches.index.isin(model_group.index) & ~matches.match_status.eq("matched")
        available_sources = set(candidates.source_game_key) - used_sources
        if available_sources:
            matches.loc[unresolved, "match_status"] = "ambiguous"
    duplicate_keys = matches.loc[matches.match_status.eq("matched"), "source_game_key"].duplicated(False)
    if duplicate_keys.any():
        raise ValueError("One-to-one matching invariant failed: duplicate source game assigned")
    return matches


def build_audit(matches):
    rows = []
    for year in YEARS:
        frame = matches[matches.season.eq(year)]
        inside = frame[frame.date.le(COVERAGE_END)]
        matched = int(inside.match_status.eq("matched").sum())
        rows.append({
            "season": year, "total_model_games": len(frame),
            "inside_advertised_coverage": len(inside),
            "outside_advertised_coverage": int(frame.date.gt(COVERAGE_END).sum()),
            "matched_games": matched,
            "unmatched_games": int(inside.match_status.eq("unmatched").sum()),
            "ambiguous_games": int(inside.match_status.eq("ambiguous").sum()),
            "duplicate_matches": int(inside.match_status.eq("duplicate_match").sum()),
            "coverage_pct_inside_advertised_range": matched / len(inside) if len(inside) else np.nan,
        })
    return pd.DataFrame(rows)


def validation_examples(matches, odds_games):
    matched = matches[matches.match_status.eq("matched")]
    samples = []
    for year in YEARS:
        season = matched[matched.season.eq(year)]
        sample = season.sample(n=min(5, len(season)), random_state=year)
        samples.append(sample)
    sample = pd.concat(samples, ignore_index=True)
    source_columns = ["source_game_key", "source_date", "start_datetime",
                      "home_team_original", "away_team_original", "home_score", "away_score"]
    source = odds_games[source_columns].rename(columns={
        "home_score": "odds_home_score", "away_score": "odds_away_score"
    })
    sample = sample.merge(source, on="source_game_key", validate="one_to_one")
    sample["date_matches"] = sample.date.eq(pd.to_datetime(sample.source_date))
    sample["home_team_matches"] = sample.home_team_normalized.eq(sample.home_team_original.map(normalize_team))
    sample["away_team_matches"] = sample.away_team_normalized.eq(sample.away_team_original.map(normalize_team))
    sample["score_matches"] = sample.home_score.eq(sample.odds_home_score) & sample.away_score.eq(sample.odds_away_score)
    return sample


def main():
    odds_games, moneylines, raw = extract_odds()
    model = load_model_games()
    matches = match_games(model, odds_games)
    audit = build_audit(matches)
    examples = validation_examples(matches, odds_games)
    match_columns = matches.loc[
        matches.match_status.eq("matched"),
        ["game_id", "season", "source_game_key", "match_status", "match_method"],
    ]
    cleaned = moneylines.merge(match_columns, on="source_game_key", how="left", validate="many_to_one")
    cleaned["match_status"] = cleaned["match_status"].fillna("unmatched_to_model")
    sportsbook_rows = []
    matched_lines = cleaned[cleaned.match_status.eq("matched")]
    for scope, frame in [(str(year), matched_lines[matched_lines.season.eq(year)]) for year in YEARS] + [("combined", matched_lines)]:
        denominator = matches.match_status.eq("matched").sum() if scope == "combined" else (
            matches.season.eq(int(scope)) & matches.match_status.eq("matched")
        ).sum()
        grouped = (frame.groupby("sportsbook", dropna=False)
                   .agg(matched_games=("game_id", "nunique"), moneyline_records=("game_id", "size"))
                   .reset_index())
        grouped["scope"] = scope
        grouped["coverage_pct_of_matched_games"] = grouped.matched_games / denominator
        sportsbook_rows.append(grouped)
    sportsbook = pd.concat(sportsbook_rows, ignore_index=True)
    sportsbook = sportsbook[["scope", "sportsbook", "matched_games", "moneyline_records",
                             "coverage_pct_of_matched_games"]].sort_values(
                                 ["scope", "matched_games"], ascending=[True, False])
    opening_available = cleaned[["opening_home_odds", "opening_away_odds"]].notna().all(axis=1)
    current_available = cleaned[["current_home_odds", "current_away_odds"]].notna().all(axis=1)

    print(f"Raw dataset: {RAW_PATH} ({os.path.getsize(RAW_PATH):,} bytes)")
    print(f"Raw dates: {len(raw):,}; raw games: {sum(len(day) for day in raw.values()):,}")
    print(f"Explicit regular-season games 2022-2025: {len(odds_games):,}")
    print(f"Sportsbook-level moneyline records: {len(cleaned):,}")
    print(f"Complete opening-line pairs: {opening_available.sum():,}/{len(cleaned):,}")
    print(f"Complete current-line pairs: {current_available.sum():,}/{len(cleaned):,}")
    print("\nMATCHING AUDIT")
    print(audit.to_string(index=False, float_format=lambda value: f"{value:.4%}"))
    print("\nSPORTSBOOK COVERAGE")
    print(sportsbook.to_string(index=False))
    print("\nRANDOM MATCH VALIDATION")
    display = ["season", "game_id", "date", "away_team", "home_team", "source_date",
               "away_team_original", "home_team_original", "away_score", "home_score",
               "odds_away_score", "odds_home_score", "date_matches", "away_team_matches",
               "home_team_matches", "score_matches", "match_method"]
    print(examples[display].to_string(index=False))
    if not examples[["date_matches", "away_team_matches", "home_team_matches", "score_matches"]].all().all():
        raise ValueError("Random matched-example validation failed")

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    cleaned.to_csv("data/processed/historical_mlb_moneylines_2022_2025.csv", index=False)
    audit.to_csv("results/historical_odds_matching_audit.csv", index=False)
    matches.to_csv("results/historical_odds_game_matches.csv", index=False)
    sportsbook.to_csv("results/historical_odds_sportsbook_coverage.csv", index=False)
    examples.to_csv("results/historical_odds_validation_examples.csv", index=False)
    print("\nAcquisition, cleaning, matching, and validation complete. No profitability analysis run.")


if __name__ == "__main__":
    main()
