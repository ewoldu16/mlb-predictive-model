"""Extract sportsbook-level regular-season MLB totals using validated game matches."""
import json, os
import pandas as pd

RAW="data/raw/historical_odds/mlb_odds_dataset.json"
MATCHES="results/historical_odds_game_matches.csv"
YEARS=range(2022,2026)

with open(RAW,encoding="utf-8") as f: raw=json.load(f)
rows=[]
for source_date,games in raw.items():
 if int(source_date[:4]) not in YEARS: continue
 for i,game in enumerate(games):
  v=game.get("gameView",{})
  if v.get("gameType")!="R": continue
  key=f"{source_date}_{i:02d}"
  for j,line in enumerate((game.get("odds") or {}).get("totals") or []):
   opening=line.get("openingLine") or {};current=line.get("currentLine") or {}
   rows.append({"source_game_key":key,"source_date":source_date,"start_datetime":v.get("startDate"),
    "away_team_original":(v.get("awayTeam") or {}).get("fullName"),"home_team_original":(v.get("homeTeam") or {}).get("fullName"),
    "away_runs":v.get("awayTeamScore"),"home_runs":v.get("homeTeamScore"),"sportsbook":line.get("sportsbook"),
    "opening_total":opening.get("total"),"opening_over_odds":opening.get("overOdds"),"opening_under_odds":opening.get("underOdds"),
    "current_total":current.get("total"),"current_over_odds":current.get("overOdds"),"current_under_odds":current.get("underOdds")})
totals=pd.DataFrame(rows);totals["actual_total_runs"]=pd.to_numeric(totals.home_runs,errors="coerce")+pd.to_numeric(totals.away_runs,errors="coerce")
matches=pd.read_csv(MATCHES);unique=matches[matches.match_status.eq("matched")][["source_game_key","game_id","season","match_status","match_method"]]
if unique.source_game_key.duplicated().any():raise ValueError("Validated mapping is not one-to-one")
totals=totals.merge(unique,on="source_game_key",how="left",validate="many_to_one");totals["match_status"]=totals.match_status.fillna("unmatched_to_model")
matched=totals[totals.match_status.eq("matched")]
audit=[]
for y in YEARS:
 model=matches[matches.season.eq(y)];lines=matched[matched.season.eq(y)]
 games=lines.game_id.nunique();audit.append({"season":y,"total_model_games":len(model),"matched_games_with_totals":games,
  "sportsbook_total_records":len(lines),"coverage_pct":games/len(model),"duplicate_book_game_records":int(lines.duplicated(["game_id","sportsbook"]).sum())})
audit=pd.DataFrame(audit)
if matched.duplicated(["game_id","sportsbook"]).any():raise ValueError("Duplicate totals book/game")
os.makedirs("data/processed",exist_ok=True);os.makedirs("results",exist_ok=True)
totals.to_csv("data/processed/historical_mlb_totals_2022_2025.csv",index=False);audit.to_csv("results/historical_totals_matching_audit.csv",index=False)
coverage=(matched.groupby(["season","sportsbook"]).game_id.nunique().rename("matched_games").reset_index())
coverage.to_csv("results/historical_totals_sportsbook_coverage.csv",index=False)
print(f"Totals records: {len(totals):,}; uniquely matched: {len(matched):,}; matched games: {matched.game_id.nunique():,}")
print(audit.to_string(index=False,float_format=lambda x:f"{x:.4%}"));print("\nSportsbooks:\n",coverage.to_string(index=False))
