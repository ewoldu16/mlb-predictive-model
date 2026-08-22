"""Final integrity report for the official MLB pitching information layer."""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd

YEARS=range(2021,2026);RAW=Path("data/raw/official_mlb_pitching");OUT=Path("data/processed");RES=Path("results")
def main():
 season=[];examples=[]
 for y in YEARS:
  games=pd.read_csv(f"data/raw/games_{y}.csv");d=pd.read_csv(RAW/f"official_pitcher_game_lines_{y}.csv");sp=pd.read_csv(f"data/raw/starting_pitchers_{y}.csv")
  starts=d[d.is_starter.astype(str).str.lower().isin(["true","1"])];wide=starts.pivot(index="game_id",columns="team_side",values="pitcher_id").reset_index().merge(sp,on="game_id")
  mismatch=int((wide.home.ne(wide.home_starter_id)|wide.away.ne(wide.away_starter_id)).sum())
  sf=pd.read_csv(OUT/f"features_official_starter_pitching_{y}.csv");bf=pd.read_csv(OUT/f"features_official_bullpen_pitching_{y}.csv");ri=pd.read_csv(OUT/f"official_reliever_pregame_{y}.csv")
  season.append({"year":y,"games":len(games),"official_pitcher_lines":len(d),"official_starters":len(starts),"official_relievers":len(d)-len(starts),"starter_id_mismatches":mismatch,
   "saves_recorded":int(pd.to_numeric(d.saves,errors="coerce").sum()),"holds_recorded":int(pd.to_numeric(d.holds,errors="coerce").sum()),"starter_feature_rows":len(sf),"bullpen_feature_rows":len(bf),"reliever_pregame_rows":len(ri),
   "starter_duplicates":int(sf.game_id.duplicated().sum()),"bullpen_duplicates":int(bf.game_id.duplicated().sum()),"starter_infinities":int(np.isinf(sf.select_dtypes('number')).sum().sum()),"bullpen_infinities":int(np.isinf(bf.select_dtypes('number')).sum().sum()),"starter_missing_cells":int(sf.drop(columns="game_id").isna().sum().sum()),"bullpen_missing_cells":int(bf.drop(columns="game_id").isna().sum().sum())})
  for gid in random.Random(9000+y).sample(sorted(games.game_id),3):
   b=json.loads((RAW/str(y)/f"boxscore_{gid}.json").read_text(encoding="utf-8"));q=d[d.game_id.eq(gid)]
   api=[]
   for side in ("away","home"):
    for pid in b["teams"][side]["pitchers"]:
     s=b["teams"][side]["players"]["ID"+str(pid)]["stats"]["pitching"];api.append((side,int(pid),int(s.get("outs",0)),int(s.get("earnedRuns",0)),int(s.get("hits",0)),int(s.get("baseOnBalls",0))))
   norm=list(q[["team_side","pitcher_id","outs","earnedRuns","hits","baseOnBalls"]].itertuples(index=False,name=None));examples.append({"year":y,"game_id":gid,"api_pitchers":len(api),"normalized_pitchers":len(norm),"official_values_match":sorted(api)==sorted(norm)})
 s=pd.DataFrame(season);e=pd.DataFrame(examples);s.to_csv(RES/"official_mlb_pitching_layer_validation.csv",index=False);e.to_csv(RES/"official_mlb_pitching_random_boxscore_validation.csv",index=False)
 audits=pd.concat([pd.read_csv(RES/"official_starter_pitching_temporal_audit.csv"),pd.read_csv(RES/"official_bullpen_pitching_temporal_audit.csv")],ignore_index=True);audits.to_csv(RES/"official_mlb_pitching_temporal_audit.csv",index=False)
 if not ((s.starter_feature_rows==s.games)&(s.bullpen_feature_rows==s.games)&s.starter_id_mismatches.eq(0)&s.starter_duplicates.eq(0)&s.bullpen_duplicates.eq(0)&s.starter_infinities.eq(0)&s.bullpen_infinities.eq(0)).all():raise ValueError("season validation failed")
 if not e.official_values_match.all() or not audits.passed.astype(bool).all():raise ValueError("boxscore/temporal validation failed")
 print(s.to_string(index=False));print(f"\nRandom official boxscores: {e.official_values_match.sum()}/{len(e)} exact; temporal audits: {audits.passed.sum()}/{len(audits)} passed")
if __name__=="__main__":main()
