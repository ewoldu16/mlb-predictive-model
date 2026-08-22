"""Build transparent, leakage-safe official run-prevention components."""
from pathlib import Path
import numpy as np
import pandas as pd

YEARS=range(2021,2026); RAW=Path("data/raw/official_mlb_pitching"); OUT=Path("data/processed")
def summary(x,p):
 starts=len(x);outs=x.outs.sum();ip=outs/3
 return {f"{p}_er_per_start":x.earnedRuns.sum()/starts if starts else np.nan,
  f"{p}_runs_per_start":x.runs.sum()/starts if starts else np.nan,
  f"{p}_ip_per_start":ip/starts if starts else np.nan,f"{p}_outs_per_start":outs/starts if starts else np.nan,
  f"{p}_era":9*x.earnedRuns.sum()/ip if ip else np.nan,f"{p}_starts":starts}
def build(y):
 games=pd.read_csv(f"data/raw/games_{y}.csv");games.date=pd.to_datetime(games.date);games=games.merge(pd.read_csv(f"data/raw/starting_pitchers_{y}.csv"),on="game_id",validate="one_to_one")
 d=pd.read_csv(RAW/f"official_pitcher_game_lines_{y}.csv");d.date=pd.to_datetime(d.date);d=d[d.is_starter.astype(str).str.lower().isin(["true","1"])]
 groups={int(k):g.sort_values(["date","game_id"]) for k,g in d.groupby("pitcher_id")};bp=pd.read_csv(OUT/f"features_official_bullpen_pitching_{y}.csv");games=games.merge(bp,on="game_id",validate="one_to_one")
 rows=[];audit=[]
 for g in games.itertuples(index=False):
  row={"game_id":g.game_id}
  for side,pid in (("home",g.home_starter_id),("away",g.away_starter_id)):
   h=groups.get(int(pid),d.iloc[0:0]);prior=h[h.date<g.date];vals=summary(prior,"season")|summary(prior[prior.date>=g.date-pd.Timedelta(days=30)],"l30")|summary(prior.tail(3),"recent3")
   expected=vals["recent3_ip_per_start"];bp_era=getattr(g,f"{side}_bp_official_available_pool_era")
   vals["expected_starter_innings"]=expected;vals["starter_er_component"]=vals["season_era"]*expected/9 if pd.notna(expected) and pd.notna(vals["season_era"]) else np.nan
   vals["expected_bullpen_innings"]=9-expected if pd.notna(expected) else np.nan;vals["available_bullpen_er_component"]=bp_era*(9-expected)/9 if pd.notna(expected) and pd.notna(bp_era) else np.nan
   vals["combined_pitching_er_component"]=vals["starter_er_component"]+vals["available_bullpen_er_component"] if pd.notna(vals["starter_er_component"]) and pd.notna(vals["available_bullpen_er_component"]) else np.nan
   row.update({f"{side}_rp_{k}":v for k,v in vals.items()})
  for c in list(row):
   if c.startswith("home_rp_") and not c.endswith("_starts"):
    base=c[8:];row[f"rp_{base}_diff"]=row[c]-row.get(f"away_rp_{base}",np.nan)
  rows.append(row)
  if len(audit)<10:
   used=[]
   for pid in (g.home_starter_id,g.away_starter_id):
    q=groups.get(int(pid),d.iloc[0:0]);q=q[q.date<g.date]
    if len(q):used.append(q.date.max())
   latest=max(used) if used else pd.NaT;audit.append({"year":y,"game_id":g.game_id,"target_date":g.date.date(),"latest_source_date":latest.date() if pd.notna(latest) else None,"passed":pd.isna(latest) or latest<g.date})
 o=pd.DataFrame(rows);nums=o.drop(columns="game_id")
 if len(o)!=len(games) or o.game_id.duplicated().any() or np.isinf(nums).any().any():raise ValueError("run-prevention validation failed")
 path=OUT/f"features_official_run_prevention_{y}.csv";o.to_csv(path,index=False);print(f"{y}: games={len(o)} duplicates=0 infinities=0 missing={nums.isna().sum().sum()} -> {path}");return audit
def main():
 OUT.mkdir(parents=True,exist_ok=True);a=[]
 for y in YEARS:a.extend(build(y))
 pd.DataFrame(a).to_csv("results/official_run_prevention_temporal_audit.csv",index=False)
if __name__=="__main__":main()
