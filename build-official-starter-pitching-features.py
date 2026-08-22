"""Build strictly pregame official starter ERA/WHIP features."""
from pathlib import Path
import numpy as np
import pandas as pd

YEARS=range(2021,2026); RAW=Path("data/raw/official_mlb_pitching"); OUT=Path("data/processed")
SUM=["outs","earnedRuns","hits","baseOnBalls","strikeOuts","homeRuns","numberOfPitches","battersFaced"]
def rates(x,p):
    if x.empty:return {f"{p}_{k}":np.nan for k in ["era","whip","k_pct","bb_pct","hr9"]}|{f"{p}_outs":0,f"{p}_starts":0}
    s=x[SUM].sum();ip=s.outs/3
    return {f"{p}_era":9*s.earnedRuns/ip if ip else np.nan,f"{p}_whip":(s.hits+s.baseOnBalls)/ip if ip else np.nan,
      f"{p}_k_pct":s.strikeOuts/s.battersFaced if s.battersFaced else np.nan,f"{p}_bb_pct":s.baseOnBalls/s.battersFaced if s.battersFaced else np.nan,
      f"{p}_hr9":9*s.homeRuns/ip if ip else np.nan,f"{p}_outs":s.outs,f"{p}_starts":len(x)}
def build(y):
    games=pd.read_csv(f"data/raw/games_{y}.csv");games.date=pd.to_datetime(games.date)
    sp=pd.read_csv(f"data/raw/starting_pitchers_{y}.csv");games=games.merge(sp,on="game_id",validate="one_to_one")
    d=pd.read_csv(RAW/f"official_pitcher_game_lines_{y}.csv");d.date=pd.to_datetime(d.date);d=d[d.is_starter.astype(str).str.lower().isin(["true","1"])].copy()
    groups={int(k):g.sort_values(["date","game_id"]) for k,g in d.groupby("pitcher_id")};rows=[];audit=[]
    for game in games.itertuples(index=False):
      row={"game_id":game.game_id}
      for side,pid in (("home",game.home_starter_id),("away",game.away_starter_id)):
        h=groups.get(int(pid),d.iloc[0:0]);prior=h[h.date<game.date]
        vals=rates(prior,"season")|rates(prior[prior.date>=game.date-pd.Timedelta(days=30)],"l30")|rates(prior.tail(3),"recent3_starts")|rates(prior[prior.team_side.eq(side)],f"season_{side}")
        row.update({f"{side}_sp_official_{k}":v for k,v in vals.items()})
      for c in list(row):
        if c.startswith("home_sp_official_") and not c.endswith(("_outs","_starts")):
          base=c[len("home_sp_official_"):];row[f"sp_official_{base}_diff"]=row[c]-row.get(f"away_sp_official_{base}",np.nan)
      rows.append(row)
      if len(audit)<10:
        used=[]
        for pid in (game.home_starter_id,game.away_starter_id):
          q=groups.get(int(pid),d.iloc[0:0]);q=q[q.date<game.date]
          if len(q):used.append(q.date.max())
        latest=max(used) if used else pd.NaT;audit.append({"year":y,"game_id":game.game_id,"target_date":game.date.date(),"latest_source_date":latest.date() if pd.notna(latest) else None,"passed":pd.isna(latest) or latest<game.date})
    o=pd.DataFrame(rows);nums=o.drop(columns="game_id")
    if len(o)!=len(games) or o.game_id.duplicated().any() or np.isinf(nums).any().any():raise ValueError("starter validation failed")
    path=OUT/f"features_official_starter_pitching_{y}.csv";o.to_csv(path,index=False)
    print(f"{y}: games={len(o)} duplicates=0 infinities=0 missing={nums.isna().sum().sum()} -> {path}")
    print(o.filter(regex="(_outs|_starts)$").describe().loc[["min","50%","max"]].round(1).to_string())
    return audit
def main():
 OUT.mkdir(parents=True,exist_ok=True);a=[]
 for y in YEARS:a.extend(build(y))
 pd.DataFrame(a).to_csv("results/official_starter_pitching_temporal_audit.csv",index=False)
if __name__=="__main__":main()
