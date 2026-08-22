"""Build pregame team-bullpen traditional/Statcast quality rates."""
from pathlib import Path
import numpy as np
import pandas as pd

YEARS=range(2021,2026); RAW=Path("data/raw"); OUT=Path("data/processed")
K={"strikeout","strikeout_double_play"}; BB={"walk","intent_walk"}; H={"single","double","triple","home_run"}
OUTS={"strikeout":1,"strikeout_double_play":2,"field_out":1,"force_out":1,"grounded_into_double_play":2,
      "double_play":2,"triple_play":3,"sac_fly":1,"sac_bunt":1,"fielders_choice_out":1}
COLS=["bf","h","bb","k","hr","outs","woba_num","woba_den","xwoba_num","xwoba_den"]

def safe(n,d): return n/d if d>0 else np.nan
def metric(c,p):
    if c is None:return {f"{p}_{x}":np.nan for x in ["whip_proxy","k_pct","bb_pct","hr9_proxy","woba_allowed","xwoba_allowed"]}|{f"{p}_bf":0,f"{p}_outs":0}
    ip=c["outs"]/3
    return {f"{p}_whip_proxy":safe(c["h"]+c["bb"],ip),f"{p}_k_pct":safe(c["k"],c["bf"]),f"{p}_bb_pct":safe(c["bb"],c["bf"]),
      f"{p}_hr9_proxy":safe(9*c["hr"],ip),f"{p}_woba_allowed":safe(c["woba_num"],c["woba_den"]),
      f"{p}_xwoba_allowed":safe(c["xwoba_num"],c["xwoba_den"]),f"{p}_bf":c["bf"],f"{p}_outs":c["outs"]}

def build(year):
    games=pd.read_csv(RAW/f"games_{year}.csv");games["date"]=pd.to_datetime(games.date)
    codes=pd.read_csv(f"data/processed/games_{year}_starter_lineup_matchup_features.csv",usecols=["game_id","home_team_code","away_team_code"])
    games=games.merge(codes,on="game_id",validate="one_to_one")
    starters=pd.read_csv(RAW/f"starting_pitchers_{year}.csv"); sm={}
    for r in starters.itertuples(index=False): sm[(r.game_id,"away")]=r.away_starter_id;sm[(r.game_id,"home")]=r.home_starter_id
    use=["game_pk","game_date","pitcher","inning_topbot","home_team","away_team","events","woba_value","woba_denom","estimated_woba_using_speedangle"]
    p=pd.read_csv(RAW/f"statcast_enriched_{year}.csv",usecols=use,low_memory=False)
    p=p[p.game_pk.isin(games.game_id)&p.events.notna()].copy();p["date"]=pd.to_datetime(p.game_date)
    p["pitcher"]=pd.to_numeric(p.pitcher,errors="coerce");p["pitching_side"]=np.where(p.inning_topbot.eq("Top"),"home","away")
    p["starter_id"]=[sm.get((g,s),np.nan) for g,s in zip(p.game_pk,p.pitching_side)]
    p=p[p.pitcher.ne(p.starter_id)]
    p["team"]=np.where(p.pitching_side.eq("home"),p.home_team,p.away_team);p["bf"]=1;p["h"]=p.events.isin(H).astype(int)
    p["bb"]=p.events.isin(BB).astype(int);p["k"]=p.events.isin(K).astype(int);p["hr"]=p.events.eq("home_run").astype(int);p["outs"]=p.events.map(OUTS).fillna(0)
    p["woba_num"]=pd.to_numeric(p.woba_value,errors="coerce").fillna(0);p["woba_den"]=pd.to_numeric(p.woba_denom,errors="coerce").fillna(0)
    est=pd.to_numeric(p.estimated_woba_using_speedangle,errors="coerce"); actual=pd.to_numeric(p.woba_value,errors="coerce");p["xwoba_num"]=est.where(est.notna(),actual).fillna(0);p["xwoba_den"]=(est.notna()|actual.notna()).astype(int)
    day=p.groupby(["team","date"],as_index=False)[COLS].sum().sort_values(["team","date"]);hist={}
    for t,g in day.groupby("team"): hist[t]=(g.date.to_numpy(dtype="datetime64[ns]"),np.vstack([np.zeros(len(COLS)),g[COLS].cumsum().to_numpy(float)]))
    def get(team,date,days=None):
      if team not in hist:return None
      d,c=hist[team];e=np.searchsorted(d,np.datetime64(date),"left");s=0 if days is None else np.searchsorted(d,np.datetime64(date-pd.Timedelta(days=days)),"left");return dict(zip(COLS,c[e]-c[s]))
    rows=[];audit=[]
    for g in games.itertuples(index=False):
      row={"game_id":g.game_id}
      for side,t in (("home",g.home_team_code),("away",g.away_team_code)):
        vals=metric(get(t,g.date),"season")|metric(get(t,g.date,30),"l30");row.update({f"{side}_{k}":v for k,v in vals.items()})
      for k in list(row):
        if k.startswith("home_") and not k.endswith(("_bf","_outs")):
          base=k[5:];row[f"bp_{base}_diff"]=row[k]-row.get(f"away_{base}",np.nan)
      rows.append(row)
      if len(audit)<8:
        used=[]
        for t in (g.home_team_code,g.away_team_code):
          if t in hist:
            ds=hist[t][0];pos=np.searchsorted(ds,np.datetime64(g.date),"left")
            if pos:used.append(pd.Timestamp(ds[pos-1]))
        latest=max(used) if used else pd.NaT
        audit.append({"year":year,"game_id":g.game_id,"target_date":g.date.date(),"latest_source_date":latest.date() if pd.notna(latest) else None,"passed":pd.isna(latest) or latest<g.date})
    out=pd.DataFrame(rows);out.to_csv(OUT/f"features_statsimpl_bullpen_traditional_{year}.csv",index=False)
    print(f"{year}: games={len(out)}, relief BF rows={len(p)}, duplicates={out.game_id.duplicated().sum()}, infinities={np.isinf(out.select_dtypes('number')).sum().sum()}")
    print(out.filter(regex="(_bf|_outs)$").describe().loc[["min","50%","max"]].round(1).to_string())
    return audit
def main():
  OUT.mkdir(parents=True,exist_ok=True);a=[]
  for y in YEARS:a.extend(build(y))
  pd.DataFrame(a).to_csv("results/statsimpl_bullpen_traditional_temporal_audit.csv",index=False)
if __name__=="__main__":main()
