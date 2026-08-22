"""Build pregame trailing-100-TBF and venue starter splits from local Statcast."""
from pathlib import Path
import numpy as np
import pandas as pd

YEARS=range(2021,2026); RAW=Path("data/raw"); OUT=Path("data/processed")
K={"strikeout","strikeout_double_play"}; BB={"walk","intent_walk"}; H={"single","double","triple","home_run"}

def summarize(x,prefix):
    n=len(x)
    if not n: return {f"{prefix}_{k}":np.nan for k in ["woba_allowed","xwoba_allowed","k_pct","bb_pct","hr_pct","hardhit_pct","barrel_pct"]}|{f"{prefix}_bf":0}
    contact=x.launch_speed.notna(); xw=x.xwoba_value.notna()
    return {f"{prefix}_woba_allowed":x.woba_value.mean(),f"{prefix}_xwoba_allowed":x.loc[xw,"xwoba_value"].mean(),
      f"{prefix}_k_pct":x.events.isin(K).mean(),f"{prefix}_bb_pct":x.events.isin(BB).mean(),f"{prefix}_hr_pct":x.events.eq("home_run").mean(),
      f"{prefix}_hardhit_pct":x.loc[contact,"launch_speed"].ge(95).mean(),f"{prefix}_barrel_pct":x.loc[contact,"launch_speed_angle"].eq(6).mean(),f"{prefix}_bf":n}

def build(year):
    games=pd.read_csv(RAW/f"games_{year}.csv"); games["date"]=pd.to_datetime(games.date)
    sp=pd.read_csv(RAW/f"starting_pitchers_{year}.csv"); games=games.merge(sp,on="game_id",how="left",validate="one_to_one")
    use=["game_pk","game_date","pitcher","stand","inning_topbot","events","woba_value","estimated_woba_using_speedangle","launch_speed","launch_speed_angle"]
    p=pd.read_csv(RAW/f"statcast_enriched_{year}.csv",usecols=use,low_memory=False)
    p=p[p.game_pk.isin(games.game_id)&p.events.notna()].copy(); p["date"]=pd.to_datetime(p.game_date)
    for c in ["pitcher","woba_value","estimated_woba_using_speedangle","launch_speed","launch_speed_angle"]: p[c]=pd.to_numeric(p[c],errors="coerce")
    p["xwoba_value"]=p.estimated_woba_using_speedangle.where(p.estimated_woba_using_speedangle.notna(),p.woba_value)
    p["pitcher"]=p.pitcher.astype("Int64"); groups={k:g.sort_values(["date","game_pk"]) for k,g in p.groupby("pitcher")}
    rows=[]; audit=[]
    for g in games.itertuples(index=False):
      row={"game_id":g.game_id}
      for side,pid,current_context in (("home",g.home_starter_id,"home"),("away",g.away_starter_id,"away")):
        h=groups.get(pid,p.iloc[0:0]); prior=h[h.date<g.date]
        vals={}
        for hand in ("L","R"):
          vals.update(summarize(prior.tail(100).loc[lambda z:z.stand.eq(hand)],f"recent100_{hand.lower()}hb"))
        venue=prior[(prior.inning_topbot.eq("Top") if current_context=="home" else prior.inning_topbot.eq("Bot"))]
        vals.update(summarize(venue,f"season_{current_context}"))
        row.update({f"{side}_{k}":v for k,v in vals.items()})
      for k in list(row):
        if k.startswith("home_") and not k.endswith("_bf"):
          base=k[5:]; row[f"sp_{base}_diff"]=row[k]-row.get(f"away_{base}",np.nan)
      rows.append(row)
      if len(audit)<8:
        used=[]
        for pid in (g.home_starter_id,g.away_starter_id):
          h=groups.get(pid,p.iloc[0:0]); d=h.loc[h.date<g.date,"date"]
          if len(d):used.append(d.max())
        latest=max(used) if used else pd.NaT
        audit.append({"year":year,"game_id":g.game_id,"target_date":g.date.date(),"latest_source_date":latest.date() if pd.notna(latest) else None,"passed":pd.isna(latest) or latest<g.date})
    out=pd.DataFrame(rows); out.to_csv(OUT/f"features_statsimpl_starter_recent100_{year}.csv",index=False)
    print(f"{year}: games={len(out)}, missing starter IDs={games[['home_starter_id','away_starter_id']].isna().sum().sum()}, duplicates={out.game_id.duplicated().sum()}, infinities={np.isinf(out.select_dtypes('number')).sum().sum()}")
    print(out.filter(regex="_bf$").describe().loc[["min","50%","max"]].round(1).to_string())
    return audit

def main():
  OUT.mkdir(parents=True,exist_ok=True); a=[]
  for y in YEARS:a.extend(build(y))
  pd.DataFrame(a).to_csv("results/statsimpl_starter_recent100_temporal_audit.csv",index=False)
if __name__=="__main__":main()
