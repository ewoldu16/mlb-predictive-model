"""Build fast, strictly pregame official bullpen and reliever ERA/WHIP features."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
YEARS=range(2021,2026);RAW=Path("data/raw/official_mlb_pitching");OUT=Path("data/processed")
SUM=["outs","earnedRuns","hits","baseOnBalls","strikeOuts","homeRuns","numberOfPitches","battersFaced"];ZERO=np.zeros(len(SUM))
def rates(v,p):
 v=dict(zip(SUM,v));ip=v["outs"]/3
 return {f"{p}_era":9*v["earnedRuns"]/ip if ip else np.nan,f"{p}_whip":(v["hits"]+v["baseOnBalls"])/ip if ip else np.nan,f"{p}_k_pct":v["strikeOuts"]/v["battersFaced"] if v["battersFaced"] else np.nan,f"{p}_bb_pct":v["baseOnBalls"]/v["battersFaced"] if v["battersFaced"] else np.nan,f"{p}_hr9":9*v["homeRuns"]/ip if ip else np.nan,f"{p}_outs":v["outs"],f"{p}_bf":v["battersFaced"]}
def make_hist(d,keys):
 out={};daily=d.groupby(keys+["date"],as_index=False)[SUM].sum().sort_values(keys+["date"]);grouper=keys[0] if len(keys)==1 else keys
 for key,g in daily.groupby(grouper,sort=False):
  if not isinstance(key,tuple):key=(int(key),)
  out[tuple(int(x) for x in key)]=(g.date.to_numpy(dtype="datetime64[ns]"),np.vstack([ZERO,g[SUM].cumsum().to_numpy(float)]))
 return out
def snap(item,date,days=None):
 if item is None:return ZERO.copy()
 ds,c=item;e=np.searchsorted(ds,np.datetime64(date),"left");s=0 if days is None else np.searchsorted(ds,np.datetime64(date-pd.Timedelta(days=days)),"left");return c[e]-c[s]
def last_before(item,date):
 if item is None:return pd.NaT
 ds=item[0];i=np.searchsorted(ds,np.datetime64(date),"left");return pd.Timestamp(ds[i-1]) if i else pd.NaT
def build(y):
 games=pd.read_csv(f"data/raw/games_{y}.csv");games.date=pd.to_datetime(games.date);games=games.merge(pd.read_csv(f"data/raw/starting_pitchers_{y}.csv"),on="game_id",validate="one_to_one")
 d=pd.read_csv(RAW/f"official_pitcher_game_lines_{y}.csv");d.date=pd.to_datetime(d.date);d["is_starter"]=d.is_starter.astype(str).str.lower().isin(["true","1"])
 ids=d[["game_id","team_side","team_id"]].drop_duplicates();home=ids[ids.team_side.eq("home")][["game_id","team_id"]].rename(columns={"team_id":"home_team_id"});away=ids[ids.team_side.eq("away")][["game_id","team_id"]].rename(columns={"team_id":"away_team_id"});team_ids=d.dropna(subset=["team_name","team_id"]).drop_duplicates("team_name").set_index("team_name").team_id.to_dict();games=games.merge(home,on="game_id",how="left",validate="one_to_one").merge(away,on="game_id",how="left",validate="one_to_one");games["home_team_id"]=games.home_team_id.fillna(games.home_team.map(team_ids));games["away_team_id"]=games.away_team_id.fillna(games.away_team.map(team_ids))
 if games[["home_team_id","away_team_id"]].isna().any().any():raise ValueError("official bullpen team-ID mapping unavailable for target game")
 rel=d[~d.is_starter].copy();team_hist=make_hist(rel,["team_id"]);pitch_hist=make_hist(rel,["team_id","pitcher_id"]);team_pitchers={t:[p for tt,p in pitch_hist if tt==t] for t in set(int(x) for x in rel.team_id)}
 rows=[];individual=[];audit=[]
 for game in games.itertuples(index=False):
  row={"game_id":game.game_id}
  for side,tid,current_sp in (("home",int(game.home_team_id),int(game.home_starter_id)),("away",int(game.away_team_id),int(game.away_starter_id))):
   vals=rates(snap(team_hist.get((tid,)),game.date),"season")|rates(snap(team_hist.get((tid,)),game.date,30),"l30");pool=[];available=[]
   for pid in team_pitchers.get(tid,[]):
    if pid==current_sp:continue
    item=pitch_hist[(tid,pid)];last=last_before(item,game.date)
    if pd.isna(last) or (game.date-last).days>30:continue
    season=snap(item,game.date);l30=snap(item,game.date,30);p1=snap(item,game.date,1)[6];p2=snap(item,game.date,2)[6];p3=snap(item,game.date,3)[6]
    dates=item[0];pos=np.searchsorted(dates,np.datetime64(game.date),"left");consec=0
    for lag in (1,2,3):
     if pos-lag>=0 and pd.Timestamp(dates[pos-lag]).normalize()==(game.date-pd.Timedelta(days=lag)).normalize():consec+=1
     else:break
    avail=p1<30 and p2<50 and consec<3;rec={"game_id":game.game_id,"team_side":side,"team_id":tid,"pitcher_id":pid,"days_rest":(game.date-last).days,"pitches_prior_1d":p1,"pitches_prior_2d":p2,"pitches_prior_3d":p3,"consecutive_days":consec,"likely_available":avail};rec.update(rates(season,"season"));rec.update(rates(l30,"l30"));individual.append(rec);pool.append(season)
    if avail:available.append(season)
   vals|=rates(np.sum(pool,axis=0) if pool else ZERO,"roster_pool")|rates(np.sum(available,axis=0) if available else ZERO,"available_pool");vals["roster_pool_pitchers"]=len(pool);vals["available_pool_pitchers"]=len(available);row.update({f"{side}_bp_official_{k}":v for k,v in vals.items()})
  for c in list(row):
   if c.startswith("home_bp_official_") and not c.endswith(("_outs","_bf","_pitchers")):
    base=c[len("home_bp_official_"):];row[f"bp_official_{base}_diff"]=row[c]-row.get(f"away_bp_official_{base}",np.nan)
  rows.append(row)
  if len(audit)<10:
   used=[last_before(team_hist.get((int(t),)),game.date) for t in (game.home_team_id,game.away_team_id)];used=[x for x in used if pd.notna(x)];latest=max(used) if used else pd.NaT;audit.append({"year":y,"game_id":game.game_id,"target_date":game.date.date(),"latest_source_date":latest.date() if pd.notna(latest) else None,"passed":pd.isna(latest) or latest<game.date})
 o=pd.DataFrame(rows);ind=pd.DataFrame(individual);nums=o.drop(columns="game_id")
 if len(o)!=len(games) or o.game_id.duplicated().any() or np.isinf(nums).any().any():raise ValueError("bullpen validation failed")
 path=OUT/f"features_official_bullpen_pitching_{y}.csv";ipath=OUT/f"official_reliever_pregame_{y}.csv";o.to_csv(path,index=False);ind.to_csv(ipath,index=False);print(f"{y}: games={len(o)} reliever-history rows={len(ind)} duplicates=0 infinities=0 missing={nums.isna().sum().sum()} -> {path}",flush=True);return audit
def main():
 p=argparse.ArgumentParser();p.add_argument("--year",type=int,choices=YEARS);x=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);a=[]
 for y in ([x.year] if x.year else YEARS):a.extend(build(y))
 pd.DataFrame(a).to_csv("results/official_bullpen_pitching_temporal_audit.csv",index=False)
if __name__=="__main__":main()
