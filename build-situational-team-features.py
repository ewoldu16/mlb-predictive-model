"""Low-cost, outcome-independent pregame team situation features."""
import argparse, os, time
import numpy as np
import pandas as pd
YEARS=range(2021,2026)

def team_games(g):
 h=g[["game_id","date","home_team","home_score","away_score"]].copy();h.columns=["game_id","date","team","rs","ra"];h["venue"]="home"
 a=g[["game_id","date","away_team","away_score","home_score"]].copy();a.columns=["game_id","date","team","rs","ra"];a["venue"]="away"
 d=pd.concat([h,a],ignore_index=True);d["win"]=(d.rs>d.ra).astype(int);return d.sort_values(["team","date","game_id"])

def state(hist,date,venue):
 p=hist[hist.date<date]
 if p.empty:return {}
 v=p[p.venue.eq(venue)];l10=p.tail(10);wins=p.win.sum();games=len(p);rs=p.rs.sum();ra=p.ra.sum();pyth=rs**2/(rs**2+ra**2) if rs**2+ra**2 else np.nan
 streak=0;last=int(p.iloc[-1].win)
 for x in p.win.iloc[::-1]:
  if int(x)==last:streak+=1
  else:break
 if last==0:streak=-streak
 return {"games":games,"win_pct":wins/games,"venue_games":len(v),"venue_win_pct":v.win.mean() if len(v) else np.nan,"l10_games":len(l10),"l10_win_pct":l10.win.mean(),"streak":streak,"run_diff":rs-ra,"run_diff_per_game":(rs-ra)/games,"pythagorean_win_pct":pyth,"actual_minus_pythagorean":wins/games-pyth,"previous_game_win":last,"l10_run_diff":l10.rs.sum()-l10.ra.sum(),"l10_run_diff_per_game":(l10.rs.sum()-l10.ra.sum())/len(l10)}

def build(y,save=True):
 t=time.perf_counter();g=pd.read_csv(f"data/raw/games_{y}.csv");g.date=pd.to_datetime(g.date);d=team_games(g);hist={team:x for team,x in d.groupby("team")};rows=[]
 for q in g.itertuples():
  row={"game_id":q.game_id}
  for side,team,venue in [("home",q.home_team,"home"),("away",q.away_team,"away")]:
   row.update({f"{side}_sit_{k}":v for k,v in state(hist[team],q.date,venue).items()})
  rows.append(row)
 o=pd.DataFrame(rows);diffs={}
 for c in [c for c in o if c.startswith("home_sit_")]:base=c[9:];diffs["sit_"+base+"_diff"]=o[c]-o["away_sit_"+base]
 o=pd.concat([o,pd.DataFrame(diffs)],axis=1);runtime=time.perf_counter()-t;validate(o,g,y,runtime)
 if save:path=f"data/processed/features_situational_{y}.csv";o.to_csv(path,index=False);print("Saved",os.path.abspath(path))

def validate(o,g,y,runtime):
 v=o.drop(columns="game_id");print(f"\n{y}: runtime={runtime:.1f}s games={len(o)} coverage={o.game_id.isin(g.game_id).sum()}/{len(g)} duplicates={o.game_id.duplicated().sum()} infinities={np.isinf(v).sum().sum()}");print("Missingness:\n"+v.isna().sum().to_string());print("Ranges:\n"+v.agg(["min","max"]).T.to_string())
 nonnegative=[c for c in v if any(k in c for k in ["_games","run_diff"]) and not c.endswith("_diff") and "run_diff" not in c];bad=sum((v[c]<0).sum() for c in nonnegative)
 rates=[c for c in v if ("win_pct" in c or "pythagorean_win_pct" in c) and "actual_minus" not in c and not c.endswith("_diff")];bad+=sum(((v[c]<0)|(v[c]>1)).sum() for c in rates)
 if len(o)!=len(g) or set(o.game_id)!=set(g.game_id) or o.game_id.duplicated().any() or np.isinf(v).any().any() or bad:raise ValueError("validation failed")

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--year",type=int);a=p.parse_args()
 for y in ([a.year] if a.year else YEARS):build(y)
