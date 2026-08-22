"""Pregame offense features adjusted for quality of previously faced starters."""
import argparse, os, time
import numpy as np
import pandas as pd

YEARS=range(2021,2026); K={"strikeout","strikeout_double_play"};BB={"walk","intent_walk"}
TOTALS=["pa","woba_num","woba_den","k","bb","quality_pa","quality_num","opp_k_num","opp_bb_num","opp_xwoba_num","opp_hardhit_num","opp_barrel_num","opp_csw_num","woba_quality_num","k_relative_num","bb_relative_num","games"]

def div(a,b):return a.div(b.where(b.ne(0)))
def load_games(y):
 g=pd.read_csv(f"data/raw/games_{y}.csv");g.date=pd.to_datetime(g.date)
 v=pd.read_csv(f"data/processed/games_{y}_starter_lineup_matchup_features.csv")
 r=pd.read_csv(f"data/processed/features_richer_starter_{y}.csv")
 return g,v.merge(r,on="game_id",validate="one_to_one")

def offense_games(y,g,v):
 cols=["game_date","game_pk","inning_topbot","events","woba_value","woba_denom"]
 p=pd.read_csv(f"data/raw/statcast_enriched_{y}.csv",usecols=cols);p.game_date=pd.to_datetime(p.game_date);p=p[p.game_pk.isin(g.game_id)].merge(g[["game_id","home_team","away_team"]].rename(columns={"game_id":"game_pk"}),on="game_pk",validate="many_to_one")
 p["team"]=np.where(p.inning_topbot.eq("Top"),p.away_team,p.home_team);p["pa"]=p.events.notna().astype(int);p["k"]=p.events.isin(K).astype(int);p["bb"]=p.events.isin(BB).astype(int);p["woba_num"]=p.woba_value.fillna(0);p["woba_den"]=p.woba_denom.fillna(0)
 d=p.groupby(["game_pk","game_date","team"],as_index=False).agg(pa=("pa","sum"),woba_num=("woba_num","sum"),woba_den=("woba_den","sum"),k=("k","sum"),bb=("bb","sum"))
 rows=[]
 for side,teamcol,opp in [("home","home_team","away"),("away","away_team","home")]:
  z=v[["game_id",teamcol,f"{opp}_sp_season_k_pct",f"{opp}_sp_season_bb_pct",f"{opp}_sp_season_xwoba_allowed",f"{opp}_sp_season_whiff_rate",f"{opp}_sp_rich_season_hardhit_pct",f"{opp}_sp_rich_season_barrel_pct",f"{opp}_sp_rich_season_csw_pct"]].copy()
  z.columns=["game_pk","team","opp_k","opp_bb","opp_xwoba","opp_whiff","opp_hardhit","opp_barrel","opp_csw"];rows.append(z)
 e=pd.concat(rows);d=d.merge(e,on=["game_pk","team"],how="left",validate="one_to_one")
 # Transparent continuous pregame SP quality: dominance/command minus expected/contact damage.
 d["starter_quality"]=d[["opp_k","opp_whiff","opp_csw"]].mean(axis=1,skipna=False)-d[["opp_bb","opp_xwoba","opp_hardhit","opp_barrel"]].mean(axis=1,skipna=False)
 d["quality_pa"]=d.pa.where(d.starter_quality.notna(),0);d["quality_num"]=d.starter_quality*d.pa
 for x in ["opp_k","opp_bb","opp_xwoba","opp_hardhit","opp_barrel","opp_csw"]:d[x+"_num"]=d[x]*d.pa
 actual_woba=div(d.woba_num,d.woba_den);actual_k=div(d.k,d.pa);actual_bb=div(d.bb,d.pa)
 d["woba_quality_num"]=(actual_woba-d.opp_xwoba)*d.pa;d["k_relative_num"]=(actual_k-d.opp_k)*d.pa;d["bb_relative_num"]=(actual_bb-d.opp_bb)*d.pa;d["games"]=1
 return d

def histories(d):
 d=d.sort_values(["team","game_date"])
 for x in TOTALS:d["c_"+x]=d.groupby("team",sort=False)[x].cumsum()
 return d

def targets(g):
 h=g[["game_id","date","home_team"]].set_axis(["game_id","date","team"],axis=1);h["side"]="home";a=g[["game_id","date","away_team"]].set_axis(["game_id","date","team"],axis=1);a["side"]="away";z=pd.concat([h,a],ignore_index=True);z["rid"]=range(len(z));return z

def snap(z,d,q,label):
 l=z[["rid","team",q]].rename(columns={q:"query"}).sort_values("query");r=d[["team","game_date"]+["c_"+x for x in TOTALS]].sort_values("game_date")
 x=pd.merge_asof(l,r,left_on="query",right_on="game_date",by="team",direction="backward",allow_exact_matches=False)
 return x[["rid"]+["c_"+v for v in TOTALS]].rename(columns={"c_"+v:label+"_"+v for v in TOTALS})

def derive(x,prefix):
 x[prefix+"avg_starter_quality"]=div(x[prefix+"quality_num"],x[prefix+"quality_pa"])
 for n in ["opp_k","opp_bb","opp_xwoba","opp_hardhit","opp_barrel","opp_csw"]:x[prefix+"avg_"+n]=div(x[prefix+n+"_num"],x[prefix+"quality_pa"])
 x[prefix+"woba"]=div(x[prefix+"woba_num"],x[prefix+"woba_den"]);x[prefix+"k_pct"]=div(x[prefix+"k"],x[prefix+"pa"]);x[prefix+"bb_pct"]=div(x[prefix+"bb"],x[prefix+"pa"])
 x[prefix+"woba_vs_expected"]=div(x[prefix+"woba_quality_num"],x[prefix+"quality_pa"]);x[prefix+"k_pct_relative"]=div(x[prefix+"k_relative_num"],x[prefix+"quality_pa"]);x[prefix+"bb_pct_relative"]=div(x[prefix+"bb_relative_num"],x[prefix+"quality_pa"])
 x[prefix+"quality_weighted_woba"]=x[prefix+"woba"]*x[prefix+"avg_starter_quality"]

def build(y,n=None,save=True):
 t=time.perf_counter();g,v=load_games(y);d=histories(offense_games(y,g,v));z=targets(g)
 if n:ids=g.sort_values(["date","game_id"]).head(n).game_id;g=g[g.game_id.isin(ids)];z=z[z.game_id.isin(ids)]
 z["q0"]=z.date;x=z.merge(snap(z,d,"q0","season"),on="rid")
 for days in [15,30]:
  q="q"+str(days);z[q]=z.date-pd.Timedelta(days=days);x=x.merge(snap(z,d,q,"old"+str(days)),on="rid")
  for col in TOTALS:x[f"l{days}_"+col]=x["season_"+col].fillna(0)-x[f"old{days}_"+col].fillna(0)
 for pre in ["season_","l15_","l30_"]:derive(x,pre)
 measures=["avg_starter_quality","avg_opp_k","avg_opp_bb","avg_opp_xwoba","avg_opp_hardhit","avg_opp_barrel","avg_opp_csw","woba","k_pct","bb_pct","woba_vs_expected","k_pct_relative","bb_pct_relative","quality_weighted_woba","pa","quality_pa","games"]
 q=x[["game_id","side"]+[pre+m for pre in ["season_","l15_","l30_"] for m in measures]].pivot(index="game_id",columns="side");q.columns=[f"{side}_oq_{v}" for v,side in q.columns];o=g[["game_id"]].merge(q.reset_index(),on="game_id",how="left",validate="one_to_one")
 diffs={}
 for c in [c for c in o if c.startswith("home_oq_")]:base=c[8:];diffs["oq_"+base+"_diff"]=o[c]-o["away_oq_"+base]
 o=pd.concat([o,pd.DataFrame(diffs)],axis=1);runtime=time.perf_counter()-t;validate(o,g,d,y,runtime)
 if save:path=f"data/processed/features_opponent_quality_offense_{y}.csv";o.to_csv(path,index=False);print("Saved",os.path.abspath(path))

def validate(o,g,d,y,runtime):
 v=o.drop(columns="game_id");print(f"\n{y}: runtime={runtime:.1f}s games={len(o)} coverage={o.game_id.isin(g.game_id).sum()}/{len(g)} offense-history rows={len(d)} duplicates={o.game_id.duplicated().sum()} infinities={np.isinf(v).sum().sum()}")
 print("Missingness:\n"+v.isna().sum().to_string());den=[c for c in v if c.endswith(("_pa","_quality_pa","_games")) and not c.endswith("_diff")];print("Sample-size distributions:\n"+v[den].describe().T[["min","25%","50%","75%","max"]].to_string())
 if len(o)!=len(g) or set(o.game_id)!=set(g.game_id) or o.game_id.duplicated().any() or np.isinf(v).any().any():raise ValueError("validation failed")

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--year",type=int);p.add_argument("--sample-games",type=int);p.add_argument("--no-save",action="store_true");a=p.parse_args()
 for y in ([a.year] if a.year else YEARS):build(y,a.sample_games,not a.no_save)
