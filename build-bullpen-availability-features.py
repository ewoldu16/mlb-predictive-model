"""Vectorized leakage-safe bullpen availability builder."""
import argparse, os, time
import numpy as np
import pandas as pd

YEARS=range(2021,2026)
K={"strikeout","strikeout_double_play"}; BB={"walk","intent_walk"}
SW={"swinging_strike","swinging_strike_blocked","foul","foul_tip","hit_into_play","hit_into_play_no_out","hit_into_play_score","missed_bunt","foul_bunt"}
WH={"swinging_strike","swinging_strike_blocked","missed_bunt"}
T=["pitches","bf","k","bb","sw","wh","xs","xn","ws","wd","hard","contact","apps","late","close"]

def ratio(a,b): return a/b.where(b.ne(0))
def inputs(y):
 g=pd.read_csv(f"data/raw/games_{y}.csv"); g.date=pd.to_datetime(g.date)
 s=pd.read_csv(f"data/raw/starting_pitchers_{y}.csv")
 c=pd.read_csv(f"data/processed/games_{y}_starter_lineup_matchup_features.csv",usecols=["game_id","home_team_code","away_team_code"])
 g=g.merge(s,on="game_id",validate="one_to_one").merge(c,on="game_id",validate="one_to_one")
 cols=["game_date","game_pk","pitcher","events","description","inning","inning_topbot","home_team","away_team","bat_score","fld_score","launch_speed","estimated_woba_using_speedangle","woba_value","woba_denom"]
 p=pd.read_csv(f"data/raw/statcast_enriched_{y}.csv",usecols=cols); p.game_date=pd.to_datetime(p.game_date); p=p[p.game_pk.isin(g.game_id)].copy()
 p["team"]=np.where(p.inning_topbot.eq("Top"),p.home_team,p.away_team)
 m=pd.concat([g[["game_id","home_team_code","home_starter_id"]].set_axis(["game_pk","team","starter"],axis=1),g[["game_id","away_team_code","away_starter_id"]].set_axis(["game_pk","team","starter"],axis=1)])
 p=p.merge(m,on=["game_pk","team"],validate="many_to_one"); return g,p[p.pitcher.ne(p.starter)&p.pitcher.notna()].copy()

def appearances(p):
 p["bf"]=p.events.notna().astype(int); p["k"]=p.events.isin(K).astype(int); p["bb"]=p.events.isin(BB).astype(int); p["sw"]=p.description.isin(SW).astype(int); p["wh"]=p.description.isin(WH).astype(int)
 p["xs"]=p.estimated_woba_using_speedangle.fillna(0); p["xn"]=p.estimated_woba_using_speedangle.notna().astype(int); p["ws"]=p.woba_value.fillna(0); p["wd"]=p.woba_denom.fillna(0)
 q=p.launch_speed.notna(); p["hard"]=(q&p.launch_speed.ge(95)).astype(int); p["contact"]=q.astype(int); p["late"]=p.inning.ge(7).astype(int); p["close"]=(p.inning.ge(7)&p.bat_score.sub(p.fld_score).abs().le(3)).astype(int)
 a=p.groupby(["team","pitcher","game_date"],as_index=False).agg(pitches=("description","size"),bf=("bf","sum"),k=("k","sum"),bb=("bb","sum"),sw=("sw","sum"),wh=("wh","sum"),xs=("xs","sum"),xn=("xn","sum"),ws=("ws","sum"),wd=("wd","sum"),hard=("hard","sum"),contact=("contact","sum"),late=("late","max"),close=("close","max")).sort_values(["team","pitcher","game_date"]); a["apps"]=1; a.pitcher=a.pitcher.astype(int)
 for x in T:a["c_"+x]=a.groupby(["team","pitcher"],sort=False)[x].cumsum()
 return a

def gamegrid(g,a,n=None):
 h=g[["game_id","date","home_team_code"]].set_axis(["game_id","date","team"],axis=1);h["side"]="home"
 w=g[["game_id","date","away_team_code"]].set_axis(["game_id","date","team"],axis=1);w["side"]="away"; tg=pd.concat([h,w])
 if n: tg=tg[tg.game_id.isin(g.sort_values(["date","game_id"]).head(n).game_id)]
 z=tg.merge(a[["team","pitcher"]].drop_duplicates(),on="team");z["rid"]=range(len(z));return tg,z

def snap(z,a,datecol,label):
 l=z[["rid","team","pitcher",datecol]].rename(columns={datecol:"q"}).sort_values("q"); cols=["team","pitcher","game_date"]+["c_"+x for x in T]
 r=pd.merge_asof(l,a[cols].sort_values("game_date"),left_on="q",right_on="game_date",by=["team","pitcher"],direction="backward",allow_exact_matches=False)
 return r[["rid","game_date"]+["c_"+x for x in T]].rename(columns={"game_date":label+"_date",**{"c_"+x:label+"_"+x for x in T}})

def states(g,a,n=None):
 tg,z=gamegrid(g,a,n); z["q0"]=z.date; out=z.merge(snap(z,a,"q0","now"),on="rid")
 for d in [1,2,3,5,7,30,45]:
  z["q"]=z.date-pd.Timedelta(days=d); out=out.merge(snap(z,a,"q","d"+str(d)),on="rid")
 out=out[out.now_date.notna()].copy();out["days_since"]=(out.date-out.now_date).dt.days;out=out[out.days_since.between(1,30)]
 for d in [1,2,3,5,7,30,45]:
  for x in T:out[f"w{d}_{x}"]=out["now_"+x].fillna(0)-out[f"d{d}_"+x].fillna(0)
 out["k_rate"]=ratio(out.now_k,out.now_bf);out["bb_rate"]=ratio(out.now_bb,out.now_bf);out["xwoba"]=ratio(out.now_xs,out.now_xn);out["woba"]=ratio(out.now_ws,out.now_wd);out["whiff"]=ratio(out.now_wh,out.now_sw);out["hardhit"]=ratio(out.now_hard,out.now_contact)
 out["l30_k"]=ratio(out.w30_k,out.w30_bf);out["l30_bb"]=ratio(out.w30_bb,out.w30_bf);out["l30_xwoba"]=ratio(out.w30_xs,out.w30_xn)
 out["fatigue"]=out.w1_pitches/30+out.w2_pitches/60+out.w3_apps/3;out["availability"]=np.exp(-out.fatigue);out["quality"]=out.k_rate-out.bb_rate-out.xwoba;out["role"]=out.w45_close+.25*out.w45_late
 out["role_rank"]=out.groupby(["game_id","side"]).role.rank(pct=True);return tg,out

def aggregate(tg,x):
 x["aqn"]=x.quality*x.availability;x["aqd"]=x.availability.where(x.quality.notna(),0);x["hqn"]=x.quality*x.availability*(.25+x.role_rank);x["hqd"]=(x.availability*(.25+x.role_rank)).where(x.quality.notna(),0)
 x["rested"]=x.fatigue.lt(.5);x["fatigued"]=x.fatigue.ge(1.5);x["yesterday"]=x.w1_apps.gt(0);x["heavy"]=x.w1_pitches.ge(30)|x.w2_pitches.ge(45)
 sums=["w1_pitches","w2_pitches","w3_pitches","w1_bf","w2_bf","w3_bf","w2_apps","w3_apps","w5_apps","w7_apps","rested","fatigued","yesterday","heavy","aqn","aqd","hqn","hqd"]
 q=x.groupby(["game_id","side"],as_index=False).agg(pool_size=("pitcher","size"),**{v:(v,"sum") for v in sums},quality_max=("quality","max"),quality_med=("quality","median"),mean_fatigue=("fatigue","mean"))
 q["available_quality"]=ratio(q.aqn,q.aqd);q["high_leverage_quality"]=ratio(q.hqn,q.hqd);q["quality_concentration"]=q.quality_max-q.quality_med
 closer=x.sort_values(["game_id","side","role"],ascending=[1,1,0]).drop_duplicates(["game_id","side"])[["game_id","side","quality","fatigue","days_since"]].rename(columns={"quality":"closer_quality","fatigue":"closer_fatigue","days_since":"closer_days_since"});q=q.merge(closer,on=["game_id","side"])
 q=q.drop(columns=["aqn","aqd","hqn","hqd","quality_max","quality_med"]);w=q.pivot(index="game_id",columns="side");w.columns=[f"{s}_bp_avail_{v}" for v,s in w.columns];w=w.reset_index();o=tg[["game_id"]].drop_duplicates().merge(w,on="game_id",how="left")
 for c in [c for c in o if c.startswith("home_bp_avail_")]:o["bp_avail_"+c[14:]+"_diff"]=o[c]-o["away_"+c[5:]]
 return o

def build(y,n=None,save=True):
 t=time.perf_counter();g,p=inputs(y);a=appearances(p);tg,x=states(g,a,n);o=aggregate(tg,x);gg=g[g.game_id.isin(tg.game_id)]
 nums=o.drop(columns="game_id");runtime=time.perf_counter()-t
 print(f"\n{y}: runtime={runtime:.1f}s games={len(o)} reliever-history rows={len(x)} duplicates={o.game_id.duplicated().sum()} infinities={np.isinf(nums).sum().sum()}")
 print("Missingness:\n",nums.isna().sum().to_string());print("Sample bullpen states:\n",x[["game_id","date","side","team","pitcher","days_since","w1_pitches","w2_pitches","w3_apps","k_rate","bb_rate","xwoba","fatigue","role"]].head(8).to_string(index=False))
 if len(o)!=len(gg) or set(o.game_id)!=set(gg.game_id) or o.game_id.duplicated().any() or np.isinf(nums).any().any():raise ValueError("validation failed")
 if save:path=f"data/processed/features_v6_bullpen_availability_{y}.csv";o.to_csv(path,index=False);print("Saved",os.path.abspath(path))
 return runtime

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--year",type=int);p.add_argument("--sample-games",type=int);p.add_argument("--no-save",action="store_true");a=p.parse_args()
 for y in ([a.year] if a.year else YEARS):build(y,a.sample_games,not a.no_save)
