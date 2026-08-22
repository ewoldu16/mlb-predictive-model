"""Build independent, strictly pregame richer starting-pitcher features."""
import argparse, os, time
import numpy as np
import pandas as pd

YEARS=range(2021,2026)
K={"strikeout","strikeout_double_play"}; BB={"walk","intent_walk"}
SW={"swinging_strike","swinging_strike_blocked","foul","foul_tip","hit_into_play","hit_into_play_no_out","hit_into_play_score","missed_bunt","foul_bunt"}
WH={"swinging_strike","swinging_strike_blocked","missed_bunt"}
BALL={"ball","blocked_ball","pitchout","hit_by_pitch","intent_ball"}
TOTALS=["pitches","pa","k","bb","sw","wh","out_sw","out_pitches","zone_pitches","first_strikes","first_pitches","called","contact","hard","barrel","ev_sum","hr","fb","gb","ld","xsum","xn","starts"]
RATE_SPECS={
 "hardhit_pct":("hard","contact"),"barrel_pct":("barrel","contact"),"avg_exit_velocity":("ev_sum","contact"),
 "hr_per_pa":("hr","pa"),"hr_per_fb":("hr","fb"),"ground_ball_pct":("gb","pa"),"fly_ball_pct":("fb","pa"),
 "line_drive_pct":("ld","pa"),"k_minus_bb_pct":(("k","bb"),"pa"),"chase_pct":("out_sw","out_pitches"),
 "zone_pct":("zone_pitches","pitches"),"first_pitch_strike_pct":("first_strikes","first_pitches"),
 "csw_pct":(("called","wh"),"pitches")
}

def div(a,b): return a.div(b.where(b.ne(0)))
def load(y):
 g=pd.read_csv(f"data/raw/games_{y}.csv");g.date=pd.to_datetime(g.date)
 s=pd.read_csv(f"data/raw/starting_pitchers_{y}.csv");g=g.merge(s,on="game_id",validate="one_to_one")
 use=["game_date","game_pk","pitcher","description","events","balls","strikes","zone","launch_speed","launch_speed_angle","bb_type","estimated_woba_using_speedangle","inning_topbot"]
 p=pd.read_csv(f"data/raw/statcast_enriched_{y}.csv",usecols=use);p.game_date=pd.to_datetime(p.game_date);p=p[p.game_pk.isin(g.game_id)].copy()
 m=pd.concat([g[["game_id","home_starter_id"]].set_axis(["game_pk","starter"],axis=1),g[["game_id","away_starter_id"]].set_axis(["game_pk","starter"],axis=1)]).dropna();m.starter=m.starter.astype(int)
 p=p.merge(m,on="game_pk",how="inner");p=p[p.pitcher.eq(p.starter)].copy();return g,p

def summaries(p):
 p["pa"]=p.events.notna().astype(int);p["k"]=p.events.isin(K).astype(int);p["bb"]=p.events.isin(BB).astype(int);p["sw"]=p.description.isin(SW).astype(int);p["wh"]=p.description.isin(WH).astype(int)
 outside=p.zone.notna()&~p.zone.between(1,9);p["out_sw"]=(outside&p.description.isin(SW)).astype(int);p["out_pitches"]=outside.astype(int);p["zone_pitches"]=p.zone.between(1,9).astype(int)
 first=p.balls.eq(0)&p.strikes.eq(0);p["first_pitches"]=first.astype(int);p["first_strikes"]=(first&~p.description.isin(BALL)).astype(int);p["called"]=p.description.eq("called_strike").astype(int)
 contact=p.launch_speed.notna();p["contact"]=contact.astype(int);p["hard"]=(contact&p.launch_speed.ge(95)).astype(int);p["barrel"]=(contact&p.launch_speed_angle.eq(6)).astype(int);p["ev_sum"]=p.launch_speed.fillna(0)
 p["hr"]=p.events.eq("home_run").astype(int);p["fb"]=(p.events.notna()&(p.bb_type.eq("fly_ball")|p.events.eq("home_run"))).astype(int);p["gb"]=(p.events.notna()&p.bb_type.eq("ground_ball")).astype(int);p["ld"]=(p.events.notna()&p.bb_type.eq("line_drive")).astype(int);p["xsum"]=p.estimated_woba_using_speedangle.fillna(0);p["xn"]=p.estimated_woba_using_speedangle.notna().astype(int)
 p["context"]=np.where(p.inning_topbot.eq("Top"),"home","road")
 a=p.groupby(["pitcher","game_date","context"],as_index=False).agg(pitches=("description","size"),**{x:(x,"sum") for x in TOTALS if x not in {"pitches","starts"}});a["starts"]=1;a.pitcher=a.pitcher.astype(int);a=a.sort_values(["pitcher","game_date"])
 for x in TOTALS:a["c_"+x]=a.groupby("pitcher",sort=False)[x].cumsum();a["cc_"+x]=a.groupby(["pitcher","context"],sort=False)[x].cumsum()
 return a

def targets(g):
 h=g[["game_id","date","home_starter_id"]].set_axis(["game_id","date","pitcher"],axis=1);h["side"]="home";h["context"]="home"
 a=g[["game_id","date","away_starter_id"]].set_axis(["game_id","date","pitcher"],axis=1);a["side"]="away";a["context"]="road"
 z=pd.concat([h,a],ignore_index=True);z=z[z.pitcher.notna()].copy();z.pitcher=z.pitcher.astype(int);z["rid"]=range(len(z));return z

def snap(z,a,q,label,context=False):
 keys=["pitcher","context"] if context else ["pitcher"];prefix="cc_" if context else "c_"
 l=z[["rid"]+keys+[q]].rename(columns={q:"query"}).sort_values("query");r=a[keys+["game_date"]+[prefix+x for x in TOTALS]].sort_values("game_date")
 x=pd.merge_asof(l,r,left_on="query",right_on="game_date",by=keys,direction="backward",allow_exact_matches=False)
 return x[["rid"]+[prefix+v for v in TOTALS]].rename(columns={prefix+v:label+"_"+v for v in TOTALS})

def metrics(x,prefix):
 for name,(num,den) in RATE_SPECS.items():
  numerator=(x[prefix+num[0]]+x[prefix+num[1]] if name=="csw_pct" else x[prefix+num[0]]-x[prefix+num[1]]) if isinstance(num,tuple) else x[prefix+num]
  x[prefix+name]=div(numerator,x[prefix+den])

def build(y,n=None,save=True):
 started=time.perf_counter();g,p=load(y);a=summaries(p);z=targets(g)
 if n:
  selected=g.sort_values(["date","game_id"]).head(n).game_id
  z=z[z.game_id.isin(selected)];g=g[g.game_id.isin(selected)].copy()
 z["q0"]=z.date;z["q30"]=z.date-pd.Timedelta(days=30)
 x=z.merge(snap(z,a,"q0","season"),on="rid").merge(snap(z,a,"q30","old30"),on="rid")
 for v in TOTALS:x["l30_"+v]=x["season_"+v].fillna(0)-x["old30_"+v].fillna(0)
 x=x.merge(snap(z,a,"q0","context_season",True),on="rid").merge(snap(z,a,"q30","context_old30",True),on="rid")
 for v in TOTALS:x["context_l30_"+v]=x["context_season_"+v].fillna(0)-x["context_old30_"+v].fillna(0)
 for prefix in ["season_","l30_","context_season_","context_l30_"]:metrics(x,prefix)
 measures=list(RATE_SPECS)+["pa","contact","fb","out_pitches","first_pitches","pitches","starts"]
 keep=["game_id","side"]+[pre+m for pre in ["season_","l30_","context_season_","context_l30_"] for m in measures]
 q=x[keep].pivot(index="game_id",columns="side");q.columns=[f"{side}_sp_rich_{v}" for v,side in q.columns];out=q.reset_index();out=g[["game_id"]].merge(out,on="game_id",how="left",validate="one_to_one")
 for c in [c for c in out if c.startswith("home_sp_rich_")]:
  base=c[len("home_sp_rich_"):];hc=c;ac="away_sp_rich_"+base
  # Lower HR/contact-damage rates are better, but raw differential remains home-away consistently.
  out["sp_rich_"+base+"_diff"]=out[hc]-out[ac]
 runtime=time.perf_counter()-started;validate(out,g,y,len(a),runtime)
 if save:path=f"data/processed/features_richer_starter_{y}.csv";out.to_csv(path,index=False);print("Saved",os.path.abspath(path))

def validate(o,g,y,rows,runtime):
 v=o.drop(columns="game_id");print(f"\n{y}: runtime={runtime:.1f}s games={len(o)} coverage={o.game_id.isin(g.game_id).sum()}/{len(g)} starter-history rows={rows} missing starter IDs={g[['home_starter_id','away_starter_id']].isna().sum().sum()} duplicates={o.game_id.duplicated().sum()} infinities={np.isinf(v).sum().sum()}")
 print("Missingness:\n"+v.isna().sum().to_string());den=[c for c in v if any(k in c for k in ["_pa","_contact","_fb","_pitches","_starts"])];print("Sample sizes:\n"+v[den].describe().T[["min","25%","50%","75%","max"]].to_string())
 rates=[c for c in v if any(c.endswith(k) or (k+"_diff") in c for k in RATE_SPECS)];checks={c:int(((v[c]<0)|(v[c]>1)).sum()) for c in rates if "avg_exit_velocity" not in c and "k_minus_bb" not in c and not c.endswith("_diff")};bad=sum(checks.values())
 if bad: print("Bad rate columns:",{k:n for k,n in checks.items() if n})
 print("Out-of-range rate values:",bad)
 if len(o)!=len(g) or set(o.game_id)!=set(g.game_id) or o.game_id.duplicated().any() or np.isinf(v).any().any() or bad:raise ValueError("validation failed")

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--year",type=int);p.add_argument("--sample-games",type=int);p.add_argument("--no-save",action="store_true");z=p.parse_args()
 for y in ([z.year] if z.year else YEARS):build(y,z.sample_games,not z.no_save)
