"""Strictly pregame starter arsenal x actual nine-hitter lineup features."""
import argparse, os, time
import numpy as np
import pandas as pd

YEARS=range(2021,2026);ORDER={i:1.03-.03*i for i in range(1,10)}
SW={"swinging_strike","swinging_strike_blocked","foul","foul_tip","hit_into_play","hit_into_play_no_out","hit_into_play_score","missed_bunt","foul_bunt"};WH={"swinging_strike","swinging_strike_blocked","missed_bunt"}
M=["pitches","sw","wh","xs","xn","ws","wd","hard","contact","barrel","velo_sum","velo_n","pfx_x_sum","pfx_x_n","pfx_z_sum","pfx_z_n"];MIN_SP=25;MIN_HITTER=10;MIN_CONTACT=3

def prep(y):
 g=pd.read_csv(f"data/raw/games_{y}.csv");g.date=pd.to_datetime(g.date);s=pd.read_csv(f"data/raw/starting_pitchers_{y}.csv");g=g.merge(s,on="game_id",validate="one_to_one")
 l=pd.read_csv(f"data/raw/lineups/starting_lineups_{y}.csv");l=l[l.batting_order.between(1,9)&l.player_id.notna()].copy()
 cols=["game_date","game_pk","pitcher","batter","pitch_type","description","events","release_speed","pfx_x","pfx_z","stand","launch_speed","launch_speed_angle","woba_value","woba_denom","estimated_woba_using_speedangle"]
 p=pd.read_csv(f"data/raw/statcast_enriched_{y}.csv",usecols=cols);p.game_date=pd.to_datetime(p.game_date);p=p[p.game_pk.isin(g.game_id)&p.pitch_type.notna()].copy()
 p["sw"]=p.description.isin(SW).astype(int);p["wh"]=p.description.isin(WH).astype(int);p["xs"]=p.estimated_woba_using_speedangle.fillna(0);p["xn"]=p.estimated_woba_using_speedangle.notna().astype(int);p["ws"]=p.woba_value.fillna(0);p["wd"]=p.woba_denom.fillna(0)
 q=p.launch_speed.notna();p["hard"]=(q&p.launch_speed.ge(95)).astype(int);p["contact"]=q.astype(int);p["barrel"]=(q&p.launch_speed_angle.eq(6)).astype(int);p["pitches"]=1
 for source,name in [("release_speed","velo"),("pfx_x","pfx_x"),("pfx_z","pfx_z")]:p[name+"_sum"]=p[source].fillna(0);p[name+"_n"]=p[source].notna().astype(int)
 return g,l,p

def history(p,who):
 d=p.groupby([who,"pitch_type","game_date"],as_index=False)[M].sum().sort_values([who,"pitch_type","game_date"]);out={}
 for key,x in d.groupby([who,"pitch_type"],sort=False):
  out[key]={"dates":x.game_date.to_numpy("datetime64[ns]"),**{m:x[m].cumsum().to_numpy(float) for m in M}}
 return out

def totals(h,key,date,days=None):
 x=h.get(key)
 if x is None:return None
 d=np.datetime64(date,"ns");r=np.searchsorted(x["dates"],d,"left");left=0 if days is None else np.searchsorted(x["dates"],d-np.timedelta64(days,"D"),"left")
 if r==0 or r<=left:return None
 return {m:x[m][r-1]-(x[m][left-1] if left else 0) for m in M}

def rate(t,n,d):return t[n]/t[d] if t is not None and t[d]>0 else np.nan
def arsenal(ph,pid,date,days=None):
 rows=[]
 for key in ph:
  if key[0]!=pid:continue
  t=totals(ph,key,date,days)
  if t and t["pitches"]>=MIN_SP:rows.append({"pitch_type":key[1],"usage_n":t["pitches"],"velocity":rate(t,"velo_sum","velo_n"),"pfx_x":rate(t,"pfx_x_sum","pfx_x_n"),"pfx_z":rate(t,"pfx_z_sum","pfx_z_n"),"xwoba_allowed":rate(t,"xs","xn"),"whiff_allowed":rate(t,"wh","sw"),"hardhit_allowed":rate(t,"hard","contact"),"barrel_allowed":rate(t,"barrel","contact")})
 if not rows:return []
 den=sum(x["usage_n"] for x in rows)
 for x in rows:x["usage"]=x["usage_n"]/den
 return rows

def hitter_match(bh,bid,date,ars,days=None):
 vals=[];known_usage=0
 for pitch in ars:
  t=totals(bh,(bid,pitch["pitch_type"]),date,days)
  if not t or t["pitches"]<MIN_HITTER:continue
  xw=rate(t,"xs","xn");wo=rate(t,"ws","wd");wh=rate(t,"wh","sw");hh=rate(t,"hard","contact") if t["contact"]>=MIN_CONTACT else np.nan;br=rate(t,"barrel","contact") if t["contact"]>=MIN_CONTACT else np.nan
  vals.append((pitch["usage"],xw,wo,wh,hh,br,pitch["xwoba_allowed"]));known_usage+=pitch["usage"]
 if not vals or known_usage==0:return None
 def avg(i):return sum(w*v for w,*z in vals for v in [z[i-1]] if pd.notna(v))/sum(w for w,*z in vals if pd.notna(z[i-1])) if any(pd.notna(z[i-1]) for w,*z in vals) else np.nan
 xw=avg(1);opp=avg(6)
 return {"xwoba":xw,"woba":avg(2),"whiff":avg(3),"hardhit":avg(4),"barrel":avg(5),"advantage":xw-opp if pd.notna(xw) and pd.notna(opp) else np.nan,"pitch_usage_coverage":known_usage,"pitch_types_known":len(vals)}

def team(side,starter,date,lineup,ph,bh):
 ars=arsenal(ph,int(starter),date);recent=arsenal(ph,int(starter),date,30);hitters=[]
 for q in lineup.itertuples():
  h=hitter_match(bh,int(q.player_id),date,ars)
  if h:h.update(order=q.batting_order,weight=ORDER[q.batting_order]);hitters.append(h)
 def agg(rows,name):
  valid=[x for x in rows if pd.notna(x[name])];return sum(x["weight"]*x[name] for x in valid)/sum(x["weight"] for x in valid) if valid else np.nan
 def arsenal_avg(name):
  known=[x for x in ars if pd.notna(x[name])];return sum(x["usage"]*x[name] for x in known)/sum(x["usage"] for x in known) if known else np.nan
 out={"arsenal_pitch_types":len(ars),"arsenal_recent_pitch_types":len(recent),"arsenal_avg_velocity":arsenal_avg("velocity"),"arsenal_avg_pfx_x":arsenal_avg("pfx_x"),"arsenal_avg_pfx_z":arsenal_avg("pfx_z"),"known_hitters":len(hitters),"lineup_size":lineup.player_id.nunique(),"lineup_pitch_usage_coverage":agg(hitters,"pitch_usage_coverage")}
 for m in ["xwoba","woba","whiff","hardhit","barrel","advantage"]:out[m]=agg(hitters,m)
 out["favorable_hitter_share"]=sum(x["weight"] for x in hitters if pd.notna(x["advantage"]) and x["advantage"]>.05)/sum(ORDER.values()) if hitters else np.nan
 out["poor_hitter_share"]=sum(x["weight"] for x in hitters if pd.notna(x["advantage"]) and x["advantage"]<-.05)/sum(ORDER.values()) if hitters else np.nan
 top=[x for x in hitters if x["order"]<=3];out["top_order_advantage"]=agg(top,"advantage")
 return {f"{side}_arsenal_{k}":v for k,v in out.items()}

def build(y,n=None,save=True):
 started=time.perf_counter();g,l,p=prep(y);ph=history(p,"pitcher");bh=history(p,"batter")
 if n:g=g.sort_values(["date","game_id"]).head(n)
 rows=[];missing_starters=0;missing_lineups=0
 for q in g.itertuples():
  row={"game_id":q.game_id}
  for side,starter,opp_side in [("home",q.home_starter_id,"away"),("away",q.away_starter_id,"home")]:
   lu=l[(l.game_id==q.game_id)&l.team_side.eq(opp_side)]
   if pd.isna(starter):missing_starters+=1;continue
   if lu.player_id.nunique()!=9:missing_lineups+=1
   row.update(team(side,starter,q.date,lu,ph,bh))
  rows.append(row)
 o=pd.DataFrame(rows);diffs={}
 for c in [c for c in o if c.startswith("home_arsenal_")]:base=c[13:];ac="away_arsenal_"+base;diffs["arsenal_"+base+"_diff"]=o[c]-o[ac]
 o=pd.concat([o,pd.DataFrame(diffs)],axis=1);runtime=time.perf_counter()-started;validate(o,g,y,runtime,missing_starters,missing_lineups)
 if save:path=f"data/processed/features_arsenal_lineup_matchup_{y}.csv";o.to_csv(path,index=False);print("Saved",os.path.abspath(path))

def validate(o,g,y,runtime,ms,ml):
 v=o.drop(columns="game_id");print(f"\n{y}: runtime={runtime:.1f}s games={len(o)} coverage={o.game_id.isin(g.game_id).sum()}/{len(g)} duplicate IDs={o.game_id.duplicated().sum()} infinities={np.isinf(v).sum().sum()} missing starter IDs={ms} incomplete lineup sides={ml}")
 print("Missingness:\n"+v.isna().sum().to_string());print("Coverage/sample sizes:\n"+v[[c for c in v if any(k in c for k in ["known_hitters","lineup_size","pitch_types","coverage"])]].describe().T.to_string())
 rates=[c for c in v if any(k in c for k in ["whiff","hardhit","barrel","share","coverage"]) and not c.endswith("_diff")];checks={c:int(((v[c]<-1e-12)|(v[c]>1+1e-12)).sum()) for c in rates};bad=sum(checks.values())
 if bad:print("Bad proportion columns:",{k:n for k,n in checks.items() if n})
 print("Out-of-range values:",bad)
 if len(o)!=len(g) or set(o.game_id)!=set(g.game_id) or o.game_id.duplicated().any() or np.isinf(v).any().any() or bad:raise ValueError("validation failed")

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--year",type=int);p.add_argument("--sample-games",type=int);p.add_argument("--no-save",action="store_true");a=p.parse_args()
 for y in ([a.year] if a.year else YEARS):build(y,a.sample_games,not a.no_save)
