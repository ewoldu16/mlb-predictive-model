"""Develop and freeze V11 unified team-run predictions without sportsbook data."""
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom, poisson, spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent; D=ROOT/"data/processed"; R=ROOT/"results"; YEARS=range(2021,2026)
FOLDS=[([2021],2022),([2021,2022],2023),([2021,2022,2023],2024)]
AUDIT_IDS=[777007,777874,777382,777481,777249,777782,778175,777067,776866,777980,778279]
STEMS=["features_v7_offensive_form","features_v8_contextual_offense","features_statsimpl_offense_risp","features_richer_starter","features_official_starter_pitching","features_official_bullpen_pitching","features_v6_bullpen_availability","features_statsimpl_starter_recent100","features_arsenal_lineup_matchup","features_opponent_quality_offense","features_situational"]

OFF=["season_woba","l30_woba","lineup_season_woba","lineup_l30_woba","season_platoon_woba","l30_platoon_woba",
 "off_season_hardhit_pct","off_season_hr_fb","off_l7_woba","off_l7_k_pct","off_l7_bb_pct","off_l7_hardhit_pct","off_l15_woba","off_l15_k_pct","off_l15_bb_pct","off_l15_hardhit_pct","off_l30_woba","off_l30_k_pct","off_l30_bb_pct","off_l30_hardhit_pct","off_l30_hr_fb",
 "season_avg","season_obp","season_slg","season_ops","l30_avg","l30_obp","l30_slg","l30_ops","season_risp_woba","season_risp_avg","season_risp_ops","l30_risp_woba","l30_risp_avg","l30_risp_ops"]
SP=["sp_season_k_pct","sp_season_bb_pct","sp_season_hr_pct","sp_season_woba_allowed","sp_l30_k_pct","sp_l30_bb_pct","sp_l30_hr_pct","sp_l30_woba_allowed","sp_days_rest","sp_prev_pitch_count","sp_season_avg_pitch_count","sp_l30_avg_pitch_count","sp_season_fastball_velocity","sp_l30_fastball_velocity","sp_season_whiff_rate","sp_l30_whiff_rate","sp_season_xwoba_allowed","sp_l30_xwoba_allowed",
 "sp_rich_season_hardhit_pct","sp_rich_season_barrel_pct","sp_rich_season_hr_per_pa","sp_rich_season_hr_per_fb","sp_rich_season_k_minus_bb_pct","sp_rich_season_chase_pct","sp_rich_season_zone_pct","sp_rich_season_csw_pct","sp_rich_l30_hardhit_pct","sp_rich_l30_barrel_pct","sp_rich_l30_hr_per_pa","sp_rich_l30_k_minus_bb_pct","sp_rich_l30_chase_pct","sp_rich_l30_csw_pct",
 "sp_official_season_era","sp_official_season_whip","sp_official_season_k_pct","sp_official_season_bb_pct","sp_official_season_hr9","sp_official_l30_era","sp_official_l30_whip","sp_official_recent3_starts_era","sp_official_recent3_starts_whip"]
BP=["bp_season_k_pct","bp_season_bb_pct","bp_season_hr_pct","bp_season_woba_allowed","bp_l30_k_pct","bp_l30_bb_pct","bp_l30_woba_allowed","bp_l7_bf",
 "bp_official_season_era","bp_official_season_whip","bp_official_season_k_pct","bp_official_season_bb_pct","bp_official_l30_era","bp_official_l30_whip","bp_official_available_pool_era","bp_official_available_pool_whip","bp_official_available_pool_k_pct","bp_official_available_pool_bb_pct",
 "bp_avail_pool_size","bp_avail_w1_pitches","bp_avail_w2_pitches","bp_avail_w3_pitches","bp_avail_rested","bp_avail_fatigued","bp_avail_yesterday","bp_avail_heavy","bp_avail_mean_fatigue","bp_avail_available_quality","bp_avail_high_leverage_quality","bp_avail_closer_quality","bp_avail_closer_fatigue","bp_avail_closer_days_since"]
MATCH=["ctx_hand_season_woba","ctx_hand_season_k_pct","ctx_hand_season_bb_pct","ctx_hand_l30_woba","ctx_hand_l30_k_pct","ctx_hand_l30_bb_pct","ctx_combined_season_woba","ctx_combined_season_k_pct","ctx_combined_season_bb_pct","ctx_combined_l30_woba","ctx_combined_l30_k_pct","ctx_combined_l30_bb_pct",
 "sp_matchup_season_xwoba_allowed","sp_matchup_season_k_pct","sp_matchup_season_bb_pct","sp_matchup_season_whiff_pct","sp_matchup_l30_xwoba_allowed","sp_matchup_l30_k_pct","sp_matchup_l30_bb_pct","sp_matchup_l30_whiff_pct",
 "recent100_lhb_woba_allowed","recent100_lhb_xwoba_allowed","recent100_lhb_k_pct","recent100_lhb_bb_pct","recent100_lhb_hr_pct","recent100_lhb_hardhit_pct","recent100_lhb_barrel_pct","recent100_rhb_woba_allowed","recent100_rhb_xwoba_allowed","recent100_rhb_k_pct","recent100_rhb_bb_pct","recent100_rhb_hr_pct","recent100_rhb_hardhit_pct","recent100_rhb_barrel_pct",
 "arsenal_xwoba","arsenal_woba","arsenal_whiff","arsenal_hardhit","arsenal_barrel","arsenal_advantage","arsenal_favorable_hitter_share","arsenal_poor_hitter_share","arsenal_top_order_advantage"]
CONTEXT=["ctx_venue_season_woba","ctx_venue_season_k_pct","ctx_venue_season_bb_pct","ctx_venue_l30_woba","ctx_venue_l30_k_pct","ctx_venue_l30_bb_pct","oq_season_woba_vs_expected","oq_season_k_pct_relative","oq_season_bb_pct_relative","oq_season_quality_weighted_woba","oq_l15_woba_vs_expected","oq_l30_woba_vs_expected"]
STRENGTH=["sit_win_pct","sit_venue_win_pct","sit_l10_win_pct","sit_streak","sit_run_diff_per_game","sit_pythagorean_win_pct","sit_actual_minus_pythagorean","sit_previous_game_win","sit_l10_run_diff_per_game"]
COMPACT=["season_woba","l30_woba","lineup_season_woba","season_ops","off_l7_woba","season_risp_woba","ctx_hand_season_woba","opp_sp_season_xwoba_allowed","opp_sp_season_k_pct","opp_sp_season_bb_pct","opp_sp_official_season_era","opp_sp_official_season_whip","opp_bp_season_woba_allowed","opp_bp_official_l30_era","opp_bp_official_available_pool_era","home_indicator"]
FAMILY_OF={**{x:"Offensive baseline" for x in OFF},**{x:"Opposing starter" for x in SP},**{x:"Bullpen" for x in BP},**{x:"Matchup" for x in MATCH},**{x:"Context" for x in CONTEXT},**{x:"Team strength" for x in STRENGTH},"home_indicator":"Context","season_scoring_avg":"Offensive baseline","recent30_scoring_avg":"Recent offense"}
BUCKET_EDGES=[-np.inf,3,3.5,4,4.5,5,5.5,6,np.inf];BUCKETS=["<3.0","3.0-3.5","3.5-4.0","4.0-4.5","4.5-5.0","5.0-5.5","5.5-6.0","6.0+"]

def merge_year(y):
 b=pd.read_csv(D/f"games_{y}_starter_lineup_matchup_features.csv")
 for stem in STEMS:
  q=pd.read_csv(D/f"{stem}_{y}.csv");drop=[c for c in q if c in b and c!="game_id"];b=b.merge(q.drop(columns=drop),on="game_id",validate="one_to_one")
 return b

def value(g,col):return getattr(g,col,np.nan)
def long_year(y):
 b=merge_year(y);rows=[]
 for g in b.itertuples(index=False):
  for side in ("away","home"):
   opp="home" if side=="away" else "away";r={"game_id":g.game_id,"date":g.date,"season":y,"team":value(g,f"{side}_team"),"opponent":value(g,f"{opp}_team"),"team_side":side,"home_indicator":int(side=="home"),"actual_team_runs":value(g,f"{side}_score"),"opponent_actual_runs":value(g,f"{opp}_score")}
   for f in OFF+CONTEXT+STRENGTH:r[f]=value(g,f"{side}_{f}")
   for f in SP+BP+MATCH:
    # Contextual offense belongs to batting team; starter/bullpen/arsenal/recent100 belong to opponent.
    owner=side if f.startswith("ctx_") else opp;r[("opp_"+f) if owner==opp else f]=value(g,f"{owner}_{f}")
   rows.append(r)
 o=pd.DataFrame(rows);o.date=pd.to_datetime(o.date);return o

def add_scoring_baselines(d):
 out=[]
 for y,g in d.groupby("season"):
  g=g.copy().sort_values(["date","game_id","team_side"]);vals=[]
  for r in g.itertuples():
   h=g[(g.team.eq(r.team))&(g.date.lt(r.date))]
   vals.append((h.actual_team_runs.mean(),h.loc[h.date.ge(r.date-pd.Timedelta(days=30)),"actual_team_runs"].mean()))
  g[["season_scoring_avg","recent30_scoring_avg"]]=vals;out.append(g)
 return pd.concat(out,ignore_index=True)

def feature_sets(columns):
 exist=lambda xs:[x for x in xs if x in columns]
 off=exist(OFF);sp=exist(["opp_"+x for x in SP]);bp=exist(["opp_"+x for x in BP]);match=exist(MATCH+["opp_"+x for x in MATCH if not x.startswith("ctx_")]);ctx=exist(CONTEXT+["home_indicator"]);strength=exist(STRENGTH)
 return {"A_offense":off,"B_offense_starter":sorted(set(off+sp)),"C_offense_starter_bullpen":sorted(set(off+sp+bp)),"D_plus_matchup":sorted(set(off+sp+bp+match)),"E_plus_context":sorted(set(off+sp+bp+match+ctx)),"F_plus_team_strength":sorted(set(off+sp+bp+match+ctx+strength)),"G_comprehensive":sorted(set(off+sp+bp+match+ctx+strength)),"H_compact":exist(COMPACT)}

CONFIGS=[("poisson_a0.1","poisson",.1),("poisson_a1","poisson",1.),("poisson_a10","poisson",10.),("ridge_a1","ridge",1.),("ridge_a10","ridge",10.),("hist_poisson","hist",None)]
def model(kind,p):
 if kind=="poisson":return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",PoissonRegressor(alpha=p,max_iter=3000))])
 if kind=="ridge":return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=p))])
 return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",HistGradientBoostingRegressor(loss="poisson",learning_rate=.05,max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=42))])
def predict(m,x):return np.clip(m.predict(x),.05,None)
def metrics(y,p):
 p=np.clip(np.asarray(p,dtype=float),.05,None)
 return {"rmse":mean_squared_error(y,p)**.5,"mae":mean_absolute_error(y,p),"poisson_deviance":mean_poisson_deviance(y,p),"mean_predicted":np.mean(p),"mean_actual":np.mean(y),"spearman":spearmanr(y,p).statistic,"pearson":pearsonr(y,p).statistic}
def alpha_moment(y,mu):return max(1e-9,float(np.sum((y-mu)**2-y)/np.sum(mu**2)))
def nll(y,mu,a,dist):
 if dist=="poisson":return float(np.mean(mu-y*np.log(mu)+gammaln(y+1)))
 size=1/a;p=size/(size+mu);return float(-np.mean(nbinom.logpmf(y,size,p)))
def calibration(d,scope):
 x=d.copy();x["bucket"]=pd.cut(x.predicted_runs,BUCKET_EDGES,labels=BUCKETS,right=False);rows=[]
 for b in BUCKETS:
  g=x[x.bucket.eq(b)];z={"scope":scope,"bucket":b,"team_games":len(g),"mean_prediction":g.predicted_runs.mean(),"actual_mean_runs":g.actual_team_runs.mean(),"median_actual_runs":g.actual_team_runs.median()}
  for k in range(2,8):z[f"pct_{k}_plus"]=g.actual_team_runs.ge(k).mean()
  rows.append(z)
 return rows
def residuals(d,scope):
 x=d.copy();x["residual"]=x.actual_team_runs-x.predicted_runs;x["pred_bucket"]=pd.cut(x.predicted_runs,BUCKET_EDGES,labels=BUCKETS,right=False);rows=[]
 def add(kind,col):
  for k,g in x.groupby(col,dropna=False,observed=True):rows.append({"scope":scope,"diagnostic":kind,"group":str(k),"team_games":len(g),"mean_prediction":g.predicted_runs.mean(),"mean_actual":g.actual_team_runs.mean(),"mean_residual":g.residual.mean(),"rmse":np.sqrt(np.mean(g.residual**2))})
 add("predicted_bucket","pred_bucket");add("team_side","team_side");add("season","season")
 if "opp_sp_official_season_era" in x:x["starter_quality"] = pd.qcut(x.opp_sp_official_season_era,3,labels=["low_ERA","mid_ERA","high_ERA"],duplicates="drop");add("starter_quality","starter_quality")
 if "opp_bp_official_season_era" in x:x["bullpen_quality"] = pd.qcut(x.opp_bp_official_season_era,3,labels=["low_ERA","mid_ERA","high_ERA"],duplicates="drop");add("bullpen_quality","bullpen_quality")
 if "season_woba" in x:x["offense_strength"]=pd.qcut(x.season_woba,3,labels=["low","mid","high"],duplicates="drop");add("offense_strength","offense_strength")
 x["season_stage"]=np.where(x.date.dt.month.le(4),"early","established");add("season_stage","season_stage")
 x["missing_count"]=x[SELECTED_FEATURES].isna().sum(axis=1);x["missing_group"]=pd.cut(x.missing_count,[-1,0,5,15,np.inf],labels=["0","1-5","6-15","16+"]);add("missingness","missing_group")
 return rows

def game_level(team):
 a=team[team.team_side.eq("away")].copy();h=team[team.team_side.eq("home")].copy();g=a.merge(h,on=["game_id","date","season"],suffixes=("_away","_home"),validate="one_to_one");g["projected_total"]=g.predicted_runs_away+g.predicted_runs_home;g["actual_total"]=g.actual_team_runs_away+g.actual_team_runs_home;g["projected_run_diff"]=g.predicted_runs_home-g.predicted_runs_away;g["actual_run_diff"]=g.actual_team_runs_home-g.actual_team_runs_away;g["home_win"]=g.actual_run_diff.gt(0).astype(int);return g

def main():
 global SELECTED_FEATURES
 warnings.filterwarnings("ignore",category=pd.errors.PerformanceWarning);R.mkdir(exist_ok=True)
 data=add_scoring_baselines(pd.concat([long_year(y) for y in YEARS],ignore_index=True));sets=feature_sets(data.columns)
 definitions=pd.DataFrame([{"feature_family":k,"feature":f,"baseball_family":FAMILY_OF.get(f.replace("opp_",""),"Matchup"),"features_in_family":len(v)} for k,v in sets.items() for f in v]);definitions.to_csv(R/"v11_feature_family_definitions.csv",index=False)
 # Development-only architecture selection.
 foldrows=[];oos_cache={}
 for fs,features in sets.items():
  for label,kind,p in CONFIGS:
   preds=[]
   for yrs,vy in FOLDS:
    tr=data[data.season.isin(yrs)];va=data[data.season.eq(vy)];m=model(kind,p);m.fit(tr[features],tr.actual_team_runs);pv=predict(m,va[features]);pt=predict(m,tr[features]);row={"feature_family":fs,"model_config":label,"validation_year":vy,"features":len(features),**metrics(va.actual_team_runs,pv),"nb_alpha_train":alpha_moment(tr.actual_team_runs.to_numpy(),pt)};foldrows.append(row);q=va[["game_id","date","season","team","opponent","team_side","home_indicator","actual_team_runs","opponent_actual_runs"]].copy();q["predicted_runs"]=pv;preds.append(q)
   oos_cache[(fs,label)]=pd.concat(preds,ignore_index=True)
 folds=pd.DataFrame(foldrows);summary=folds.groupby(["feature_family","model_config","features"],as_index=False).agg(mean_rmse=("rmse","mean"),sd_rmse=("rmse","std"),mean_mae=("mae","mean"),mean_poisson_deviance=("poisson_deviance","mean"),mean_spearman=("spearman","mean"),mean_pearson=("pearson","mean")).sort_values(["mean_rmse","mean_mae","mean_poisson_deviance"]);win=summary.iloc[0];fs,label=win.feature_family,win.model_config;kind,p=next((k,p) for l,k,p in CONFIGS if l==label);SELECTED_FEATURES=sets[fs];dev0=oos_cache[(fs,label)];meta=["game_id","date","season","team","opponent","team_side","home_indicator","actual_team_runs","opponent_actual_runs"];dev=data[data.season.between(2022,2024)][meta+[f for f in SELECTED_FEATURES if f not in meta]].merge(dev0[["game_id","team_side","predicted_runs"]],on=["game_id","team_side"],validate="one_to_one")
 # Baselines on identical folds.
 brows=[]
 for yrs,vy in FOLDS:
  tr=data[data.season.isin(yrs)];va=data[data.season.eq(vy)]
  for name,pv in [("league_average",np.repeat(tr.actual_team_runs.mean(),len(va))),("pregame_team_season_avg",va.season_scoring_avg.fillna(tr.actual_team_runs.mean()).to_numpy()),("pregame_recent30_avg",va.recent30_scoring_avg.fillna(tr.actual_team_runs.mean()).to_numpy())]:brows.append({"baseline":name,"validation_year":vy,**metrics(va.actual_team_runs,pv)})
 baselines=pd.DataFrame(brows)
 # Select distribution using development predictions and fold-training alpha only.
 dist=[]
 for vy,g in dev.groupby("season"):
  a=folds[(folds.feature_family.eq(fs))&(folds.model_config.eq(label))&(folds.validation_year.eq(vy))].nb_alpha_train.iloc[0]
  for name in ("poisson","negative_binomial"):dist.append({"scope":str(vy),"distribution":name,"alpha":0 if name=="poisson" else a,"nll":nll(g.actual_team_runs.to_numpy(),g.predicted_runs.to_numpy(),a,name),"variance_actual":g.actual_team_runs.var(),"variance_implied":np.mean(g.predicted_runs if name=="poisson" else g.predicted_runs+a*g.predicted_runs**2)})
 dist=pd.DataFrame(dist);dist_sum=dist.groupby("distribution",as_index=False).nll.mean().sort_values("nll");dist_choice=dist_sum.iloc[0].distribution
 # Fixed predictive distribution diagnostics (development only).
 dd=[]
 for name in ("poisson","negative_binomial"):
  probs_by_k=[]
  for _,r in dev.iterrows():
   a=folds[(folds.feature_family.eq(fs))&(folds.model_config.eq(label))&(folds.validation_year.eq(r.season))].nb_alpha_train.iloc[0];mu=r.predicted_runs
   probs_by_k.append([poisson.pmf(k,mu) if name=="poisson" else nbinom.pmf(k,1/a,(1/a)/(1/a+mu)) for k in range(11)])
  pr=np.asarray(probs_by_k)
  for k in range(10):dd.append({"scope":"combined_development","distribution":name,"diagnostic":"team_run_frequency","group":str(k),"actual":dev.actual_team_runs.eq(k).mean(),"predicted":pr[:,k].mean()})
  dd.append({"scope":"combined_development","distribution":name,"diagnostic":"team_run_frequency","group":"10+","actual":dev.actual_team_runs.ge(10).mean(),"predicted":1-pr[:,:10].sum(axis=1).mean()})
  for k in (4,5,6):dd.append({"scope":"combined_development","distribution":name,"diagnostic":"team_run_threshold","group":f"{k}+","actual":dev.actual_team_runs.ge(k).mean(),"predicted":1-pr[:,:k].sum(axis=1).mean()})
  rng=np.random.default_rng(1100 if name=="poisson" else 1101);ga=dev[dev.team_side.eq("away")].sort_values("game_id");gh=dev[dev.team_side.eq("home")].sort_values("game_id");draws=[]
  for arow,hrow in zip(ga.itertuples(),gh.itertuples()):
   aa=folds[(folds.feature_family.eq(fs))&(folds.model_config.eq(label))&(folds.validation_year.eq(arow.season))].nb_alpha_train.iloc[0]
   if name=="poisson":x=rng.poisson(arow.predicted_runs,200);z=rng.poisson(hrow.predicted_runs,200)
   else:size=1/aa;x=rng.negative_binomial(size,size/(size+arow.predicted_runs),200);z=rng.negative_binomial(size,size/(size+hrow.predicted_runs),200)
   draws.append((x+z,z-x))
  st=np.concatenate([x[0] for x in draws]);sd=np.concatenate([x[1] for x in draws]);actualg=game_level(dev)
  for what,av,pv in [("game_total_variance",actualg.actual_total.var(),st.var()),("run_difference_variance",actualg.actual_run_diff.var(),sd.var()),("game_total_10+",actualg.actual_total.ge(10).mean(),np.mean(st>=10)),("absolute_run_diff_4+",actualg.actual_run_diff.abs().ge(4).mean(),np.mean(np.abs(sd)>=4))]:dd.append({"scope":"combined_development","distribution":name,"diagnostic":what,"group":"all","actual":av,"predicted":pv})
 dist_detail=pd.DataFrame(dd)
 # Freeze specification before touching 2025 outcomes.
 spec={"label":"V11 UNIFIED TEAM RUN MODEL","development_winner":{"feature_family":fs,"model_config":label,"features":len(SELECTED_FEATURES)},"features":SELECTED_FEATURES,"distribution":dist_choice,"selection_primary":"mean chronological RMSE","development_folds":["2021->2022","2021-22->2023","2021-23->2024"],"holdout":"2025 loaded only after frozen selection","sportsbook_data_used":False}
 (R/"v11_frozen_specification.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
 # Untouched 2025 once.
 train=data[data.season.le(2024)];hold=data[data.season.eq(2025)];final=model(kind,p);final.fit(train[SELECTED_FEATURES],train.actual_team_runs);ph=predict(final,hold[SELECTED_FEATURES]);meta=["game_id","date","season","team","opponent","team_side","home_indicator","actual_team_runs","opponent_actual_runs"];holdpred=hold[meta+[f for f in SELECTED_FEATURES if f not in meta]].copy();holdpred["predicted_runs"]=ph
 combined=pd.concat([dev,holdpred[dev.columns]],ignore_index=True);hm=pd.DataFrame([{"scope":"2025",**metrics(hold.actual_team_runs,ph)}]);devcal=pd.DataFrame(sum([calibration(g,str(y)) for y,g in dev.groupby("season")],[])+calibration(dev,"combined_development"));combcal=pd.DataFrame(sum([calibration(g,str(y)) for y,g in combined.groupby("season")],[])+calibration(combined,"combined_oos"))
 for name,pv in [("league_average",np.repeat(train.actual_team_runs.mean(),len(hold))),("pregame_team_season_avg",hold.season_scoring_avg.fillna(train.actual_team_runs.mean()).to_numpy()),("pregame_recent30_avg",hold.recent30_scoring_avg.fillna(train.actual_team_runs.mean()).to_numpy())]:brows.append({"baseline":name,"validation_year":2025,**metrics(hold.actual_team_runs,pv)})
 baselines=pd.DataFrame(brows)
 # Contributions: exact additive terms for linear champion; local median replacement deltas for hist.
 contrib=[]
 if kind in ("poisson","ridge"):
  imp=final.named_steps["imputer"];sc=final.named_steps["scale"];mod=final.named_steps["model"];xi=imp.transform(hold[SELECTED_FEATURES]);z=sc.transform(xi);co=mod.coef_.ravel()
  for i,r in enumerate(holdpred.itertuples()):
   for j,f in enumerate(SELECTED_FEATURES):contrib.append({"game_id":r.game_id,"team_side":r.team_side,"team":r.team,"feature":f,"feature_family":FAMILY_OF.get(f.replace("opp_",""),"Matchup"),"raw_value":getattr(r,f),"training_reference":sc.mean_[j],"training_median":imp.statistics_[j],"standardized_value":z[i,j],"direction":"positive" if co[j]>=0 else "negative","coefficient":co[j],"contribution":co[j]*z[i,j],"contribution_scale":"log_mean" if kind=="poisson" else "runs"})
 else:
  med=pd.Series(final.named_steps["imputer"].statistics_,index=SELECTED_FEATURES)
  for _,r in holdpred.iterrows():
   base=r[SELECTED_FEATURES].to_frame().T;basep=predict(final,base)[0]
   for f in SELECTED_FEATURES:
    alt=base.copy();alt[f]=med[f];delta=basep-predict(final,alt)[0];contrib.append({"game_id":r.game_id,"team_side":r.team_side,"team":r.team,"feature":f,"feature_family":FAMILY_OF.get(f.replace("opp_",""),"Matchup"),"raw_value":r[f],"training_reference":med[f],"training_median":med[f],"standardized_value":np.nan,"direction":"local_delta","coefficient":np.nan,"contribution":delta,"contribution_scale":"nonadditive_local_median_replacement"})
 contrib=pd.DataFrame(contrib)
 residual=pd.DataFrame(residuals(combined,"combined_oos"));games=game_level(combined);gm={"team_rmse":metrics(combined.actual_team_runs,combined.predicted_runs)["rmse"],"total_rmse":mean_squared_error(games.actual_total,games.projected_total)**.5,"total_mae":mean_absolute_error(games.actual_total,games.projected_total),"run_diff_rmse":mean_squared_error(games.actual_run_diff,games.projected_run_diff)**.5,"winner_auc":roc_auc_score(games.home_win,games.projected_run_diff),"total_spearman":spearmanr(games.actual_total,games.projected_total).statistic,"run_diff_spearman":spearmanr(games.actual_run_diff,games.projected_run_diff).statistic};gamecal=pd.DataFrame([gm])
 # Predictive comparison only after holdout is complete.
 oldml=pd.read_csv(R/"v5_team_strength_oos_predictions_2022_2025.csv");oldt=pd.read_csv(R/"totals_oos_predictions_2022_2025.csv");oldrs=pd.read_csv(R/"team_run_score_complete_team_games_2021_2025.csv",usecols=["game_id","team_side","run_score"])
 g25=games[games.season.eq(2025)].merge(oldml[["game_id","predicted_home_probability"]],on="game_id").merge(oldt[["game_id","predicted_total"]],on="game_id");rs25=holdpred.merge(oldrs,on=["game_id","team_side"])
 compare=pd.DataFrame([{"comparison":"team_runs_v11","rmse":mean_squared_error(holdpred.actual_team_runs,holdpred.predicted_runs)**.5,"mae":mean_absolute_error(holdpred.actual_team_runs,holdpred.predicted_runs),"spearman":spearmanr(holdpred.actual_team_runs,holdpred.predicted_runs).statistic},{"comparison":"team_run_score_rescaled_diagnostic","rmse":np.nan,"mae":np.nan,"spearman":spearmanr(rs25.actual_team_runs,rs25.run_score).statistic},{"comparison":"game_total_v11","rmse":mean_squared_error(g25.actual_total,g25.projected_total)**.5,"mae":mean_absolute_error(g25.actual_total,g25.projected_total),"spearman":spearmanr(g25.actual_total,g25.projected_total).statistic},{"comparison":"game_total_old","rmse":mean_squared_error(g25.actual_total,g25.predicted_total)**.5,"mae":mean_absolute_error(g25.actual_total,g25.predicted_total),"spearman":spearmanr(g25.actual_total,g25.predicted_total).statistic},{"comparison":"winner_v11_diff","rmse":np.nan,"mae":np.nan,"spearman":roc_auc_score(g25.home_win,g25.projected_run_diff)},{"comparison":"winner_old_ml","rmse":np.nan,"mae":np.nan,"spearman":roc_auc_score(g25.home_win,g25.predicted_home_probability)}])
 audit=g25[g25.game_id.isin(AUDIT_IDS)].merge(oldrs[oldrs.game_id.isin(AUDIT_IDS)].pivot(index="game_id",columns="team_side",values="run_score").add_prefix("old_run_score_").reset_index(),on="game_id");audit=audit.sort_values("game_id");ar=["# V11 re-audit of the same 11 games",""]
 for r in audit.itertuples():
  cc=contrib[contrib.game_id.eq(r.game_id)];ar += [f"## {r.game_id}: {r.team_away} at {r.team_home}",f"Pregame: away λ={r.predicted_runs_away:.3f}, home λ={r.predicted_runs_home:.3f}, total={r.projected_total:.3f}, home-away={r.projected_run_diff:+.3f}; old ML home P={r.predicted_home_probability:.3f}, old total={r.predicted_total:.3f}, old Run Scores={r.old_run_score_away:.2f}/{r.old_run_score_home:.2f}."]
  for side in ("away","home"):
   q=cc[cc.team_side.eq(side)].reindex(cc[cc.team_side.eq(side)].contribution.abs().sort_values(ascending=False).index).head(5);ar.append(f"{side.title()} largest V11 factors: "+", ".join(f"{x.feature}={x.raw_value:.4g} ({x.contribution:+.4f})" for x in q.itertuples()))
  ar += [f"Outcome (shown after prediction): {r.actual_team_runs_away:.0f}-{r.actual_team_runs_home:.0f}.",""]
 # Save everything.
 folds.to_csv(R/"v11_model_selection_folds.csv",index=False);summary.to_csv(R/"v11_model_selection_summary.csv",index=False);baselines.to_csv(R/"v11_baseline_comparison.csv",index=False);dev.to_csv(R/"v11_development_oos_team_predictions.csv",index=False);devcal.to_csv(R/"v11_development_run_calibration.csv",index=False);holdpred.to_csv(R/"v11_untouched_2025_team_predictions.csv",index=False);hm.to_csv(R/"v11_untouched_2025_metrics.csv",index=False);combined.to_csv(R/"v11_combined_oos_team_predictions.csv",index=False);combcal.to_csv(R/"v11_combined_run_calibration.csv",index=False);residual.to_csv(R/"v11_residual_diagnostics.csv",index=False);contrib.to_csv(R/"v11_feature_contributions.csv",index=False);games.to_csv(R/"v11_game_level_predictions.csv",index=False);gamecal.to_csv(R/"v11_game_level_calibration.csv",index=False);pd.concat([dist,dist_sum.assign(scope="development_mean",alpha=np.nan,variance_actual=np.nan,variance_implied=np.nan)],ignore_index=True).to_csv(R/"v11_distribution_diagnostics.csv",index=False);dist_detail.to_csv(R/"v11_distribution_frequency_diagnostics.csv",index=False);compare.to_csv(R/"v11_vs_existing_models.csv",index=False);audit.to_csv(R/"v11_11_game_reaudit.csv",index=False);(R/"v11_11_game_reaudit_report.md").write_text("\n\n".join(ar),encoding="utf-8")
 monotonic=combcal[combcal.scope.eq("combined_oos")].dropna(subset=["actual_mean_runs"]).actual_mean_runs.is_monotonic_increasing;bestbase=baselines.groupby("baseline").rmse.mean().min();report=f"""# V11 final report\n\nDevelopment winner: {fs} / {label} ({len(SELECTED_FEATURES)} features), mean RMSE {win.mean_rmse:.4f}. Distribution: {dist_choice}.\n\nUntouched 2025: RMSE {hm.rmse.iloc[0]:.4f}, MAE {hm.mae.iloc[0]:.4f}, Poisson deviance {hm.poisson_deviance.iloc[0]:.4f}, Spearman {hm.spearman.iloc[0]:.4f}.\n\nBest simple development baseline mean RMSE: {bestbase:.4f}. V11 delta: {win.mean_rmse-bestbase:+.4f}. Combined OOS bucket means monotonic: {monotonic}.\n\n2025 V11 total RMSE {compare.loc[compare.comparison.eq('game_total_v11'),'rmse'].iloc[0]:.4f} versus old totals {compare.loc[compare.comparison.eq('game_total_old'),'rmse'].iloc[0]:.4f}. Winner AUC V11 {compare.loc[compare.comparison.eq('winner_v11_diff'),'spearman'].iloc[0]:.4f} versus old ML {compare.loc[compare.comparison.eq('winner_old_ml'),'spearman'].iloc[0]:.4f}.\n\nNo sportsbook odds, EV, ROI, or bet selection were used.\n""";(R/"v11_final_report.md").write_text(report,encoding="utf-8")
 print(summary.head(12).to_string(index=False));print("\nWINNER",fs,label,len(SELECTED_FEATURES));print("\n2025",hm.to_string(index=False));print("\nCOMPARISON",compare.to_string(index=False));print("\nMONOTONIC",monotonic)

if __name__=="__main__":main()
