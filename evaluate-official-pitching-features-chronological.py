"""Chronologically evaluate official pitching families for ML and totals."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,PoissonRegressor
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score,mean_absolute_error,mean_squared_error,mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDS=[([2021],2022),([2021,2022],2023),([2021,2022,2023],2024)];RES=Path("results")
V5=["season_woba_diff","l30_woba_diff","sp_season_k_pct_diff","sp_season_bb_pct_diff","sp_season_woba_allowed_diff","sp_l30_k_pct_diff","sp_l30_bb_pct_diff","sp_l30_woba_allowed_diff","bp_season_k_pct_diff","bp_season_bb_pct_diff","bp_season_woba_allowed_diff","bp_l30_k_pct_diff","bp_l30_bb_pct_diff","bp_l30_woba_allowed_diff","bp_l7_bf_diff","sp_days_rest_diff","sp_prev_pitch_count_diff","sp_season_velocity_diff","sp_l30_velocity_diff","sp_season_whiff_diff","sp_l30_whiff_diff","sp_season_xwoba_allowed_diff","sp_l30_xwoba_allowed_diff","season_platoon_woba_diff","l30_platoon_woba_diff","sp_matchup_season_xwoba_allowed_diff","sp_matchup_season_k_pct_diff","sp_matchup_season_bb_pct_diff","sp_matchup_season_whiff_pct_diff","sp_matchup_l30_xwoba_allowed_diff","sp_matchup_l30_k_pct_diff","sp_matchup_l30_bb_pct_diff","sp_matchup_l30_whiff_pct_diff"]
STRENGTH=["sit_run_diff_diff","sit_run_diff_per_game_diff","sit_pythagorean_win_pct_diff","sit_actual_minus_pythagorean_diff"];ML_BASE=V5+STRENGTH
METRICS=["era","whip","k_pct","bb_pct","hr9"]
SP_DIFF=[f"sp_official_{w}_{m}_diff" for w in ["season","l30","recent3_starts"] for m in METRICS]+[f"sp_official_venue_{m}_diff" for m in METRICS]
BP_DIFF=[f"bp_official_{w}_{m}_diff" for w in ["season","l30","roster_pool","available_pool"] for m in METRICS]
RP_DIFF=[f"rp_{w}_{m}_diff" for w in ["season","l30","recent3"] for m in ["er_per_start","runs_per_start","ip_per_start"]]+[f"rp_{x}_diff" for x in ["expected_starter_innings","starter_er_component","expected_bullpen_innings","available_bullpen_er_component","combined_pitching_er_component"]]
PRED_EDGES=[0,6,7,8,9,10,11,12,np.inf];PRED_LABELS=["<6","6-7","7-8","8-9","9-10","10-11","11-12","12+"]

def load(y):
 b=pd.read_csv(f"data/processed/games_{y}_starter_lineup_matchup_features.csv");files=[f"data/processed/features_situational_{y}.csv",f"data/processed/features_official_starter_pitching_{y}.csv",f"data/processed/features_official_bullpen_pitching_{y}.csv",f"data/processed/features_official_run_prevention_{y}.csv"]
 for f in files:
  q=pd.read_csv(f)
  if q.game_id.duplicated().any() or set(q.game_id)!=set(b.game_id):raise ValueError(f"coverage failure {f}")
  overlap=[c for c in q if c in b and c!="game_id"]
  b=b.merge(q.drop(columns=overlap),on="game_id",validate="one_to_one")
 for m in METRICS:b[f"sp_official_venue_{m}_diff"]=b[f"home_sp_official_season_home_{m}"]-b[f"away_sp_official_season_away_{m}"]
 b["season"]=y;b["actual_total_runs"]=b.home_score+b.away_score;return b
def ml_model():return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000))])
def total_model():return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",PoissonRegressor(alpha=1,max_iter=2000))])
def ml_metrics(y,p):return {"log_loss":log_loss(y,p),"brier":brier_score_loss(y,p),"auc":roc_auc_score(y,p),"accuracy":accuracy_score(y,p>=.5),"mean_prediction":p.mean(),"actual_mean":y.mean()}
def total_metrics(y,p):return {"rmse":mean_squared_error(y,p)**.5,"mae":mean_absolute_error(y,p),"poisson_deviance":mean_poisson_deviance(y,np.clip(p,.05,None)),"mean_prediction":p.mean(),"actual_mean":y.mean()}
def calibration(y,p,scope,model):
 x=pd.DataFrame({"actual":y,"prediction":p});x["bucket"]=pd.cut(x.prediction,PRED_EDGES,labels=PRED_LABELS,right=False)
 return [{"model":model,"scope":scope,"bucket":b,"games":len(g),"mean_prediction":g.prediction.mean(),"actual_mean":g.actual.mean(),"error":g.actual.mean()-g.prediction.mean()} for b in PRED_LABELS for g in [x[x.bucket.eq(b)]]]
def coefs(pipe,features,model,year):return pd.DataFrame({"experiment":model,"validation_year":year,"feature":features,"standardized_coefficient":pipe.named_steps["model"].coef_.ravel()})
def summarize(folds,metric):
 nums=[c for c in folds if c not in ["experiment","validation_year","features"]];s=folds.groupby("experiment",as_index=False)[nums].agg(["mean","std"]);s.columns=["experiment"]+[f"{a}_{b}" for a,b in s.columns.tolist()[1:]];return s.sort_values(f"{metric}_mean")
def run_dev(data,specs,kind):
 rows=[];coef=[];cal=[]
 for name,features in specs.items():
  for yrs,vy in FOLDS:
   tr=pd.concat([data[y] for y in yrs],ignore_index=True);va=data[vy];m=ml_model() if kind=="ml" else total_model();target="home_win" if kind=="ml" else "actual_total_runs";m.fit(tr[features],tr[target]);p=m.predict_proba(va[features])[:,1] if kind=="ml" else np.clip(m.predict(va[features]),.05,None);met=ml_metrics(va[target],p) if kind=="ml" else total_metrics(va[target],p);rows.append({"experiment":name,"validation_year":vy,"features":len(features),**met});coef.append(coefs(m,features,name,vy));
   if kind=="totals":cal+=calibration(va[target],p,str(vy),name)
 return pd.DataFrame(rows),pd.concat(coef,ignore_index=True),pd.DataFrame(cal)
def holdout(dev,hold,specs,kind,names):
 tr=pd.concat([dev[y] for y in range(2021,2025)],ignore_index=True);rows=[];preds=[];coef=[];cal=[];target="home_win" if kind=="ml" else "actual_total_runs"
 for name in names:
  f=specs[name];m=ml_model() if kind=="ml" else total_model();m.fit(tr[f],tr[target]);p=m.predict_proba(hold[f])[:,1] if kind=="ml" else np.clip(m.predict(hold[f]),.05,None);met=ml_metrics(hold[target],p) if kind=="ml" else total_metrics(hold[target],p);rows.append({"experiment":name,"validation_year":2025,"features":len(f),**met});preds.append(pd.DataFrame({"game_id":hold.game_id,"experiment":name,"actual":hold[target],"prediction":p}));coef.append(coefs(m,f,name,2025));
  if kind=="totals":cal+=calibration(hold[target],p,"2025",name)
 return pd.DataFrame(rows),pd.concat(preds),pd.concat(coef),pd.DataFrame(cal)
def correlations(dev,new):
 x=pd.concat([dev[y] for y in range(2021,2025)],ignore_index=True);c=x[ML_BASE+new].corr();rows=[]
 for n in new:
  q=c.loc[n,ML_BASE].dropna();
  if len(q):
   k=q.abs().idxmax();rows.append({"official_feature":n,"highest_correlated_existing_feature":k,"correlation":q[k],"absolute_correlation":abs(q[k]),"above_0_80":abs(q[k])>.8})
 return pd.DataFrame(rows).sort_values("absolute_correlation",ascending=False)
def main():
 RES.mkdir(exist_ok=True);dev={y:load(y) for y in range(2021,2025)}
 ml_specs={"baseline":ML_BASE,"official_starter":ML_BASE+SP_DIFF,"official_bullpen":ML_BASE+BP_DIFF,"official_both":ML_BASE+SP_DIFF+BP_DIFF,"structural_run_prevention":ML_BASE+RP_DIFF}
 ml_fold,ml_coef,_=run_dev(dev,ml_specs,"ml");ml_summary=summarize(ml_fold,"log_loss");raw=ml_summary[ml_summary.experiment.isin(["baseline","official_starter","official_bullpen","official_both"])];ml_winner=raw.iloc[0].experiment;ml_struct="structural_run_prevention"
 selected=pd.read_csv("results/totals_selected_features.csv").feature.tolist();sp_abs=[f"{s}_sp_official_{w}_{m}" for s in ["home","away"] for w in ["season","l30","recent3_starts"] for m in METRICS]+[f"{s}_sp_official_season_{s}_{m}" for s in ["home","away"] for m in METRICS];bp_abs=[f"{s}_bp_official_{w}_{m}" for s in ["home","away"] for w in ["season","l30","roster_pool","available_pool"] for m in METRICS];rp_abs=[f"{s}_rp_{w}_{m}" for s in ["home","away"] for w in ["season","l30","recent3"] for m in ["er_per_start","runs_per_start","ip_per_start"]]+[f"{s}_rp_{x}" for s in ["home","away"] for x in ["expected_starter_innings","starter_er_component","expected_bullpen_innings","available_bullpen_er_component","combined_pitching_er_component"]]
 total_specs={"baseline":selected,"official_starter":selected+sp_abs,"official_bullpen":selected+bp_abs,"official_both":selected+sp_abs+bp_abs,"structural_run_prevention":selected+rp_abs}
 t_fold,t_coef,t_cal=run_dev(dev,total_specs,"totals");t_summary=summarize(t_fold,"rmse");rawt=t_summary[t_summary.experiment.isin(["baseline","official_starter","official_bullpen","official_both"])];t_winner=rawt.iloc[0].experiment
 # Specifications are frozen above using development data only. 2025 is first loaded here.
 hold=load(2025);ml_25,ml_pred,ml_c25,_=holdout(dev,hold,ml_specs,"ml",list(dict.fromkeys(["baseline",ml_winner,ml_struct])));t_25,t_pred,t_c25,t_cal25=holdout(dev,hold,total_specs,"totals",list(dict.fromkeys(["baseline",t_winner,"structural_run_prevention"])))
 corr=correlations(dev,SP_DIFF+BP_DIFF);allcoef=pd.concat([ml_coef,ml_c25],ignore_index=True);official_coeff=allcoef[allcoef.feature.isin(SP_DIFF+BP_DIFF+RP_DIFF)]
 ml_fold.to_csv(RES/"official_pitching_ml_development_folds.csv",index=False);ml_summary.to_csv(RES/"official_pitching_ml_development_summary.csv",index=False);ml_25.to_csv(RES/"official_pitching_ml_untouched_2025.csv",index=False);ml_pred.to_csv(RES/"official_pitching_ml_2025_predictions.csv",index=False)
 t_fold.to_csv(RES/"official_pitching_totals_development_folds.csv",index=False);t_summary.to_csv(RES/"official_pitching_totals_development_summary.csv",index=False);t_25.to_csv(RES/"official_pitching_totals_untouched_2025.csv",index=False);t_pred.to_csv(RES/"official_pitching_totals_2025_predictions.csv",index=False);pd.concat([t_cal,t_cal25]).to_csv(RES/"official_pitching_totals_calibration.csv",index=False)
 corr.to_csv(RES/"official_pitching_vs_statcast_correlations.csv",index=False);official_coeff.to_csv(RES/"official_pitching_ml_standardized_coefficients.csv",index=False);t_coef[t_coef.feature.isin(sp_abs+bp_abs+rp_abs)].to_csv(RES/"official_pitching_totals_standardized_coefficients.csv",index=False)
 decision={"moneyline_development_winner":ml_winner,"totals_development_winner":t_winner,"moneyline_structural_dev_log_loss":float(ml_summary.loc[ml_summary.experiment.eq(ml_struct),"log_loss_mean"].iloc[0]),"totals_structural_dev_rmse":float(t_summary.loc[t_summary.experiment.eq("structural_run_prevention"),"rmse_mean"].iloc[0])};(RES/"official_pitching_experiment_decision.json").write_text(json.dumps(decision,indent=2))
 print("\nML DEVELOPMENT\n",ml_summary.to_string(index=False));print("\nML 2025\n",ml_25.to_string(index=False));print("\nTOTALS DEVELOPMENT\n",t_summary.to_string(index=False));print("\nTOTALS 2025\n",t_25.to_string(index=False));print("\nHighest official-vs-Statcast correlations\n",corr.head(20).to_string(index=False));print("\nFrozen winners:",decision)
if __name__=="__main__":main()
