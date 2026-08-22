"""Chronological, development-only evaluation of independent new feature families."""
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

YEARS=[2021,2022,2023,2024]
FOLDS=[([2021],2022),([2021,2022],2023),([2021,2022,2023],2024)]
TARGET="home_win";CORR_THRESHOLD=.80
V5=[
"season_woba_diff","l30_woba_diff","sp_season_k_pct_diff","sp_season_bb_pct_diff","sp_season_woba_allowed_diff","sp_l30_k_pct_diff","sp_l30_bb_pct_diff","sp_l30_woba_allowed_diff",
"bp_season_k_pct_diff","bp_season_bb_pct_diff","bp_season_woba_allowed_diff","bp_l30_k_pct_diff","bp_l30_bb_pct_diff","bp_l30_woba_allowed_diff","bp_l7_bf_diff",
"sp_days_rest_diff","sp_prev_pitch_count_diff","sp_season_velocity_diff","sp_l30_velocity_diff","sp_season_whiff_diff","sp_l30_whiff_diff","sp_season_xwoba_allowed_diff","sp_l30_xwoba_allowed_diff",
"season_platoon_woba_diff","l30_platoon_woba_diff","sp_matchup_season_xwoba_allowed_diff","sp_matchup_season_k_pct_diff","sp_matchup_season_bb_pct_diff","sp_matchup_season_whiff_pct_diff","sp_matchup_l30_xwoba_allowed_diff","sp_matchup_l30_k_pct_diff","sp_matchup_l30_bb_pct_diff","sp_matchup_l30_whiff_pct_diff"]

FILES={
"V5 + V6 bullpen availability":"features_v6_bullpen_availability",
"V5 + richer starter":"features_richer_starter",
"V5 + opponent-quality offense":"features_opponent_quality_offense",
"V5 + arsenal x actual lineup":"features_arsenal_lineup_matchup",
"V5 + situational":"features_situational"}

def predictive_columns(family,columns):
 d=[c for c in columns if c.endswith("_diff")]
 if "richer_starter" in family:
  diagnostics=("pa_diff","contact_diff","fb_diff","out_pitches_diff","first_pitches_diff","pitches_diff","starts_diff")
  return [c for c in d if not c.endswith(diagnostics)]
 if "opponent_quality" in family:return [c for c in d if not c.endswith(("pa_diff","quality_pa_diff","games_diff"))]
 if "arsenal_lineup" in family:return [c for c in d if not any(x in c for x in ["pitch_types_diff","known_hitters_diff","lineup_size_diff","coverage_diff"])]
 if "situational" in family:
  return [c for c in d if c.startswith("sit_") and not c.endswith(("games_diff","venue_games_diff","l10_games_diff"))]
 return d

def pipeline():
 return Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",LogisticRegression(max_iter=3000))])

def combine(store,years):return pd.concat([store[y] for y in years],ignore_index=True)
def score(name,features,train,valid,year):
 m=pipeline();m.fit(train[features],train[TARGET]);prob=m.predict_proba(valid[features])[:,1];pred=(prob>=.5).astype(int)
 row={"model":name,"validation_year":year,"features_added":len(features)-len(V5),"log_loss":log_loss(valid[TARGET],prob),"brier":brier_score_loss(valid[TARGET],prob),"auc":roc_auc_score(valid[TARGET],prob),"accuracy":accuracy_score(valid[TARGET],pred)}
 return row,m

def pairs(frame,left,right=None):
 corr=frame[left+(right or [])].corr();rows=[]
 if right is None:
  for i,a in enumerate(left):
   for b in left[i+1:]:
    v=corr.loc[a,b]
    if pd.notna(v) and abs(v)>CORR_THRESHOLD:rows.append((a,b,v))
 else:
  for a in left:
   for b in right:
    v=corr.loc[a,b]
    if pd.notna(v) and abs(v)>CORR_THRESHOLD:rows.append((a,b,v))
 return sorted(rows,key=lambda x:abs(x[2]),reverse=True)

print("Loading 2021-2024 only. 2025 paths are never constructed.")
base={};family_frames={name:{} for name in FILES};family_features={}
for y in YEARS:
 b=pd.read_csv(f"data/processed/games_{y}_starter_lineup_matchup_features.csv")
 if b.game_id.duplicated().any():raise ValueError(f"Duplicate V5 IDs {y}")
 base[y]=b
 for name,stem in FILES.items():
  f=pd.read_csv(f"data/processed/{stem}_{y}.csv")
  if f.game_id.duplicated().any() or set(f.game_id)!=set(b.game_id):raise ValueError(f"Coverage failure: {name} {y}")
  if name not in family_features:family_features[name]=predictive_columns(stem,f.columns.tolist())
  missing=[c for c in family_features[name] if c not in f]
  if missing:raise ValueError(f"Missing features {name} {y}: {missing}")
  family_frames[name][y]=b.merge(f[["game_id"]+family_features[name]],on="game_id",validate="one_to_one")
 print(y,len(b))

missing_rows=[];corr_rows=[]
for name,stem in FILES.items():
 f=family_features[name]
 for y in YEARS:
  x=family_frames[name][y][f]
  for c in f:missing_rows.append({"model":name,"year":y,"feature":c,"missing_count":int(x[c].isna().sum()),"missing_pct":x[c].isna().mean()})
 dev=combine(family_frames[name],YEARS)
 for scope,found in [("new_vs_v5",pairs(dev,f,V5)),("within_new",pairs(dev,f))]:
  for a,b,v in found:corr_rows.append({"model":name,"scope":scope,"feature_1":a,"feature_2":b,"correlation":v})

rows=[];coef_rows=[]
for train_years,val_year in FOLDS:
 train=combine(base,train_years);valid=base[val_year]
 r,_=score("Baseline V5",V5,train,valid,val_year);rows.append(r)
 print(f"\nTrain {train_years} -> Validate {val_year}")
 print("Baseline V5",r)
 for name in FILES:
  features=V5+family_features[name];train=combine(family_frames[name],train_years);valid=family_frames[name][val_year]
  r,m=score(name,features,train,valid,val_year);rows.append(r);print(name,r)
  coefs=m.named_steps["model"].coef_[0]
  if len(coefs)!=len(features):raise ValueError(f"Imputer removed an all-missing feature for {name}, fold {val_year}")
  for feature,value in zip(family_features[name],coefs[len(V5):]):coef_rows.append({"model":name,"validation_year":val_year,"feature":feature,"standardized_coefficient":value,"abs_coefficient":abs(value)})

fold=pd.DataFrame(rows);baseline=fold[fold.model.eq("Baseline V5")].set_index("validation_year")
for metric in ["log_loss","brier","auc","accuracy"]:
 fold["delta_"+metric]=fold.apply(lambda r:r[metric]-baseline.loc[r.validation_year,metric] if r.model!="Baseline V5" else 0,axis=1)

summary=fold.groupby("model",as_index=False).agg(features_added=("features_added","first"),mean_log_loss=("log_loss","mean"),std_log_loss=("log_loss","std"),mean_brier=("brier","mean"),std_brier=("brier","std"),mean_auc=("auc","mean"),std_auc=("auc","std"),mean_accuracy=("accuracy","mean"),std_accuracy=("accuracy","std"))
b=summary[summary.model.eq("Baseline V5")].iloc[0]
summary["delta_log_loss"]=summary.mean_log_loss-b.mean_log_loss;summary["delta_brier"]=summary.mean_brier-b.mean_brier;summary["delta_auc"]=summary.mean_auc-b.mean_auc;summary["delta_accuracy"]=summary.mean_accuracy-b.mean_accuracy
improved=fold.assign(improved=fold.delta_log_loss.lt(0)).groupby("model").improved.sum();summary["years_logloss_improved"]=summary.model.map(improved).fillna(0).astype(int)

def classify(r):
 if r.model=="Baseline V5":return "BASELINE"
 if r.delta_log_loss<=-.001 and r.delta_brier<=0 and r.delta_auc>0 and r.years_logloss_improved>=2:return "STRONG PASS"
 if r.delta_log_loss<0 and (r.delta_brier<0 or r.delta_auc>0) and r.years_logloss_improved>=2:return "PASS"
 if r.delta_log_loss>0 and r.delta_brier>=0 and r.delta_auc<=0:return "REJECT"
 return "MIXED"
summary["classification"]=summary.apply(classify,axis=1);summary=summary.sort_values("mean_log_loss")

print("\nFOLD-LEVEL RESULTS AND CHANGES (NEW MINUS V5)")
print(fold.sort_values(["validation_year","log_loss"]).to_string(index=False,float_format=lambda x:f"{x:.6f}"))
print("\nMEAN / STANDARD DEVIATION SUMMARY")
print(summary.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
print("\nMISSINGNESS BY FAMILY/YEAR")
miss=pd.DataFrame(missing_rows);print(miss.groupby(["model","year"]).agg(features=("feature","nunique"),missing_cells=("missing_count","sum"),mean_feature_missing_pct=("missing_pct","mean"),max_feature_missing_pct=("missing_pct","max")).to_string(float_format=lambda x:f"{x:.4f}"))
print("\nCORRELATIONS ABOVE |0.80|")
corr=pd.DataFrame(corr_rows)
if corr.empty:print("None")
else:print(corr.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
print("\nNEW-FEATURE STANDARDIZED COEFFICIENTS BY FOLD")
coef=pd.DataFrame(coef_rows).sort_values(["model","validation_year","abs_coefficient"],ascending=[True,True,False]);print(coef.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
qualifiers=summary[summary.classification.isin(["STRONG PASS","PASS"])].model.tolist()
print("\nQUALIFY FOR NEXT COMBINATION EXPERIMENT:",", ".join(qualifiers) if qualifiers else "None")
print("2025 was not loaded, inspected, or evaluated.")

os.makedirs("results",exist_ok=True)
fold.to_csv("results/new_feature_families_chronological_fold_results.csv",index=False)
summary.to_csv("results/new_feature_families_chronological_summary.csv",index=False)
pd.DataFrame(missing_rows).to_csv("results/new_feature_families_missingness.csv",index=False)
pd.DataFrame(corr_rows,columns=["model","scope","feature_1","feature_2","correlation"]).to_csv("results/new_feature_families_correlations.csv",index=False)
pd.DataFrame(coef_rows).to_csv("results/new_feature_families_coefficients.csv",index=False)
