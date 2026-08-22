"""Fixed-weight, transparent StatsImpl-style ML score and OOS backtest."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,brier_score_loss,log_loss,roc_auc_score

R=Path("results");YEARS=range(2021,2026);FOLDS=[([2021],2022),([2021,2022],2023),([2021,2022,2023],2024)];CLIP=3.;POINT_SCALE=20.;SEED=20260819
TOP={"sp":.35,"trend":.15,"homeaway":.14,"hand":.20,"combined":.08,"streak":.08}
COMP={
"sp":{"sp_season_xwoba_allowed_diff":(-1,.15),"sp_official_season_era_diff":(-1,.10),"sp_official_season_whip_diff":(-1,.05),"sp_l30_xwoba_allowed_diff":(-1,.10),"sp_official_recent3_starts_era_diff":(-1,.10),"sp_official_venue_era_diff":(-1,.05),"sp_matchup_season_xwoba_allowed_diff":(-1,.15),"sp_matchup_l30_xwoba_allowed_diff":(-1,.10),"sp_matchup_season_k_pct_diff":(1,.05),"sp_recent100_xwoba_allowed_diff":(-1,.10),"sp_days_rest_diff":(1,.05)},
"trend":{"season_woba_diff":(1,.65),"off_l7_woba_diff":(1,.25),"off_l7_hardhit_pct_diff":(1,.05),"off_l7_bb_pct_diff":(1,.025),"off_l7_k_pct_diff":(-1,.025)},
"homeaway":{"sit_venue_win_pct_diff":(1,.45),"ctx_venue_season_woba_diff":(1,.35),"ctx_venue_l30_woba_diff":(1,.10),"ctx_venue_season_k_pct_diff":(-1,.05),"ctx_venue_season_bb_pct_diff":(1,.05)},
"hand":{"season_platoon_woba_diff":(1,.35),"l30_platoon_woba_diff":(1,.15),"ctx_hand_season_woba_diff":(1,.25),"ctx_hand_l30_woba_diff":(1,.10),"sp_matchup_season_xwoba_allowed_diff":(-1,.10),"sp_matchup_l30_xwoba_allowed_diff":(-1,.05)},
"combined":{"ctx_combined_season_woba_diff":(1,.60),"ctx_combined_l30_woba_diff":(1,.25),"ctx_combined_season_k_pct_diff":(-1,.075),"ctx_combined_season_bb_pct_diff":(1,.075)},
"streak":{"sit_streak_diff":(1,.35),"sit_l10_win_pct_diff":(1,.45),"sit_l10_run_diff_per_game_diff":(1,.20)}}
THRESH=[0,.02,.04,.05,.075,.10];SCORE_EDGES=[0,5,10,15,20,25,30,np.inf];SCORE_LABELS=["0-5","5-10","10-15","15-20","20-25","25-30","30+"]
def load(y):
 b=pd.read_csv(f"data/processed/games_{y}_starter_lineup_matchup_features.csv")
 for stem in ["features_v7_offensive_form","features_v8_contextual_offense","features_situational","features_official_starter_pitching","features_statsimpl_starter_recent100"]:
  q=pd.read_csv(f"data/processed/{stem}_{y}.csv");over=[c for c in q if c in b and c!="game_id"];b=b.merge(q.drop(columns=over),on="game_id",validate="one_to_one")
 b["sp_official_venue_era_diff"]=b.home_sp_official_season_home_era-b.away_sp_official_season_away_era
 b["sp_recent100_xwoba_allowed_diff"]=(b.sp_recent100_lhb_xwoba_allowed_diff+b.sp_recent100_rhb_xwoba_allowed_diff)/2;b["season"]=y;return b
class Scorer:
 def fit(self,x):
  features=sorted({f for spec in COMP.values() for f in spec});self.med=x[features].median();self.mean=x[features].fillna(self.med).mean();self.sd=x[features].fillna(self.med).std().replace(0,1);raw=self._raw(x);self.cm={k:raw[k].mean() for k in COMP};self.cs={k:(raw[k].std() or 1) for k in COMP};return self
 def _raw(self,x):
  z=((x[list(self.mean.index)].fillna(self.med)-self.mean)/self.sd).clip(-CLIP,CLIP);return pd.DataFrame({k:sum(sign*w*z[f] for f,(sign,w) in spec.items()) for k,spec in COMP.items()},index=x.index)
 def transform(self,x):
  raw=self._raw(x);std=pd.DataFrame({k:((raw[k]-self.cm[k])/self.cs[k]).clip(-CLIP,CLIP) for k in COMP},index=x.index);contrib=pd.DataFrame({k:TOP[k]*std[k] for k in COMP},index=x.index);score=contrib.sum(axis=1);out=pd.DataFrame(index=x.index)
  for k in COMP:out[f"{k}_raw"]=raw[k];out[f"{k}_standardized"]=std[k];out[f"{k}_weighted_contribution"]=contrib[k]
  out["final_score_standardized"]=score;out["final_score_points"]=POINT_SCALE*score;return out
def fit_cal(method,score,y):
 if method=="logistic":m=LogisticRegression(max_iter=2000);m.fit(np.asarray(score).reshape(-1,1),y);return m
 m=IsotonicRegression(out_of_bounds="clip",y_min=.001,y_max=.999);m.fit(score,y);return m
def predict_cal(m,method,score):return m.predict_proba(np.asarray(score).reshape(-1,1))[:,1] if method=="logistic" else m.predict(score)
def metrics(y,p):return {"log_loss":log_loss(y,p),"brier":brier_score_loss(y,p),"auc":roc_auc_score(y,p),"accuracy":accuracy_score(y,p>=.5)}
def american(s):
 s=pd.to_numeric(s,errors="coerce");o=pd.Series(np.nan,index=s.index);o[s.ge(100)]=1+s[s.ge(100)]/100;o[s.le(-100)]=1+100/s[s.le(-100)].abs();return o
def maxdd(g):
 path=np.r_[0,g.sort_values(["date","game_id"]).profit_units.cumsum().to_numpy()];return float((np.maximum.accumulate(path)-path).max())
def profit_summary(g,total,model,threshold,scope):
 n=len(g);w=int(g.result.eq("W").sum());l=int(g.result.eq("L").sum());u=g.profit_units.sum()
 return {"model":model,"threshold":threshold,"scope":scope,"eligible_games":total,"bets":n,"passes":total-n,"wins":w,"losses":l,"win_rate":w/n if n else np.nan,"average_odds":g.decimal_odds.mean(),"break_even_win_rate":1/g.decimal_odds.mean() if n else np.nan,"average_model_probability":g.model_probability.mean(),"average_market_no_vig_probability":g.market_no_vig_probability.mean(),"average_claimed_edge":g.edge.mean(),"realized_advantage":g.result.eq("W").mean()-g.market_no_vig_probability.mean() if n else np.nan,"units":u,"roi":u/n if n else np.nan,"maximum_drawdown":maxdd(g) if n else 0}
def calibration_rows(d,model):
 q=d.copy();q["bin"]=pd.cut(q.model_probability,np.linspace(0,1,11),include_lowest=True)
 return [{"model":model,"bin":str(b),"games":len(g),"average_probability":g.model_probability.mean(),"actual_home_win_rate":g.home_win.mean(),"error":g.home_win.mean()-g.model_probability.mean()} for b,g in q.groupby("bin",observed=False)]
def main():
 R.mkdir(exist_ok=True);dev={y:load(y) for y in range(2021,2025)};foldrows=[];pred_by_method={m:[] for m in ["logistic","isotonic"]};breakdowns=[]
 for yrs,vy in FOLDS:
  tr=pd.concat([dev[y] for y in yrs],ignore_index=True);va=dev[vy];sc=Scorer().fit(tr);st=sc.transform(tr);sv=sc.transform(va);base=pd.DataFrame({"game_id":va.game_id,"date":va.date,"season":vy,"home_team":va.home_team,"away_team":va.away_team,"home_win":va.home_win});breakdowns.append(pd.concat([base.reset_index(drop=True),sv.reset_index(drop=True)],axis=1))
  for method in pred_by_method:
   cal=fit_cal(method,st.final_score_standardized,tr.home_win);p=predict_cal(cal,method,sv.final_score_standardized);foldrows.append({"method":method,"validation_year":vy,**metrics(va.home_win,p)});q=base.copy();q["model_probability"]=p;q["final_score_points"]=sv.final_score_points.to_numpy();pred_by_method[method].append(q)
 folds=pd.DataFrame(foldrows);summ=folds.groupby("method",as_index=False).agg(mean_log_loss=("log_loss","mean"),mean_brier=("brier","mean"),mean_auc=("auc","mean"),mean_accuracy=("accuracy","mean")).sort_values(["mean_log_loss","mean_brier"]);winner=summ.iloc[0].method
 # Frozen calibration method and component formula; only now load 2025.
 hold=load(2025);tr=pd.concat([dev[y] for y in range(2021,2025)],ignore_index=True);sc=Scorer().fit(tr);st=sc.transform(tr);sh=sc.transform(hold);cal=fit_cal(winner,st.final_score_standardized,tr.home_win);p=predict_cal(cal,winner,sh.final_score_standardized);base=pd.DataFrame({"game_id":hold.game_id,"date":hold.date,"season":2025,"home_team":hold.home_team,"away_team":hold.away_team,"home_win":hold.home_win});base["model_probability"]=p;base["final_score_points"]=sh.final_score_points.to_numpy();pred_by_method[winner].append(base);breakdowns.append(pd.concat([base.drop(columns=["model_probability","final_score_points"]).reset_index(drop=True),sh.reset_index(drop=True)],axis=1))
 oos=pd.concat(pred_by_method[winner],ignore_index=True);components=pd.concat(breakdowns,ignore_index=True);oos["selected_home"]=oos.model_probability.ge(.5);oos["model_selected_probability"]=np.maximum(oos.model_probability,1-oos.model_probability);oos["selected_won"]=np.where(oos.selected_home,oos.home_win.eq(1),oos.home_win.eq(0));oos["absolute_score_points"]=oos.final_score_points.abs()
 oos["score_bucket"]=pd.cut(oos.absolute_score_points,SCORE_EDGES,labels=SCORE_LABELS,right=False);score=[]
 for scope,d in [(str(y),oos[oos.season.eq(y)]) for y in range(2022,2026)]+[("combined",oos)]:
  for b in SCORE_LABELS:
   g=d[d.score_bucket.eq(b)];score.append({"scope":scope,"score_bucket":b,"games":len(g),"wins":int(g.selected_won.sum()),"losses":int((~g.selected_won).sum()),"win_rate":g.selected_won.mean()})
 # Odds are introduced only after score/calibration selection is frozen.
 odds=pd.read_csv("data/processed/historical_mlb_moneylines_2022_2025.csv");book=odds[odds.sportsbook.eq("bet365")&odds.match_status.eq("matched")].copy();book.game_id=book.game_id.astype(int);book["home_decimal"]=american(book.current_home_odds);book["away_decimal"]=american(book.current_away_odds);rh=1/book.home_decimal;ra=1/book.away_decimal;book["home_novig"]=rh/(rh+ra);book["away_novig"]=ra/(rh+ra)
 def market_ledger(pred,name):
  pred=pred.rename(columns={"model_probability":"home_probability"});z=pred.merge(book[["game_id","home_decimal","away_decimal","home_novig","away_novig"]],on="game_id",validate="one_to_one");home=z.home_probability.ge(.5);z["model_probability"]=np.where(home,z.home_probability,1-z.home_probability);z["decimal_odds"]=np.where(home,z.home_decimal,z.away_decimal);z["market_no_vig_probability"]=np.where(home,z.home_novig,z.away_novig);z["edge"]=z.model_probability-z.market_no_vig_probability;z["result"]=np.where(np.where(home,z.home_win.eq(1),z.home_win.eq(0)),"W","L");z["profit_units"]=np.where(z.result.eq("W"),z.decimal_odds-1,-1);z["model"]=name;return z
 stats=market_ledger(oos,"statsimpl_weighted");v5=pd.read_csv(R/"v5_team_strength_oos_predictions_2022_2025.csv").rename(columns={"predicted_home_probability":"model_probability"});v5=market_ledger(v5,"v5_team_strength");common=set(stats[stats.decimal_odds.ge(1.70)].game_id)&set(v5[v5.decimal_odds.ge(1.70)].game_id)
 rows=[]
 for name,d in [("statsimpl_weighted",stats),("v5_team_strength",v5)]:
  eligible=d[d.game_id.isin(common)]
  for th in THRESH:
   for scope,g0 in [(str(y),eligible[eligible.season.eq(y)]) for y in range(2022,2026)]+[("combined",eligible)]:rows.append(profit_summary(g0[g0.edge.ge(th)],len(g0),name,th,scope))
 profit=pd.DataFrame(rows);high=pd.concat([stats[stats.game_id.isin(common)&stats.edge.ge(.10)],v5[v5.game_id.isin(common)&v5.edge.ge(.10)]],ignore_index=True)
 rng=np.random.default_rng(SEED);g=stats[stats.game_id.isin(common)&stats.edge.ge(.10)];boot=rng.choice(g.profit_units.to_numpy(),size=(10000,len(g)),replace=True).mean(axis=1);unc=pd.DataFrame([{"seed":SEED,"samples":10000,"bets":len(g),"observed_roi":g.profit_units.mean(),"roi_ci_2_5":np.quantile(boot,.025),"roi_ci_97_5":np.quantile(boot,.975),"ci_excludes_zero":np.quantile(boot,.025)>0 or np.quantile(boot,.975)<0}])
 predictive=[];calrows=[]
 for name,d in [("statsimpl_weighted",stats),("v5_team_strength",v5)]:
  q=d[d.game_id.isin(common)];predictive.append({"model":name,"games":len(q),**metrics(q.home_win,q.home_probability)});qc=q.copy();qc["model_probability"]=qc.home_probability;calrows+=calibration_rows(qc,name)
 spec={"label":"transparent StatsImpl approximation","top_level_weights":TOP,"component_inputs":COMP,"input_and_component_z_clip":CLIP,"point_scale":POINT_SCALE,"calibration_selected":winner,"selection_basis":"2022-2024 mean Log Loss, Brier secondary","totals_note":"No StatsImpl totals score was built. It would require run-environment, starter/bullpen expected innings and runs, park/weather, and lineup run-production components with totals-specific frozen weights."}
 components.to_csv(R/"statsimpl_weighted_component_breakdown.csv",index=False);folds.to_csv(R/"statsimpl_weighted_calibration_folds.csv",index=False);summ.to_csv(R/"statsimpl_weighted_calibration_selection.csv",index=False);oos.to_csv(R/"statsimpl_weighted_chronological_predictions.csv",index=False);pd.DataFrame(score).to_csv(R/"statsimpl_weighted_score_buckets.csv",index=False);profit.to_csv(R/"statsimpl_weighted_edge_profitability.csv",index=False);high.to_csv(R/"statsimpl_weighted_ge10_game_ledger.csv",index=False);unc.to_csv(R/"statsimpl_weighted_ge10_bootstrap.csv",index=False);pd.DataFrame(predictive).to_csv(R/"statsimpl_weighted_vs_v5_predictive_metrics.csv",index=False);pd.DataFrame(calrows).to_csv(R/"statsimpl_weighted_vs_v5_calibration.csv",index=False);(R/"statsimpl_weighted_frozen_specification.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
 print("\nCALIBRATION SELECTION\n",summ.to_string(index=False));print("\nSCORE BUCKETS COMBINED\n",pd.DataFrame(score).query("scope=='combined'").to_string(index=False));print("\nPROFITABILITY COMBINED\n",profit.query("scope=='combined'").to_string(index=False));print("\nPREDICTIVE COMPARISON\n",pd.DataFrame(predictive).to_string(index=False));print("\n>=10% BOOTSTRAP\n",unc.to_string(index=False))
if __name__=="__main__":main()
