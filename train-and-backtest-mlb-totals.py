"""Baseball-only MLB totals model selection, frozen 2025 test, and odds diagnostics."""
import os
import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FOLDS=[([2021],2022),([2021,2022],2023),([2021,2022,2023],2024)]
PRED_BUCKETS=[0,6,7,8,9,10,11,12,np.inf];PRED_LABELS=["<6","6-7","7-8","8-9","9-10","10-11","11-12","12+"]
EDGE_EDGES=[-np.inf,0,.02,.04,.06,.08,.10,np.inf];EDGE_LABELS=["edge_le_0","edge_0_2","edge_2_4","edge_4_6","edge_6_8","edge_8_10","edge_10_plus"]
EV_EDGES=[-np.inf,0,.02,.05,.10,.15,np.inf];EV_LABELS=["ev_le_0","ev_0_2","ev_2_5","ev_5_10","ev_10_15","ev_15_plus"]

def load_year(y):
 b=pd.read_csv(f"data/processed/games_{y}_starter_lineup_matchup_features.csv")
 extras=[]
 for stem in ["features_v7_offensive_form","features_richer_starter","features_v6_bullpen_availability","features_situational"]:
  f=pd.read_csv(f"data/processed/{stem}_{y}.csv");
  if f.game_id.duplicated().any() or set(f.game_id)!=set(b.game_id):raise ValueError(f"Coverage failure {stem} {y}")
  extras.append(f)
 out=b
 for f in extras:out=out.merge(f,on="game_id",validate="one_to_one")
 out["season"]=y;out["actual_total_runs"]=out.home_score+out.away_score
 return out

def feature_sets(frame):
 numeric=set(frame.select_dtypes(include=np.number).columns)
 exclude=("score","_id","_code","known_hitters","_pa","_pa_log","_weight","_denom","_bbe","fly_balls","_contact","_fb","out_pitches","first_pitches","_pitches","_starts","_games")
 core=[c for c in numeric if (c.startswith("home_") or c.startswith("away_")) and c!="home_win" and not any(x in c for x in exclude)
       and not c.startswith(("home_off_","away_off_","home_sp_rich_","away_sp_rich_","home_bp_avail_","away_bp_avail_","home_sit_","away_sit_"))]
 offense=[c for c in numeric if c.startswith(("home_off_","away_off_")) and not any(x in c for x in exclude)]
 rich=[c for c in numeric if c.startswith(("home_sp_rich_","away_sp_rich_")) and not any(x in c for x in exclude)]
 bullpen=[c for c in numeric if c.startswith(("home_bp_avail_","away_bp_avail_")) and not any(x in c for x in ("_diff",))]
 context=[c for c in numeric if c.startswith(("home_sit_","away_sit_")) and not any(x in c for x in ("_games",))]
 return {"core":sorted(core),"core_plus_offense":sorted(set(core+offense)),"comprehensive":sorted(set(core+offense+rich+bullpen+context))}

def make_model(name,param):
 if name=="ridge":return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",Ridge(alpha=param))])
 if name=="poisson":return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",PoissonRegressor(alpha=param,max_iter=2000))])
 return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",HistGradientBoostingRegressor(loss="poisson",learning_rate=.05,max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=42))])

CONFIGS=[(f"ridge_alpha_{a}","ridge",a) for a in [.1,1,10]]+[(f"poisson_alpha_{a}","poisson",a) for a in [.01,.1,1]]+[("hist_poisson","hist",None)]
def model_metrics(y,p):
 p=np.clip(p,.05,None);return {"rmse":mean_squared_error(y,p)**.5,"mae":mean_absolute_error(y,p),"poisson_deviance":mean_poisson_deviance(y,p),"mean_predicted":p.mean(),"actual_mean":y.mean()}
def dispersion(y,mu):
 mu=np.clip(np.asarray(mu),.05,None);return max(0,float(np.sum((np.asarray(y)-mu)**2-np.asarray(y))/np.sum(mu**2)))
def calibration_rows(frame,scope):
 x=frame.copy();x["bucket"]=pd.cut(x.predicted_total,PRED_BUCKETS,labels=PRED_LABELS,right=False)
 return [{"scope":scope,"bucket":b,"games":len(g),"mean_prediction":g.predicted_total.mean(),"actual_mean":g.actual_total_runs.mean(),"rmse":mean_squared_error(g.actual_total_runs,g.predicted_total)**.5 if len(g) else np.nan} for b in PRED_LABELS for g in [x[x.bucket.eq(b)]]]
def probs(mu,alpha,line):
 integer=float(line).is_integer();floor=int(np.floor(line))
 if alpha>1e-10:
  size=1/alpha;p=size/(size+mu);cdf=lambda k:nbinom.cdf(k,size,p);pmf=lambda k:nbinom.pmf(k,size,p);sf=lambda k:nbinom.sf(k,size,p)
 else:cdf=lambda k:poisson.cdf(k,mu);pmf=lambda k:poisson.pmf(k,mu);sf=lambda k:poisson.sf(k,mu)
 if integer:return float(sf(int(line))),float(cdf(int(line)-1)),float(pmf(int(line)))
 return float(sf(floor)),float(cdf(floor)),0.0
def american_decimal(s):
 s=pd.to_numeric(s,errors="coerce");out=pd.Series(np.nan,index=s.index);valid=s.ne(0)&s.abs().ge(100);out[valid&s.gt(0)]=1+s[valid&s.gt(0)]/100;out[valid&s.lt(0)]=1+100/s[valid&s.lt(0)].abs();return out
def maxdd(p):
 c=np.asarray(p).cumsum();path=np.r_[0,c];return float((np.maximum.accumulate(path)-path).max())
def bet_summary(f,scope,strategy):
 n=len(f);w=int(f.result.eq("W").sum());l=int(f.result.eq("L").sum());push=int(f.result.eq("P").sum());u=f.profit_units.sum() if n else 0
 return {"scope":scope,"strategy":strategy,"bets":n,"wins":w,"losses":l,"pushes":push,"win_rate_ex_push":w/(w+l) if w+l else np.nan,"average_odds":f.decimal_odds.mean(),"units":u,"roi":u/n if n else np.nan,"maximum_drawdown":maxdd(f.sort_values(["date","game_id"]).profit_units) if n else np.nan}

def main():
 dev={y:load_year(y) for y in range(2021,2025)};sets=feature_sets(dev[2021]);rows=[]
 for fs,features in sets.items():
  for label,kind,param in CONFIGS:
   for yrs,vy in FOLDS:
    tr=pd.concat([dev[y] for y in yrs],ignore_index=True);va=dev[vy];m=make_model(kind,param);m.fit(tr[features],tr.actual_total_runs);pred=np.clip(m.predict(va[features]),.05,None)
    rows.append({"feature_set":fs,"model_config":label,"validation_year":vy,"features":len(features),**model_metrics(va.actual_total_runs,pred)})
 folds=pd.DataFrame(rows);summ=(folds.groupby(["feature_set","model_config","features"],as_index=False).agg(mean_rmse=("rmse","mean"),mean_mae=("mae","mean"),mean_poisson_deviance=("poisson_deviance","mean"))).sort_values(["mean_rmse","mean_mae"])
 winner=summ.iloc[0];fs=winner.feature_set;label=winner.model_config;features=sets[fs];kind,param=next((k,p) for l,k,p in CONFIGS if l==label)
 print("Frozen development selection:",fs,label,"features",len(features));oos=[]
 for yrs,vy in FOLDS:
  tr=pd.concat([dev[y] for y in yrs],ignore_index=True);va=dev[vy];m=make_model(kind,param);m.fit(tr[features],tr.actual_total_runs);pt=np.clip(m.predict(tr[features]),.05,None);pv=np.clip(m.predict(va[features]),.05,None);a=dispersion(tr.actual_total_runs,pt)
  oos.append(pd.DataFrame({"game_id":va.game_id,"date":va.date,"season":vy,"actual_total_runs":va.actual_total_runs,"predicted_total":pv,"dispersion_alpha":a}))
 oos=pd.concat(oos,ignore_index=True);dev_results=pd.DataFrame([{"validation_year":y,**model_metrics(g.actual_total_runs,g.predicted_total)} for y,g in oos.groupby("season")]);dev_results=pd.concat([dev_results,pd.DataFrame([{"validation_year":"combined_2022_2024",**model_metrics(oos.actual_total_runs,oos.predicted_total)}])],ignore_index=True)
 cal=[]
 for y,g in oos.groupby("season"):cal+=calibration_rows(g,str(y))
 cal+=calibration_rows(oos,"combined_2022_2024")
 # Freeze, then load 2025 exactly once.
 tr=pd.concat([dev[y] for y in range(2021,2025)],ignore_index=True);va=load_year(2025);m=make_model(kind,param);m.fit(tr[features],tr.actual_total_runs);pt=np.clip(m.predict(tr[features]),.05,None);pv=np.clip(m.predict(va[features]),.05,None);a=dispersion(tr.actual_total_runs,pt)
 hold=pd.DataFrame({"game_id":va.game_id,"date":va.date,"season":2025,"actual_total_runs":va.actual_total_runs,"predicted_total":pv,"dispersion_alpha":a});hold_results=pd.DataFrame([model_metrics(hold.actual_total_runs,hold.predicted_total)]);hold_cal=pd.DataFrame(calibration_rows(hold,"2025"));allpred=pd.concat([oos,hold],ignore_index=True)
 # Only now load totals odds/currentLine.
 odds=pd.read_csv("data/processed/historical_mlb_totals_2022_2025.csv");book=odds[odds.sportsbook.eq("bet365")&odds.match_status.eq("matched")].copy();book["over_decimal"]=american_decimal(book.current_over_odds);book["under_decimal"]=american_decimal(book.current_under_odds);book["opening_over_decimal"]=american_decimal(book.opening_over_odds);book["opening_under_decimal"]=american_decimal(book.opening_under_odds);book=book.dropna(subset=["current_total","over_decimal","under_decimal"])
 merged=allpred.merge(book[["game_id","opening_total","opening_over_odds","opening_under_odds","current_total","current_over_odds","current_under_odds","over_decimal","under_decimal","opening_over_decimal","opening_under_decimal"]],on="game_id",validate="one_to_one")
 pr=[probs(r.predicted_total,r.dispersion_alpha,r.current_total) for r in merged.itertuples()];merged[["p_over","p_under","p_push"]]=pd.DataFrame(pr,index=merged.index)
 rawo=1/merged.over_decimal;rawu=1/merged.under_decimal;merged["market_over_no_vig"]=rawo/(rawo+rawu);merged["market_under_no_vig"]=rawu/(rawo+rawu)
 sides=[]
 for side in ["over","under"]:
  x=merged.copy();x["side"]=side;x["model_probability"]=x[f"p_{side}"];x["market_no_vig_probability"]=x[f"market_{side}_no_vig"];x["decimal_odds"]=x[f"{side}_decimal"];x["model_edge"]=x.model_probability-x.market_no_vig_probability;x["model_ev"]=x.model_probability*x.decimal_odds-1
  if side=="over":x["result"]=np.where(x.actual_total_runs>x.current_total,"W",np.where(x.actual_total_runs<x.current_total,"L","P"))
  else:x["result"]=np.where(x.actual_total_runs<x.current_total,"W",np.where(x.actual_total_runs>x.current_total,"L","P"))
  x["profit_units"]=np.where(x.result.eq("W"),x.decimal_odds-1,np.where(x.result.eq("L"),-1,0));sides.append(x)
 ledger=pd.concat(sides,ignore_index=True);ledger["edge_bucket"]=pd.cut(ledger.model_edge,EDGE_EDGES,labels=EDGE_LABELS,right=False);ledger["ev_bucket"]=pd.cut(ledger.model_ev,EV_EDGES,labels=EV_LABELS,right=False)
 summaries=[]
 for scope,g in [(str(y),ledger[ledger.season.eq(y)]) for y in range(2022,2026)]+[("combined",ledger)]:
  pos=g[g.model_ev.gt(0)];summaries += [bet_summary(pos,scope,"positive_ev_all"),bet_summary(pos[pos.side.eq("over")],scope,"positive_ev_over"),bet_summary(pos[pos.side.eq("under")],scope,"positive_ev_under")]
 profit=pd.DataFrame(summaries);edge=pd.DataFrame([bet_summary(g[g.edge_bucket.eq(b)],scope,b) for scope,g in [(str(y),ledger[ledger.season.eq(y)]) for y in range(2022,2026)]+[("combined",ledger)] for b in EDGE_LABELS]);ev=pd.DataFrame([bet_summary(g[g.ev_bucket.eq(b)],scope,b) for scope,g in [(str(y),ledger[ledger.season.eq(y)]) for y in range(2022,2026)]+[("combined",ledger)] for b in EV_LABELS])
 pc=[]
 ledger["prob_bucket"]=pd.cut(ledger.model_probability,[0,.4,.45,.5,.55,.6,.65,1],right=False)
 for (side,b),g in ledger.groupby(["side","prob_bucket"],observed=True):pc.append({"side":side,"probability_bucket":str(b),"bets":len(g),"average_unconditional_win_probability":g.model_probability.mean(),"actual_unconditional_win_rate":g.result.eq("W").mean(),"average_conditional_win_probability_ex_push":(g.model_probability/(1-g.p_push)).mean(),"win_rate_ex_push":g.result.eq("W").sum()/g.result.ne("P").sum(),"push_rate":g.result.eq("P").mean()})
 probcal=pd.DataFrame(pc)
 merged["projected_direction"]=np.sign(merged.predicted_total-merged.opening_total);merged["market_total_movement"]=merged.current_total-merged.opening_total;oro=1/merged.opening_over_decimal;oru=1/merged.opening_under_decimal;merged["opening_over_no_vig"]=oro/(oro+oru);merged["current_over_no_vig"]=merged.market_over_no_vig;merged["over_price_probability_movement"]=merged.current_over_no_vig-merged.opening_over_no_vig;line_comp=merged.projected_direction.ne(0)&merged.market_total_movement.ne(0);price_comp=merged.projected_direction.ne(0)&merged.market_total_movement.eq(0)&merged.over_price_probability_movement.ne(0);merged["combined_market_direction"]=np.where(merged.market_total_movement.ne(0),np.sign(merged.market_total_movement),np.sign(merged.over_price_probability_movement));all_comp=merged.projected_direction.ne(0)&merged.combined_market_direction.ne(0);movement=pd.DataFrame([{"games":len(merged),"line_changed_comparable_games":int(line_comp.sum()),"line_direction_agreement_pct":np.sign(merged.loc[line_comp,"market_total_movement"]).eq(merged.loc[line_comp,"projected_direction"]).mean(),"unchanged_line_price_comparable_games":int(price_comp.sum()),"price_direction_agreement_pct":np.sign(merged.loc[price_comp,"over_price_probability_movement"]).eq(merged.loc[price_comp,"projected_direction"]).mean(),"combined_comparable_games":int(all_comp.sum()),"combined_direction_agreement_pct":merged.loc[all_comp,"combined_market_direction"].eq(merged.loc[all_comp,"projected_direction"]).mean(),"average_total_movement":merged.market_total_movement.mean(),"average_movement_in_model_direction":(merged.projected_direction*merged.market_total_movement).mean(),"average_over_price_probability_movement":merged.over_price_probability_movement.mean()}])
 os.makedirs("results",exist_ok=True);folds.to_csv("results/totals_model_selection_folds.csv",index=False);summ.to_csv("results/totals_model_selection_summary.csv",index=False);pd.DataFrame({"feature":features,"feature_set":fs}).to_csv("results/totals_selected_features.csv",index=False);dev_results.to_csv("results/totals_chronological_results.csv",index=False);pd.DataFrame(cal).to_csv("results/totals_development_calibration.csv",index=False);hold_results.to_csv("results/totals_untouched_2025_results.csv",index=False);hold_cal.to_csv("results/totals_2025_calibration.csv",index=False);allpred.to_csv("results/totals_oos_predictions_2022_2025.csv",index=False);ledger.to_csv("results/totals_bet365_game_side_ledger.csv",index=False);profit.to_csv("results/totals_bet365_positive_ev_summary.csv",index=False);edge.to_csv("results/totals_bet365_edge_buckets.csv",index=False);ev.to_csv("results/totals_bet365_ev_buckets.csv",index=False);probcal.to_csv("results/totals_probability_calibration.csv",index=False);movement.to_csv("results/totals_opening_to_current_movement.csv",index=False)
 print("\nMODEL SELECTION\n",summ.head(12).to_string(index=False));print("\nDEV RESULTS\n",dev_results.to_string(index=False));print("\n2025\n",hold_results.to_string(index=False));print("\nPOSITIVE EV\n",profit.to_string(index=False));print("\nEDGE COMBINED\n",edge[edge.scope.eq("combined")].to_string(index=False));print("\nEV COMBINED\n",ev[ev.scope.eq("combined")].to_string(index=False));print("\nMOVEMENT\n",movement.to_string(index=False));print("Frozen model was not modified after 2025; no thresholds optimized.")
if __name__=="__main__":main()
