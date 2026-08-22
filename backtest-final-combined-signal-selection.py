"""Frozen one-signal-per-game ML/totals selector and flat-stake backtest."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

R=Path("results");SEED=20260819;BOOT=10000;ODDS_FLOOR=1.70;ML_MIN_P=.55;ML_MIN_EDGE=.04
BUCKET_EDGES=[-np.inf,1.5,1.6,1.7,1.8,1.9,2,2.25,np.inf];BUCKETS=["<1.50","1.50-1.59","1.60-1.69","1.70-1.79","1.80-1.89","1.90-1.99","2.00-2.24","2.25+"]
def decimal(s):
 s=pd.to_numeric(s,errors="coerce");o=pd.Series(np.nan,index=s.index);o[s.ge(100)]=1+s[s.ge(100)]/100;o[s.le(-100)]=1+100/s[s.le(-100)].abs();return o
def pnl(result,odds):return np.where(result.eq("W"),odds-1,np.where(result.eq("L"),-1,0.))
def maxdd(f):
 if not len(f):return 0.
 path=np.r_[0,f.sort_values(["date","game_id"]).profit_units.cumsum().to_numpy()];return float((np.maximum.accumulate(path)-path).max())
def candidates():
 pred=pd.read_csv(R/"v5_team_strength_oos_predictions_2022_2025.csv");pred.game_id=pred.game_id.astype(int)
 odds=pd.read_csv("data/processed/historical_mlb_moneylines_2022_2025.csv");book=odds[odds.sportsbook.eq("bet365")&odds.match_status.eq("matched")].copy();book.game_id=book.game_id.astype(int)
 if book.game_id.duplicated().any():raise ValueError("duplicate Bet365 moneyline game")
 book["home_decimal"]=decimal(book.current_home_odds);book["away_decimal"]=decimal(book.current_away_odds);rh=1/book.home_decimal;ra=1/book.away_decimal;book["home_novig"]=rh/(rh+ra);book["away_novig"]=ra/(rh+ra)
 ml=pred.merge(book[["game_id","home_decimal","away_decimal","home_novig","away_novig"]],on="game_id",how="inner",validate="one_to_one");is_home=ml.model_selected_team.eq(ml.home_team)
 mlc=pd.DataFrame({"game_id":ml.game_id,"date":ml.date,"season":ml.season,"home_team":ml.home_team,"away_team":ml.away_team,"candidate_type":"ML","selection":ml.model_selected_team,"model_probability":ml.model_selected_probability,"market_no_vig_probability":np.where(is_home,ml.home_novig,ml.away_novig),"decimal_odds":np.where(is_home,ml.home_decimal,ml.away_decimal),"result":np.where(ml.model_selection_won.eq(1),"W","L")})
 mlc["edge"]=mlc.model_probability-mlc.market_no_vig_probability;mlc["model_ev"]=mlc.model_probability*mlc.decimal_odds-1;mlc["model_qualified"]=mlc.model_probability.ge(ML_MIN_P)&mlc.edge.ge(ML_MIN_EDGE);mlc["qualification_rule"]="ML probability >= 0.55 and edge >= 0.04"
 tl=pd.read_csv(R/"totals_bet365_game_side_ledger.csv");tl.game_id=tl.game_id.astype(int);tc=pd.DataFrame({"game_id":tl.game_id,"date":tl.date,"season":tl.season,"candidate_type":tl.side.str.upper(),"selection":tl.side.str.title(),"model_probability":tl.model_probability,"market_no_vig_probability":tl.market_no_vig_probability,"decimal_odds":tl.decimal_odds,"edge":tl.model_edge,"model_ev":tl.model_ev,"result":tl.result});teams=pred[["game_id","home_team","away_team"]];tc=tc.merge(teams,on="game_id",how="left",validate="many_to_one");tc["model_qualified"]=tc.model_ev.gt(0);tc["qualification_rule"]="Totals model EV > 0"
 allc=pd.concat([mlc,tc],ignore_index=True);allc["passes_odds_floor"]=allc.decimal_odds.ge(ODDS_FLOOR);return allc
def choose(group,use_floor):
 q=group[group.model_qualified].copy()
 if use_floor:q=q[q.passes_odds_floor]
 if not len(q):return None,"no qualifying candidate"
 # Highest edge normally. Exception compares the strongest ML and O/U candidates.
 ml=q[q.candidate_type.eq("ML")].sort_values("edge",ascending=False);ou=q[~q.candidate_type.eq("ML")].sort_values("edge",ascending=False);reason="only qualifying candidate" if len(q)==1 else "higher-edge rule"
 winner=q.sort_values(["edge","decimal_odds"],ascending=False).iloc[0]
 if len(ml) and len(ou):
  a=ml.iloc[0];b=ou.iloc[0]
  if a.edge>.05 and b.edge>.05 and abs(a.edge-b.edge)<=.02:
   winner=pd.DataFrame([a,b]).sort_values(["decimal_odds","edge"],ascending=False).iloc[0];reason="5% / within-2pp higher-odds tiebreak"
 return int(winner.name),reason
def select(allc,use_floor):
 selected={};reasons={}
 for gid,g in allc.groupby("game_id"):
  idx,reason=choose(g,use_floor);selected[gid]=idx;reasons[gid]=reason
 out=allc.copy();out["selected"]=False;out["selection_reason"]=""
 for gid,idx in selected.items():
  if idx is not None:out.loc[idx,"selected"]=True;out.loc[idx,"selection_reason"]=reasons[gid]
 def reject(r):
  if r.selected:return "selected"
  if not r.model_qualified:return "failed model qualification"
  if use_floor and not r.passes_odds_floor:return "below 1.70 odds"
  win=out[(out.game_id.eq(r.game_id))&out.selected]
  if not len(win):return "no surviving candidate"
  why=win.iloc[0].selection_reason
  return "lost higher-odds tiebreak" if "tiebreak" in why else "lower edge (higher-edge rule)"
 out["rejection_reason"]=out.apply(reject,axis=1);return out
def game_universe(allc):
 meta=allc.sort_values("candidate_type").drop_duplicates("game_id")[["game_id","date","season","home_team","away_team"]];return meta
def ledger(selected,universe):
 s=selected[selected.selected].copy();s["profit_units"]=pnl(s.result,s.decimal_odds);s=s.sort_values(["date","game_id"]);s["cumulative_units"]=s.profit_units.cumsum()
 rejected=selected[selected.model_qualified&~selected.selected].groupby("game_id").apply(lambda g:" | ".join(f"{r.candidate_type}:{r.selection}; {r.rejection_reason}; edge={r.edge:.4f}; odds={r.decimal_odds:.3f}" for r in g.itertuples()),include_groups=False).rename("rejected_competing_candidates")
 s=s.merge(rejected,on="game_id",how="left");return s
def summary(s,universe,scope,kind="all"):
 total=len(universe);n=len(s);w=int(s.result.eq("W").sum());l=int(s.result.eq("L").sum());p=int(s.result.eq("P").sum());u=float(s.profit_units.sum())
 wins=s[s.result.eq("W")];loss=s[s.result.eq("L")]
 return {"scope":scope,"market_type":kind,"total_games":total,"signals":n,"passes":total-n,"pct_games_bet":n/total if total else np.nan,"ml_signals":int(s.candidate_type.eq("ML").sum()),"over_signals":int(s.candidate_type.eq("OVER").sum()),"under_signals":int(s.candidate_type.eq("UNDER").sum()),"wins":w,"losses":l,"pushes":p,"win_rate_ex_push":w/(w+l) if w+l else np.nan,"average_odds":s.decimal_odds.mean(),"median_odds":s.decimal_odds.median(),"average_model_probability":s.model_probability.mean(),"average_market_no_vig_probability":s.market_no_vig_probability.mean(),"average_edge":s.edge.mean(),"total_units":u,"roi":u/n if n else np.nan,"maximum_drawdown":maxdd(s),"winning_bet_average_odds":wins.decimal_odds.mean(),"winning_bet_median_odds":wins.decimal_odds.median(),"losing_bet_average_odds":loss.decimal_odds.mean(),"losing_bet_median_odds":loss.decimal_odds.median()}
def scopes(s,u):return [(str(y),s[s.season.eq(y)],u[u.season.eq(y)]) for y in range(2022,2026)]+[("combined",s,u)]
def main():
 allc=candidates();universe=game_universe(allc);nf=select(allc,False);ff=select(allc,True);no_floor=ledger(nf,universe);final=ledger(ff,universe)
 # Independent P&L audit.
 expected=pnl(final.result,final.decimal_odds)
 if not np.allclose(expected,final.profit_units,atol=1e-12) or not np.isclose(expected.sum(),final.profit_units.sum(),atol=1e-12):raise ValueError("P&L audit failed")
 season=pd.DataFrame([summary(s,u,scope) for scope,s,u in scopes(final,universe)]);market=pd.DataFrame([summary(g,universe,kind,kind) for kind,g in final.groupby("candidate_type")])
 # Pre-floor otherwise-selected price audit.
 no_floor["odds_bucket"]=pd.cut(no_floor.decimal_odds,BUCKET_EDGES,labels=BUCKETS,right=False)
 buckets=[]
 for b in BUCKETS:
  g=no_floor[no_floor.odds_bucket.eq(b)];z=summary(g,universe,b,b);z["break_even_win_rate"]=1/g.decimal_odds.mean() if len(g) else np.nan;buckets.append(z)
 buckets=pd.DataFrame(buckets)
 final_keys=set(zip(final.game_id,final.candidate_type,final.selection));no_floor_keys=set(zip(no_floor.game_id,no_floor.candidate_type,no_floor.selection));removed=no_floor[[ (g,t,s) not in final_keys for g,t,s in zip(no_floor.game_id,no_floor.candidate_type,no_floor.selection) ]];replacements=final[[ (g,t,s) not in no_floor_keys for g,t,s in zip(final.game_id,final.candidate_type,final.selection) ]];base=summary(no_floor,universe,"without_floor");floor=summary(final,universe,"with_1.70_floor");rem=summary(removed,universe,"removed_or_replaced_below_1.70")
 comparison=pd.DataFrame([{**base,"displaced_selections":0,"replacement_signals":0,"net_bets_removed":0,"change_total_units":0,"change_roi":0,"change_maximum_drawdown":0},{**floor,"displaced_selections":len(removed),"replacement_signals":len(replacements),"net_bets_removed":len(no_floor)-len(final),"change_total_units":floor["total_units"]-base["total_units"],"change_roi":floor["roi"]-base["roi"],"change_maximum_drawdown":floor["maximum_drawdown"]-base["maximum_drawdown"]},{**rem,"displaced_selections":len(removed),"replacement_signals":len(replacements),"net_bets_removed":len(no_floor)-len(final),"change_total_units":np.nan,"change_roi":np.nan,"change_maximum_drawdown":np.nan}])
 rng=np.random.default_rng(SEED);profits=final.profit_units.to_numpy();boot=rng.choice(profits,size=(BOOT,len(profits)),replace=True).mean(axis=1);bootstrap=pd.DataFrame([{"seed":SEED,"replicates":BOOT,"bets":len(profits),"observed_roi":profits.mean(),"roi_ci_2_5":np.quantile(boot,.025),"roi_ci_97_5":np.quantile(boot,.975),"ci_excludes_zero":np.quantile(boot,.025)>0 or np.quantile(boot,.975)<0}])
 passes=universe[~universe.game_id.isin(final.game_id)].copy();pass_reasons=ff[~ff.selected].groupby("game_id").apply(lambda g:" | ".join(f"{r.candidate_type}:{r.rejection_reason}" for r in g.itertuples()),include_groups=False).rename("pass_reasons");passes=passes.merge(pass_reasons,on="game_id",how="left")
 allc=ff.rename(columns={"selected":"selected_with_floor"});allc["selected_without_floor"]=nf["selected"].to_numpy()
 allc.to_csv(R/"combined_signal_complete_candidate_ledger.csv",index=False);final.to_csv(R/"combined_signal_final_one_per_game_ledger.csv",index=False);passes.to_csv(R/"combined_signal_pass_ledger.csv",index=False);buckets.to_csv(R/"combined_signal_odds_bucket_audit.csv",index=False);season.to_csv(R/"combined_signal_season_results.csv",index=False);market.to_csv(R/"combined_signal_market_type_results.csv",index=False);comparison.to_csv(R/"combined_signal_170_floor_comparison.csv",index=False);bootstrap.to_csv(R/"combined_signal_bootstrap_uncertainty.csv",index=False)
 consistency=(season[season.scope.ne("combined")].total_units.gt(0).sum());final_summary={"rules":{"ml":"probability >= 0.55 and edge >= 0.04","totals":"model EV > 0","minimum_decimal_odds":1.70,"one_signal_per_game":True},"combined":season[season.scope.eq("combined")].iloc[0].to_dict(),"profitable_seasons":int(consistency),"bootstrap":bootstrap.iloc[0].to_dict(),"pnl_audit":{"individual_sum":float(expected.sum()),"reported_total":float(final.profit_units.sum()),"exact_match":True}}
 (R/"combined_signal_final_summary.json").write_text(json.dumps(final_summary,indent=2,default=str),encoding="utf-8")
 print("\nODDS BUCKET AUDIT\n",buckets[["scope","signals","wins","losses","pushes","win_rate_ex_push","average_odds","break_even_win_rate","total_units","roi"]].to_string(index=False));print("\nFINAL RESULTS\n",season.to_string(index=False));print("\nMARKET TYPE\n",market.to_string(index=False));print("\n1.70 COUNTERFACTUAL\n",comparison.to_string(index=False));print("\nBOOTSTRAP\n",bootstrap.to_string(index=False));print("\nP&L audit exact:",expected.sum(),final.profit_units.sum())
if __name__=="__main__":main()
