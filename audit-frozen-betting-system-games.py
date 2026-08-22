"""Outcome-blind audit of the frozen ML, totals and Team Run Score system."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent; R=ROOT/"results"; D=ROOT/"data"/"processed"
SEED=20250819; PRIMARY=777007; ODDS_FLOOR=1.70; ML_MIN_P=.55; ML_MIN_EDGE=.04

def imod(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def dec(x): return 1+x/100 if x>=100 else 1+100/abs(x)
def norm_team(x):
 x=str(x).strip(); return {"Athletics Athletics":"Athletics","Oakland Athletics":"Athletics"}.get(x,x)

def raw_odds():
 raw=json.loads((ROOT/"data/raw/historical_odds/mlb_odds_dataset.json").read_text(encoding="utf-8"));rows=[]
 for ds,games in raw.items():
  if not ds.startswith("2025-"):continue
  for j,g in enumerate(games):
   v=g.get("gameView",{}); away=norm_team((v.get("awayTeam") or {}).get("fullName"));home=norm_team((v.get("homeTeam") or {}).get("fullName"))
   ml=next((z for z in (g.get("odds",{}).get("moneyline") or []) if z.get("sportsbook")=="bet365"),None)
   tt=next((z for z in (g.get("odds",{}).get("totals") or []) if z.get("sportsbook")=="bet365"),None)
   if not ml or not tt:continue
   mc=ml.get("currentLine",{});tc=tt.get("currentLine",{})
   need=[mc.get("homeOdds"),mc.get("awayOdds"),tc.get("total"),tc.get("overOdds"),tc.get("underOdds")]
   if any(x is None for x in need):continue
   rows.append({"source_key":f"{ds}_{j}","date":ds,"away_team":away,"home_team":home,"start":v.get("startDate"),
    "ml_home_american":need[0],"ml_away_american":need[1],"total_line":need[2],"over_american":need[3],"under_american":need[4]})
 return pd.DataFrame(rows)

def score_objects():
 rs=imod("rs","build-team-run-score-experiment.py");hist=pd.concat([rs.long(y) for y in range(2021,2025)],ignore_index=True);sc=rs.Score().fit(hist)
 base=rs.long(2025);z=sc.transform(base);out=pd.concat([base.reset_index(drop=True),z.reset_index(drop=True)],axis=1)
 return rs,sc,out

def fit_models():
 ml=imod("ml","backtest-v5-underlying-team-strength.py"); tr=pd.concat([ml.load_year(y) for y in range(2021,2025)],ignore_index=True);va=ml.load_year(2025);mm=ml.pipeline();mm.fit(tr[ml.FEATURES],tr[ml.TARGET])
 tm=imod("tm","train-and-backtest-mlb-totals.py");td={y:tm.load_year(y) for y in range(2021,2026)};features=pd.read_csv(R/"totals_selected_features.csv").feature.tolist();mt=tm.make_model("poisson",1);ttrain=pd.concat([td[y] for y in range(2021,2025)],ignore_index=True);mt.fit(ttrain[features],ttrain.actual_total_runs);alpha=tm.dispersion(ttrain.actual_total_runs,np.clip(mt.predict(ttrain[features]),.05,None))
 return ml,tr,va,mm,tm,td[2025],features,ttrain,mt,alpha

def contributions(pipe,train,row,features,model_type):
 imp=pipe.named_steps["imputer"];scale=pipe.named_steps.get("scaler",pipe.named_steps.get("scale"));model=pipe.named_steps["model"]
 vals=row[features].to_frame().T;xi=imp.transform(vals)[0];z=scale.transform(vals)[0];med=imp.statistics_;means=scale.mean_;sd=scale.scale_;coef=model.coef_.ravel();terms=coef*z
 return pd.DataFrame({"feature":features,"raw_value":pd.to_numeric(vals.iloc[0],errors="coerce").to_numpy(),"imputed_value":xi,"training_median":med,"training_mean":means,"training_scale_sd":sd,"standardized_value":z,"coefficient":coef,"log_scale_contribution":terms,"model":model_type})

def select_priced(cands):
 q=[x for x in cands if x["qualified"] and x["odds"]>=ODDS_FLOOR]
 if not q:return None,"no qualifying priced candidate"
 winner=max(q,key=lambda x:(x["edge"],x["odds"]));reason="only qualifying candidate" if len(q)==1 else "higher-edge rule"
 ml=[x for x in q if x["type"]=="ML"];ou=[x for x in q if x["type"]!="ML"]
 if ml and ou:
  a=max(ml,key=lambda x:x["edge"]);b=max(ou,key=lambda x:x["edge"])
  if a["edge"]>.05 and b["edge"]>.05 and abs(a["edge"]-b["edge"])<=.02:
   winner=max([a,b],key=lambda x:(x["odds"],x["edge"]));reason="5% / within-2pp higher-odds tiebreak"
 return winner,reason

def explain(points):
 pos=points.sort_values(ascending=False);neg=points.sort_values();
 good=", ".join(f"{k} ({v:+.2f})" for k,v in pos.head(2).items());bad=", ".join(f"{k} ({v:+.2f})" for k,v in neg.head(2).items())
 return f"Strongest upward components: {good}. Strongest downward components: {bad}."

def mdtable(frame):
 cols=list(frame.columns);lines=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
 for row in frame.itertuples(index=False,name=None):lines.append("| "+" | ".join("NaN" if pd.isna(v) else (f"{v:.6f}" if isinstance(v,(float,np.floating)) else str(v)) for v in row)+" |")
 return "\n".join(lines)

def main():
 odds=raw_odds();rs,rsc,run=score_objects();ml,mtrain,m25,mm,tm,t25,tfeatures,ttrain,mt,alpha=fit_models()
 official_sp=pd.read_csv(D/"features_official_starter_pitching_2025.csv").set_index("game_id");official_bp=pd.read_csv(D/"features_official_bullpen_pitching_2025.csv").set_index("game_id")
 games=m25[["game_id","date","home_team","away_team","home_score","away_score","home_win"]].copy();games.date=games.date.astype(str)
 # Unique date/team matches only; no scores participate in matching or selection.
 merged=games.merge(odds,on=["date","home_team","away_team"],how="inner",validate="many_to_many");counts=merged.groupby("game_id").size();merged=merged[merged.game_id.isin(counts[counts.eq(1)].index)].drop_duplicates("game_id")
 pred=pd.read_csv(R/"v5_team_strength_oos_predictions_2022_2025.csv");tp=pd.read_csv(R/"totals_oos_predictions_2022_2025.csv")
 gate=run.groupby("game_id").agg(gate=("both_starters_15ip_gate","all"),inputs_complete=(list({f for x in rs.COMP.values() for f in x})[0],"size"))
 eligible=merged.merge(pred[["game_id","predicted_home_probability"]],on="game_id").merge(tp[["game_id","predicted_total","dispersion_alpha"]],on="game_id").merge(gate[["gate"]],on="game_id")
 eligible=eligible[eligible.gate].copy();ids=eligible.game_id.astype(int).tolist()
 if PRIMARY not in ids:raise RuntimeError("Primary game is not uniquely odds-matched and gated")
 others=pd.Series(sorted(set(ids)-{PRIMARY})).sample(10,random_state=SEED).astype(int).tolist();selected=[PRIMARY]+others
 # Selection has now frozen without any outcome columns being consulted.
 summary=[];feature_ledgers=[];run_ledgers=[];report=["# Frozen betting-system audit: 11 OOS games","",f"Fixed seed: `{SEED}`. Outcomes were not used for selection. Games: {selected}",""]
 for n,gid in enumerate(selected,1):
  g=merged[merged.game_id.eq(gid)].iloc[0];mr=m25[m25.game_id.eq(gid)].iloc[0];tr=t25[t25.game_id.eq(gid)].iloc[0]
  mc=contributions(mm,mtrain,mr,ml.FEATURES,"moneyline");tc=contributions(mt,ttrain,tr,tfeatures,"totals");mc["game_id"]=gid;tc["game_id"]=gid;feature_ledgers += [mc,tc]
  ph=float(mm.predict_proba(mr[ml.FEATURES].to_frame().T)[0,1]);pa=1-ph;pred_saved=float(pred.loc[pred.game_id.eq(gid),"predicted_home_probability"].iloc[0]);
  if not np.isclose(ph,pred_saved,atol=1e-12):raise RuntimeError(f"ML identity failure {gid}")
  mu=float(np.clip(mt.predict(tr[tfeatures].to_frame().T)[0],.05,None));mu_saved=float(tp.loc[tp.game_id.eq(gid),"predicted_total"].iloc[0]);
  if not np.isclose(mu,mu_saved,atol=1e-10):raise RuntimeError(f"totals identity failure {gid}")
  hd,ad=dec(g.ml_home_american),dec(g.ml_away_american);rh,ra=1/hd,1/ad;hm,am=rh/(rh+ra),ra/(rh+ra)
  od,ud=dec(g.over_american),dec(g.under_american);ro,ru=1/od,1/ud;om,um=ro/(ro+ru),ru/(ro+ru);po,pu,pp=tm.probs(mu,alpha,g.total_line)
  mlhome={"type":"ML","side":g.home_team,"prob":ph,"market":hm,"edge":ph-hm,"odds":hd,"qualified":ph>=ML_MIN_P and ph-hm>=ML_MIN_EDGE};mlaway={"type":"ML","side":g.away_team,"prob":pa,"market":am,"edge":pa-am,"odds":ad,"qualified":pa>=ML_MIN_P and pa-am>=ML_MIN_EDGE}
  # Existing ML system generates only the model-preferred side.
  mlcand=mlhome if ph>=.5 else mlaway
  over={"type":"OVER","side":"Over","prob":po,"market":om,"edge":po-om,"odds":od,"qualified":po*od-1>0};under={"type":"UNDER","side":"Under","prob":pu,"market":um,"edge":pu-um,"odds":ud,"qualified":pu*ud-1>0}
  winner,why=select_priced([mlcand,over,under])
  rr=run[run.game_id.eq(gid)].set_index("team_side");awayrs=rr.loc["away"];homers=rr.loc["home"];ttsignals=[x for x,r in [(g.away_team,awayrs),(g.home_team,homers)] if r.run_score>=66 and r.both_starters_15ip_gate]
  final="PASS" if winner is None else f"{winner['type']} {winner['side']}"; unpriced="; ".join(f"{x} TT OVER" for x in ttsignals) if ttsignals else "none"
  osp=official_sp.loc[gid];obp=official_bp.loc[gid]
  snapshot=[]
  for side in ("away","home"):
   snapshot.append({"side":side,"team":getattr(mr,f"{side}_team"),"starter":getattr(mr,f"{side}_starter_name"),"season_woba":getattr(mr,f"{side}_season_woba"),"l30_woba":getattr(mr,f"{side}_l30_woba"),"lineup_woba":getattr(mr,f"{side}_lineup_season_woba"),"platoon_woba":getattr(mr,f"{side}_season_platoon_woba"),"SP_xwOBA":getattr(mr,f"{side}_sp_season_xwoba_allowed"),"SP_K%":getattr(mr,f"{side}_sp_season_k_pct"),"SP_BB%":getattr(mr,f"{side}_sp_season_bb_pct"),"SP_matchup_xwOBA":getattr(mr,f"{side}_sp_matchup_season_xwoba_allowed"),"BP_wOBA":getattr(mr,f"{side}_bp_season_woba_allowed"),"BP_K%":getattr(mr,f"{side}_bp_season_k_pct"),"BP_BB%":getattr(mr,f"{side}_bp_season_bb_pct"),"unused_official_SP_ERA":osp[f"{side}_sp_official_season_era"],"unused_official_SP_WHIP":osp[f"{side}_sp_official_season_whip"],"unused_official_BP_ERA":obp[f"{side}_bp_official_season_era"],"unused_official_BP_WHIP":obp[f"{side}_bp_official_season_whip"]})
  report += [f"## {n}. {g.away_team} at {g.home_team} — {g.date} — game {gid}","",f"Starters: {snapshot[0]['starter']} ({g.away_team}) vs {snapshot[1]['starter']} ({g.home_team}).","","Raw pregame baseball snapshot (`unused_official_*` columns were available but NOT USED by either frozen champion):","",mdtable(pd.DataFrame(snapshot)),"","### Pregame moneyline","",f"Home {g.home_team}: P={ph:.6f}, Bet365 {g.ml_home_american:+g} ({hd:.4f}), no-vig={hm:.6f}, edge={ph-hm:+.6f}, passes={mlhome['qualified'] and hd>=ODDS_FLOOR}.",f"Away {g.away_team}: P={pa:.6f}, Bet365 {g.ml_away_american:+g} ({ad:.4f}), no-vig={am:.6f}, edge={pa-am:+.6f}, passes={mlaway['qualified'] and ad>=ODDS_FLOOR}.",f"Model preference: **{mlcand['side']}**. Exact logit intercept={mm.named_steps['model'].intercept_[0]:.6f}; feature terms sum={mc.log_scale_contribution.sum():.6f}.","","Largest ML feature terms (USED):","",mdtable(mc.reindex(mc.log_scale_contribution.abs().sort_values(ascending=False).index).head(10)[["feature","raw_value","training_median","standardized_value","coefficient","log_scale_contribution"]]),"", "### Pregame full-game total","",f"Prediction={mu:.6f}; Bet365 line={g.total_line:g}; Over {g.over_american:+g} ({od:.4f}), Under {g.under_american:+g} ({ud:.4f}).",f"P(Over)={po:.6f}, P(Under)={pu:.6f}, P(push)={pp:.6f}; market no-vig Over={om:.6f}, Under={um:.6f}; edges Over={po-om:+.6f}, Under={pu-um:+.6f}.",f"Existing result: Over qualified={over['qualified'] and od>=ODDS_FLOOR}; Under qualified={under['qualified'] and ud>=ODDS_FLOOR}.","","Largest totals log-mean feature terms (USED):","",mdtable(tc.reindex(tc.log_scale_contribution.abs().sort_values(ascending=False).index).head(10)[["feature","raw_value","training_median","standardized_value","coefficient","log_scale_contribution"]]),"","### Team Run Score","",f"Away {g.away_team}: {awayrs.run_score:.6f}; >=66={awayrs.run_score>=66}. Home {g.home_team}: {homers.run_score:.6f}; >=66={homers.run_score>=66}. Both-starter 15-IP gate={bool(awayrs.both_starters_15ip_gate)}."]
  for side,row in [("away",awayrs),("home",homers)]:
   pts=pd.Series({c:row[f"{c}_point_contribution"] for c in rs.COMP});run_ledgers.append(pd.DataFrame({"game_id":gid,"team_side":side,"team":row.team,"component":pts.index,"point_contribution":pts.values,"run_score":row.run_score}))
   report += [f"{row.team}: "+", ".join(f"{c}={pts[c]:+.4f}" for c in rs.COMP)+f"; sum={pts.sum():+.6f}; 50+sum={50+pts.sum():.6f}. {explain(pts)}"]
  report += ["",f"Unpriced TT signal(s): **{unpriced}**. These did not compete against priced ML/O/U candidates.","", "### Final frozen selection","",f"**FINAL SYSTEM PICK = {final}** — {why}.","", "### Why (pregame only)","",f"The priced selector chose {final} from the exact qualification and edge rules. The ML model's strongest raw standardized terms and the totals model's strongest raw standardized terms are shown above; negative terms are the strongest opposing evidence. Available official ERA/WHIP, recent-100 starter splits, and RISP/OPS layers were not inputs to either frozen champion unless their exact column appears in the USED tables.",""]
  # Outcome revealed only here.
  actual=int(g.home_score+g.away_score);mlout="W" if winner and winner['type']=="ML" and ((winner['side']==g.home_team and g.home_win==1) or (winner['side']==g.away_team and g.home_win==0)) else ("L" if winner and winner['type']=="ML" else "N/A")
  if winner and winner['type'] in ("OVER","UNDER"):
   totout="P" if actual==g.total_line else ("W" if (winner['type']=="OVER" and actual>g.total_line) or (winner['type']=="UNDER" and actual<g.total_line) else "L")
  else:totout="N/A"
  report += ["### Actual result — revealed after reconstruction","",f"Final: {g.away_team} {int(g.away_score)}, {g.home_team} {int(g.home_score)}; total runs={actual}; selected-wager outcome={mlout if winner and winner['type']=='ML' else totout}.",""]
  summary.append({"game_id":gid,"date":g.date,"away_team":g.away_team,"home_team":g.home_team,"away_starter":snapshot[0]["starter"],"home_starter":snapshot[1]["starter"],"home_probability":ph,"away_probability":pa,"ml_home_odds":hd,"ml_away_odds":ad,"ml_home_novig":hm,"ml_away_novig":am,"predicted_total":mu,"total_line":g.total_line,"p_over":po,"p_under":pu,"p_push":pp,"over_odds":od,"under_odds":ud,"market_over_novig":om,"market_under_novig":um,"away_run_score":awayrs.run_score,"home_run_score":homers.run_score,"starter_gate":bool(awayrs.both_starters_15ip_gate),"unpriced_tt_signals":unpriced,"final_pick":final,"selection_reason":why,"away_runs":g.away_score,"home_runs":g.home_score,"actual_total":actual,"selected_outcome":mlout if winner and winner['type']=='ML' else totout})
 # Cross-game diagnosis is descriptive, not adaptive.
 s=pd.DataFrame(summary);report += ["# Cross-game diagnosis","",f"Audited {len(s)} fixed OOS games. Mean ML selected confidence={np.maximum(s.home_probability,1-s.home_probability).mean():.4f}; mean absolute ML-market disagreement={np.maximum((s.home_probability-s.ml_home_novig).abs(),(s.away_probability-s.ml_away_novig).abs()).mean():.4f}.","The complete ledgers preserve all feature terms, including correlated representations whose individual coefficients can mask family importance. Official ERA/WHIP and recent-100 diagnostics remained available but were not champion inputs; the Team Run Score used a separate transparent component architecture and did not affect priced historical selection.","No signs, coefficients, thresholds, probabilities, or outcomes were changed during this audit."]
 pd.DataFrame(summary).to_csv(R/"frozen_system_11_game_audit_summary.csv",index=False);pd.concat(feature_ledgers,ignore_index=True).to_csv(R/"frozen_system_11_game_feature_contributions.csv",index=False);pd.concat(run_ledgers,ignore_index=True).to_csv(R/"frozen_system_11_game_run_score_components.csv",index=False);(R/"frozen_system_11_game_audit_report.md").write_text("\n".join(report),encoding="utf-8")
 print("Selected games:",selected);print(s[["game_id","date","away_team","home_team","home_probability","predicted_total","total_line","away_run_score","home_run_score","final_pick","selected_outcome"]].to_string(index=False));print("Saved report and three ledgers.")

if __name__=="__main__":main()
