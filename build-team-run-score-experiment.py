"""Build a fixed, transparent StatsImpl-inspired team Run Score (not proprietary)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

YEARS=range(2021,2026);OUT=Path("data/processed");RES=Path("results");CLIP=3.;SCALE=15.
TOP={"offense":.27,"recent":.13,"risp":.10,"hand":.10,"combined":.07,"starter":.25,"bullpen":.06,"context":.02}
COMP={
"offense":{"off_season_woba":.40,"off_season_ops":.30,"actual_lineup_season_woba":.30},
"recent":{"off_l7_woba":.50,"off_l30_ops":.25,"off_l7_hardhit":.15,"off_l7_bb":.05,"off_l7_k":-.05},
"risp":{"risp_season_woba":.60,"risp_season_ops":.25,"risp_l30_woba":.15},
"hand":{"hand_season_woba":.55,"hand_l30_woba":.25,"hand_season_bb":.10,"hand_season_k":-.10},
"combined":{"combined_season_woba":.60,"combined_l30_woba":.25,"combined_season_bb":.075,"combined_season_k":-.075},
"starter":{"opp_sp_season_xwoba":.20,"opp_sp_official_era":.15,"opp_sp_official_whip":.10,"opp_sp_l30_xwoba":.15,"opp_sp_recent3_era":.15,"opp_sp_official_hr9":.05,"opp_sp_lineup_matchup_xwoba":.20},
"bullpen":{"opp_bp_official_era":.20,"opp_bp_l30_era":.25,"opp_bp_available_era":.25,"opp_bp_available_whip":.15,"opp_bp_mean_fatigue":.15},
"context":{"is_home":1.0}}
BUCKET_EDGES=[-np.inf,40,50,60,66,70,75,80,np.inf];BUCKETS=["<40","40-50","50-60","60-66","66-70","70-75","75-80","80+"]
def wide(y):
 b=pd.read_csv(OUT/f"games_{y}_starter_lineup_matchup_features.csv")
 for stem in ["features_v7_offensive_form","features_v8_contextual_offense","features_statsimpl_offense_risp","features_official_starter_pitching","features_official_bullpen_pitching","features_v6_bullpen_availability"]:
  q=pd.read_csv(OUT/f"{stem}_{y}.csv");over=[c for c in q if c in b and c!="game_id"];b=b.merge(q.drop(columns=over),on="game_id",validate="one_to_one")
 return b
def long(y):
 b=wide(y);rows=[]
 for g in b.itertuples(index=False):
  gate=(getattr(g,"home_sp_official_season_outs")>=45) and (getattr(g,"away_sp_official_season_outs")>=45)
  for side in ["away","home"]:
   opp="home" if side=="away" else "away";v=lambda name:getattr(g,name,np.nan)
   rows.append({"game_id":g.game_id,"date":g.date,"season":y,"team_side":side,"team":v(f"{side}_team"),"opponent":v(f"{opp}_team"),"actual_team_runs":v(f"{side}_score"),"both_starters_15ip_gate":gate,"opposing_starter_season_ip":v(f"{opp}_sp_official_season_outs")/3,
    "off_season_woba":v(f"{side}_season_woba"),"off_season_ops":v(f"{side}_season_ops"),"actual_lineup_season_woba":v(f"{side}_lineup_season_woba"),
    "off_l7_woba":v(f"{side}_off_l7_woba"),"off_l30_ops":v(f"{side}_l30_ops"),"off_l7_hardhit":v(f"{side}_off_l7_hardhit_pct"),"off_l7_bb":v(f"{side}_off_l7_bb_pct"),"off_l7_k":v(f"{side}_off_l7_k_pct"),
    "risp_season_woba":v(f"{side}_season_risp_woba"),"risp_season_ops":v(f"{side}_season_risp_ops"),"risp_l30_woba":v(f"{side}_l30_risp_woba"),
    "hand_season_woba":v(f"{side}_ctx_hand_season_woba"),"hand_l30_woba":v(f"{side}_ctx_hand_l30_woba"),"hand_season_bb":v(f"{side}_ctx_hand_season_bb_pct"),"hand_season_k":v(f"{side}_ctx_hand_season_k_pct"),
    "combined_season_woba":v(f"{side}_ctx_combined_season_woba"),"combined_l30_woba":v(f"{side}_ctx_combined_l30_woba"),"combined_season_bb":v(f"{side}_ctx_combined_season_bb_pct"),"combined_season_k":v(f"{side}_ctx_combined_season_k_pct"),
    "opp_sp_season_xwoba":v(f"{opp}_sp_season_xwoba_allowed"),"opp_sp_official_era":v(f"{opp}_sp_official_season_era"),"opp_sp_official_whip":v(f"{opp}_sp_official_season_whip"),"opp_sp_l30_xwoba":v(f"{opp}_sp_l30_xwoba_allowed"),"opp_sp_recent3_era":v(f"{opp}_sp_official_recent3_starts_era"),"opp_sp_official_hr9":v(f"{opp}_sp_official_season_hr9"),"opp_sp_lineup_matchup_xwoba":v(f"{opp}_sp_matchup_season_xwoba_allowed"),
    "opp_bp_official_era":v(f"{opp}_bp_official_season_era"),"opp_bp_l30_era":v(f"{opp}_bp_official_l30_era"),"opp_bp_available_era":v(f"{opp}_bp_official_available_pool_era"),"opp_bp_available_whip":v(f"{opp}_bp_official_available_pool_whip"),"opp_bp_mean_fatigue":v(f"{opp}_bp_avail_mean_fatigue"),"is_home":1 if side=="home" else 0})
 return pd.DataFrame(rows)
class Score:
 def fit(self,d):
  fs=sorted({f for x in COMP.values() for f in x});self.med=d[fs].median();x=d[fs].fillna(self.med);self.mean=x.mean();self.sd=x.std().replace(0,1);raw=self.raw(d);self.cm=raw.mean();self.cs=raw.std().replace(0,1);return self
 def raw(self,d):
  z=((d[list(self.mean.index)].fillna(self.med)-self.mean)/self.sd).clip(-CLIP,CLIP);return pd.DataFrame({k:sum(w*z[f] for f,w in spec.items()) for k,spec in COMP.items()},index=d.index)
 def transform(self,d):
  raw=self.raw(d);std=((raw-self.cm)/self.cs).clip(-CLIP,CLIP);weighted=pd.DataFrame({k:SCALE*TOP[k]*std[k] for k in COMP},index=d.index);o=pd.DataFrame(index=d.index)
  for k in COMP:o[f"{k}_raw"]=raw[k];o[f"{k}_standardized"]=std[k];o[f"{k}_point_contribution"]=weighted[k]
  o["run_score_unclipped"]=50+weighted.sum(axis=1);o["run_score"]=o.run_score_unclipped.clip(0,100);den=weighted.abs().sum(axis=1);o["largest_component_share"]=weighted.abs().max(axis=1)/den.replace(0,np.nan);o["dominant_component_over_60pct"]=o.largest_component_share.gt(.60);return o
def bucket_rows(d,scope):
 d=d.copy();d["score_bucket"]=pd.cut(d.run_score,BUCKET_EDGES,labels=BUCKETS,right=False);rows=[]
 for b in BUCKETS:
  g=d[d.score_bucket.eq(b)];r={"scope":scope,"score_bucket":b,"team_games":len(g),"mean_run_score":g.run_score.mean(),"average_actual_runs":g.actual_team_runs.mean(),"median_runs":g.actual_team_runs.median()}
  for n in [0,2,3,4,5,6,7]:r[f"pct_scoring_{n}_plus"]=g.actual_team_runs.ge(n).mean()
  rows.append(r)
 return rows
def gate_rows(d):
 rows=[]
 for scope,g0 in [(str(y),d[d.season.eq(y)]) for y in YEARS]+[("combined",d)]:
  for gate,g in g0.groupby("both_starters_15ip_gate"):
   rows.append({"scope":scope,"passes_15ip_gate":gate,"team_games":len(g),"games":g.game_id.nunique(),"mean_score":g.run_score.mean(),"average_runs":g.actual_team_runs.mean(),"spearman":g.run_score.corr(g.actual_team_runs,method="spearman")})
 return rows
def cutoff_rows(d):
 rows=[]
 for scope,g0 in [(str(y),d[d.season.eq(y)]) for y in range(2022,2026)]+[("combined",d[d.season.ge(2022)])]:
  for label,mask in [("<66",g0.run_score.lt(66)),(">=66",g0.run_score.ge(66))]:
   g=g0[mask];r={"scope":scope,"cutoff_group":label,"team_games":len(g),"average_score":g.run_score.mean(),"average_actual_runs":g.actual_team_runs.mean(),"median_runs":g.actual_team_runs.median()}
   for n in range(0,8):r[f"pct_scoring_{n}_plus"]=g.actual_team_runs.ge(n).mean()
   rows.append(r)
 return rows
def main():
 RES.mkdir(exist_ok=True);data={y:long(y) for y in YEARS};allrows=[]
 for y in YEARS:
  train=data[2021] if y==2021 else pd.concat([data[k] for k in range(2021,y)],ignore_index=True);sc=Score().fit(train);z=sc.transform(data[y]);base=data[y].reset_index(drop=True);base["transformation_fit_years"]="2021 reference" if y==2021 else f"2021-{y-1}";base["genuinely_oos_transformation"]=y>2021;allrows.append(pd.concat([base,z.reset_index(drop=True)],axis=1))
 out=pd.concat(allrows,ignore_index=True);evald=out[out.season.ge(2022)].copy();b=[]
 for y in range(2022,2026):b+=bucket_rows(evald[evald.season.eq(y)],str(y))
 b+=bucket_rows(evald,"combined");b=pd.DataFrame(b)
 seasons=pd.DataFrame([{"scope":str(y),"team_games":len(g),"mean_score":g.run_score.mean(),"average_runs":g.actual_team_runs.mean(),"spearman":g.run_score.corr(g.actual_team_runs,method="spearman"),"score_bucket_average_runs_monotonic":pd.Series([r["average_actual_runs"] for r in bucket_rows(g,str(y))]).dropna().is_monotonic_increasing} for y,g in evald.groupby("season")]+[{"scope":"combined","team_games":len(evald),"mean_score":evald.run_score.mean(),"average_runs":evald.actual_team_runs.mean(),"spearman":evald.run_score.corr(evald.actual_team_runs,method="spearman"),"score_bucket_average_runs_monotonic":pd.Series([r["average_actual_runs"] for r in bucket_rows(evald,"combined")]).dropna().is_monotonic_increasing}])
 cutoff=pd.DataFrame(cutoff_rows(out));gate=pd.DataFrame(gate_rows(out));contrib=[]
 for label,g in evald.groupby(evald.run_score.ge(66)):
  for k in COMP:contrib.append({"cutoff_group":">=66" if label else "<66","component":k,"team_games":len(g),"mean_raw":g[f"{k}_raw"].mean(),"mean_standardized":g[f"{k}_standardized"].mean(),"mean_point_contribution":g[f"{k}_point_contribution"].mean()})
 contrib=pd.DataFrame(contrib);dominance=pd.DataFrame([{"scope":"combined","team_games":len(evald),"dominant_over_60pct":int(evald.dominant_component_over_60pct.sum()),"pct_dominant_over_60pct":evald.dominant_component_over_60pct.mean(),"median_largest_component_share":evald.largest_component_share.median(),"max_largest_component_share":evald.largest_component_share.max()}])
 spec={"label":"StatsImpl-inspired transparent team Run Score; not proprietary","component_weights":TOP,"component_inputs":COMP,"input_and_component_z_clip":CLIP,"score_formula":"clip(50 + 15 * sum(component_weight * component_z), 0, 100)","starter_gate":"both listed starters have >=15.0 official current-season IP (>=45 outs), diagnostic only","subweights_optimized":False,"outcomes_used_to_fit_score":False,"park_context_note":"No leakage-safe historical park/weather layer exists locally; context is limited to home batting status, while venue x hand is represented separately."}
 out.to_csv(RES/"team_run_score_complete_team_games_2021_2025.csv",index=False);out[[c for c in out if c.endswith(("_raw","_standardized","_point_contribution"))]+["game_id","season","team_side","team","run_score","largest_component_share","dominant_component_over_60pct"]].to_csv(RES/"team_run_score_component_breakdown.csv",index=False);b.to_csv(RES/"team_run_score_bucket_results.csv",index=False);seasons.to_csv(RES/"team_run_score_season_results.csv",index=False);cutoff.to_csv(RES/"team_run_score_66_cutoff_diagnostic.csv",index=False);gate.to_csv(RES/"team_run_score_starter_15ip_gate_diagnostic.csv",index=False);contrib.to_csv(RES/"team_run_score_66_component_differences.csv",index=False);dominance.to_csv(RES/"team_run_score_component_dominance.csv",index=False);out[out.season.eq(2025)].to_csv(RES/"team_run_score_untouched_2025.csv",index=False);(RES/"team_run_score_frozen_specification.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")
 print("\nSEASON RESULTS\n",seasons.to_string(index=False));print("\nCOMBINED BUCKETS\n",b[b.scope.eq("combined")].to_string(index=False));print("\n66 CUTOFF\n",cutoff.to_string(index=False));print("\nCOMPONENT DIFFERENCES\n",contrib.to_string(index=False));print("\nDOMINANCE\n",dominance.to_string(index=False));print("\n15 IP GATE\n",gate.to_string(index=False))
if __name__=="__main__":main()
