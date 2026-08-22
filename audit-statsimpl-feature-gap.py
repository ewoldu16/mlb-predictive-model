"""Create the StatsImpl feature-gap audit and validate new information layers."""
from pathlib import Path
import numpy as np
import pandas as pd

R=Path("results"); P=Path("data/processed"); YEARS=range(2021,2026)
def row(feature,status,equiv,kind,local,download,risk,rec):
 return dict(StatsImpl_feature=feature,Already_in_project=status,Equivalent_existing_feature=equiv,Exact_approximate_missing=kind,Source_available_locally=local,New_download_required=download,Leakage_risk=risk,Implementation_recommendation=rec)

ROWS=[
row("Offense: season wOBA","Yes","season_woba_diff","Exact","Yes","No","Current game","Reuse existing"),
row("Offense: L30/L15/L7 wOBA","Yes","V7 off_l7/l15/l30_woba_diff","Exact","Yes","No","Current date","Reuse existing"),
row("Offense: season/L30 AVG, OBP, SLG, OPS","New","features_statsimpl_offense_risp_*: home/away/diffs","Exact public equivalent","Yes","No","Current date","Built"),
row("Offense: K% / BB%","Yes","season/l30 offense plus V7 L7/L15/L30","Exact","Yes","No","Current date","Reuse existing"),
row("Offense: HardHit%","Yes","V7 season/L7/L15/L30 HardHit%","Exact","Yes","No","Missing contact/current date","Reuse existing"),
row("Offense: HR/FB","Yes","V7 season and L30 HR/FB","Exact","Yes","No","Fly-ball denominator/current date","Reuse existing"),
row("Offense: home/away splits","Yes","V8 venue season/L30 wOBA/K/BB","Exact","Yes","No","Current date","Reuse existing"),
row("Offense: vs LHP/RHP","Yes","V8 opposing-hand season/L30 wOBA/K/BB","Exact","Yes","No","Starter hand/current date","Reuse existing"),
row("Offense: venue x starter hand","Yes","V8 combined season/L30 wOBA/K/BB","Exact","Yes","No","Sparse samples/current date","Reuse existing"),
row("Offense: RISP season wOBA/AVG/OBP/SLG/OPS","New","features_statsimpl_offense_risp_*","Exact public equivalent","Yes","No","Base state/current date","Built with PA/AB"),
row("Offense: RISP recent performance","New","L30 RISP wOBA/AVG/OBP/SLG/OPS","Exact public equivalent","Yes","No","Sparse samples/current date","Built with PA/AB; retain NaN"),
row("Offense: recent form","Yes","V7 rolling offense and situational L10","Exact/transparent","Yes","No","Current date","Reuse existing"),
row("Offense vs opposing-starter quality","Yes","features_opponent_quality_offense_* continuous quality interactions","Transparent approximation","Yes","No","Opponent quality must be pregame","Reuse existing"),
row("Starter: season/recent ERA","No","None","Missing exact","No boxscore ER","Yes","Future/full-season ER","Do not approximate as ERA; official boxscore download needed"),
row("Starter: season/recent WHIP","New","starter recent/venue plus bullpen WHIP proxy only","Approximate","Yes","No","Out attribution/baserunning outs","Use explicitly labelled proxy; exact needs boxscores"),
row("Starter: K/9 or K%","Yes","season/l30 sp_k_pct; richer starter K-BB","Exact K%","Yes","No","Current game","Prefer K%"),
row("Starter: BB/9 or BB%","Yes","season/l30 sp_bb_pct","Exact BB%","Yes","No","Current game","Prefer BB%"),
row("Starter: HR/9 / HR/FB","Yes","richer starter HR/PA and HR/FB","Transparent equivalent","Yes","No","Current game","Reuse existing"),
row("Starter: wOBA/xwOBA allowed","Yes","advanced and matchup layers","Exact/Statcast-derived","Yes","No","Current game","Reuse existing"),
row("Starter: HardHit% / Barrel%","Yes","richer starter season/L30/context","Exact","Yes","No","Measured contact/current game","Reuse existing"),
row("Starter: velocity","Yes","advanced velocity_diff; arsenal characteristics","Exact","Yes","No","Current game","Reuse existing"),
row("Starter: whiff/chase/zone/CSW","Yes","advanced + richer starter","Exact definitions","Yes","No","Current game","Reuse existing"),
row("Starter: home/road","New","features_statsimpl_starter_recent100_* season_home/away","Exact public split","Yes","No","Current venue/current date","Built"),
row("Starter: vs LHB/RHB season/L30","Yes","starter-lineup matchup layer","Exact","Yes","No","Current game","Reuse existing"),
row("Starter: recent 100 TBF vs LHB/RHB","New","features_statsimpl_starter_recent100_*","Transparent trailing window","Yes","No","Ordering/current date","Built; trailing 100 overall then hand split"),
row("Starter: pitch arsenal x actual lineup","Yes","features_arsenal_lineup_matchup_*","Transparent approximation","Yes","No","Actual lineup/current game","Reuse existing"),
row("Starter: workload/rest/pitch count","Yes","advanced rest and pitch-count features","Exact/transparent","Yes","No","Current game","Reuse existing"),
row("Bullpen: season/recent ERA","No","None","Missing exact","No boxscore ER","Yes","Earned-run attribution","Do not mislabel proxies as ERA"),
row("Bullpen: season/L30 WHIP/K/BB/wOBA/xwOBA","New","features_statsimpl_bullpen_traditional_*","Exact rates except labelled WHIP proxy","Yes","No","Starter exclusion/current date","Built"),
row("Bullpen: individual reliever quality","Yes","V6 availability quality","Transparent composite ingredients","Yes","No","Future roster assignment","Reuse existing"),
row("Bullpen: recent pitch counts/rest/consecutive days","Yes","V6 W1/2/3/5/7 workload states","Exact","Yes","No","Current-game relievers","Reuse existing"),
row("Bullpen: closer/high-leverage roles","Yes","V6 trailing-45-day close/late usage rank","Approximate role inference","Yes","No","Future saves/roles","Reuse existing prior-only labels"),
row("Bullpen: closer/high-leverage availability","Yes","V6 closer fatigue/days since + high-leverage quality","Transparent approximation","Yes","No","Future role/current game","Reuse existing"),
row("Bullpen: depth/available quality","Yes","V6 pool/rested/fatigued/available quality","Transparent approximation","Yes","No","Future assignments","Reuse existing"),
row("Team: overall/home-away record","Yes","situational overall and venue win%","Exact","Yes","No","Current game","Reuse existing"),
row("Team: L10/streak/previous-game result","Yes","situational L10, streak, previous result","Exact","Yes","No","Same-day ordering","Reuse existing date-strict version"),
row("Team: run differential/recent run differential","Yes","situational run diff and L10 run diff","Exact","Yes","No","Current game","Reuse existing"),
row("Team: Pythagorean/actual-minus-Pythagorean","Yes","situational Pythagorean features","Transparent formula","Yes","No","Current game","Reuse existing"),
row("Team: series game/series record entering game","No","None","Missing unreliable","No authoritative series metadata/time","Yes","Doubleheader ordering/series boundaries","Omitted rather than inferred"),
row("Team: game after win/loss","Yes","previous-game result differential","Exact date-strict","Yes","No","Doubleheaders","Reuse; same-day excluded"),
row("Derived: starter matchup score","Ingredients","platoon + starter-lineup + arsenal features","Transparent components","Yes","No","Proprietary formula","Do not copy score; retain raw components"),
row("Derived: lineup pressure / pitcher difficulty","Ingredients","actual lineup quality, opponent quality, arsenal matchup","Transparent components","Yes","No","Proprietary formula","Do not copy score; retain raw components"),
row("Derived: bullpen edge / closer-depth grade","Ingredients","V6 home/away/differential states","Transparent components","Yes","No","Proprietary formula/roles","Do not copy grade; retain raw components"),
]

def validate():
 records=[]
 families=["offense_risp","starter_recent100","bullpen_traditional"]
 for fam in families:
  for y in YEARS:
   path=P/f"features_statsimpl_{fam}_{y}.csv"; d=pd.read_csv(path)
   expected=len(pd.read_csv(f"data/raw/games_{y}.csv",usecols=["game_id"]))
   records.append({"family":fam,"year":y,"rows":len(d),"expected":expected,"coverage_pct":100*len(d)/expected,
    "duplicate_game_ids":int(d.game_id.duplicated().sum()),"infinite_values":int(np.isinf(d.select_dtypes('number')).sum().sum()),
    "missing_cells":int(d.drop(columns="game_id").isna().sum().sum())})
 v=pd.DataFrame(records)
 if not ((v.rows==v.expected)&(v.duplicate_game_ids==0)&(v.infinite_values==0)).all():raise ValueError("output validation failed")
 return v

def main():
 R.mkdir(exist_ok=True); audit=pd.DataFrame(ROWS);audit.to_csv(R/"statsimpl_feature_gap_audit.csv",index=False)
 validation=validate();validation.to_csv(R/"statsimpl_new_feature_validation.csv",index=False)
 temporal=[]
 for f in R.glob("statsimpl_*_temporal_audit.csv"):
  x=pd.read_csv(f);x["audit_file"]=f.name;temporal.append(x)
 t=pd.concat(temporal,ignore_index=True);t.to_csv(R/"statsimpl_master_temporal_leakage_audit.csv",index=False)
 if not t.passed.astype(bool).all():raise ValueError("temporal audit failed")
 lines=["# StatsImpl feature-gap audit and completion report","",f"Audited {len(audit)} feature/statistic categories. New output validation: {len(validation)} season-family files passed.","",
 "## Completed information layers","","- Traditional overall offense (AVG/OBP/SLG/OPS) and season/L30 RISP wOBA/AVG/OBP/SLG/OPS with PA/AB denominators.",
 "- Starter trailing-100-TBF handedness splits and current-venue season splits.","- Team bullpen season/L30 K%, BB%, wOBA, xwOBA and explicitly labelled WHIP/HR9 proxies.","",
 "## Deliberate omissions","","- Exact ERA and WHIP require official pitcher boxscore earned runs/innings. The local cache lacks them; no run proxy is called ERA.",
 "- Series position/entering record is omitted because local schedules lack authoritative series IDs and start ordering needed for doubleheaders.",
 "- Proprietary StatsImpl scores/grades are not copied; their public-data ingredients remain separate and inspectable.","",
 "## Leakage and integrity","","All new histories use source_date < target_date, exclude all same-date games, preserve genuine NaN, and are restricted to known regular-season game IDs. Every output has exact game coverage, zero duplicate IDs and zero infinities. Random temporal audit rows all passed."]
 (R/"statsimpl_feature_gap_master_report.md").write_text("\n".join(lines),encoding="utf-8")
 print(audit.groupby("Exact_approximate_missing").size().to_string());print("\nValidation:\n",validation.to_string(index=False));print(f"\nTemporal audit rows={len(t)}, passed={t.passed.sum()}")
if __name__=="__main__":main()
