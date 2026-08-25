"""Append one completed day of genuine pre-first-pitch V11.2 snapshots."""
from __future__ import annotations
import argparse, hashlib, json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'/'live_tracking';LEDGER=OUT/'v11_2_live_predictions.csv';DAILY=OUT/'v11_2_live_daily_summary.csv';CUMULATIVE=OUT/'v11_2_live_cumulative_summary.csv'

def validate_snapshots(day):
 """Freeze eligibility from snapshot content only. Outcomes are not loaded here."""
 folder=ROOT/'data'/'live'/day;meta=json.loads((ROOT/'artifacts/v11_2_compact_metadata.json').read_text());digest=hashlib.sha256((ROOT/'artifacts/v11_2_compact_pipeline.joblib').read_bytes()).hexdigest()
 if digest!=meta['artifact_sha256']:raise RuntimeError('production artifact hash does not match metadata')
 eligible=[];rejected=[]
 for path in sorted(folder.glob('prediction_*.json')):
  game=json.loads(path.read_text());snapshot=game.get('snapshot',{});reasons=[];generated=pd.to_datetime(snapshot.get('generated_at'),utc=True,errors='coerce');pitch=pd.to_datetime(snapshot.get('scheduled_start'),utc=True,errors='coerce')
  if str(game.get('date'))!=day:reasons.append('wrong_date')
  if pd.isna(generated) or pd.isna(pitch) or not generated<pitch:reasons.append('not_provably_pregame')
  if snapshot.get('model_version')!=meta['model_version']:reasons.append('wrong_model_version')
  if snapshot.get('artifact_sha256')!=digest:reasons.append('wrong_artifact_hash')
  if snapshot.get('immutable_after_first_pitch') is not True:reasons.append('not_marked_immutable')
  if snapshot.get('forecast_type')=='PROVISIONAL_PREDICTION' or game.get('forecast_type')=='PROVISIONAL_PREDICTION':reasons.append('provisional_not_official_live_tracking')
  if not game.get('prediction'):reasons.append('missing_prediction')
  if path.stem!=f"prediction_{game.get('game_id')}":reasons.append('filename_game_id_mismatch')
  (rejected if reasons else eligible).append({'file':path.name,'game_id':game.get('game_id'),'reasons':'|'.join(reasons)}) if reasons else eligible.append(game)
 return eligible,rejected

def final_result(game_id):
 with urllib.request.urlopen(f'https://statsapi.mlb.com/api/v1.1/game/{int(game_id)}/feed/live',timeout=30) as response:data=json.loads(response.read())
 if data['gameData']['status']['abstractGameState']!='Final':raise RuntimeError(f'game {game_id} is not final')
 teams=data['liveData']['linescore']['teams'];return int(teams['away']['runs']),int(teams['home']['runs'])

def grade(game,away_runs,home_runs):
 p=game['prediction'];away=float(p['away']['expected_runs']);home=float(p['home']['expected_runs']);actual_winner=game['home_team'] if home_runs>away_runs else game['away_team'];ae=away-away_runs;he=home-home_runs;total=float(p['projected_total']);actual_total=away_runs+home_runs;pdiff=float(p['projected_run_difference']);adiff=home_runs-away_runs
 return {'date':game['date'],'game_id':int(game['game_id']),'snapshot_timestamp':game['snapshot']['generated_at'],'first_pitch':game['snapshot']['scheduled_start'],'away_team':game['away_team'],'home_team':game['home_team'],'pred_away_runs':away,'pred_home_runs':home,'pred_total':total,'predicted_winner':p['predicted_winner'],'winner_probability':float(p['winner_probability']),'confidence':p['confidence'],'actual_away_runs':away_runs,'actual_home_runs':home_runs,'actual_total':actual_total,'actual_winner':actual_winner,'winner_correct':int(p['predicted_winner']==actual_winner),'away_abs_error':abs(ae),'home_abs_error':abs(he),'away_squared_error':ae**2,'home_squared_error':he**2,'total_abs_error':abs(total-actual_total),'total_squared_error':(total-actual_total)**2,'predicted_run_diff_home_minus_away':pdiff,'actual_run_diff_home_minus_away':adiff,'run_diff_abs_error':abs(pdiff-adiff),'model_version':game['snapshot']['model_version'],'artifact_hash':game['snapshot']['artifact_sha256']}

def metrics(frame):
 team_abs=np.r_[frame.away_abs_error,frame.home_abs_error];team_sq=np.r_[frame.away_squared_error,frame.home_squared_error];high=frame[frame.winner_probability.ge(.60)]
 return {'predictions':len(frame),'winner_correct':int(frame.winner_correct.sum()),'winner_incorrect':int((1-frame.winner_correct).sum()),'winner_accuracy':float(frame.winner_correct.mean()),'team_run_MAE':float(team_abs.mean()),'team_run_RMSE':float(np.sqrt(team_sq.mean())),'total_MAE':float(frame.total_abs_error.mean()),'total_RMSE':float(np.sqrt(frame.total_squared_error.mean())),'mean_projected_total':float(frame.pred_total.mean()),'mean_actual_total':float(frame.actual_total.mean()),'60plus_predictions':len(high),'60plus_correct':int(high.winner_correct.sum()),'60plus_accuracy':float(high.winner_correct.mean()) if len(high) else None}

def append_day(day,graded):
 OUT.mkdir(parents=True,exist_ok=True);new=pd.DataFrame(graded);old=pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=new.columns)
 existing=set(pd.to_numeric(old.get('game_id',pd.Series(dtype=float)),errors='coerce').dropna().astype(int));new=new[~pd.to_numeric(new.game_id,errors='coerce').astype(int).isin(existing)] if len(new) else new
 ledger=pd.concat([old,new],ignore_index=True).sort_values(['date','first_pitch','game_id']) if len(old) or len(new) else old;ledger.to_csv(LEDGER,index=False)
 daily=[]
 for tracked_day in sorted(ledger.date.astype(str).unique()):daily.append({'date':tracked_day,**metrics(ledger[ledger.date.astype(str).eq(tracked_day)])})
 pd.DataFrame(daily).to_csv(DAILY,index=False);cumulative=[]
 for cutoff in sorted(ledger.date.astype(str).unique()):cumulative.append({'through_date':cutoff,'tracking_label':'PROSPECTIVE LIVE 2026 TRACKING',**metrics(ledger[ledger.date.astype(str).le(cutoff)])})
 pd.DataFrame(cumulative).to_csv(CUMULATIVE,index=False);return new,pd.DataFrame(daily),pd.DataFrame(cumulative)

def grade_available(day):
 eligible,rejected=validate_snapshots(day);existing_frame=pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame();existing_ids=set(pd.to_numeric(existing_frame.get('game_id',pd.Series(dtype=float)),errors='coerce').dropna().astype(int));graded=[];not_final=[]
 for game in eligible:
  if int(game['game_id']) in existing_ids:continue
  try:graded.append(grade(game,*final_result(game['game_id'])))
  except RuntimeError as exc:not_final.append({'game_id':int(game['game_id']),'reason':str(exc)})
 if not graded:
  date_records=existing_frame[existing_frame.date.astype(str).eq(day)].to_dict('records') if len(existing_frame) and 'date' in existing_frame else []
  return {'date':day,'eligible_snapshots':len(eligible),'newly_graded':0,'graded':[],'date_records':date_records,'rejected':rejected,'not_final':not_final}
 new,daily,cumulative=append_day(day,graded)
 date_records=pd.read_csv(LEDGER);date_records=date_records[date_records.date.astype(str).eq(day)]
 return {'date':day,'eligible_snapshots':len(eligible),'newly_graded':len(new),'graded':new.to_dict('records'),'date_records':date_records.to_dict('records'),'rejected':rejected,'not_final':not_final,'latest_tracked_date':str(daily.date.max()),'cumulative_predictions':int(cumulative.iloc[-1].predictions)}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--date',required=True);args=parser.parse_args();print(json.dumps(grade_available(args.date),indent=2,default=str))
if __name__=='__main__':main()
