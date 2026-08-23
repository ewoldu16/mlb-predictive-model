from pathlib import Path
import pandas as pd
from .storage import database_url,load_state,normalize_json_value

def _mean(frame,column):
 if column not in frame:return None
 value=pd.to_numeric(frame[column],errors='coerce').mean()
 return None if pd.isna(value) else float(value)

def load_live_tracking(root):
 if database_url():
  stored=load_state('live_tracking_summary')
  return stored or {'available':False,'label':'PROSPECTIVE LIVE 2026 TRACKING','daily':[]}
 folder=Path(root)/'results'/'live_tracking';daily_path=folder/'v11_2_live_daily_summary.csv';ledger_path=folder/'v11_2_live_predictions.csv'
 if not daily_path.exists() or not ledger_path.exists():return {'available':False,'label':'PROSPECTIVE LIVE 2026 TRACKING','daily':[]}
 daily=pd.read_csv(daily_path);ledger=pd.read_csv(ledger_path)
 high=ledger[pd.to_numeric(ledger.get('winner_probability',pd.Series(index=ledger.index,dtype=float)),errors='coerce').ge(.60)]
 error_columns=[pd.to_numeric(ledger[column],errors='coerce') for column in ('away_abs_error','home_abs_error') if column in ledger]
 team_abs=pd.concat(error_columns,ignore_index=True) if error_columns else pd.Series(dtype=float)
 summary={'available':bool(len(ledger)),'label':'PROSPECTIVE LIVE 2026 TRACKING','predictions':len(ledger),'winner_accuracy':_mean(ledger,'winner_correct'),'high_predictions':len(high),'high_accuracy':_mean(high,'winner_correct'),'team_run_mae':None if team_abs.empty or pd.isna(team_abs.mean()) else float(team_abs.mean()),'total_mae':_mean(ledger,'total_abs_error'),'daily':daily.sort_values('date',ascending=False).to_dict('records') if 'date' in daily else daily.to_dict('records')}
 return normalize_json_value(summary)
