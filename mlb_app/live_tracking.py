from pathlib import Path
import pandas as pd
def load_live_tracking(root):
 folder=Path(root)/'results'/'live_tracking';daily_path=folder/'v11_2_live_daily_summary.csv';ledger_path=folder/'v11_2_live_predictions.csv'
 if not daily_path.exists() or not ledger_path.exists():return {'available':False,'label':'PROSPECTIVE LIVE 2026 TRACKING','daily':[]}
 daily=pd.read_csv(daily_path);ledger=pd.read_csv(ledger_path);high=ledger[ledger.winner_probability.ge(.60)];team_abs=pd.concat([ledger.away_abs_error,ledger.home_abs_error],ignore_index=True)
 return {'available':True,'label':'PROSPECTIVE LIVE 2026 TRACKING','predictions':len(ledger),'winner_accuracy':ledger.winner_correct.mean(),'high_predictions':len(high),'high_accuracy':high.winner_correct.mean() if len(high) else None,'team_run_mae':team_abs.mean(),'total_mae':ledger.total_abs_error.mean(),'daily':daily.sort_values('date',ascending=False).to_dict('records')}
