from pathlib import Path
import subprocess,sys,time
from .live_pipeline import generate_predictions

def refresh_cycle(root,service,day=None,assembler=None):
 root=Path(root);started=time.perf_counter();payload=generate_predictions(root,service,day);candidates=[g for g in payload['games'] if g.get('status')=='INSUFFICIENT_DATA' and g.get('lineup_status')=='confirmed' and not (root/'data/live'/payload['date']/f"prediction_{g['game_id']}.json").exists()]
 if candidates:
  command=assembler or [sys.executable,str(root/'assemble-v11-2-current-day-features.py'),'--date',payload['date']]
  subprocess.run(command,cwd=root,check=True);payload=generate_predictions(root,service,payload['date'])
 payload['refresh_seconds']=time.perf_counter()-started;return payload
