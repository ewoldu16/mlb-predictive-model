"""One-shot, cache-aware live refresh for GitHub Actions or manual execution."""
from __future__ import annotations
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import argparse,importlib.util,json,os,subprocess,sys,time

from mlb_app.live_pipeline import atomic_json,fetch_schedule
from mlb_app.model_service import V112ModelService
from mlb_app.owner_controls import load_team_state
from mlb_app.refresh_service import refresh_cycle
from mlb_app.storage import database_url,finish_rebuild_request,load_snapshots_for_date,load_state,pending_rebuild_requests,save_state,snapshot_dates_before

ROOT=Path(__file__).resolve().parent

def _state_path():return Path(os.getenv('MLB_STATE_DIR',ROOT/'data/live'))/'actions_refresh_state.json'
def _read_state():
 path=_state_path();return json.loads(path.read_text()) if path.exists() else {}
def _full_refresh_required(day,games,force=False):
 year=int(day[:4]);required=[ROOT/'data/raw'/f'games_{year}.csv',ROOT/'data/raw'/f'statcast_enriched_{year}.csv',ROOT/'data/processed'/f'features_arsenal_lineup_matchup_{year}.csv']
 completed=sorted(int(g['game_id']) for g in games if g.get('status') in {'Final','Game Over','Completed Early'});state=_read_state()
 return force or any(not p.exists() for p in required) or state.get('season_refresh_date')!=day,completed
def _run_full(day):subprocess.run([sys.executable,str(ROOT/'refresh-v11-2-2026-features.py'),'--date',day],cwd=ROOT,check=True)
def _resolve_requests(root,payload):
 by_id={int(g['game_id']):g for g in payload.get('games',[])};resolved=[]
 for request in pending_rebuild_requests():
  gid=int(request['game_id']);game=by_id.get(gid);states=[load_team_state(root,gid,side) for side in ('away','home')]
  if game and game.get('forecast_type')=='PROVISIONAL_PREDICTION' and game.get('lineup_status')=='owner_managed':status,reason='complete',None
  elif not all(states) or any(not state.get('valid') for state in states):status,reason='incomplete','PROVISIONAL_LINEUP_INCOMPLETE'
  elif game and game.get('status') in {'IN_PROGRESS','FINAL','POSTPONED'}:status,reason='locked','first-pitch lock prevents a new provisional forecast'
  else:continue
  finish_rebuild_request(gid,status,reason);resolved.append({'game_id':gid,'status':status,'reason':reason})
 return resolved
def _publish_tracking():
 from mlb_app.live_tracking import load_live_tracking
 # Temporarily force the local-file branch while publishing the compact summary.
 old={key:os.environ.pop(key,None) for key in ('DATABASE_URL','SUPABASE_DATABASE_URL')}
 try:summary=load_live_tracking(ROOT)
 finally:
  for key,value in old.items():
   if value is not None:os.environ[key]=value
 if summary.get('available'):save_state('live_tracking_summary',summary)
def _grade_completed_days(day):
 candidates=set(snapshot_dates_before(day));state_root=Path(os.getenv('MLB_STATE_DIR',ROOT/'data/live'))
 candidates.update(path.name for path in state_root.iterdir() if path.is_dir() and path.name<day and path.name[:4].isdigit()) if state_root.exists() else None
 spec=importlib.util.spec_from_file_location('live_day_evaluator',ROOT/'evaluate-live-v11-2-day.py');evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator);reports=[]
 for target in sorted(candidates):
  snapshots=load_snapshots_for_date(target)
  if not snapshots:continue
  folder=state_root/target;folder.mkdir(parents=True,exist_ok=True)
  for game in snapshots:atomic_json(game,folder/f"prediction_{int(game['game_id'])}.json")
  report=evaluator.grade_available(target);reports.append(report)
  if report.get('date_records'):save_state('live_results:'+target,report['date_records'])
 return reports
def run(day,force_season=False,skip_season=False):
 started=time.perf_counter();previous=load_state('refresh_status') or {};now=datetime.now(timezone.utc).isoformat();save_state('refresh_status',{'status':'running','started_at':now,'last_successful_refresh':previous.get('last_successful_refresh'),'executor':'github_actions','current_data_date':day})
 try:
  try:grading=_grade_completed_days(day);_publish_tracking()
  except Exception as grading_error:grading=[{'status':'deferred','error':type(grading_error).__name__}]
  state_root=Path(os.getenv('MLB_STATE_DIR',ROOT/'data/live'));games,_=fetch_schedule(day,state_root/day/'schedule_cache');refresh_needed,completed=_full_refresh_required(day,games,force_season)
  if refresh_needed and not skip_season:_run_full(day)
  service=V112ModelService(ROOT);payload=refresh_cycle(ROOT,service,day);resolved=_resolve_requests(ROOT,payload);finished=datetime.now(timezone.utc).isoformat();state={'season_refresh_date':day if refresh_needed and not skip_season else _read_state().get('season_refresh_date'),'completed_today':completed,'last_run':finished};atomic_json(state,_state_path());health={'status':'ok','executor':'github_actions','started_at':now,'last_successful_refresh':finished,'current_data_date':day,'games':len(payload.get('games',[])),'queued_requests_resolved':resolved,'grading':grading,'full_season_refresh':bool(refresh_needed and not skip_season),'refresh_seconds':time.perf_counter()-started};save_state('refresh_status',health);print(json.dumps(health,indent=2));return payload
 except Exception as exc:
  save_state('refresh_status',{'status':'error','executor':'github_actions','started_at':now,'last_successful_refresh':previous.get('last_successful_refresh'),'current_data_date':day,'message':str(exc)[:1000]});raise
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--date',default=date.today().isoformat());parser.add_argument('--force-season-refresh',action='store_true');parser.add_argument('--skip-season-refresh',action='store_true');args=parser.parse_args();run(args.date,args.force_season_refresh,args.skip_season_refresh)
if __name__=='__main__':main()
