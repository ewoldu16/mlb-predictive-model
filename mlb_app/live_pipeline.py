from pathlib import Path
from datetime import date,datetime,timezone
import json,os,urllib.parse,urllib.request
import pandas as pd
from .model_service import confidence_label
from .feature_builder import build_daily_feature_rows

STARTED={'In Progress','Manager Challenge','Delayed','Final','Game Over','Completed Early'}
def atomic_json(data,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def fetch_schedule(day,cache_dir):
 cache=Path(cache_dir)/f'schedule_{day}.json';cache.parent.mkdir(parents=True,exist_ok=True);url='https://statsapi.mlb.com/api/v1/schedule?'+urllib.parse.urlencode({'sportId':1,'date':day,'hydrate':'probablePitcher,venue'})
 try:
  with urllib.request.urlopen(url,timeout=15) as res:data=json.loads(res.read())
  atomic_json(data,cache);source='live_mlb_stats_api'
 except Exception as exc:
  if cache.exists():data=json.loads(cache.read_text());source='cached_schedule_after_api_failure'
  else:return [],{'status':'api_error','message':str(exc)}
 games=[]
 for block in data.get('dates',[]):
  for g in block.get('games',[]):
   away=g['teams']['away'];home=g['teams']['home'];detail=g.get('status',{}).get('detailedState','Scheduled')
   games.append({'game_id':int(g['gamePk']),'date':day,'start_time':g.get('gameDate'),'status':detail,'away_team':away['team']['name'],'home_team':home['team']['name'],'away_starter':away.get('probablePitcher',{}).get('fullName'),'home_starter':home.get('probablePitcher',{}).get('fullName'),'away_starter_id':away.get('probablePitcher',{}).get('id'),'home_starter_id':home.get('probablePitcher',{}).get('id'),'venue':g.get('venue',{}).get('name'),'schedule_source':source})
 return games,{'status':'ok','source':source}
def fetch_lineup_status(game,cache_dir):
 cache=Path(cache_dir)/f"boxscore_{game['game_id']}.json";cache.parent.mkdir(parents=True,exist_ok=True);url=f"https://statsapi.mlb.com/api/v1/game/{game['game_id']}/boxscore"
 try:
  with urllib.request.urlopen(url,timeout=12) as response:data=json.loads(response.read());atomic_json(data,cache)
 except Exception:
  if not cache.exists():return 'unavailable',{}
  data=json.loads(cache.read_text())
 counts={}
 for side in ('away','home'):
  orders=set()
  for player in data.get('teams',{}).get(side,{}).get('players',{}).values():
   raw=player.get('battingOrder')
   if raw not in (None,'') and int(raw)%100==0:orders.add(int(raw)//100)
  counts[side]=len(orders)
 if counts=={'away':9,'home':9}:return 'confirmed',counts
 if counts.get('away',0) or counts.get('home',0):return 'partial',counts
 return 'unavailable',counts
def status_for(game,lineup_status,source_status):
 detail=game['status']
 if 'Postponed' in detail:return 'POSTPONED','Game postponed.'
 if detail in STARTED:return ('FINAL' if detail in {'Final','Game Over','Completed Early'} else 'IN_PROGRESS'),None
 try:
  if datetime.now(timezone.utc)>=pd.Timestamp(game['start_time']).to_pydatetime():return 'IN_PROGRESS','First-pitch cutoff has passed; no new prediction may be generated.'
 except Exception:pass
 if not game.get('away_starter_id') or not game.get('home_starter_id'):return 'PENDING_STARTER','Waiting for probable starter.'
 if lineup_status!='confirmed':return 'PENDING_LINEUP',('Waiting for confirmed lineup.' if lineup_status=='unavailable' else 'Waiting for complete confirmed lineup.')
 if source_status!='ok':return 'INSUFFICIENT_DATA','Insufficient validated pregame data.'
 return 'SCHEDULED',None
def generate_predictions(root,service,day=None,refresh=True):
 root=Path(root);day=day or date.today().isoformat();folder=root/'data/live'/day;folder.mkdir(parents=True,exist_ok=True);out=folder/'predictions.json';legacy=root/'data/live'/f'predictions_{day}.json'
 games,meta=fetch_schedule(day,folder/'schedule_cache') if refresh else ([],{'status':'not_refreshed'});feature_path=folder/'features.csv';features=pd.read_csv(feature_path) if feature_path.exists() else None
 if features is None and games:
  features,build_meta=build_daily_feature_rows(root,day,[g['game_id'] for g in games],service.features)
  if features is not None:features.to_csv(feature_path,index=False)
 else:build_meta={'status':'ok' if features is not None else 'not_built','source':'daily_cache' if features is not None else None}
 results=[]
 for g in games:
  snapshot=folder/f"prediction_{g['game_id']}.json";lineup_status,lineup_counts=fetch_lineup_status(g,folder/'boxscore_cache') if refresh else ('unavailable',{});state,message=status_for(g,lineup_status,build_meta['status'])
  if snapshot.exists():
   saved=json.loads(snapshot.read_text());saved['status']=state if state in {'IN_PROGRESS','FINAL','POSTPONED'} else 'PREDICTION_READY';results.append(saved);continue
  item={**g,'status':state,'forecast_status':state.lower(),'forecast_message':message,'lineup_status':lineup_status,'lineup_counts':lineup_counts,'prediction':None}
  if state=='SCHEDULED' and features is not None:
   rows=features[features.game_id.eq(g['game_id'])]
   try:
    cutoff=g['start_time'];away_row=rows[rows.team_side.eq('away')].iloc[0];home_row=rows[rows.team_side.eq('home')].iloc[0];away_audit=service.validated_vector(away_row,cutoff);home_audit=service.validated_vector(home_row,cutoff);away=service.predict_team(away_row);home=service.predict_team(home_row);hp=service.home_probability(away['expected_runs'],home['expected_runs']);fav=max(hp,1-hp)
    item['status']='PREDICTION_READY';item['forecast_status']='ready';item['forecast_message']=None;item['prediction']={'away':away,'home':home,'projected_total':away['expected_runs']+home['expected_runs'],'projected_run_difference':home['expected_runs']-away['expected_runs'],'home_win_probability':hp,'away_win_probability':1-hp,'predicted_winner':g['home_team'] if hp>=.5 else g['away_team'],'winner_probability':fav,'confidence':confidence_label(fav)};item['feature_vectors']={'away':away_audit,'home':home_audit};item['snapshot']={'generated_at':datetime.now(timezone.utc).isoformat(),'model_version':service.meta['model_version'],'artifact_sha256':service.meta['artifact_sha256'],'scheduled_start':g['start_time'],'immutable_after_first_pitch':True};atomic_json(item,snapshot)
   except Exception as exc:item['status']='INSUFFICIENT_DATA';item['forecast_status']='insufficient_data';item['forecast_message']='Insufficient pregame data: '+str(exc)
  results.append(item)
 payload={'schema_version':'site-predictions-2.0','date':day,'generated_from':meta,'feature_build':build_meta,'games':results};atomic_json(payload,out);atomic_json(payload,legacy);atomic_json({'date':day,'schedule':meta,'feature_build':build_meta,'model_version':service.meta['model_version'],'artifact_sha256':service.meta['artifact_sha256']},folder/'metadata.json');return payload
def load_today(root,day=None):
 day=day or date.today().isoformat();root=Path(root)
 for path in (root/'data/live'/day/'predictions.json',root/'data/live'/f'predictions_{day}.json'):
  if path.exists():
   payload=json.loads(path.read_text());build=payload.get('feature_build',{})
   if build.get('status')=='target_games_missing_from_feature_universe':
    for game in payload.get('games',[]):
     if game.get('status')=='INSUFFICIENT_DATA':game['forecast_message']='Exact target-game feature row is missing from the validated 2026 feature build.'
   return payload
 return {'schema_version':'site-predictions-2.0','date':day,'generated_from':{'status':'not_generated'},'games':[]}
