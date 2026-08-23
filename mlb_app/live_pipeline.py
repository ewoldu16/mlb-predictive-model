from pathlib import Path
from datetime import date,datetime,timezone
import json,os,urllib.parse,urllib.request
import pandas as pd
from .model_service import confidence_label
from .feature_builder import build_daily_feature_rows
from .storage import append_provisional_history,load_provisional_snapshot,load_snapshot,load_state,save_provisional_snapshot,save_snapshot,save_state
from .probable_lineups import fetch_probable_lineups,lineup_fingerprint
from .owner_controls import offense_confidence,owner_lineup_for_game,load_team_state

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
   games.append({'game_id':int(g['gamePk']),'date':day,'start_time':g.get('gameDate'),'status':detail,'away_team':away['team']['name'],'home_team':home['team']['name'],'away_team_id':away['team'].get('id'),'home_team_id':home['team'].get('id'),'away_starter':away.get('probablePitcher',{}).get('fullName'),'home_starter':home.get('probablePitcher',{}).get('fullName'),'away_starter_id':away.get('probablePitcher',{}).get('id'),'home_starter_id':home.get('probablePitcher',{}).get('id'),'venue':g.get('venue',{}).get('name'),'schedule_source':source})
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
def confirmed_lineup_details(game,cache_dir):
 path=Path(cache_dir)/f"boxscore_{game['game_id']}.json"
 if not path.exists():return None
 data=json.loads(path.read_text());teams={}
 for side in ('away','home'):
  players=[]
  for player in data.get('teams',{}).get(side,{}).get('players',{}).values():
   raw=player.get('battingOrder')
   if raw not in (None,'') and int(raw)%100==0:players.append({'order':int(raw)//100,'player_id':int(player['person']['id']),'source_player_id':int(player['person']['id']),'name':player['person'].get('fullName'),'position':player.get('position',{}).get('abbreviation')})
  teams[side]=sorted(players,key=lambda x:x['order'])
 if any(len(teams[s])!=9 for s in teams):return None
 return {'status':'confirmed','source':'mlb_stats_api_boxscore','retrieval_timestamp':datetime.now(timezone.utc).isoformat(),'source_status':{'away':'Confirmed','home':'Confirmed'},'teams':teams}
def status_for(game,lineup_status,source_status):
 detail=game['status']
 if 'Postponed' in detail:return 'POSTPONED','Game postponed.'
 if detail in STARTED:return ('FINAL' if detail in {'Final','Game Over','Completed Early'} else 'IN_PROGRESS'),None
 try:
  if datetime.now(timezone.utc)>=pd.Timestamp(game['start_time']).to_pydatetime():return 'IN_PROGRESS','First-pitch cutoff has passed; no new prediction may be generated.'
 except Exception:pass
 if not game.get('away_starter_id') or not game.get('home_starter_id'):return 'PENDING_STARTER','Waiting for probable starter.'
 if lineup_status not in {'confirmed','probable','owner_managed'}:return 'PENDING_LINEUP','Waiting for a legitimate owner-managed, projected, or confirmed lineup.'
 if source_status!='ok':return 'INSUFFICIENT_DATA','Insufficient validated pregame data.'
 return ('PROVISIONAL_PREDICTION','Using provisional lineup; forecast may update before first pitch.') if lineup_status in {'probable','owner_managed'} else ('SCHEDULED',None)
def generate_predictions(root,service,day=None,refresh=True):
 root=Path(root);day=day or date.today().isoformat();state_root=Path(os.getenv('MLB_STATE_DIR',root/'data/live'));folder=state_root/day;folder.mkdir(parents=True,exist_ok=True);out=folder/'predictions.json';legacy=state_root/f'predictions_{day}.json'
 games,meta=fetch_schedule(day,folder/'schedule_cache') if refresh else ([],{'status':'not_refreshed'})
 probable,probable_meta=fetch_probable_lineups(games,day,folder/'probable_lineup_cache') if refresh else ({},{'status':'not_refreshed'})
 feature_path=folder/'features.csv';features=pd.read_csv(feature_path) if feature_path.exists() else None
 if features is None and games:
  features,build_meta=build_daily_feature_rows(root,day,[g['game_id'] for g in games],service.features)
  if features is not None:features.to_csv(feature_path,index=False)
 else:build_meta={'status':'ok' if features is not None else 'not_built','source':'daily_cache' if features is not None else None}
 results=[]
 for g in games:
  gid=g['game_id'];final_path=folder/f'prediction_{gid}.json';provisional_path=folder/f'provisional_prediction_{gid}.json';confirmed_status,lineup_counts=fetch_lineup_status(g,folder/'boxscore_cache') if refresh else ('unavailable',{})
  owner_lineup=owner_lineup_for_game(root,g) if confirmed_status!='confirmed' else None
  lineup=confirmed_lineup_details(g,folder/'boxscore_cache') if confirmed_status=='confirmed' else (owner_lineup or probable.get(gid));lineup_status='confirmed' if confirmed_status=='confirmed' else ('owner_managed' if owner_lineup else ('probable' if lineup else 'unavailable'));lineup_counts={side:len(lineup['teams'][side]) for side in ('away','home')} if lineup else lineup_counts
  state,message=status_for(g,lineup_status,build_meta['status']);stored_final=load_snapshot(gid)
  if final_path.exists() or stored_final:
   saved=json.loads(final_path.read_text()) if final_path.exists() else stored_final;saved['status']=state if state in {'IN_PROGRESS','FINAL','POSTPONED'} else 'FINAL_PREGAME_PREDICTION';saved['forecast_status']='final_pregame_prediction';results.append(saved);continue
  stored_provisional=json.loads(provisional_path.read_text()) if provisional_path.exists() else load_provisional_snapshot(gid)
  if stored_provisional and state in {'IN_PROGRESS','FINAL','POSTPONED'}:
   stored_provisional['status']=state;stored_provisional['forecast_message']='Provisional forecast retained for research; no official final pregame forecast was created.';results.append(stored_provisional);continue
  if not lineup and stored_provisional:
   stored_provisional['status']=state;stored_provisional['forecast_message']=message;results.append(stored_provisional);continue
  fingerprint=lineup_fingerprint(lineup)
  if stored_provisional and lineup and stored_provisional.get('lineup_fingerprint')==fingerprint and stored_provisional.get('lineup_status')==lineup_status:
   results.append(stored_provisional);continue
  marker_path=folder/f'lineup_build_{gid}.json';marker=json.loads(marker_path.read_text()) if marker_path.exists() else (load_state('lineup_build:'+str(gid)) or {});needs_rebuild=bool(lineup and (marker.get('lineup_fingerprint')!=fingerprint or marker.get('lineup_status')!=lineup_status))
  item={**g,'status':state,'forecast_status':state.lower(),'forecast_message':message,'forecast_type':None,'lineup_status':lineup_status,'lineup_counts':lineup_counts,'lineup_details':lineup,'lineup_source':lineup.get('source') if lineup else None,'lineup_retrieved_at':lineup.get('retrieval_timestamp') if lineup else None,'lineup_fingerprint':fingerprint,'owner_lineup_version':lineup.get('owner_lineup_version') if lineup else None,'owner_modified':lineup.get('owner_modified',False) if lineup else False,'prediction':None}
  if needs_rebuild:
   item['status']='INSUFFICIENT_DATA';item['forecast_status']='insufficient_data';item['forecast_message']='Lineup changed; exact frozen features are being rebuilt.';item['lineup_needs_rebuild']=True;results.append(item);continue
  if state in {'SCHEDULED','PROVISIONAL_PREDICTION'} and features is not None:
   rows=features[features.game_id.eq(gid)]
   try:
    cutoff=g['start_time'];away_row=rows[rows.team_side.eq('away')].iloc[0];home_row=rows[rows.team_side.eq('home')].iloc[0];away_audit=service.validated_vector(away_row,cutoff);home_audit=service.validated_vector(home_row,cutoff);away=service.predict_team(away_row);home=service.predict_team(home_row);hp=service.home_probability(away['expected_runs'],home['expected_runs']);fav=max(hp,1-hp)
    kind='PROVISIONAL_PREDICTION' if lineup_status in {'probable','owner_managed'} else 'FINAL_PREGAME_PREDICTION'
    item['status']=kind;item['forecast_status']=kind.lower();item['forecast_type']=kind;item['forecast_message']='Using owner-managed provisional lineup; lineup uncertainty is provisional.' if lineup_status=='owner_managed' else ('Using probable lineup; lineup uncertainty is provisional.' if kind.startswith('PROVISIONAL') else None);item['prediction']={'away':away,'home':home,'projected_total':away['expected_runs']+home['expected_runs'],'projected_run_difference':home['expected_runs']-away['expected_runs'],'home_win_probability':hp,'away_win_probability':1-hp,'predicted_winner':g['home_team'] if hp>=.5 else g['away_team'],'winner_probability':fav,'confidence':confidence_label(fav)};item['feature_vectors']={'away':away_audit,'home':home_audit};item['snapshot']={'generated_at':datetime.now(timezone.utc).isoformat(),'model_version':service.meta['model_version'],'artifact_sha256':service.meta['artifact_sha256'],'scheduled_start':g['start_time'],'forecast_type':kind,'immutable_after_first_pitch':kind=='FINAL_PREGAME_PREDICTION'}
    if lineup_status=='owner_managed':
     item['team_offense_confidence']={side:offense_confidence(load_team_state(root,gid,side),item['feature_vectors'][side]) for side in ('away','home')}
    if kind=='FINAL_PREGAME_PREDICTION':
     if stored_provisional:item['provisional_comparison']=compare_forecasts(stored_provisional,item)
     atomic_json(item,final_path);save_snapshot(item)
    else:atomic_json(item,provisional_path);save_provisional_snapshot(item);append_provisional_history(item)
   except Exception as exc:item['status']='INSUFFICIENT_DATA';item['forecast_status']='insufficient_data';item['forecast_message']='Insufficient pregame data: '+str(exc)
  results.append(item)
 payload={'schema_version':'site-predictions-3.0','date':day,'generated_from':meta,'probable_lineups':probable_meta,'feature_build':build_meta,'games':results};atomic_json(payload,out);atomic_json(payload,legacy);atomic_json({'date':day,'schedule':meta,'probable_lineups':probable_meta,'feature_build':build_meta,'model_version':service.meta['model_version'],'artifact_sha256':service.meta['artifact_sha256']},folder/'metadata.json');save_state('today:'+day,payload);return payload

def compare_forecasts(provisional,final):
 changes=[]
 for side in ('away','home'):
  old={p['order']:p for p in provisional.get('lineup_details',{}).get('teams',{}).get(side,[])};new={p['order']:p for p in final.get('lineup_details',{}).get('teams',{}).get(side,[])}
  for order in sorted(set(old)|set(new)):
   if old.get(order,{}).get('player_id')!=new.get(order,{}).get('player_id'):changes.append({'team_side':side,'batting_order':order,'provisional_player':old.get(order,{}).get('name'),'confirmed_player':new.get(order,{}).get('name')})
 return {'provisional_generated_at':provisional.get('snapshot',{}).get('generated_at'),'final_generated_at':final.get('snapshot',{}).get('generated_at'),'away_expected_runs_change':final['prediction']['away']['expected_runs']-provisional['prediction']['away']['expected_runs'],'home_expected_runs_change':final['prediction']['home']['expected_runs']-provisional['prediction']['home']['expected_runs'],'projected_total_change':final['prediction']['projected_total']-provisional['prediction']['projected_total'],'home_win_probability_change':final['prediction']['home_win_probability']-provisional['prediction']['home_win_probability'],'winner_probability_change':final['prediction']['winner_probability']-provisional['prediction']['winner_probability'],'predicted_winner_changed':final['prediction']['predicted_winner']!=provisional['prediction']['predicted_winner'],'lineup_substitutions':changes}
def load_today(root,day=None):
 day=day or date.today().isoformat();root=Path(root);stored=load_state('today:'+day)
 if stored:return stored
 state_root=Path(os.getenv('MLB_STATE_DIR',root/'data/live'))
 for path in (state_root/day/'predictions.json',state_root/f'predictions_{day}.json'):
  if path.exists():
   payload=json.loads(path.read_text());build=payload.get('feature_build',{})
   if build.get('status')=='target_games_missing_from_feature_universe':
    for game in payload.get('games',[]):
     if game.get('status')=='INSUFFICIENT_DATA':game['forecast_message']='Exact target-game feature row is missing from the validated 2026 feature build.'
   return payload
 return {'schema_version':'site-predictions-2.0','date':day,'generated_from':{'status':'not_generated'},'games':[]}
