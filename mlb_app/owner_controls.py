"""Owner-managed provisional lineups around (never inside) frozen V11.2."""
from __future__ import annotations

from datetime import datetime,timedelta,timezone
from pathlib import Path
import json,os,secrets,urllib.parse,urllib.request

import pandas as pd

from .storage import append_owner_audit,load_owner_lineup,save_owner_lineup,owner_audit_rows

AVAILABILITY_STATUSES=('AVAILABLE','DOUBTFUL','UNAVAILABLE','INJURED','REST_EXPECTED','SUSPENDED','MINORS_OR_INACTIVE','UNKNOWN')
LINEUP_DEPENDENT_PREFIXES=('lineup_','opp_sp_matchup_','opp_arsenal_')

def _root(root):return Path(os.getenv('MLB_STATE_DIR',Path(root)/'data/live'))/'owner'
def _path(root,game_id,side):return _root(root)/f'game_{int(game_id)}_{side}.json'
def _read(path):return json.loads(Path(path).read_text()) if Path(path).exists() else None
def atomic_json(data,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def load_team_state(root,game_id,side):return load_owner_lineup(game_id,side) or _read(_path(root,game_id,side))
def save_team_state(root,state):
 atomic_json(state,_path(root,state['game_id'],state['team_side']));save_owner_lineup(state['game_id'],state['team_side'],state);return state
def _cache_json(url,path):
 path=Path(path)
 try:
  with urllib.request.urlopen(url,timeout=20) as response:data=json.loads(response.read())
  atomic_json(data,path);return data
 except Exception:
  if path.exists():return json.loads(path.read_text())
  raise
def _person(row):
 person=row.get('person',{});position=row.get('position',{});bat=person.get('batSide',{})
 return {'player_id':int(person['id']),'name':person.get('fullName'),'position':position.get('abbreviation') or position.get('code'),'handedness':bat.get('code')}

def active_roster(root,team_id,as_of=None):
 day=as_of or datetime.now(timezone.utc).date().isoformat();cache=_root(root)/'api_cache'/day/f'roster_{int(team_id)}.json'
 data=_cache_json(f'https://statsapi.mlb.com/api/v1/teams/{int(team_id)}/roster?rosterType=active&hydrate=person',cache)
 players=(_person(row) for row in data.get('roster',[]) if row.get('person',{}).get('id'))
 return sorted((p for p in players if p.get('position') not in {'P','SP','RP'}),key=lambda x:(x['position'] or '',x['name'] or ''))

def previous_completed_lineup(root,team_id,before_day):
 end=pd.Timestamp(before_day)-pd.Timedelta(days=1);start=end-pd.Timedelta(days=21);cache_dir=_root(root)/'api_cache'/str(before_day)
 query=urllib.parse.urlencode({'sportId':1,'teamId':int(team_id),'startDate':start.date().isoformat(),'endDate':end.date().isoformat(),'gameType':'R'})
 schedule=_cache_json('https://statsapi.mlb.com/api/v1/schedule?'+query,cache_dir/f'previous_schedule_{int(team_id)}.json');games=[]
 for block in schedule.get('dates',[]):
  for game in block.get('games',[]):
   if game.get('status',{}).get('abstractGameState')=='Final':games.append((block.get('date'),int(game['gamePk'])))
 for source_date,game_id in sorted(games,reverse=True):
  data=_cache_json(f'https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore',cache_dir/f'previous_boxscore_{game_id}.json')
  side='home' if int(data.get('teams',{}).get('home',{}).get('team',{}).get('id',-1))==int(team_id) else 'away';players=[]
  for row in data.get('teams',{}).get(side,{}).get('players',{}).values():
   raw=row.get('battingOrder')
   if raw not in (None,'') and int(raw)%100==0:
    p=_person(row);p['order']=int(raw)//100;players.append(p)
  if len(players)==9 and len({x['order'] for x in players})==9:
   return {'source_game_id':game_id,'source_date':source_date,'players':sorted(players,key=lambda x:x['order'])}
 return {'source_game_id':None,'source_date':None,'players':[]}

def _recent_usage(root,player_ids,before_day):
 year=pd.Timestamp(before_day).year;lineups=Path(root)/'data/raw/lineups'/f'starting_lineups_{year}.csv';games=Path(root)/'data/raw'/f'games_{year}.csv'
 if not lineups.exists() or not games.exists():return {}
 use=pd.read_csv(lineups,usecols=lambda c:c in {'game_id','player_id','position'});dates=pd.read_csv(games,usecols=['game_id','date']);use=use.merge(dates,on='game_id',how='left');use['date']=pd.to_datetime(use.date,errors='coerce');cut=pd.Timestamp(before_day);use=use[(use.date<cut)&(use.date>=cut-pd.Timedelta(days=30))&use.player_id.isin(player_ids)]
 return {int(pid):{'recent_start_count':int(len(group)),'recent_position_counts':{str(k):int(v) for k,v in group.position.fillna('UNKNOWN').value_counts().items()}} for pid,group in use.groupby('player_id')}

def bootstrap_team(root,game,side):
 existing=load_team_state(root,game['game_id'],side)
 if existing:return existing
 team_id=int(game[f'{side}_team_id']);template=previous_completed_lineup(root,team_id,game['date']);roster=[{**p,'active_roster':True} for p in active_roster(root,team_id,game['date'])];usage=_recent_usage(root,[p['player_id'] for p in roster],game['date']);roster=[{**p,**usage.get(p['player_id'],{'recent_start_count':0,'recent_position_counts':{}})} for p in roster];roster_ids={p['player_id'] for p in roster};active_by_id={p['player_id']:p for p in roster}
 template['players']=[{**p,'handedness':p.get('handedness') or active_by_id.get(p['player_id'],{}).get('handedness')} for p in template['players']]
 roster.extend({**p,'active_roster':False} for p in template['players'] if p['player_id'] not in roster_ids)
 availability={str(p['player_id']):('AVAILABLE' if p['player_id'] in roster_ids else 'MINORS_OR_INACTIVE') for p in template['players']}
 for player in roster:availability.setdefault(str(player['player_id']),'AVAILABLE')
 now=datetime.now(timezone.utc).isoformat();state={'game_id':int(game['game_id']),'game_date':game['date'],'team_side':side,'team_id':team_id,'team_name':game[f'{side}_team'],'template':template,'roster':roster,'availability':availability,'lineup':[p for p in template['players'] if availability.get(str(p['player_id']))=='AVAILABLE'],'version':1,'updated_at':now,'owner_modified':False,'last_impact':None}
 state.update(validate_state(state));return save_team_state(root,state)

def locked(game,now=None):
 now=now or datetime.now(timezone.utc)
 if game.get('status') not in {'Scheduled','Pre-Game','Warmup'}:return True
 try:return now>=pd.Timestamp(game['start_time']).to_pydatetime()
 except Exception:return True

def validate_state(state):
 lineup=state.get('lineup',[]);availability=state.get('availability',{});roster={int(p['player_id']) for p in state.get('roster',[]) if p.get('active_roster',True)}
 orders=[p.get('order') for p in lineup];ids=[int(p['player_id']) for p in lineup]
 errors=[]
 if len(lineup)!=9:errors.append(f'exactly nine hitters required; {max(0,9-len(lineup))} still required')
 if len(ids)!=len(set(ids)):errors.append('duplicate player')
 if set(orders)!=set(range(1,10)):errors.append('batting positions must be exactly 1-9')
 if any(pid not in roster for pid in ids):errors.append('player is not on active roster')
 if any(availability.get(str(pid))!='AVAILABLE' for pid in ids):errors.append('every selected player must be AVAILABLE')
 empty=sorted(set(range(1,10))-set(x for x in orders if isinstance(x,int)))
 return {'valid':not errors,'validation_errors':errors,'empty_positions':empty,'hitters_required':max(0,9-len(lineup)),'status':'PROVISIONAL_READY' if not errors else 'PROVISIONAL_LINEUP_INCOMPLETE'}

def replacement_suggestions(state,order,limit=5):
 lineup_ids={int(x['player_id']) for x in state.get('lineup',[])};target=next((x for x in state.get('template',{}).get('players',[]) if x.get('order')==order),{});position=target.get('position')
 prior={int(x['player_id']) for x in state.get('template',{}).get('players',[])}
 candidates=[p for p in state.get('roster',[]) if p.get('active_roster',True) and state.get('availability',{}).get(str(p['player_id']))=='AVAILABLE' and int(p['player_id']) not in lineup_ids]
 return sorted(candidates,key=lambda p:(p.get('position')!=position,-int(p.get('recent_position_counts',{}).get(position,0)),-int(p.get('recent_start_count',0)),int(p['player_id']) not in prior,p.get('name') or ''))[:limit]

def _audit(root,state,owner_id,action,changes,resulting_snapshot_version=None):
 entry={'timestamp':datetime.now(timezone.utc).isoformat(),'owner_id':owner_id,'game_id':state['game_id'],'team_side':state['team_side'],'team':state['team_name'],'action':action,'changes':changes,'lineup_version':state['version'],'resulting_prediction_snapshot_version':resulting_snapshot_version}
 path=_root(root)/'audit_log.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a',encoding='utf-8') as handle:handle.write(json.dumps(entry,allow_nan=False)+'\n')
 append_owner_audit(entry);return entry

def record_rebuild(root,state,owner_id,snapshot):
 return _audit(root,state,owner_id,'provisional_forecast_rebuilt',[],snapshot.get('snapshot',{}).get('generated_at') if snapshot else None)

def set_availability(root,game,side,player_id,status,owner_id):
 if status not in AVAILABILITY_STATUSES:raise ValueError('invalid availability status')
 if locked(game):raise PermissionError('first-pitch lock is active')
 state=bootstrap_team(root,game,side);pid=int(player_id)
 if str(pid) not in state['availability']:raise ValueError('player is not on relevant roster')
 old=state['availability'][str(pid)];before=[dict(x) for x in state['lineup']];state['availability'][str(pid)]=status
 if status!='AVAILABLE':state['lineup']=[x for x in state['lineup'] if int(x['player_id'])!=pid]
 state['version']+=1;state['updated_at']=datetime.now(timezone.utc).isoformat();state['owner_modified']=True;state.update(validate_state(state));save_team_state(root,state)
 _audit(root,state,owner_id,'availability_changed',[{'player_id':pid,'field':'availability','old_value':old,'new_value':status},{'field':'lineup','old_value':before,'new_value':state['lineup']}]);return state

def save_lineup(root,game,side,lineup,owner_id):
 if locked(game):raise PermissionError('first-pitch lock is active')
 state=bootstrap_team(root,game,side);before=[dict(x) for x in state['lineup']];roster={int(x['player_id']):x for x in state['roster'] if x.get('active_roster',True)};normalized=[]
 for item in lineup:
  pid=int(item['player_id']);base=roster.get(pid)
  if not base:raise ValueError(f'player {pid} is not on active roster')
  if state['availability'].get(str(pid))!='AVAILABLE':raise ValueError(f'player {pid} is not AVAILABLE')
  normalized.append({**base,'order':int(item['order']),'position':item.get('position') or base.get('position')})
 candidate={**state,'lineup':normalized};candidate.update(validate_state(candidate))
 if not candidate['valid']:raise ValueError('; '.join(candidate['validation_errors']))
 candidate['version']+=1;candidate['updated_at']=datetime.now(timezone.utc).isoformat();candidate['owner_modified']=True;save_team_state(root,candidate)
 _audit(root,candidate,owner_id,'lineup_saved',[{'field':'lineup','old_value':before,'new_value':normalized}]);return candidate

def owner_lineup_for_game(root,game):
 states={side:load_team_state(root,game['game_id'],side) for side in ('away','home')}
 if not all(states.values()) or not all(s.get('valid') for s in states.values()):return None
 latest=max(s['updated_at'] for s in states.values());return {'status':'owner_managed','source':'previous_game_template_plus_owner_adjustments','retrieval_timestamp':latest,'source_status':{'away':'Owner managed','home':'Owner managed'},'owner_lineup_version':max(s['version'] for s in states.values()),'owner_modified':any(s['owner_modified'] for s in states.values()),'teams':{side:[{'order':p['order'],'player_id':p['player_id'],'source_player_id':p['player_id'],'name':p['name'],'position':p.get('position'),'handedness':p.get('handedness')} for p in states[side]['lineup']] for side in states}}

def offense_confidence(state,feature_vector=None):
 template={int(p['player_id']) for p in state.get('template',{}).get('players',[])};selected={int(p['player_id']) for p in state.get('lineup',[])}
 retained=len(template&selected)/9 if template else 0;position_known=sum(bool(p.get('position')) for p in state.get('lineup',[]))/9
 dependent=[r for r in (feature_vector or []) if r.get('feature','').startswith(LINEUP_DEPENDENT_PREFIXES)];coverage=(sum(not r.get('imputed') for r in dependent)/len(dependent)) if dependent else 0
 score=.45*retained+.20*position_known+.35*coverage;label='HIGH' if score>=.85 else ('MODERATE' if score>=.65 else 'LOW')
 return {'label':label,'score':score,'regulars_retained':len(template&selected),'replacements':len(selected-template),'position_coverage':position_known,'lineup_feature_observed_coverage':coverage,'formula':'0.45*previous-game starters retained + 0.20*known defensive positions + 0.35*non-imputed frozen lineup-feature coverage'}

def audit_history(root,game_id=None):
 db=owner_audit_rows(game_id)
 if db:return db
 path=_root(root)/'audit_log.jsonl'
 if not path.exists():return []
 rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
 return [x for x in rows if game_id is None or int(x['game_id'])==int(game_id)]

def csrf_token(session):
 if '_csrf_token' not in session:session['_csrf_token']=secrets.token_urlsafe(32)
 return session['_csrf_token']
