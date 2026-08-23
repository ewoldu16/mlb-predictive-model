"""Conservative adapter for RotoWire's documented projected-lineups API."""
from datetime import datetime,timezone
from pathlib import Path
import json,os,re,unicodedata,urllib.parse,urllib.request

TEAM_CODES={'Arizona Diamondbacks':'ARI','Atlanta Braves':'ATL','Baltimore Orioles':'BAL','Boston Red Sox':'BOS','Chicago Cubs':'CHC','Chicago White Sox':'CWS','Cincinnati Reds':'CIN','Cleveland Guardians':'CLE','Colorado Rockies':'COL','Detroit Tigers':'DET','Houston Astros':'HOU','Kansas City Royals':'KC','Los Angeles Angels':'LAA','Los Angeles Dodgers':'LAD','Miami Marlins':'MIA','Milwaukee Brewers':'MIL','Minnesota Twins':'MIN','New York Mets':'NYM','New York Yankees':'NYY','Athletics':'ATH','Philadelphia Phillies':'PHI','Pittsburgh Pirates':'PIT','San Diego Padres':'SD','Seattle Mariners':'SEA','San Francisco Giants':'SF','St. Louis Cardinals':'STL','Tampa Bay Rays':'TB','Texas Rangers':'TEX','Toronto Blue Jays':'TOR','Washington Nationals':'WAS'}
CODE_ALIASES={'OAK':'ATH','WSH':'WAS','CHW':'CWS','KCR':'KC','SDP':'SD','SFG':'SF','TBR':'TB'}

def _name(value):
 value=unicodedata.normalize('NFKD',str(value or '')).encode('ascii','ignore').decode().lower();return re.sub(r'[^a-z0-9]','',value)
def _code(value):
 value=str(value or '').upper();return CODE_ALIASES.get(value,value)
def _atomic(data,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(tmp,path)
def _json(url,cache,headers=None):
 try:
  request=urllib.request.Request(url,headers=headers or {})
  with urllib.request.urlopen(request,timeout=20) as response:data=json.loads(response.read())
  _atomic(data,cache);return data,'live'
 except Exception:
  if Path(cache).exists():return json.loads(Path(cache).read_text()),'cache'
  raise
def _roster(team_id,cache_dir):
 data,_=_json(f'https://statsapi.mlb.com/api/v1/teams/{int(team_id)}/roster?rosterType=active',Path(cache_dir)/f'mlb_roster_{team_id}.json');by_name={}
 for row in data.get('roster',[]):
  person=row.get('person',{});by_name.setdefault(_name(person.get('fullName')),[]).append({'player_id':int(person['id']),'player_name':person.get('fullName'),'position':row.get('position',{}).get('abbreviation')})
 return by_name
def _source_status(game,team):
 for obj in (team,game):
  for key in ('LineupStatus','Status','lineupStatus','status'):
   if obj.get(key) not in (None,''):return str(obj[key])
  if obj.get('Confirmed') is True:return 'Confirmed'
 return 'Expected'

def fetch_probable_lineups(games,day,cache_dir):
 """Return only complete, uniquely MLB-mapped projected orders; otherwise omit."""
 key=os.getenv('ROTOWIRE_API_KEY','').strip();meta={'source':'rotowire_projected_lineups_api','retrieved_at':datetime.now(timezone.utc).isoformat(),'status':'not_configured' if not key else 'requested','games_available':0,'rejections':[]}
 if not key:return {},meta
 query=urllib.parse.urlencode({'date':day,'format':'json','key':key});data,transport=_json('https://api.rotowire.com/Baseball/MLB/ProjectedLineups.php?'+query,Path(cache_dir)/f'rotowire_projected_{day}.json');meta['transport']=transport;out={}
 schedule={}
 for g in games:schedule.setdefault((_code(TEAM_CODES.get(g['away_team'])),_code(TEAM_CODES.get(g['home_team']))),[]).append(g)
 roster_cache={}
 for source_game in data.get('Games',[]):
  teams=source_game.get('Teams',[]);away=next((t for t in teams if not bool(int(t.get('IsHome',0)))),None);home=next((t for t in teams if bool(int(t.get('IsHome',0)))),None)
  if not away or not home:continue
  candidates=schedule.get((_code(away.get('Code')),_code(home.get('Code'))),[])
  if len(candidates)==1:game=candidates[0]
  elif len(candidates)>1:
   try:
    source_time=datetime.fromisoformat(str(source_game.get('DateTime')).replace('Z','+00:00'));ranked=sorted((abs((datetime.fromisoformat(str(g['start_time']).replace('Z','+00:00'))-source_time).total_seconds()),g) for g in candidates)
    game=ranked[0][1] if len(ranked)==1 or ranked[0][0]<ranked[1][0] else None
   except Exception:game=None
  else:game=None
  if not game:meta['rejections'].append({'teams':f"{away.get('Code')}@{home.get('Code')}",'reason':'game_match_missing_or_ambiguous'});continue
  normalized={'status':'probable','source':'rotowire_projected_lineups_api','retrieval_timestamp':meta['retrieved_at'],'source_status':{'away':_source_status(source_game,away),'home':_source_status(source_game,home)},'teams':{}}
  valid=True
  for side,source_team in (('away',away),('home',home)):
   team_id=game.get(f'{side}_team_id');roster_cache.setdefault(team_id,_roster(team_id,cache_dir));players=[]
   for player in source_team.get('Players',[]):
    try:order=int(player.get('BattingSpot'))
    except Exception:continue
    if not 1<=order<=9:continue
    source_name=' '.join(x for x in (player.get('FirstName'),player.get('LastName')) if x).strip();matches=roster_cache[team_id].get(_name(source_name),[])
    if len(matches)!=1:meta['rejections'].append({'game_id':game['game_id'],'side':side,'player':source_name,'reason':'mlb_identity_not_unique'});valid=False;continue
    match=matches[0];players.append({'order':order,'player_id':match['player_id'],'source_player_id':player.get('Id'),'name':match['player_name'],'position':player.get('Position') or match['position']})
   if len(players)!=9 or len({p['order'] for p in players})!=9 or len({p['player_id'] for p in players})!=9:valid=False
   normalized['teams'][side]=sorted(players,key=lambda x:x['order'])
  if valid:out[int(game['game_id'])]=normalized
 meta['status']='ok';meta['games_available']=len(out);return out,meta

def lineup_fingerprint(lineup):
 if not lineup:return None
 return '|'.join(f"{side}:"+','.join(f"{p['order']}={p['player_id']}" for p in lineup['teams'][side]) for side in ('away','home'))
