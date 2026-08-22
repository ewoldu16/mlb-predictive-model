"""Resumable exact-definition 2026 data refresh for frozen V11.2 live inference."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import argparse,importlib.util,json,os,re,subprocess,sys,time,urllib.parse,urllib.request
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parent;YEAR=2026;RAW=ROOT/'data/raw';PROCESSED=ROOT/'data/processed';CACHE=RAW/'live_2026'
BASE_CHAIN=['build-offensive-features.py','build-pitching-features.py','build-bullpen-features.py','build-advanced-pitching-features.py','build-platoon-features.py','build-lineup-features.py','build-starter-lineup-matchup-features.py']
FAMILY_BUILDERS=['build-richer-offensive-form-features.py','build-contextual-offense-features.py','build-statsimpl-offense-risp-features.py','build-richer-starter-features.py','build-official-starter-pitching-features.py','build-official-bullpen-pitching-features.py','build-bullpen-availability-features.py','build-statsimpl-starter-recent100-features.py','build-arsenal-lineup-matchup-features.py','build-opponent-quality-offense-features.py','build-situational-team-features.py']

def request_json(url,path,retries=3):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(retries):
  try:
   with urllib.request.urlopen(url,timeout=30) as response:data=json.loads(response.read())
   tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data),encoding='utf-8');os.replace(tmp,path);return data
  except Exception:
   if attempt==retries-1:
    if path.exists():return json.loads(path.read_text())
    raise
   time.sleep(2*(attempt+1))

def schedule(day):
 url='https://statsapi.mlb.com/api/v1/schedule?'+urllib.parse.urlencode({'sportId':1,'startDate':f'{YEAR}-03-01','endDate':day,'gameType':'R','hydrate':'probablePitcher'})
 data=request_json(url,CACHE/'season_schedule.json');rows=[];probables={}
 for block in data.get('dates',[]):
  for g in block.get('games',[]):
   teams=g['teams'];gid=int(g['gamePk']);status=g.get('status',{}).get('detailedState');away_score=teams['away'].get('score') if status in ('Final','Game Over','Completed Early') else np.nan;home_score=teams['home'].get('score') if status in ('Final','Game Over','Completed Early') else np.nan
   rows.append({'date':block['date'],'game_id':gid,'away_team':teams['away']['team']['name'],'home_team':teams['home']['team']['name'],'away_score':away_score,'home_score':home_score,'home_win':(int(home_score>away_score) if pd.notna(home_score) and pd.notna(away_score) else np.nan),'game_status':status})
   probables[gid]={s:{'id':teams[s].get('probablePitcher',{}).get('id'),'name':teams[s].get('probablePitcher',{}).get('fullName')} for s in ('away','home')}
 games=pd.DataFrame(rows).drop_duplicates('game_id').sort_values(['date','game_id']);games.to_csv(RAW/f'games_{YEAR}.csv',index=False);return games,probables

def boxscores_and_identity(games,probables,workers=8):
 folder=CACHE/'boxscores';completed=games[games.away_score.notna()];targets=pd.concat([completed,games[games.date.eq(games.date.max())]]).drop_duplicates('game_id')
 def fetch(gid):
  path=folder/f'{gid}.json'
  if path.exists():return gid,json.loads(path.read_text())
  return gid,request_json(f'https://statsapi.mlb.com/api/v1/game/{gid}/boxscore',path)
 boxes={}
 with ThreadPoolExecutor(max_workers=workers) as pool:
  for gid,data in pool.map(fetch,targets.game_id.astype(int)):boxes[gid]=data
 starter=[];lineups=[]
 for g in games.itertuples(index=False):
  b=boxes.get(int(g.game_id));row={'game_id':int(g.game_id)}
  for side in ('away','home'):
   if b and b.get('teams',{}).get(side,{}).get('pitchers'):
    team=b['teams'][side];pid=int(team['pitchers'][0]);name=team['players']['ID'+str(pid)]['person']['fullName']
   else:pid=probables[int(g.game_id)][side]['id'];name=probables[int(g.game_id)][side]['name']
   row[f'{side}_starter_id']=pid;row[f'{side}_starter_name']=name
   if b:
    for key,p in b['teams'][side].get('players',{}).items():
     raw=p.get('battingOrder')
     if raw not in (None,'') and int(raw)%100==0:lineups.append({'game_id':int(g.game_id),'team_side':side,'batting_order':int(raw)//100,'player_id':int(p['person']['id']),'player_name':p['person']['fullName'],'position':p.get('position',{}).get('abbreviation')})
  starter.append(row)
 pd.DataFrame(starter).to_csv(RAW/f'starting_pitchers_{YEAR}.csv',index=False);ld=pd.DataFrame(lineups).drop_duplicates(['game_id','team_side','batting_order']);(RAW/'lineups').mkdir(exist_ok=True);ld.to_csv(RAW/'lineups'/f'starting_lineups_{YEAR}.csv',index=False)
 complete_lineup_ids=set(ld.groupby('game_id').size().loc[lambda x:x.eq(18)].index);build_games=games[games.away_score.notna()|games.game_id.isin(complete_lineup_ids)].copy();build_games.to_csv(RAW/f'games_{YEAR}.csv',index=False)
 # Reuse the official normalization exactly.
 spec=importlib.util.spec_from_file_location('official_download',ROOT/'download-official-mlb-pitching-boxscores.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);official=[]
 for g in completed.itertuples(index=False):official.extend(m.extract(int(g.game_id),g.date,boxes[int(g.game_id)]))
 official_dir=RAW/'official_mlb_pitching';official_dir.mkdir(exist_ok=True);pd.DataFrame(official).to_csv(official_dir/f'official_pitcher_game_lines_{YEAR}.csv',index=False)

def statcast_refresh(games,day):
 from pybaseball import statcast
 import pybaseball;pybaseball.cache.enable()
 union=['game_date','game_pk','game_type','at_bat_number','pitch_number','pitcher','batter','stand','p_throws','pitch_type','release_speed','effective_speed','release_spin_rate','release_extension','spin_axis','pfx_x','pfx_z','plate_x','plate_z','zone','description','events','bb_type','launch_speed','launch_angle','launch_speed_angle','estimated_woba_using_speedangle','woba_value','woba_denom','balls','strikes','outs_when_up','inning','inning_topbot','home_team','away_team','bat_score','fld_score','on_1b','on_2b','on_3b','delta_run_exp','delta_home_win_exp']
 folder=CACHE/'statcast_chunks';folder.mkdir(parents=True,exist_ok=True);start=pd.Timestamp(f'{YEAR}-03-01');end=min(pd.Timestamp(day),pd.Timestamp.now().normalize());parts=[]
 while start<=end:
  stop=min(start+pd.Timedelta(days=6),end);path=folder/f'{start.date()}_{stop.date()}.csv'
  if not path.exists():
   data=statcast(start.strftime('%Y-%m-%d'),stop.strftime('%Y-%m-%d'))
   expected_games=games[pd.to_datetime(games.date).between(start,stop)&games.away_score.notna()]
   if data.empty and expected_games.empty:data=pd.DataFrame(columns=union)
   elif data.empty:raise ValueError(f'Statcast returned no pitches for {len(expected_games)} completed games in {start.date()} to {stop.date()}')
   missing=[c for c in union if c not in data]
   if missing:raise ValueError(f'Statcast chunk {start.date()} missing columns {missing}')
   data=data[union];tmp=path.with_suffix('.tmp');data.to_csv(tmp,index=False);os.replace(tmp,path)
  parts.append(pd.read_csv(path,low_memory=False));start=stop+pd.Timedelta(days=1)
 data=pd.concat(parts,ignore_index=True);ids=set(games.game_id.astype(int));data=data[pd.to_numeric(data.game_pk,errors='coerce').isin(ids)].drop_duplicates(['game_pk','at_bat_number','pitch_number']).sort_values(['game_date','game_pk','at_bat_number','pitch_number']);data.to_csv(RAW/f'statcast_enriched_{YEAR}.csv',index=False);pa=data[data.events.notna()].copy();pa['batting_team']=np.where(pa.inning_topbot.eq('Top'),pa.away_team,pa.home_team);pa[['game_date','game_pk','batter','pitcher','events','home_team','away_team','inning_topbot','woba_value','woba_denom','batting_team']].to_csv(RAW/f'statcast_{YEAR}.csv',index=False);pitch_dir=RAW/'pitching';pitch_dir.mkdir(exist_ok=True);data.to_csv(pitch_dir/f'statcast_pitches_{YEAR}.csv',index=False)

def run_builder(script):
 source=(ROOT/script).read_text(encoding='utf-8');source=re.sub(r'YEARS\s*=\s*\[[^\]]+\]',f'YEARS = [{YEAR}]',source,count=1);source=re.sub(r'YEARS\s*=\s*range\(2021,\s*2026\)',f'YEARS = [{YEAR}]',source,count=1);runtime=CACHE/'builder_runtime';runtime.mkdir(parents=True,exist_ok=True);path=runtime/script;path.write_text(source,encoding='utf-8');env=os.environ.copy();env['PYTHONWARNINGS']='ignore';print(f'Running exact builder: {script}',flush=True);subprocess.run([sys.executable,str(path)],cwd=ROOT,env=env,check=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--date',default=date.today().isoformat());p.add_argument('--skip-statcast',action='store_true');args=p.parse_args();CACHE.mkdir(parents=True,exist_ok=True);games,probables=schedule(args.date);boxscores_and_identity(games,probables)
 if not args.skip_statcast:statcast_refresh(games,args.date)
 for script in BASE_CHAIN+FAMILY_BUILDERS:run_builder(script)
 print(f'2026 exact-source refresh complete: {len(games)} games through {args.date}')
if __name__=='__main__':main()
