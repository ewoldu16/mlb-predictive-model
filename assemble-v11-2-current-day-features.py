"""Stage eligible current games as outcome-free targets and run exact V11 builders."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,importlib.util,json,time
import pandas as pd
from mlb_app.live_pipeline import fetch_schedule,fetch_lineup_status

ROOT=Path(__file__).resolve().parent;RAW=ROOT/'data/raw';YEAR=2026
BUILDERS=['build-offensive-features.py','build-pitching-features.py','build-bullpen-features.py','build-advanced-pitching-features.py','build-platoon-features.py','build-lineup-features.py','build-starter-lineup-matchup-features.py','build-richer-offensive-form-features.py','build-contextual-offense-features.py','build-statsimpl-offense-risp-features.py','build-richer-starter-features.py','build-official-starter-pitching-features.py','build-official-bullpen-pitching-features.py','build-bullpen-availability-features.py','build-statsimpl-starter-recent100-features.py','build-arsenal-lineup-matchup-features.py','build-opponent-quality-offense-features.py','build-situational-team-features.py']

def parse_lineups(path,gid):
 data=json.loads(Path(path).read_text());rows=[]
 for side in ('away','home'):
  for player in data.get('teams',{}).get(side,{}).get('players',{}).values():
   raw=player.get('battingOrder')
   if raw not in (None,'') and int(raw)%100==0:rows.append({'game_id':gid,'team_side':side,'batting_order':int(raw)//100,'player_id':int(player['person']['id']),'player_name':player['person']['fullName'],'position':player.get('position',{}).get('abbreviation')})
 frame=pd.DataFrame(rows)
 if len(frame)!=18 or frame.duplicated(['team_side','batting_order']).any():raise ValueError(f'{gid}: confirmed lineup did not normalize to 18 unique slots')
 return frame

def stage(day):
 folder=ROOT/'data/live'/day;games,_=fetch_schedule(day,folder/'schedule_cache');now=datetime.now(timezone.utc);eligible=[]
 base=pd.read_csv(RAW/f'games_{YEAR}.csv');starters=pd.read_csv(RAW/f'starting_pitchers_{YEAR}.csv');all_lineups=pd.read_csv(RAW/'lineups'/f'starting_lineups_{YEAR}.csv')
 for game in games:
  start=pd.Timestamp(game['start_time']).to_pydatetime()
  if start.tzinfo is None:start=start.replace(tzinfo=timezone.utc)
  if game['status'] not in {'Scheduled','Pre-Game','Warmup'} or now>=start:continue
  lineup_status,_=fetch_lineup_status(game,folder/'boxscore_cache')
  if lineup_status!='confirmed' or not game.get('away_starter_id') or not game.get('home_starter_id'):continue
  gid=game['game_id'];box=folder/'boxscore_cache'/f'boxscore_{gid}.json';lineup=parse_lineups(box,gid);cache=RAW/'lineups'/str(YEAR);cache.mkdir(parents=True,exist_ok=True);lineup.to_csv(cache/f'lineup_{gid}.csv',index=False);all_lineups=pd.concat([all_lineups[~all_lineups.game_id.eq(gid)],lineup],ignore_index=True)
  row={'date':day,'game_id':gid,'away_team':game['away_team'],'home_team':game['home_team'],'away_score':None,'home_score':None,'home_win':None,'game_status':'Scheduled'};base=pd.concat([base[~base.game_id.eq(gid)],pd.DataFrame([row])],ignore_index=True)
  sr={'game_id':gid,'away_starter_id':game['away_starter_id'],'away_starter_name':game['away_starter'],'home_starter_id':game['home_starter_id'],'home_starter_name':game['home_starter']};starters=pd.concat([starters[~starters.game_id.eq(gid)],pd.DataFrame([sr])],ignore_index=True);eligible.append(gid)
 base.sort_values(['date','game_id']).to_csv(RAW/f'games_{YEAR}.csv',index=False);starters.to_csv(RAW/f'starting_pitchers_{YEAR}.csv',index=False);all_lineups.to_csv(RAW/'lineups'/f'starting_lineups_{YEAR}.csv',index=False);return eligible

def run(day):
 started=time.perf_counter();eligible=stage(day)
 if not eligible:return {'eligible':[],'seconds':time.perf_counter()-started,'status':'no_unstarted_confirmed_lineups'}
 spec=importlib.util.spec_from_file_location('refresh',ROOT/'refresh-v11-2-2026-features.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 for script in BUILDERS:module.run_builder(script)
 return {'eligible':eligible,'seconds':time.perf_counter()-started,'status':'complete'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--date',default=datetime.now().date().isoformat());a=p.parse_args();result=run(a.date);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
