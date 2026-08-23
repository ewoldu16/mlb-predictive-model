from pathlib import Path
import json,os

SECTIONS={
 'Starting pitchers':('starter_quality','starter_recent','starter_workload','starter_matchup'),
 'Offense / lineup':('offensive_baseline','recent_offense','lineup_quality','opponent_quality_offense'),
 'Bullpens':('bullpen_quality','bullpen_recent','bullpen_availability'),
 'Team / context':('venue_context','run_differential_pythagorean','team_record_form'),
}
def build_transparency(root,game,service,payload):
 root=Path(root);prov=json.loads((root/'results/v11_2_live_feature_provenance.json').read_text());by_name={x['feature']:x for x in prov};vectors=game.get('feature_vectors',{});rows=[]
 for side,team in [('away',game['away_team']),('home',game['home_team'])]:
  existing={x['feature']:x for x in vectors.get(side,[])}
  for feature in service.features:
   p=by_name[feature];v=existing.get(feature);status=('IMPUTED' if v and v['imputed'] else 'VALID') if v else 'NOT_AVAILABLE'
   imputation=v.get('imputation_source') if v else None;source=p['source_dataset']+(f' · {imputation}' if imputation else '')
   lineup_dependent=feature.startswith(('lineup_','opp_sp_matchup_','opp_arsenal_','opp_recent100_'));derivation=('DERIVED FROM PROBABLE LINEUP' if game.get('lineup_status')=='probable' else 'DERIVED FROM CONFIRMED LINEUP') if lineup_dependent else None
   rows.append({'team_side':side,'team':team,'feature':feature,'raw_value':v.get('raw_value') if v else None,'final_value':v.get('final_value') if v else None,'status':status,'source':source+(f' · {derivation}' if derivation else ''),'as_of':v.get('feature_cutoff') if v else game.get('start_time'),'used':True,'family':p['family'],'imputation_source':imputation,'lineup_derivation':derivation})
 available=sum(r['status'] in ('VALID','IMPUTED') for r in rows);imputed=sum(r['status']=='IMPUTED' for r in rows);missing=len(rows)-available
 reason=game.get('forecast_message')
 build=payload.get('feature_build',{})
 if build.get('status')=='target_games_missing_from_feature_universe':reason='Exact 2026 target-game feature rows are absent from the validated feature universe. The lineup/processed feature build has not completed for this game.'
 elif build.get('status')=='source_data_incomplete':reason='Required exact-source feature datasets are incomplete: '+', '.join(build.get('missing_files',[]))
 missing_rows=[]
 for r in rows:
  if r['status'] not in ('VALID','IMPUTED'):missing_rows.append({**r,'reason':reason or 'Validated source value unavailable.','eligible':True,'blocks':True})
 sections={name:[r for r in rows if r['family'] in families] for name,families in SECTIONS.items()}
 return {'rows':rows,'sections':sections,'missing':missing_rows,'counts':{'available':available,'required':100,'imputed':imputed,'missing':missing},'reason':reason,'refreshed':game.get('snapshot',{}).get('generated_at') or _mtime(root,game),'prediction_status':game.get('status') or game.get('forecast_status'),'lineups':game.get('lineup_counts',{}),'lineup_status':game.get('lineup_status','unavailable'),'lineup_source':game.get('lineup_source'),'lineup_retrieved_at':game.get('lineup_retrieved_at'),'batting_orders':_lineups(root,game),'diagnostics':game.get('diagnostics',[]),'provisional_comparison':game.get('provisional_comparison')}
def _mtime(root,game):
 path=Path(os.getenv('MLB_STATE_DIR',root/'data/live'))/str(game.get('date'))/'predictions.json'
 return __import__('datetime').datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat() if path.exists() else 'N/A'
def _lineups(root,game):
 details=game.get('lineup_details')
 if details:return {side:[{'order':p['order'],'name':p['name'],'position':p.get('position'),'player_id':p.get('player_id'),'source_player_id':p.get('source_player_id')} for p in details.get('teams',{}).get(side,[])] for side in ('away','home')}
 path=Path(os.getenv('MLB_STATE_DIR',root/'data/live'))/str(game.get('date'))/'boxscore_cache'/f"boxscore_{game['game_id']}.json";out={'away':[],'home':[]}
 if not path.exists():return out
 data=json.loads(path.read_text())
 for side in out:
  for player in data.get('teams',{}).get(side,{}).get('players',{}).values():
   raw=player.get('battingOrder')
   if raw not in (None,'') and int(raw)%100==0:out[side].append({'order':int(raw)//100,'name':player.get('person',{}).get('fullName'),'position':player.get('position',{}).get('abbreviation')})
  out[side].sort(key=lambda x:x['order'])
 return out
