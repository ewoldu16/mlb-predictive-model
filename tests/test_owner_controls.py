from datetime import datetime,timezone
from pathlib import Path
import hashlib,re

import pytest
from werkzeug.security import generate_password_hash

ROOT=Path(__file__).resolve().parents[1]

def game():return {'game_id':991,'date':'2099-06-01','start_time':'2099-06-01T20:00:00Z','status':'Scheduled','away_team':'Away','home_team':'Home','away_team_id':1,'home_team_id':2}
def state():
 roster=[{'player_id':i,'name':f'P{i}','position':'OF' if i<10 else 'IF','handedness':'R','active_roster':True} for i in range(1,12)]
 lineup=[{**roster[i-1],'order':i} for i in range(1,10)]
 return {'game_id':991,'game_date':'2099-06-01','team_side':'away','team_id':1,'team_name':'Away','template':{'source_game_id':900,'source_date':'2099-05-31','players':[dict(x) for x in lineup]},'roster':roster,'availability':{str(i):'AVAILABLE' for i in range(1,12)},'lineup':lineup,'version':1,'updated_at':'2099-06-01T10:00:00Z','owner_modified':False,'last_impact':None,'valid':True,'validation_errors':[],'empty_positions':[],'hitters_required':0,'status':'PROVISIONAL_READY'}

@pytest.fixture()
def local_state(monkeypatch,tmp_path):
 monkeypatch.setenv('MLB_STATE_DIR',str(tmp_path/'live'))
 from mlb_app.owner_controls import save_team_state
 value=state();save_team_state(tmp_path,value);return tmp_path,value

@pytest.mark.parametrize('status',['DOUBTFUL','UNAVAILABLE','INJURED','REST_EXPECTED','SUSPENDED','MINORS_OR_INACTIVE','UNKNOWN'])
def test_every_non_available_status_removes_player_and_fails_closed(local_state,status):
 from mlb_app.owner_controls import set_availability
 root,value=local_state;updated=set_availability(root,game(),'away',1,status,'owner')
 assert 1 not in {p['player_id'] for p in updated['lineup']}
 assert updated['availability']['1']==status and not updated['valid']
 assert updated['status']=='PROVISIONAL_LINEUP_INCOMPLETE' and updated['hitters_required']==1

def test_suggestions_and_selection_exclude_unavailable(local_state):
 from mlb_app.owner_controls import replacement_suggestions,set_availability,save_lineup
 root,value=local_state;updated=set_availability(root,game(),'away',1,'INJURED','owner')
 suggestions=replacement_suggestions(updated,1)
 assert 1 not in {p['player_id'] for p in suggestions}
 with pytest.raises(ValueError,match='not AVAILABLE'):save_lineup(root,game(),'away',[{**p,'player_id':1 if p['order']==2 else p['player_id']} for p in updated['lineup']],'owner')

def test_lineup_validation_duplicate_short_and_valid_replacement(local_state):
 from mlb_app.owner_controls import save_lineup,set_availability
 root,value=local_state;updated=set_availability(root,game(),'away',1,'INJURED','owner')
 with pytest.raises(ValueError):save_lineup(root,game(),'away',[{'order':i,'player_id':2} for i in range(1,10)],'owner')
 with pytest.raises(ValueError):save_lineup(root,game(),'away',[{'order':i,'player_id':i+1} for i in range(1,9)],'owner')
 lineup=[{'order':1,'player_id':10}]+[{'order':i,'player_id':i} for i in range(2,10)]
 saved=save_lineup(root,game(),'away',lineup,'owner');assert saved['valid'] and len(saved['lineup'])==9 and saved['lineup'][0]['player_id']==10

def test_confidence_is_deterministic_and_does_not_enter_model_vector():
 from mlb_app.owner_controls import offense_confidence
 s=state();vector=[{'feature':'lineup_season_woba','imputed':False},{'feature':'opp_sp_matchup_season_k_pct','imputed':False},{'feature':'sit_win_pct','imputed':False}]
 result=offense_confidence(s,vector);assert result['label']=='HIGH' and result['score']==pytest.approx(1.0)
 assert set(result).isdisjoint({'expected_runs','home_win_probability'})

def test_first_pitch_lock_blocks_owner_mutation(local_state):
 from mlb_app.owner_controls import set_availability
 root,value=local_state;past={**game(),'start_time':'2000-01-01T00:00:00Z'}
 with pytest.raises(PermissionError):set_availability(root,past,'away',1,'INJURED','owner')

def test_owner_authentication_csrf_and_logout(monkeypatch):
 import app as site
 monkeypatch.setattr(site,'load_today',lambda root:{'date':'2099-06-01','generated_from':{'status':'test'},'games':[]})
 app=site.create_app({'TESTING':True,'SECRET_KEY':'test-secret','OWNER_USERNAME':'boss','OWNER_PASSWORD_HASH':generate_password_hash('safe-pass')});client=app.test_client()
 assert client.get('/owner').status_code==302
 html=client.get('/owner/login').get_data(as_text=True);token=re.search(r'name="_csrf_token" value="([^"]+)"',html).group(1)
 assert client.post('/owner/login',data={'_csrf_token':token,'username':'boss','password':'wrong'}).status_code==401
 html=client.get('/owner/login').get_data(as_text=True);token=re.search(r'name="_csrf_token" value="([^"]+)"',html).group(1)
 assert client.post('/owner/login',data={'_csrf_token':token,'username':'boss','password':'safe-pass'}).status_code==302
 assert client.get('/owner').status_code==200
 assert client.post('/owner/logout',data={}).status_code==400

def test_frozen_artifact_hash_unchanged():
 path=ROOT/'artifacts/v11_2_compact_pipeline.joblib'
 assert hashlib.sha256(path.read_bytes()).hexdigest()=='4249dfdf08b4326f051f8a845924048aab8d94aaa497a56c884cf65af7493dbf'
