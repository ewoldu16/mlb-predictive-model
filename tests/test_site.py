from pathlib import Path
from datetime import date
import json
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def service():
    from mlb_app.model_service import V112ModelService
    return V112ModelService(ROOT)

@pytest.fixture()
def client():
    from app import create_app
    return create_app({"TESTING": True}).test_client()

def test_artifact_and_feature_contract(service):
    assert service.meta["model_version"] == "V11.2_COMPACT_TEAM_RUN"
    assert len(service.features) == 50
    assert service.features == json.loads((ROOT / "results/v11_2_compact_frozen_specification.json").read_text())["features"]
    with pytest.raises(ValueError, match="missing frozen features"):
        service.predict_team({})

def test_predictions_are_valid_and_missing_values_are_imputed(service):
    row = {feature: np.nan for feature in service.features}
    prediction = service.predict_team(row)
    assert prediction["expected_runs"] >= 0
    assert prediction["variance"] >= prediction["expected_runs"]
    assert prediction["interval_50"][0] <= prediction["interval_50"][1]
    assert len(prediction["all_contributions"]) == 50
    probability = service.home_probability(prediction["expected_runs"], prediction["expected_runs"] + 0.5)
    assert 0 <= probability <= 1

def test_confidence_labels_are_frozen():
    from mlb_app.model_service import confidence_label
    assert confidence_label(.5499) == "LOW"
    assert confidence_label(.55) == "MODERATE"
    assert confidence_label(.60) == "HIGH"

def test_pages_and_apis(client):
    for route in ["/", "/methodology", "/performance", "/live-tracking", "/about", "/history", "/api/games/today", "/api/model/performance", "/api/model/metadata", "/health"]:
        response = client.get(route)
        assert response.status_code == 200, route
    assert client.get("/api/game/777007").status_code == 200
    assert client.get("/game/777007").status_code == 200
    assert client.get("/api/game/1").status_code == 404

def test_history_schema_and_arithmetic(client):
    payload = client.get("/api/predictions/history?season=2025&limit=5").get_json()
    assert payload["count"] == 5
    required = {"game_id", "date", "team_away", "team_home", "predicted_runs_away", "predicted_runs_home", "home_win_probability"}
    for game in payload["predictions"]:
        assert required <= game.keys()
        assert game["predicted_runs_away"] >= 0 and game["predicted_runs_home"] >= 0
        assert game["predicted_runs_away"] + game["predicted_runs_home"] >= 0
        assert 0 <= game["home_win_probability"] <= 1

def test_performance_api_matches_verified_source(client):
    api = client.get("/api/model/performance").get_json()
    source = pd.read_csv(ROOT / "results/v11_2_untouched_2025_comparison.csv").set_index("model").loc["V11_2_compact"]
    assert api["forecast"]["rmse"] == pytest.approx(source.rmse)
    assert api["forecast"]["winner_accuracy"] == pytest.approx(source.winner_accuracy)

def test_fixed_historical_games_reconstruct_preserved_oos_values(client):
    source = pd.read_csv(ROOT / "results/v11_2_confidence_oos_game_predictions_2022_2025.csv").set_index("game_id")
    for game_id in [777007, 661042, 746776]:
        expected = source.loc[game_id]
        actual = client.get(f"/api/game/{game_id}").get_json()
        assert actual["predicted_runs_away"] == pytest.approx(expected.predicted_runs_away)
        assert actual["predicted_runs_home"] == pytest.approx(expected.predicted_runs_home)
        assert actual["home_win_probability"] == pytest.approx(expected.home_win_probability)

def test_today_api_schema_and_honest_pending_state(client):
    payload = client.get("/api/games/today").get_json()
    assert payload["schema_version"] in {"site-predictions-1.0", "site-predictions-2.0", "site-predictions-3.0"}
    for game in payload["games"]:
        assert {"game_id", "away_team", "home_team", "forecast_status", "prediction"} <= game.keys()
        if game["forecast_status"] not in {"ready","in_progress","final"}:
            assert game["prediction"] is None
            assert game["forecast_message"]

def test_live_game_exposes_complete_transparency(client):
    if date.today().isoformat()!='2026-08-22' or not (ROOT/'data/live/2026-08-22/prediction_823831.json').exists():
        pytest.skip('live transparency fixture is intentionally absent from a fresh clone')
    response=client.get('/game/823831');assert response.status_code==200
    html=response.get_data(as_text=True)
    assert 'DATA STATUS' in html and '100 / 100' in html
    assert 'PREDICTION_READY' in html
    assert html.count('data-feature-status=')==100
    assert 'frozen 2021-2024 training median' in html

def test_scheduled_outcome_free_target_has_exact_rows():
    if not (ROOT/'data/raw/games_2026.csv').exists():
        pytest.skip('live current-season cache is intentionally absent from a fresh clone')
    from mlb_app.feature_builder import build_daily_feature_rows
    features=json.loads((ROOT/'results/v11_2_compact_frozen_specification.json').read_text())['features']
    rows,meta=build_daily_feature_rows(ROOT,'2026-08-22',[823831],features)
    assert meta['status']=='ok' and len(rows)==2 and set(rows.team_side)=={'away','home'}
    game=pd.read_csv(ROOT/'data/raw/games_2026.csv').query('game_id == 823831').iloc[0]
    assert pd.isna(game.away_score) and pd.isna(game.home_score)

def test_first_pitch_cutoff_blocks_new_prediction():
    from mlb_app.live_pipeline import status_for
    game={'status':'Scheduled','start_time':'2000-01-01T00:00:00Z','away_starter_id':1,'home_starter_id':2}
    status,message=status_for(game,'confirmed','ok')
    assert status=='IN_PROGRESS' and 'cutoff' in message

def test_probable_lineup_source_fails_closed_without_key(monkeypatch,tmp_path):
    from mlb_app.probable_lineups import fetch_probable_lineups
    monkeypatch.delenv('ROTOWIRE_API_KEY',raising=False)
    lineups,meta=fetch_probable_lineups([],'2099-06-01',tmp_path)
    assert lineups=={} and meta['status']=='not_configured'

def test_probable_status_uses_frozen_confidence_thresholds():
    from mlb_app.live_pipeline import status_for
    game={'status':'Scheduled','start_time':'2099-06-01T20:00:00Z','away_starter_id':1,'home_starter_id':2}
    assert status_for(game,'probable','ok')[0]=='PROVISIONAL_PREDICTION'

def test_provisional_snapshot_upgrades_to_separate_immutable_final(monkeypatch,tmp_path):
    import mlb_app.live_pipeline as live
    from mlb_app.probable_lineups import lineup_fingerprint
    day='2099-06-01';game={'game_id':99,'date':day,'start_time':day+'T20:00:00Z','status':'Scheduled','away_team':'Away','home_team':'Home','away_team_id':1,'home_team_id':2,'away_starter':'A','home_starter':'H','away_starter_id':11,'home_starter_id':22,'venue':'Park'}
    def lineup(status,home_id=202):return {'status':status,'source':'test_projected_source' if status=='probable' else 'mlb_stats_api_boxscore','retrieval_timestamp':'2099-06-01T10:00:00Z','source_status':{'away':status,'home':status},'teams':{'away':[{'order':i,'player_id':100+i,'source_player_id':1000+i,'name':f'A{i}','position':'X'} for i in range(1,10)],'home':[{'order':i,'player_id':home_id+i,'source_player_id':2000+i,'name':f'H{i}','position':'X'} for i in range(1,10)]}}
    probable=lineup('probable');confirmed=lineup('confirmed',302);state={'confirmed':False}
    monkeypatch.setattr(live,'fetch_schedule',lambda *a:([game],{'status':'ok','source':'test'}));monkeypatch.setattr(live,'fetch_probable_lineups',lambda *a:({99:probable},{'status':'ok'}));monkeypatch.setattr(live,'fetch_lineup_status',lambda *a:(('confirmed' if state['confirmed'] else 'unavailable'),{}));monkeypatch.setattr(live,'confirmed_lineup_details',lambda *a:(confirmed if state['confirmed'] else None))
    class Service:
        features=['x'];meta={'model_version':'V11.2_COMPACT_TEAM_RUN','artifact_sha256':'hash'}
        def validated_vector(self,row,cutoff):return [{'feature':'x','raw_value':float(row.x),'final_value':float(row.x),'imputed':False,'feature_cutoff':cutoff}]
        def predict_team(self,row):
            value=float(row.x);return {'expected_runs':value,'variance':value,'interval_50':[1,2],'interval_80':[0,3],'positive_factors':[],'negative_factors':[],'all_contributions':[]}
        def home_probability(self,away,home):return .55
    folder=tmp_path/'data/live'/day;folder.mkdir(parents=True);pd.DataFrame([{'game_id':99,'team_side':'away','x':4.0},{'game_id':99,'team_side':'home','x':4.5}]).to_csv(folder/'features.csv',index=False);(folder/'lineup_build_99.json').write_text(json.dumps({'game_id':99,'lineup_status':'probable','lineup_fingerprint':lineup_fingerprint(probable)}));monkeypatch.setenv('MLB_STATE_DIR',str(tmp_path/'data/live'))
    first=live.generate_predictions(tmp_path,Service(),day);assert first['games'][0]['forecast_type']=='PROVISIONAL_PREDICTION';provisional_path=folder/'provisional_prediction_99.json';assert provisional_path.exists() and not (folder/'prediction_99.json').exists();before=provisional_path.read_bytes()
    state['confirmed']=True;(folder/'lineup_build_99.json').write_text(json.dumps({'game_id':99,'lineup_status':'confirmed','lineup_fingerprint':lineup_fingerprint(confirmed)}));second=live.generate_predictions(tmp_path,Service(),day);final=second['games'][0];assert final['forecast_type']=='FINAL_PREGAME_PREDICTION';assert (folder/'prediction_99.json').exists() and provisional_path.read_bytes()==before;assert final['snapshot']['immutable_after_first_pitch'] is True;assert final['provisional_comparison']['lineup_substitutions']

def test_provisional_website_labels(monkeypatch):
    import app as site
    players=[{'order':i,'player_id':i,'source_player_id':100+i,'name':f'Player {i}','position':'X'} for i in range(1,10)]
    game={'game_id':909090,'date':'2099-06-01','start_time':'2099-06-01T20:00:00Z','status':'PROVISIONAL_PREDICTION','forecast_status':'provisional_prediction','forecast_type':'PROVISIONAL_PREDICTION','away_team':'Away','home_team':'Home','away_starter':'A','home_starter':'H','away_starter_id':1,'home_starter_id':2,'venue':'Park','lineup_status':'probable','lineup_counts':{'away':9,'home':9},'lineup_source':'rotowire_projected_lineups_api','lineup_retrieved_at':'2099-06-01T10:00:00Z','lineup_details':{'teams':{'away':players,'home':players}},'prediction':{'away':{'expected_runs':4.2,'positive_factors':[],'negative_factors':[]},'home':{'expected_runs':4.7,'positive_factors':[],'negative_factors':[]},'projected_total':8.9,'predicted_winner':'Home','winner_probability':.55,'confidence':'MODERATE'}}
    payload={'date':'2099-06-01','generated_from':{'status':'test'},'feature_build':{'status':'ok'},'games':[game]};monkeypatch.setattr(site,'load_today',lambda root:payload);client=site.create_app({'TESTING':True}).test_client();home=client.get('/').get_data(as_text=True);detail=client.get('/game/909090').get_data(as_text=True);assert 'PROVISIONAL FORECAST' in home and 'USING PROBABLE LINEUP' in home;assert 'PROVISIONAL FORECAST' in detail and 'Lineup uncertainty: provisional' in detail and 'rotowire_projected_lineups_api' in detail and 'DERIVED FROM PROBABLE LINEUP' in detail

@pytest.mark.parametrize('lineup_status,forecast_type',[('probable','PROVISIONAL_PREDICTION'),('confirmed','FINAL_PREGAME_PREDICTION')])
def test_refresh_cycle_rebuilds_eligible_lineup_state(monkeypatch,tmp_path,lineup_status,forecast_type):
    import mlb_app.refresh_service as refresh
    day='2099-06-01';folder=tmp_path/'data/live'/day;folder.mkdir(parents=True);(folder/'features.csv').write_text('stale');calls=[];payloads=[{'date':day,'games':[{'game_id':1,'status':'INSUFFICIENT_DATA','lineup_status':lineup_status}]},{'date':day,'games':[{'game_id':1,'status':forecast_type,'forecast_type':forecast_type}]}]
    monkeypatch.setenv('MLB_STATE_DIR',str(tmp_path/'data/live'));monkeypatch.setattr(refresh,'generate_predictions',lambda *a,**k:payloads.pop(0));monkeypatch.setattr(refresh.subprocess,'run',lambda command,**kwargs:calls.append(command))
    result=refresh.refresh_cycle(tmp_path,object(),day,assembler=['assembler']);assert calls==[['assembler']] and result['games'][0]['forecast_type']==forecast_type and not (folder/'features.csv').exists()

def test_nationals_snapshot_is_frozen_and_audited():
    path=ROOT/'data/live/2026-08-22/prediction_823831.json'
    if not path.exists():
        pytest.skip('live immutable snapshot is intentionally absent from a fresh clone')
    before=path.read_bytes();game=json.loads(before)
    assert game['snapshot']['immutable_after_first_pitch'] is True
    assert len(game['feature_vectors']['away'])==50 and len(game['feature_vectors']['home'])==50
    assert path.read_bytes()==before
