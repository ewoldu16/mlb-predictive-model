from pathlib import Path
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
    for route in ["/", "/methodology", "/performance", "/about", "/history", "/api/games/today", "/api/model/performance", "/api/model/metadata"]:
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
    assert payload["schema_version"] in {"site-predictions-1.0", "site-predictions-2.0"}
    for game in payload["games"]:
        assert {"game_id", "away_team", "home_team", "forecast_status", "prediction"} <= game.keys()
        if game["forecast_status"] not in {"ready","in_progress","final"}:
            assert game["prediction"] is None
            assert game["forecast_message"]

def test_live_game_exposes_complete_transparency(client):
    response=client.get('/game/823831');assert response.status_code==200
    html=response.get_data(as_text=True)
    assert 'DATA STATUS' in html and '100 / 100' in html
    assert 'PREDICTION_READY' in html
    assert html.count('data-feature-status=')==100
    assert 'frozen 2021-2024 training median' in html

def test_scheduled_outcome_free_target_has_exact_rows():
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

def test_nationals_snapshot_is_frozen_and_audited():
    path=ROOT/'data/live/2026-08-22/prediction_823831.json';before=path.read_bytes();game=json.loads(before)
    assert game['snapshot']['immutable_after_first_pitch'] is True
    assert len(game['feature_vectors']['away'])==50 and len(game['feature_vectors']['home'])==50
    assert path.read_bytes()==before
