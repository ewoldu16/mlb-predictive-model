import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
from werkzeug.security import generate_password_hash

ROOT = Path(__file__).parents[1]


def _evaluator(tmp_path):
    spec = importlib.util.spec_from_file_location("tracking_evaluator_test", ROOT / "evaluate-live-v11-2-day.py")
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.OUT = tmp_path;module.LEDGER = tmp_path / "ledger.csv";module.DAILY = tmp_path / "daily.csv";module.CUMULATIVE = tmp_path / "cumulative.csv"
    return module


def _graded(day, game_id, correct=1):
    return {"date": day, "game_id": game_id, "snapshot_timestamp": day + "T10:00:00Z", "first_pitch": day + "T18:00:00Z", "away_team": "Away", "home_team": "Home", "pred_away_runs": 4.0, "pred_home_runs": 4.5, "pred_total": 8.5, "predicted_winner": "Home", "winner_probability": .58, "confidence": "MODERATE", "actual_away_runs": 3, "actual_home_runs": 5, "actual_total": 8, "actual_winner": "Home", "winner_correct": correct, "away_abs_error": 1.0, "home_abs_error": .5, "away_squared_error": 1.0, "home_squared_error": .25, "total_abs_error": .5, "total_squared_error": .25, "predicted_run_diff_home_minus_away": .5, "actual_run_diff_home_minus_away": 2, "run_diff_abs_error": 1.5, "model_version": "V11.2_COMPACT_TEAM_RUN", "artifact_hash": "hash"}


def test_tracking_append_is_idempotent_and_cumulative(tmp_path):
    evaluator = _evaluator(tmp_path)
    new, daily, cumulative = evaluator.append_day("2026-08-23", [_graded("2026-08-23", 1)])
    assert len(new) == 1 and cumulative.iloc[-1].predictions == 1
    duplicate, daily, cumulative = evaluator.append_day("2026-08-23", [_graded("2026-08-23", 1)])
    assert duplicate.empty and cumulative.iloc[-1].predictions == 1
    evaluator.append_day("2026-08-24", [_graded("2026-08-24", 2, 0)])
    daily = pd.read_csv(evaluator.DAILY);cumulative = pd.read_csv(evaluator.CUMULATIVE)
    assert daily.date.astype(str).tolist() == ["2026-08-23", "2026-08-24"]
    assert cumulative.iloc[-1].predictions == 2


def _game(status="Scheduled"):
    return {"game_id": 9, "date": "2026-08-25", "start_time": "2099-08-25T18:00:00Z", "status": status, "away_team": "Away", "home_team": "Home", "away_starter": "A", "home_starter": "H", "away_starter_id": 1, "home_starter_id": 2}


def _snapshot(kind="PROVISIONAL_PREDICTION"):
    return {**_game(), "status": kind, "forecast_status": kind.lower(), "forecast_type": kind, "lineup_status": "owner_managed", "lineup_fingerprint": "saved", "prediction": {"away": {"expected_runs": 4.1}, "home": {"expected_runs": 4.6}, "projected_total": 8.7, "predicted_winner": "Home", "winner_probability": .57, "confidence": "MODERATE"}, "snapshot": {"generated_at": "2026-08-25T10:00:00Z"}}


def test_provisional_snapshot_beats_pending_official_lineup(monkeypatch, tmp_path):
    import mlb_app.live_pipeline as live
    monkeypatch.setattr(live, "fetch_schedule", lambda *a: ([_game()], {"status": "ok"}));monkeypatch.setattr(live, "fetch_probable_lineups", lambda *a: ({}, {}));monkeypatch.setattr(live, "fetch_lineup_status", lambda *a: ("unavailable", {}));monkeypatch.setattr(live, "owner_lineup_for_game", lambda *a: None);monkeypatch.setattr(live, "build_daily_feature_rows", lambda *a: (None, {"status": "source_data_incomplete"}));monkeypatch.setattr(live, "load_snapshot", lambda *a: None);monkeypatch.setattr(live, "load_provisional_snapshot", lambda *a: _snapshot());monkeypatch.setattr(live, "save_state", lambda *a: True)
    service=type("S",(),{"features":[],"meta":{"model_version":"test","artifact_sha256":"test"}})()
    result=live.generate_predictions(tmp_path,service,"2026-08-25")["games"][0]
    assert result["status"] == "PROVISIONAL_PREDICTION" and result["prediction"]
    assert result["official_lineup_status"] == "unavailable"


def test_final_snapshot_supersedes_provisional(monkeypatch, tmp_path):
    import mlb_app.live_pipeline as live
    final=_snapshot("FINAL_PREGAME_PREDICTION")
    monkeypatch.setattr(live,"fetch_schedule",lambda *a:([_game()],{"status":"ok"}));monkeypatch.setattr(live,"fetch_probable_lineups",lambda *a:({},{}));monkeypatch.setattr(live,"fetch_lineup_status",lambda *a:("unavailable",{}));monkeypatch.setattr(live,"owner_lineup_for_game",lambda *a:None);monkeypatch.setattr(live,"build_daily_feature_rows",lambda *a:(None,{"status":"source_data_incomplete"}));monkeypatch.setattr(live,"load_snapshot",lambda *a:final);monkeypatch.setattr(live,"load_provisional_snapshot",lambda *a:_snapshot());monkeypatch.setattr(live,"save_state",lambda *a:True)
    result=live.generate_predictions(tmp_path,type("S",(),{"features":[],"meta":{"model_version":"test","artifact_sha256":"test"}})(),"2026-08-25")["games"][0]
    assert result["forecast_type"] == "FINAL_PREGAME_PREDICTION"


def test_date_read_uses_snapshots_without_regeneration(monkeypatch, tmp_path):
    import mlb_app.live_pipeline as live
    monkeypatch.setattr(live,"load_state",lambda key: [] if key.startswith("live_results:") else None);monkeypatch.setattr(live,"load_snapshots_for_date",lambda day:[_snapshot("FINAL_PREGAME_PREDICTION")]);monkeypatch.setattr(live,"load_provisional_snapshots_for_date",lambda day:[]);monkeypatch.setattr(live,"build_daily_feature_rows",lambda *a:(_ for _ in ()).throw(AssertionError("must not rebuild")))
    payload=live.load_game_date(tmp_path,"2026-08-24")
    assert len(payload["games"]) == 1 and payload["generated_from"]["status"] == "immutable_snapshots"


def test_date_navigation_owner_login_and_logout(monkeypatch):
    import app as web
    monkeypatch.setattr(web,"load_game_date",lambda *a:{"date":"2026-08-24","generated_from":{"status":"stored"},"games":[]});monkeypatch.setattr(web,"load_today",lambda *a:{"date":"2026-08-25","games":[]});monkeypatch.setattr(web,"health_state",lambda:{})
    app=web.create_app({"TESTING":True,"SECRET_KEY":"test","OWNER_USERNAME":"owner","OWNER_PASSWORD_HASH":generate_password_hash("password")});client=app.test_client()
    page=client.get("/games/2026-08-24");html=page.get_data(as_text=True)
    assert page.status_code==200 and "Previous day" in html and 'type="date"' in html and "Owner Login" in html
    login=client.get("/owner/login");token=re.search(r'name="_csrf_token" value="([^"]+)"',login.get_data(as_text=True)).group(1)
    assert client.post("/owner/login",data={"_csrf_token":token,"username":"owner","password":"wrong"}).status_code==401
    login=client.get("/owner/login");token=re.search(r'name="_csrf_token" value="([^"]+)"',login.get_data(as_text=True)).group(1)
    assert client.post("/owner/login",data={"_csrf_token":token,"username":"owner","password":"password"}).status_code==302
    dashboard=client.get("/owner");assert dashboard.status_code==200 and "Logout" in dashboard.get_data(as_text=True)
    with client.session_transaction() as session:token=session["_csrf_token"]
    assert client.post("/owner/logout",data={"_csrf_token":token}).status_code==302
