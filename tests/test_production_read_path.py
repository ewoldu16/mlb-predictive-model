from datetime import datetime, timezone


def test_static_assets_use_public_static_endpoint():
    from app import create_app

    client = create_app({"TESTING": True}).test_client()
    home = client.get("/")
    assert home.status_code == 200
    html = home.get_data(as_text=True)
    assert 'href="/static/css/product.css"' in html
    assert 'data-ui="editorial-v11-2"' in html
    assert 'src="/static/js/site.js"' in home.get_data(as_text=True)
    css = client.get("/static/css/site.css")
    js = client.get("/static/js/site.js")
    assert css.status_code == 200 and css.content_type.startswith("text/css")
    assert js.status_code == 200 and "javascript" in js.content_type


def test_health_exposes_sanitized_persistence_and_today_counts(monkeypatch):
    import app as web

    payload = {
        "games": [
            {"status": "FINAL_PREGAME_PREDICTION", "forecast_type": "FINAL_PREGAME_PREDICTION"},
            {"status": "PENDING_LINEUP", "forecast_type": None},
        ]
    }
    monkeypatch.setattr(web, "health_state", lambda: {"status": "ok", "executor": "github_actions", "last_successful_refresh": "2026-08-23T20:04:39+00:00", "current_data_date": "2026-08-23"})
    monkeypatch.setattr(web, "load_today", lambda root, day=None: payload)
    monkeypatch.setattr(web, "persistence_health", lambda day: {"configured": True, "connected": True, "mode": "supabase_postgres", "today_payload_present": True, "queued_rebuild_requests": 2})
    monkeypatch.setattr(web, "database_url", lambda: "configured")
    response = web.create_app({"TESTING": True}).test_client().get("/health")
    data = response.get_json()
    assert data["healthy"] is True
    assert data["supabase"]["connected"] is True
    assert data["today"]["game_count"] == 2
    assert data["today"]["final_predictions"] == 1
    assert data["today"]["pending_lineup"] == 1


def test_persistence_failure_is_credential_safe(monkeypatch):
    import mlb_app.storage as storage

    class BrokenConnection:
        def __enter__(self):
            raise RuntimeError('password authentication failed for user "postgres.secret-project"')
        def __exit__(self, *args): return False

    monkeypatch.setattr(storage, "database_url", lambda: "configured")
    monkeypatch.setattr(storage, "_connect", lambda: BrokenConnection())
    result = storage.persistence_health("2026-08-23")
    assert result["connected"] is False
    assert result["error"] == "database authentication failed"
    assert "secret-project" not in str(result)


def test_started_game_without_snapshot_is_not_called_insufficient(monkeypatch, tmp_path):
    import mlb_app.live_pipeline as live

    game = {"game_id": 1, "date": "2026-08-23", "start_time": "2026-08-23T17:00:00Z", "status": "In Progress", "away_team": "Away", "home_team": "Home", "away_starter": "A", "home_starter": "H", "away_starter_id": 10, "home_starter_id": 20}
    lineup = {"status": "confirmed", "source": "test", "retrieval_timestamp": datetime.now(timezone.utc).isoformat(), "teams": {side: [{"order": i, "player_id": i + (0 if side == "away" else 20), "name": str(i)} for i in range(1, 10)] for side in ("away", "home")}}
    monkeypatch.setattr(live, "fetch_schedule", lambda *args: ([game], {"status": "ok", "source": "test"}))
    monkeypatch.setattr(live, "fetch_probable_lineups", lambda *args: ({}, {"status": "not_configured"}))
    monkeypatch.setattr(live, "fetch_lineup_status", lambda *args: ("confirmed", {"away": 9, "home": 9}))
    monkeypatch.setattr(live, "confirmed_lineup_details", lambda *args: lineup)
    monkeypatch.setattr(live, "build_daily_feature_rows", lambda *args: (None, {"status": "source_data_incomplete"}))
    monkeypatch.setattr(live, "load_snapshot", lambda *args: None)
    monkeypatch.setattr(live, "load_provisional_snapshot", lambda *args: None)
    monkeypatch.setattr(live, "load_state", lambda *args: {})
    monkeypatch.setattr(live, "save_state", lambda *args: True)
    service = type("Service", (), {"features": [], "meta": {"model_version": "test", "artifact_sha256": "test"}})()
    payload = live.generate_predictions(tmp_path, service, "2026-08-23")
    result = payload["games"][0]
    assert result["status"] == "IN_PROGRESS"
    assert result["prediction"] is None
    assert "No immutable pregame forecast" in result["forecast_message"]
