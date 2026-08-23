import json

import pandas as pd


def _write_tracking(root, ledger_rows, daily_rows):
    folder = root / "results" / "live_tracking"
    folder.mkdir(parents=True)
    ledger_columns = [
        "winner_probability", "winner_correct", "away_abs_error",
        "home_abs_error", "total_abs_error",
    ]
    daily_columns = [
        "date", "predictions", "winner_correct", "winner_incorrect",
        "winner_accuracy", "team_run_MAE", "total_MAE",
        "60plus_predictions", "60plus_accuracy",
    ]
    pd.DataFrame(ledger_rows, columns=ledger_columns).to_csv(
        folder / "v11_2_live_predictions.csv", index=False
    )
    pd.DataFrame(daily_rows, columns=daily_columns).to_csv(
        folder / "v11_2_live_daily_summary.csv", index=False
    )


def test_publish_tracking_with_zero_60plus_forecasts(monkeypatch, tmp_path):
    import github_actions_refresh as worker

    _write_tracking(tmp_path, [{
        "winner_probability": .57, "winner_correct": 1,
        "away_abs_error": 1.0, "home_abs_error": 2.0,
        "total_abs_error": 3.0,
    }], [{
        "date": "2026-08-22", "predictions": 1, "winner_correct": 1,
        "winner_incorrect": 0, "winner_accuracy": 1.0,
        "team_run_MAE": 1.5, "total_MAE": 3.0,
        "60plus_predictions": 0, "60plus_accuracy": None,
    }])
    saved = []
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "save_state", lambda key, value: saved.append((key, value)))
    worker._publish_tracking()
    assert saved[0][0] == "live_tracking_summary"
    summary = saved[0][1]
    assert summary["high_predictions"] == 0
    assert summary["high_accuracy"] is None
    assert summary["daily"][0]["60plus_accuracy"] is None
    json.dumps(summary, allow_nan=False)


def test_publish_tracking_with_empty_partial_summary(monkeypatch, tmp_path):
    import github_actions_refresh as worker

    _write_tracking(tmp_path, [], [])
    saved = []
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "save_state", lambda key, value: saved.append((key, value)))
    worker._publish_tracking()
    assert saved == []


def test_normal_finite_live_summary_is_preserved(monkeypatch, tmp_path):
    from mlb_app.live_tracking import load_live_tracking

    _write_tracking(tmp_path, [{
        "winner_probability": .64, "winner_correct": 1,
        "away_abs_error": .75, "home_abs_error": 1.25,
        "total_abs_error": .5,
    }], [{
        "date": "2026-08-22", "predictions": 1, "winner_correct": 1,
        "winner_incorrect": 0, "winner_accuracy": 1.0,
        "team_run_MAE": 1.0, "total_MAE": .5,
        "60plus_predictions": 1, "60plus_accuracy": 1.0,
    }])
    monkeypatch.delenv("SUPABASE_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    summary = load_live_tracking(tmp_path)
    assert summary["winner_accuracy"] == 1.0
    assert summary["high_accuracy"] == 1.0
    assert summary["team_run_mae"] == 1.0
    assert summary["total_mae"] == .5


def test_undefined_template_values_render_as_na():
    from app import create_app

    app = create_app({"TESTING": True})
    assert app.jinja_env.filters["pct"](None) == "N/A"
    assert app.jinja_env.filters["pct"](float("nan")) == "N/A"
    assert app.jinja_env.filters["num"](float("inf")) == "N/A"
