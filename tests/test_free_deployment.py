from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]

def test_render_blueprint_contains_only_one_free_web_service():
 text=(ROOT/'render.yaml').read_text();assert text.count('type: web')==1 and 'plan: free' in text
 assert 'type: worker' not in text and 'disk:' not in text and 'databases:' not in text and 'plan: standard' not in text
 assert 'SUPABASE_DATABASE_URL' in text

def test_actions_workflow_is_scheduled_manual_cached_and_secret_safe():
 text=(ROOT/'.github/workflows/mlb-live-refresh.yml').read_text()
 for required in ('workflow_dispatch:','schedule:','actions/cache/restore@v4','actions/cache/save@v4','github_actions_refresh.py','secrets.SUPABASE_DATABASE_URL','pip install -r requirements.txt','runtime_import_preflight.py'):assert required in text
 assert 'ROTOWIRE_API_KEY' not in text and 'contents: write' not in text
 assert '*/15' in text and '*/30' in text
 assert 'github.run_id' in text and 'github.run_attempt' in text and 'if: always()' in text

def test_supabase_url_alias(monkeypatch):
 from mlb_app.storage import database_url
 monkeypatch.delenv('DATABASE_URL',raising=False);monkeypatch.setenv('SUPABASE_DATABASE_URL','postgresql://example.invalid/db');assert database_url()=='postgresql://example.invalid/db'

def test_conflicting_database_urls_fail_closed(monkeypatch):
 import pytest
 from mlb_app.storage import database_url
 monkeypatch.setenv('SUPABASE_DATABASE_URL','postgresql://one.invalid/db');monkeypatch.setenv('DATABASE_URL','postgresql://two.invalid/db')
 with pytest.raises(RuntimeError,match='Conflicting'):database_url()

def test_cache_refresh_decision_is_daily_not_every_completed_game(monkeypatch,tmp_path):
 import github_actions_refresh as worker
 monkeypatch.setattr(worker,'ROOT',tmp_path);monkeypatch.setenv('MLB_STATE_DIR',str(tmp_path/'live'))
 for path in (tmp_path/'data/raw/games_2026.csv',tmp_path/'data/raw/statcast_enriched_2026.csv',tmp_path/'data/processed/features_arsenal_lineup_matchup_2026.csv'):
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text('x')
 state=tmp_path/'live/actions_refresh_state.json';state.parent.mkdir(parents=True);state.write_text(json.dumps({'season_refresh_date':'2026-08-23','completed_today':[]}))
 needed,_=worker._full_refresh_required('2026-08-23',[{'game_id':1,'status':'Final'}]);assert needed is False
 needed,_=worker._full_refresh_required('2026-08-24',[]);assert needed is True

def test_fresh_clone_fixture_reproduces_exact_predictions():
 import subprocess,sys
 result=subprocess.run([sys.executable,str(ROOT/'validate-v11-2-artifact-fixture.py')],cwd=ROOT,capture_output=True,text=True,check=True)
 summary=json.loads(result.stdout);assert summary['passed'] and summary['maximum_absolute_prediction_difference']==0

def test_final_snapshot_sql_remains_insert_once():
 text=(ROOT/'mlb_app/storage.py').read_text();assert 'ON CONFLICT (game_id) DO NOTHING' in text

def test_web_process_does_not_run_refresh_pipeline():
 render=(ROOT/'render.yaml').read_text();assert 'deployment_worker.py' not in render and 'refresh-v11-2-2026-features.py' not in render
