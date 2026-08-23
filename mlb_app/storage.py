"""Persistence for live operational state and immutable prediction snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os

_initialized = False


def database_url():
    """Return the configured DSN verbatim and reject ambiguous configuration."""
    supabase_url = os.getenv("SUPABASE_DATABASE_URL") or None
    legacy_url = os.getenv("DATABASE_URL") or None
    if supabase_url and legacy_url and supabase_url != legacy_url:
        raise RuntimeError(
            "Conflicting SUPABASE_DATABASE_URL and DATABASE_URL values are set"
        )
    selected = supabase_url or legacy_url or ""
    if selected and selected != selected.strip():
        raise RuntimeError("Database URL contains leading or trailing whitespace")
    return selected


def _connect():
    import psycopg
    return psycopg.connect(database_url())


def initialize():
    global _initialized
    if not database_url():
        return False
    if _initialized:
        return True
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_snapshots (
                game_id BIGINT PRIMARY KEY,
                game_date DATE NOT NULL,
                payload JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS operational_state (
                state_key TEXT PRIMARY KEY,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS provisional_prediction_snapshots (
                game_id BIGINT PRIMARY KEY,
                game_date DATE NOT NULL,
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS owner_lineups (
                game_id BIGINT NOT NULL,
                team_side TEXT NOT NULL CHECK (team_side IN ('away','home')),
                payload JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (game_id, team_side)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS owner_player_availability (
                game_id BIGINT NOT NULL,
                team_side TEXT NOT NULL CHECK (team_side IN ('away','home')),
                player_id BIGINT NOT NULL,
                availability TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (game_id, team_side, player_id)
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS rebuild_requests (
                game_id BIGINT PRIMARY KEY,
                requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status TEXT NOT NULL,
                reason TEXT,
                payload JSONB NOT NULL,
                completed_at TIMESTAMPTZ
            )
        """)
        connection.execute("""
            DO $$ BEGIN
                IF to_regclass('public.owner_lineup_state') IS NOT NULL THEN
                    INSERT INTO owner_lineups(game_id,team_side,payload,updated_at)
                    SELECT game_id,team_side,payload,updated_at FROM owner_lineup_state
                    ON CONFLICT(game_id,team_side) DO NOTHING;
                END IF;
            END $$
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS owner_audit_log (
                audit_id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                game_id BIGINT NOT NULL,
                team_side TEXT,
                owner_id TEXT NOT NULL,
                payload JSONB NOT NULL
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS provisional_prediction_history (
                history_id BIGSERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                game_id BIGINT NOT NULL,
                game_date DATE NOT NULL,
                lineup_version BIGINT,
                payload JSONB NOT NULL
            )
        """)
    _initialized = True
    return True


def save_snapshot(game):
    """Insert once. Existing snapshots are deliberately never overwritten."""
    if not database_url():
        return False
    initialize()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO prediction_snapshots(game_id, game_date, payload) "
            "VALUES (%s, %s, %s::jsonb) ON CONFLICT (game_id) DO NOTHING",
            (int(game["game_id"]), game["date"], json.dumps(game, allow_nan=False)),
        )
    return True


def load_snapshot(game_id):
    if not database_url():
        return None
    initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT payload FROM prediction_snapshots WHERE game_id=%s", (int(game_id),)
        ).fetchone()
    return row[0] if row else None


def load_snapshots_for_date(game_date):
    if not database_url():return []
    initialize()
    with _connect() as connection:rows=connection.execute("SELECT payload FROM prediction_snapshots WHERE game_date=%s ORDER BY game_id",(game_date,)).fetchall()
    return [row[0] for row in rows]


def save_provisional_snapshot(game):
    """Provisional forecasts may update before first pitch; finals never do."""
    if not database_url(): return False
    initialize()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO provisional_prediction_snapshots(game_id, game_date, payload, updated_at) VALUES (%s,%s,%s::jsonb,NOW()) "
            "ON CONFLICT (game_id) DO UPDATE SET payload=EXCLUDED.payload, updated_at=NOW()",
            (int(game['game_id']),game['date'],json.dumps(game,allow_nan=False)))
    return True


def load_provisional_snapshot(game_id):
    if not database_url(): return None
    initialize()
    with _connect() as connection:row=connection.execute("SELECT payload FROM provisional_prediction_snapshots WHERE game_id=%s",(int(game_id),)).fetchone()
    return row[0] if row else None


def save_owner_lineup(game_id, team_side, payload):
    if not database_url(): return False
    initialize()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO owner_lineups(game_id,team_side,payload,updated_at) VALUES (%s,%s,%s::jsonb,NOW()) "
            "ON CONFLICT(game_id,team_side) DO UPDATE SET payload=EXCLUDED.payload,updated_at=NOW()",
            (int(game_id),team_side,json.dumps(payload,allow_nan=False)))
        for player_id,status in payload.get('availability',{}).items():
            connection.execute("INSERT INTO owner_player_availability(game_id,team_side,player_id,availability,updated_at) VALUES (%s,%s,%s,%s,NOW()) ON CONFLICT(game_id,team_side,player_id) DO UPDATE SET availability=EXCLUDED.availability,updated_at=NOW()",(int(game_id),team_side,int(player_id),status))
    return True


def load_owner_lineup(game_id, team_side):
    if not database_url(): return None
    initialize()
    with _connect() as connection:
        row=connection.execute("SELECT payload FROM owner_lineups WHERE game_id=%s AND team_side=%s",(int(game_id),team_side)).fetchone()
    return row[0] if row else None


def append_owner_audit(entry):
    if not database_url(): return False
    initialize()
    with _connect() as connection:
        connection.execute("INSERT INTO owner_audit_log(game_id,team_side,owner_id,payload) VALUES (%s,%s,%s,%s::jsonb)",(int(entry['game_id']),entry.get('team_side'),entry['owner_id'],json.dumps(entry,allow_nan=False)))
    return True


def owner_audit_rows(game_id=None):
    if not database_url(): return []
    initialize();query="SELECT payload FROM owner_audit_log";args=()
    if game_id is not None:query+=" WHERE game_id=%s";args=(int(game_id),)
    query+=" ORDER BY audit_id"
    with _connect() as connection:rows=connection.execute(query,args).fetchall()
    return [row[0] for row in rows]


def append_provisional_history(game):
    if not database_url():
        from pathlib import Path
        root=Path(os.getenv('MLB_STATE_DIR','data/live'))/'owner'/'provisional_history'/str(int(game['game_id']));root.mkdir(parents=True,exist_ok=True)
        stamp=str(game.get('snapshot',{}).get('generated_at') or datetime.now(timezone.utc).isoformat()).replace(':','-')
        path=root/f"version_{game.get('owner_lineup_version') or 0}_{stamp}.json";path.write_text(json.dumps(game,indent=2,allow_nan=False),encoding='utf-8');return True
    initialize()
    with _connect() as connection:
        connection.execute("INSERT INTO provisional_prediction_history(game_id,game_date,lineup_version,payload) VALUES (%s,%s,%s,%s::jsonb)",(int(game['game_id']),game['date'],game.get('owner_lineup_version'),json.dumps(game,allow_nan=False)))
    return True


def queue_rebuild_request(game_id,payload,reason='owner_lineup_changed'):
    if not database_url():return save_state('owner_rebuild_request:'+str(int(game_id)),{'status':'queued','reason':reason,**payload})
    initialize()
    with _connect() as connection:
        connection.execute("INSERT INTO rebuild_requests(game_id,status,reason,payload,requested_at,completed_at) VALUES (%s,'queued',%s,%s::jsonb,NOW(),NULL) ON CONFLICT(game_id) DO UPDATE SET status='queued',reason=EXCLUDED.reason,payload=EXCLUDED.payload,requested_at=NOW(),completed_at=NULL",(int(game_id),reason,json.dumps(payload,allow_nan=False)))
    return True


def pending_rebuild_requests():
    if not database_url():return []
    initialize()
    with _connect() as connection:rows=connection.execute("SELECT game_id,payload,reason,requested_at FROM rebuild_requests WHERE status='queued' ORDER BY requested_at").fetchall()
    return [{'game_id':int(row[0]),'payload':row[1],'reason':row[2],'requested_at':row[3].isoformat()} for row in rows]


def load_rebuild_request(game_id):
    if not database_url():return load_state('owner_rebuild_request:'+str(int(game_id)))
    initialize()
    with _connect() as connection:row=connection.execute("SELECT status,reason,payload,requested_at,completed_at FROM rebuild_requests WHERE game_id=%s",(int(game_id),)).fetchone()
    return {'game_id':int(game_id),'status':row[0],'reason':row[1],'payload':row[2],'requested_at':row[3].isoformat(),'completed_at':row[4].isoformat() if row[4] else None} if row else None


def finish_rebuild_request(game_id,status='complete',reason=None):
    if not database_url():return save_state('owner_rebuild_request:'+str(int(game_id)),{'game_id':int(game_id),'status':status,'reason':reason,'completed_at':datetime.now(timezone.utc).isoformat()})
    initialize()
    with _connect() as connection:connection.execute("UPDATE rebuild_requests SET status=%s,reason=COALESCE(%s,reason),completed_at=NOW() WHERE game_id=%s",(status,reason,int(game_id)))
    return True


def save_state(key, payload):
    if not database_url():
        return False
    initialize()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO operational_state(state_key, payload, updated_at) "
            "VALUES (%s, %s::jsonb, NOW()) ON CONFLICT (state_key) DO UPDATE "
            "SET payload=EXCLUDED.payload, updated_at=NOW()",
            (key, json.dumps(payload, allow_nan=False)),
        )
    return True


def load_state(key):
    if not database_url():
        return None
    initialize()
    with _connect() as connection:
        row = connection.execute(
            "SELECT payload FROM operational_state WHERE state_key=%s", (key,)
        ).fetchone()
    return row[0] if row else None


def health_state():
    state = load_state("refresh_status") if database_url() else None
    return state or {
        "status": "not_started",
        "last_successful_refresh": None,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def acquire_worker_lock(connection, lock_id=112002026):
    """Hold a PostgreSQL session advisory lock for one refresh worker."""
    return bool(connection.execute("SELECT pg_try_advisory_lock(%s)", (lock_id,)).fetchone()[0])


def worker_connection():
    return _connect()
