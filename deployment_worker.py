"""Single deployment-safe refresh process for the frozen V11.2 website."""
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time

from mlb_app.model_service import V112ModelService
from mlb_app.refresh_service import refresh_cycle
from mlb_app.storage import acquire_worker_lock, database_url, initialize, load_state, save_state, worker_connection

ROOT = Path(__file__).resolve().parent
INTERVAL_SECONDS = max(60, int(os.getenv("REFRESH_INTERVAL_SECONDS", "600")))
_prepared_date = None


def prepare_current_season():
    global _prepared_date
    today = datetime.now(timezone.utc).date().isoformat()
    if _prepared_date == today:
        return
    year = datetime.now(timezone.utc).year
    required = ROOT / "data" / "raw" / f"games_{year}.csv"
    if year != 2026:
        raise RuntimeError("The exact live refresh bootstrap is currently frozen for the 2026 season.")
    # Refresh once per UTC day. Resumable API caches make subsequent starts cheap,
    # while the 10-minute loop below handles starters and confirmed lineups.
    subprocess.run([sys.executable, str(ROOT / "refresh-v11-2-2026-features.py")], cwd=ROOT, check=True)
    if not required.exists():
        raise RuntimeError("Current-season refresh did not produce the required game universe.")
    _prepared_date = today


def main():
    lock_connection = None
    if database_url():
        initialize()
        lock_connection = worker_connection()
        if not acquire_worker_lock(lock_connection):
            raise RuntimeError("Another live-refresh worker already holds the deployment lock.")
    service = V112ModelService(ROOT)
    while True:
        previous = load_state("refresh_status") or {}
        started = datetime.now(timezone.utc).isoformat()
        save_state("refresh_status", {"status": "running", "started_at": started, "last_successful_refresh": previous.get("last_successful_refresh")})
        try:
            prepare_current_season()
            payload = refresh_cycle(ROOT, service)
            save_state("refresh_status", {"status": "ok", "started_at": started, "last_successful_refresh": datetime.now(timezone.utc).isoformat(), "games": len(payload.get("games", [])), "refresh_seconds": payload.get("refresh_seconds")})
        except Exception as exc:
            save_state("refresh_status", {"status": "error", "started_at": started, "last_successful_refresh": previous.get("last_successful_refresh"), "message": str(exc)[:500]})
            print(f"Live refresh failed safely: {exc}", flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
