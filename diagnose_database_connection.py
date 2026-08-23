"""Sanitized database configuration audit and connectivity preflight."""
from __future__ import annotations

import json
import os
import sys
from urllib.parse import unquote, urlsplit

from mlb_app.storage import database_url


def selected_variable() -> str:
    if os.getenv("SUPABASE_DATABASE_URL"):
        return "SUPABASE_DATABASE_URL"
    if os.getenv("DATABASE_URL"):
        return "DATABASE_URL (legacy alias)"
    return "none"


def safe_metadata(url: str) -> dict[str, object]:
    parsed = urlsplit(url)
    username = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    database = unquote(parsed.path.lstrip("/").split("/", 1)[0])
    return {
        "SUPABASE_DATABASE_URL_present": bool(os.getenv("SUPABASE_DATABASE_URL")),
        "DATABASE_URL_present": bool(os.getenv("DATABASE_URL")),
        "selected_configuration": selected_variable(),
        "scheme": parsed.scheme or None,
        "hostname": parsed.hostname or None,
        "port": parsed.port,
        "database_name": database or None,
        "username_format": (
            "postgres.<project_ref>"
            if username.startswith("postgres.") and len(username) > len("postgres.")
            else "bare_postgres" if username == "postgres"
            else "other" if username else "missing"
        ),
        "username_has_project_ref": username.startswith("postgres.")
        and len(username) > len("postgres."),
        "password_present": bool(password),
        "password_character_count": len(password),
        "leading_or_trailing_whitespace": url != url.strip(),
    }


def sanitized_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "password authentication failed" in message:
        return "password authentication failed for the selected username format"
    if "could not translate host name" in message or "name or service not known" in message:
        return "database hostname could not be resolved"
    if "timeout" in message:
        return "database connection timed out"
    if "connection refused" in message:
        return "database connection was refused"
    return "database connection failed; credentials and URL are intentionally redacted"


def main() -> int:
    try:
        url = database_url()
    except Exception as exc:
        print(json.dumps({
            "database_connection": "failed_before_connect",
            "configuration_error": str(exc),
            "SUPABASE_DATABASE_URL_present": bool(os.getenv("SUPABASE_DATABASE_URL")),
            "DATABASE_URL_present": bool(os.getenv("DATABASE_URL")),
        }, indent=2))
        return 1
    if not url:
        print(json.dumps({
            "database_connection": "failed_before_connect",
            "configuration_error": "No database URL is configured",
            "SUPABASE_DATABASE_URL_present": False,
            "DATABASE_URL_present": False,
        }, indent=2))
        return 1
    try:
        metadata = safe_metadata(url)
    except Exception:
        print(json.dumps({
            "database_connection": "failed_before_connect",
            "selected_configuration": selected_variable(),
            "configuration_error": "Selected database URL could not be parsed",
        }, indent=2))
        return 1
    print(json.dumps(metadata, indent=2))
    try:
        import psycopg
        with psycopg.connect(url, connect_timeout=15) as connection:
            row = connection.execute("SELECT 1").fetchone()
            if not row or row[0] != 1:
                raise RuntimeError("Database probe returned an unexpected result")
    except Exception as exc:
        print(json.dumps({
            "database_connection": "failed",
            "selected_configuration": selected_variable(),
            "sanitized_error": sanitized_failure(exc),
            "exception_type": type(exc).__name__,
        }, indent=2))
        return 1
    print(json.dumps({
        "database_connection": "success",
        "selected_configuration": selected_variable(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
