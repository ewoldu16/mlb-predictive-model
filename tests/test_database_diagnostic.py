from diagnose_database_connection import safe_metadata, sanitized_failure


def test_safe_metadata_never_exposes_credentials(monkeypatch):
    secret = "NeverPrintThis"
    project = "projectref123"
    url = f"postgresql://postgres.{project}:{secret}@example.invalid:6543/postgres"
    monkeypatch.setenv("SUPABASE_DATABASE_URL", url)
    metadata = safe_metadata(url)
    rendered = str(metadata)
    assert secret not in rendered
    assert project not in rendered
    assert metadata["username_format"] == "postgres.<project_ref>"
    assert metadata["password_character_count"] == len(secret)


def test_password_failure_is_sanitized():
    message = sanitized_failure(
        RuntimeError('password authentication failed for user "postgres.secretref"')
    )
    assert "secretref" not in message
    assert message == "password authentication failed for the selected username format"
