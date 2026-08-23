from runtime_import_preflight import RUNTIME_IMPORTS, audit_imports


def test_refresh_runtime_import_manifest_covers_statsapi():
    assert RUNTIME_IMPORTS["statsapi"] == "MLB-StatsAPI"


def test_import_audit_returns_all_missing_modules():
    def missing(module):
        if module in {"statsapi", "psycopg"}:
            raise ModuleNotFoundError(name=module)
        return object()

    result = audit_imports(missing)
    assert result["status"] == "failed"
    assert {row["module"] for row in result["missing_modules"]} == {
        "statsapi", "psycopg"
    }


def test_requirements_declares_every_refresh_distribution():
    from pathlib import Path

    requirements = (Path(__file__).parents[1] / "requirements.txt").read_text().lower()
    for distribution in set(RUNTIME_IMPORTS.values()):
        assert distribution.split("[")[0].lower() in requirements
