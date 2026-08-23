"""Fail-fast import audit for the GitHub Actions live-refresh runtime."""
from __future__ import annotations

import importlib
import json
import sys


# Direct third-party modules reachable from the Actions refresh, its subprocess
# scripts, local application modules, and the serialized model runtime.
RUNTIME_IMPORTS = {
    "joblib": "joblib",
    "numpy": "numpy",
    "pandas": "pandas",
    "psycopg": "psycopg[binary]",
    "pybaseball": "pybaseball",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "statsapi": "MLB-StatsAPI",
}


def audit_imports(importer=importlib.import_module) -> dict[str, object]:
    imported: list[str] = []
    missing: list[dict[str, str]] = []
    broken: list[dict[str, str]] = []
    for module, distribution in RUNTIME_IMPORTS.items():
        try:
            importer(module)
            imported.append(module)
        except ModuleNotFoundError as exc:
            missing.append({
                "module": module,
                "distribution": distribution,
                "error": f"ModuleNotFoundError: {exc.name or module}",
            })
        except Exception as exc:
            broken.append({
                "module": module,
                "distribution": distribution,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            })
    return {
        "status": "ok" if not missing and not broken else "failed",
        "python": sys.version.split()[0],
        "required_modules": len(RUNTIME_IMPORTS),
        "imported_modules": imported,
        "missing_modules": missing,
        "broken_modules": broken,
    }


def main() -> int:
    result = audit_imports()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
