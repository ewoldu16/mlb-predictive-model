import json

import numpy as np
import pandas as pd

from mlb_app.storage import json_payload, normalize_json_value


def test_recursive_nonfinite_normalization():
    payload = {
        "python_nan": float("nan"),
        "numpy_nan": np.nan,
        "numpy_scalar_nan": np.float64(np.nan),
        "positive_infinity": float("inf"),
        "negative_infinity": float("-inf"),
        "pandas_na": pd.NA,
        "pandas_nat": pd.NaT,
        "nested": [1.25, {"missing": np.float64(np.nan)}, (2, np.inf)],
        "finite_numpy": np.float64(3.5),
    }
    normalized = normalize_json_value(payload)
    assert normalized == {
        "python_nan": None,
        "numpy_nan": None,
        "numpy_scalar_nan": None,
        "positive_infinity": None,
        "negative_infinity": None,
        "pandas_na": None,
        "pandas_nat": None,
        "nested": [1.25, {"missing": None}, [2, None]],
        "finite_numpy": 3.5,
    }
    assert json.loads(json_payload(payload)) == normalized
    assert "NaN" not in json_payload(payload)
    assert "Infinity" not in json_payload(payload)


def test_save_state_sends_strict_json_null_to_supabase(monkeypatch):
    import mlb_app.storage as storage

    calls = []

    class Connection:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, sql, parameters=None):
            calls.append((sql, parameters))

    monkeypatch.setattr(storage, "database_url", lambda: "postgresql://configured")
    monkeypatch.setattr(storage, "initialize", lambda: True)
    monkeypatch.setattr(storage, "_connect", lambda: Connection())
    assert storage.save_state("tracking", {"accuracy": np.nan, "games": 0})
    persisted = calls[-1][1][1]
    assert json.loads(persisted) == {"accuracy": None, "games": 0}
    assert "NaN" not in persisted
