# MLB Forecasting Model — V11.2 Compact

A local, production-style Flask website for the frozen V11.2 compact team-run model. It presents model forecasts, uncertainty, historical out-of-sample performance, methodology, and feature-level explanations. It does not contain sportsbook odds, wagering recommendations, or generated placeholder predictions.

## Model contract

- Model: `PoissonRegressor(alpha=10)`
- Inputs: the exact 50-feature V11.2 compact specification
- Preprocessing: training-only median imputation and standardization
- Training window: 2021–2024
- Uncertainty: training-only negative-binomial dispersion, with `variance = mu + alpha * mu²`
- Historical evaluation: chronological out-of-sample forecasts

The production artifact builder serializes this already-frozen specification. It does not select features or tune the model.

## Install and run

```powershell
python -m pip install -r requirements.txt
python build-v11-2-production-artifact.py
python generate-site-assets.py
python generate-site-metric-audit.py
python run_site.py
```

Open <http://127.0.0.1:5000>. To run without refreshing the MLB schedule:

```powershell
python run_site.py --no-refresh
```

The launch command verifies the artifact checksum. If the binary artifact is absent, `run_site.py` rebuilds it from the frozen specification and existing local datasets.

## Live prediction data

The app downloads and caches the official MLB schedule. A forecast is emitted only when `data/live/features_YYYY-MM-DD.csv` contains two validated rows for the game (`team_side` equal to `away` and `home`) and all frozen feature columns are present. Until then the UI explicitly says that the forecast is pending. Starter changes and incomplete inputs therefore cannot silently produce a made-up forecast.

Live prediction JSON is saved under `data/live/`. This operational state is intentionally ignored by Git.

## Routes

- `/` — current schedule and forecast readiness
- `/game/<game_id>` — live forecast breakdown or a genuine historical OOS forecast
- `/methodology` — frozen model and leakage-control explanation
- `/performance` — verified metrics and calibration evidence
- `/history` — searchable chronological OOS archive
- `/about` — purpose and limitations
- `/api/games/today`, `/api/game/<id>`, `/api/model/performance`, `/api/model/metadata`, `/api/predictions/history`

## Verification

```powershell
python -m py_compile app.py run_site.py build-v11-2-production-artifact.py generate-site-assets.py generate-site-metric-audit.py
python -m pytest -q
```

`results/site_metric_source_audit.csv` records the source file for each headline metric displayed by the site. Historical forecasts come directly from `results/v11_2_confidence_oos_game_predictions_2022_2025.csv`.

## Security and reproducibility

- The joblib artifact is loaded only after SHA-256 verification against its metadata.
- Secrets and `.env` files are ignored; the application requires no secret keys.
- Binary artifacts and live/cache state are reproducible and excluded from source control.
- Inputs are selected in exact frozen feature order before inference.
- Missing values are handled only by the imputer fitted on 2021–2024.

## Limitations

Forecasts are uncertain estimates, not guarantees. Historical accuracy varies by season and confidence level. Lineups, probable starters, weather, postponements, and late roster news can change after a forecast is produced. The interface surfaces pending or insufficient-data states instead of filling gaps with invented values.
