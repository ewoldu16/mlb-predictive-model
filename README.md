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

The launch command verifies the artifact checksum and fails loudly if the exact committed production artifact is absent.

## Live prediction data

The app downloads and caches the official MLB schedule. A forecast is emitted only when `data/live/features_YYYY-MM-DD.csv` contains two validated rows for the game (`team_side` equal to `away` and `home`) and all frozen feature columns are present. Until then the UI explicitly says that the forecast is pending. Starter changes and incomplete inputs therefore cannot silently produce a made-up forecast.

Live prediction JSON is saved under `data/live/`. This operational state is intentionally ignored by Git.

## Routes

- `/` — current schedule and forecast readiness
- `/game/<game_id>` — live forecast breakdown or a genuine historical OOS forecast
- `/methodology` — frozen model and leakage-control explanation
- `/performance` — verified metrics and calibration evidence
- `/history` — searchable chronological OOS archive
- `/live-tracking` — prospective predictions recorded before first pitch
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
- All model binaries except the exact 5 KB frozen production pipeline are excluded; live/cache state also remains excluded.
- Inputs are selected in exact frozen feature order before inference.
- Missing values are handled only by the imputer fitted on 2021–2024.

## Limitations

Forecasts are uncertain estimates, not guarantees. Historical accuracy varies by season and confidence level. Lineups, probable starters, weather, postponements, and late roster news can change after a forecast is produced. The interface surfaces pending or insufficient-data states instead of filling gaps with invented values.

## Local development

The frozen production model is now supplied directly by the repository at
`artifacts/v11_2_compact_pipeline.joblib`. The file is only about 5 KB, contains
no raw source data, and is verified against the SHA-256 digest in
`artifacts/v11_2_compact_metadata.json` before it is loaded. A fresh clone does
not rebuild the model or require the 2021-2024 training datasets.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_site.py --no-refresh
```

For a Linux production-style web smoke test:

```bash
gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 120 app:app
```

`run_site.py` remains a local launcher. It respects `PORT` and `HOST`; Render
does not use Flask's development server.

## Public deployment

The project is prepared for Render through `render.yaml`, which declares:

1. `mlb-v11-2-web`, a Gunicorn web service.
2. `mlb-v11-2-refresh`, one dedicated 10-minute refresh worker with a 10 GB
   persistent disk mounted at `/opt/render/project/src/data`.
3. `mlb-v11-2-db`, a small Render Postgres database shared by web and worker.

The worker alone owns schedule and lineup polling. A PostgreSQL advisory lock
prevents duplicate workers. It refreshes current-season public source state once
per UTC day and checks starter and confirmed-lineup readiness every 600 seconds.

Postgres stores the latest daily payload, refresh status, and immutable per-game
prediction snapshots. Snapshot insertion uses `ON CONFLICT DO NOTHING`, so a
restart or deployment cannot replace an existing forecast. The worker disk stores
resumable public MLB/Statcast downloads and feature caches.

No 2021-2025 raw research dataset is required in deployment. Initial worker start
does acquire the current 2026 regular-season public data required by the exact
feature definitions, so the first bootstrap can take substantially longer than
later resumable refreshes.

### Exact Render steps

1. Commit and push these deployment changes.
2. In Render, select **New > Blueprint**.
3. Connect `ewoldu16/mlb-predictive-model` and select the repository root.
4. Review and apply the three resources from `render.yaml`.
5. Do not manually enter `DATABASE_URL`; the Blueprint supplies the internal
   connection string to both services.
6. Wait for the web health check and the worker's initial current-season refresh.
7. Open `https://<your-render-host>/health`. Confirm `model_loaded` and
   `artifact_verified` are `true`; `refresh_status` should become `ok` after a
   successful worker cycle.

Render supplies `PORT` automatically. No MLB API key is required. Safe local
configuration examples are listed in `.env.example`; never commit a real `.env`.

The `/health` endpoint reports model readiness, model version, artifact integrity,
persistence mode, last successful refresh, and current refresh status without
exposing paths, credentials, or database details.

## Probable-lineup forecasts

The optional `ROTOWIRE_API_KEY` enables RotoWire's documented MLB Projected
Lineups API. Before official MLB lineups are posted, a complete projected order
whose players map uniquely to active MLB roster IDs can generate a clearly marked
`PROVISIONAL_PREDICTION`. No lineup is inferred from a previous game or fabricated.

When both official MLB boxscore lineups become available, the exact lineup-driven
features are rebuilt and a separate immutable `FINAL_PREGAME_PREDICTION` is
created. Expected-run and winner-probability changes, plus identifiable lineup
substitutions, are retained. Only the final confirmed-lineup snapshot is eligible
for prospective live-performance tracking.

## Owner-managed provisional lineups

The private `/owner` route can seed a provisional order from each team's most
recent completed-game lineup, then restrict replacements to the current MLB
active roster. The previous lineup is labelled as a template, never as an
official or independently projected lineup. Configure `OWNER_USERNAME`, a
Werkzeug `OWNER_PASSWORD_HASH`, and `SECRET_KEY`; no credentials are committed.

Only `AVAILABLE` players are eligible. Saving any other availability state
removes that player immediately, leaves the batting-order position empty, and
fails closed until the owner selects an eligible replacement. Valid orders must
contain nine unique active-roster MLB IDs in positions 1–9.

Once both orders validate, the normal current-day assembler stages those exact
MLB player IDs and reruns the existing frozen feature builders. The frozen model,
50-feature contract, preprocessing, coefficients, and uncertainty settings are
unchanged. Owner snapshots are provisional and versioned; confirmed MLB lineups
remain authoritative and create the separate immutable final forecast.

Team Offense Confidence is descriptive only. It is calculated as
`45% × previous-game starters retained + 20% × known defensive-position coverage
+ 35% × observed (non-imputed) frozen lineup-feature coverage`, labelled HIGH at
85%+, MODERATE at 65–84.99%, and LOW below 65%. It never enters V11.2.
