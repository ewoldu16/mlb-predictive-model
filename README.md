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

## Free-tier public deployment

The production design targets $0/month using three independently replaceable
services:

1. One Render **Free** Gunicorn web service from `render.yaml`.
2. Supabase Free PostgreSQL for operational state and immutable snapshots.
3. GitHub Actions scheduled/manual one-shot refresh jobs.

The Render process serves pages and owner mutations only. It never downloads
Statcast or runs feature builders. Owner edits are committed immediately to
Supabase and create a queued rebuild request. The next Action processes that
request; the website communicates an expected delay of roughly 15 minutes without
promising an exact execution time.

`.github/workflows/mlb-live-refresh.yml` runs every 30 minutes from 12:00–15:59
UTC, every 15 minutes from 16:00–05:59 UTC, only during March–November, plus
manual `workflow_dispatch`. GitHub cron is UTC and may run late. A versioned
daily Actions cache retains only current-season source chunks, normalized
2026 tables, processed 2026 feature layers, live snapshots, and tracking state.

The first cache miss performs the resumable current-season bootstrap. Later jobs
run the full season refresh only when the UTC date changes or today's completed
game universe advances. Routine lineup polls do not redownload the season.

Supabase is accessed through its PostgreSQL connection string; no Supabase browser
key is exposed. Final snapshots use insert-once `ON CONFLICT DO NOTHING` semantics.

### Required secrets

GitHub repository Actions secret:

- `SUPABASE_DATABASE_URL`

Render environment variables:

- `SUPABASE_DATABASE_URL`
- `OWNER_USERNAME`
- `OWNER_PASSWORD_HASH`
- `SECRET_KEY` (the Blueprint can generate this)
- `OWNER_SESSION_MINUTES` (defaults to 30)

No RotoWire key is required. The adapter remains optional and inactive without a
key; owner-managed provisional orders are the supported free fallback.

### Deployment sequence

1. Create a Supabase Free project and copy its server-side PostgreSQL connection
   string, preferably the documented pooler URL suitable for transient clients.
2. Add it to GitHub Actions as `SUPABASE_DATABASE_URL`.
3. Run **MLB live refresh** manually once. This creates/migrates the schema and
   warms the current-season cache.
4. Create the single Render Free web service from `render.yaml`.
5. Set the Render variables above using the same Supabase connection string.
6. Open `/health` and confirm `persistence=supabase_postgres`,
   `refresh_executor=github_actions`, and a recent successful refresh.

Free tiers, quotas, sleep policies, and Actions policies can change. See
`results/deployment_free_tier_report.md` for limits and cost assumptions.

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
