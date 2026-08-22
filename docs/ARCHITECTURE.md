# Site architecture

The application has four deliberately separate layers:

1. `build-v11-2-production-artifact.py` serializes the exact frozen V11.2 pipeline and records its SHA-256 digest and inference metadata.
2. `mlb_app/model_service.py` validates the digest and 50-feature contract, serves expected runs, derives negative-binomial uncertainty, and calculates non-tie winner probabilities.
3. `mlb_app/live_pipeline.py` obtains the official schedule and joins only validated daily feature snapshots. Missing features or starters produce explicit pending states.
4. `app.py` exposes HTML and JSON routes. `mlb_app/performance.py` sources descriptive metrics directly from preserved result files.

The frontend is server-rendered HTML with a small progressive-enhancement script. This keeps pages useful without JavaScript and makes failure states visible rather than hiding them behind client-side loading.

## Data flow

`official schedule -> validated daily features -> checksum-verified frozen pipeline -> expected team runs -> uncertainty/winner derivation -> cached JSON -> HTML/API`

Historical pages bypass live inference and read the preserved chronological OOS ledger. Outcomes and forecasts remain separate fields.

## Operational boundary

This package is the presentation and inference boundary. It does not tune, recalibrate, select features, scrape unneeded data, or silently substitute full-season/future values. A separate upstream pregame process must create a correctly keyed daily feature snapshot before a forecast becomes ready.
