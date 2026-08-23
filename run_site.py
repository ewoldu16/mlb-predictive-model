from pathlib import Path
import argparse
import os
from mlb_app.model_service import V112ModelService
from mlb_app.live_pipeline import generate_predictions
from mlb_app.refresh_service import refresh_cycle
ROOT=Path(__file__).resolve().parent
def ensure_artifact():
 if not (ROOT/'artifacts/v11_2_compact_pipeline.joblib').exists():
  raise FileNotFoundError('Frozen production artifact is missing. Restore artifacts/v11_2_compact_pipeline.joblib from the repository.')
def main():
 p=argparse.ArgumentParser();p.add_argument('--date');p.add_argument('--no-refresh',action='store_true');p.add_argument('--port',type=int,default=int(os.getenv('PORT','5000')));p.add_argument('--refresh-minutes',type=float,default=10);args=p.parse_args();ensure_artifact();service=V112ModelService(ROOT)
 if not args.no_refresh:
  payload=refresh_cycle(ROOT,service,args.date);ready=sum(x['forecast_status']=='ready' for x in payload['games']);print(f"Schedule {payload['date']}: {len(payload['games'])} games, {ready} forecasts ready ({payload['refresh_seconds']:.1f}s).")
  print('Inline refresh is intended for local development only. Production uses deployment_worker.py.')
 from app import create_app
 create_app().run(host=os.getenv('HOST','127.0.0.1'),port=args.port,debug=False)
if __name__=='__main__':main()
