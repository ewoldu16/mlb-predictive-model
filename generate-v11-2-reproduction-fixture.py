"""Create a tiny fresh-clone artifact-reproduction fixture from preserved OOS games."""
from pathlib import Path
import importlib.util,json
import pandas as pd
from mlb_app.model_service import V112ModelService
ROOT=Path(__file__).resolve().parent
def main():
 spec=importlib.util.spec_from_file_location('v11',ROOT/'evaluate-v11-unified-team-runs.py');v11=importlib.util.module_from_spec(spec);spec.loader.exec_module(v11);service=V112ModelService(ROOT);cases=[]
 for game_id in (777007,661042,746776):
  year=int(pd.read_csv(ROOT/'results/v11_2_confidence_oos_game_predictions_2022_2025.csv').set_index('game_id').loc[game_id,'season']);game=v11.long_year(year).query('game_id == @game_id').sort_values('team_side');pred={}
  for row in game.itertuples(index=False):
   raw={feature:(None if pd.isna(getattr(row,feature)) else float(getattr(row,feature))) for feature in service.features};expected=float(service.predict_team(raw)['expected_runs']);pred[row.team_side]=expected;cases.append({'game_id':game_id,'season':year,'team_side':row.team_side,'features':raw,'expected_runs':expected})
  hp=service.home_probability(pred['away'],pred['home'])
  for case in cases[-2:]:case['expected_home_win_probability']=hp
 output={'model_version':service.meta['model_version'],'artifact_sha256':service.meta['artifact_sha256'],'tolerance':1e-12,'cases':cases};(ROOT/'artifacts/v11_2_reproduction_fixture.json').write_text(json.dumps(output,indent=2,allow_nan=False),encoding='utf-8')
if __name__=='__main__':main()
