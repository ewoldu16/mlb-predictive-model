"""Fresh-clone-safe exact prediction gate for the frozen production artifact."""
from pathlib import Path
import json
from mlb_app.model_service import V112ModelService
ROOT=Path(__file__).resolve().parent
def main():
 fixture=json.loads((ROOT/'artifacts/v11_2_reproduction_fixture.json').read_text());service=V112ModelService(ROOT)
 if fixture['artifact_sha256']!=service.meta['artifact_sha256'] or fixture['model_version']!=service.meta['model_version']:raise SystemExit('fixture/artifact identity mismatch')
 diffs=[]
 for case in fixture['cases']:diffs.append(abs(service.predict_team(case['features'])['expected_runs']-case['expected_runs']))
 grouped={}
 for case in fixture['cases']:grouped.setdefault(case['game_id'],{})[case['team_side']]=case
 for sides in grouped.values():diffs.append(abs(service.home_probability(sides['away']['expected_runs'],sides['home']['expected_runs'])-sides['home']['expected_home_win_probability']))
 maximum=max(diffs);summary={'cases':len(fixture['cases']),'maximum_absolute_prediction_difference':maximum,'tolerance':fixture['tolerance'],'passed':maximum<=fixture['tolerance']};print(json.dumps(summary,indent=2))
 if not summary['passed']:raise SystemExit('frozen artifact reproduction failed')
if __name__=='__main__':main()
