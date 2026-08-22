"""Serialize the already-frozen V11.2 specification; performs no model selection."""
from pathlib import Path
import importlib.util,json,hashlib,joblib
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent;R=ROOT/'results';OUT=ROOT/'artifacts';OUT.mkdir(exist_ok=True)
def load_v11():
 s=importlib.util.spec_from_file_location('v11',ROOT/'evaluate-v11-unified-team-runs.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 spec=json.loads((R/'v11_2_compact_frozen_specification.json').read_text());features=spec['features'];v=load_v11();data=pd.concat([v.long_year(y) for y in range(2021,2025)],ignore_index=True)
 pipe=Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',PoissonRegressor(alpha=10,max_iter=3000))]);pipe.fit(data[features],data.actual_team_runs);mu=np.clip(pipe.predict(data[features]),.05,None);alpha=max(1e-9,float(np.sum((data.actual_team_runs.to_numpy()-mu)**2-data.actual_team_runs.to_numpy())/np.sum(mu**2)))
 artifact=OUT/'v11_2_compact_pipeline.joblib';joblib.dump(pipe,artifact);digest=hashlib.sha256(artifact.read_bytes()).hexdigest();meta={'model_version':'V11.2_COMPACT_TEAM_RUN','artifact_format':'joblib','artifact_sha256':digest,'features':features,'feature_count':len(features),'poisson_alpha':10,'training_seasons':[2021,2022,2023,2024],'training_team_games':len(data),'imputation':'median fitted on 2021-2024','scaling':'standard fitted on 2021-2024','prediction_floor':.05,'negative_binomial':{'parameterization':'variance=mu+alpha*mu^2','dispersion_alpha_training_only':alpha},'frozen_specification':'results/v11_2_compact_frozen_specification.json'};(OUT/'v11_2_compact_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'artifact':str(artifact),'sha256':digest,'nb_alpha':alpha,'features':len(features)},indent=2))
if __name__=='__main__':main()

