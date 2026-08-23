from pathlib import Path
from importlib.metadata import version
import hashlib,json,joblib
import numpy as np,pandas as pd
from scipy.stats import nbinom

class ModelArtifactError(RuntimeError):pass
class V112ModelService:
 def __init__(self,root):
  self.root=Path(root);self.meta=json.loads((self.root/'artifacts/v11_2_compact_metadata.json').read_text());path=self.root/'artifacts/v11_2_compact_pipeline.joblib'
  mismatched={name:(expected,version(name)) for name,expected in self.meta.get('runtime_versions',{}).items() if version(name)!=expected}
  if mismatched:raise ModelArtifactError(f'frozen model runtime version mismatch: {mismatched}')
  if hashlib.sha256(path.read_bytes()).hexdigest()!=self.meta['artifact_sha256']:raise ModelArtifactError('model artifact integrity check failed')
  self.pipeline=joblib.load(path);self.features=self.meta['features'];self.alpha=self.meta['negative_binomial']['dispersion_alpha_training_only']
 def validate_features(self,frame):
  missing=[x for x in self.features if x not in frame]
  if missing:raise ValueError('missing frozen features: '+', '.join(missing))
  return frame[self.features]
 def predict_team(self,row):
  x=self.validate_features(pd.DataFrame([row]));mu=max(.05,float(self.pipeline.predict(x)[0]));imp=self.pipeline.named_steps['imputer'];sc=self.pipeline.named_steps['scale'];model=self.pipeline.named_steps['model'];z=sc.transform(imp.transform(x))[0];items=[]
  for f,raw,standardized,coef in zip(self.features,x.iloc[0],z,model.coef_):items.append({'feature':f,'raw_value':None if pd.isna(raw) else float(raw),'standardized_value':float(standardized),'coefficient':float(coef),'log_mean_contribution':float(standardized*coef),'family':feature_family(f)})
  return {'expected_runs':mu,'variance':mu+self.alpha*mu*mu,'interval_50':self.interval(mu,.25,.75),'interval_80':self.interval(mu,.10,.90),'positive_factors':sorted(items,key=lambda q:q['log_mean_contribution'],reverse=True)[:5],'negative_factors':sorted(items,key=lambda q:q['log_mean_contribution'])[:5],'all_contributions':items}
 def validated_vector(self,row,cutoff):
  x=self.validate_features(pd.DataFrame([row]));raw=x.iloc[0];imp=self.pipeline.named_steps['imputer'];final=imp.transform(x)[0];records=[]
  for i,f in enumerate(self.features):
   records.append({'position':i,'feature':f,'raw_value':None if pd.isna(raw.iloc[i]) else float(raw.iloc[i]),'final_value':float(final[i]),'imputed':bool(pd.isna(raw.iloc[i])),'imputation_source':'frozen 2021-2024 training median' if pd.isna(raw.iloc[i]) else None,'feature_cutoff':cutoff,'validation_status':'valid'})
  values=np.asarray(final,dtype=float)
  if len(records)!=50 or len({r['feature'] for r in records})!=50 or not np.isfinite(values).all():raise ValueError('frozen feature-vector integrity validation failed')
  return records
 def interval(self,mu,lo,hi):
  size=1/self.alpha;p=size/(size+mu);return [int(nbinom.ppf(lo,size,p)),int(nbinom.ppf(hi,size,p))]
 def home_probability(self,away,home):
  k=np.arange(61);size=1/self.alpha
  def pmf(mu):
   p=size/(size+mu);x=nbinom.pmf(k,size,p);x[-1]+=max(0,1-x.sum());return x
  joint=np.outer(pmf(away),pmf(home));pa=np.tril(joint,-1).sum();ph=np.triu(joint,1).sum();return float(ph/(ph+pa))
def feature_family(f):
 z=f.removeprefix('opp_')
 if z.startswith(('sp_matchup','recent100','arsenal')):return 'Starting pitcher matchup'
 if z.startswith('sp_'):return 'Starting pitcher quality'
 if z.startswith('bp_'):return 'Bullpen'
 if z.startswith('lineup'):return 'Lineup / offense'
 if z.startswith(('ctx_venue','ctx_combined')):return 'Venue / context'
 if z.startswith('sit_'):return 'Team strength'
 if z.startswith('oq_'):return 'Quality-adjusted offense'
 return 'Lineup / offense'
def confidence_label(p):
 if p<.55:return 'LOW'
 if p<.60:return 'MODERATE'
 return 'HIGH'
