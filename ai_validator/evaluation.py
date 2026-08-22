"""Outcome evaluation helpers. Not invoked until genuine external decisions are frozen."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr

def status_error_analysis(joined):
 x=joined.copy();x['absolute_error']=abs(x.actual_team_runs-x.predicted_runs);x['squared_error']=(x.actual_team_runs-x.predicted_runs)**2;x['signed_error']=x.actual_team_runs-x.predicted_runs
 return x.groupby('status').agg(observations=('absolute_error','size'),mae=('absolute_error','mean'),median_ae=('absolute_error','median'),rmse=('squared_error',lambda z:np.sqrt(z.mean())),mean_signed_error=('signed_error','mean'),p90_ae=('absolute_error',lambda z:z.quantile(.9)),p95_ae=('absolute_error',lambda z:z.quantile(.95))).reset_index()
def confidence_analysis(joined):
 x=joined.copy();x['absolute_error']=abs(x.actual_team_runs-x.predicted_runs);x['extreme_error']=x.absolute_error.ge(5);x['confidence_bucket']=pd.cut(x.confidence_score,[0,20,40,60,80,100],include_lowest=True)
 table=x.groupby('confidence_bucket',observed=True).agg(observations=('absolute_error','size'),mae=('absolute_error','mean'),rmse=('absolute_error',lambda z:np.sqrt(np.mean(z**2))),extreme_error_rate=('extreme_error','mean')).reset_index();return table,spearmanr(x.confidence_score,-x.absolute_error).statistic

