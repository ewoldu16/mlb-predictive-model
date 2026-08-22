"""Outcome-only confidence diagnostic for frozen V11.2; no sportsbook data."""
from pathlib import Path
import importlib.util,json
import numpy as np
import pandas as pd
from scipy.stats import nbinom,spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor,LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent;R=ROOT/'results';SEED=20260822;BOOT=10000
FOLDS=[([2021],2022),([2021,2022],2023),([2021,2022,2023],2024),([2021,2022,2023,2024],2025)]
PROB_EDGES=[.5,.525,.55,.575,.60,.625,.65,.70,1.000001]
PROB_LABELS=['50-52.5%','52.5-55%','55-57.5%','57.5-60%','60-62.5%','62.5-65%','65-70%','70%+']
DIFF_EDGES=[0,.10,.25,.50,.75,1.0,1.5,2.0,np.inf]
DIFF_LABELS=['0-.10','.10-.25','.25-.50','.50-.75','.75-1.0','1.0-1.5','1.5-2.0','2.0+']

def load_v11():
 s=importlib.util.spec_from_file_location('v11',ROOT/'evaluate-v11-unified-team-runs.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def alpha_moment(y,mu):return max(1e-9,float(np.sum((y-mu)**2-y)/np.sum(mu**2)))
def pmf(mu,a,n=60):
 k=np.arange(n+1);size=1/a;p=size/(size+mu);x=nbinom.pmf(k,size,p);x[-1]+=max(0,1-x.sum());return x
def home_probability(away,home,a):
 joint=np.outer(pmf(away,a),pmf(home,a));pa=np.tril(joint,-1).sum();ph=np.triu(joint,1).sum();return ph/(ph+pa)
def ci_accuracy(values,rng):
 x=np.asarray(values,float)
 if not len(x):return np.nan,np.nan
 b=rng.choice(x,(BOOT,len(x)),replace=True).mean(1);return float(np.quantile(b,.025)),float(np.quantile(b,.975))
def calibration(y,p):
 if len(y)<20 or len(np.unique(y))<2:return np.nan,np.nan
 p=np.clip(np.asarray(p),1e-6,1-1e-6);x=np.log(p/(1-p)).reshape(-1,1);m=LogisticRegression(C=1e6,solver='lbfgs').fit(x,y);return float(m.intercept_[0]),float(m.coef_[0,0])
def summary(g,scheme,bucket,scope,rng):
 lo,hi=ci_accuracy(g.correct,rng);intercept,slope=calibration(g.correct.astype(int),g.favorite_probability)
 return {'scheme':scheme,'bucket':str(bucket),'scope':scope,'games':len(g),'correct_predictions':int(g.correct.sum()),'accuracy':g.correct.mean(),'expected_accuracy':g.favorite_probability.mean(),'calibration_error':g.correct.mean()-g.favorite_probability.mean(),'accuracy_ci_low':lo,'accuracy_ci_high':hi,'confidence_correctness_auc':roc_auc_score(g.correct,g.favorite_probability) if len(g)>1 and g.correct.nunique()>1 else np.nan,'confidence_error_spearman':spearmanr(g.favorite_probability,g.correct.astype(float)).statistic if len(g)>2 else np.nan,'calibration_intercept':intercept,'calibration_slope':slope}
def main():
 R.mkdir(exist_ok=True);v=load_v11();features=json.loads((R/'v11_2_compact_frozen_specification.json').read_text())['features'];parts=[];disp=[]
 # Build each OOS season independently; 2025 is used only as the already-frozen final fold.
 for yrs,vy in FOLDS:
  tr=pd.concat([v.long_year(y) for y in yrs],ignore_index=True);va=v.long_year(vy);imp=SimpleImputer(strategy='median');sc=StandardScaler();xt=sc.fit_transform(imp.fit_transform(tr[features]));xv=sc.transform(imp.transform(va[features]));m=PoissonRegressor(alpha=10,max_iter=3000).fit(xt,tr.actual_team_runs);pt=np.clip(m.predict(xt),.05,None);pv=np.clip(m.predict(xv),.05,None);a=alpha_moment(tr.actual_team_runs.to_numpy(),pt);disp.append({'validation_year':vy,'training_years':'-'.join(map(str,yrs)),'nb_alpha_training_only':a,'training_team_games':len(tr)});q=va[['game_id','date','season','team','team_side','actual_team_runs']].copy();q['predicted_runs']=pv;parts.append(q)
 team=pd.concat(parts,ignore_index=True);games=[]
 for y,g in team.groupby('season'):
  alpha=next(x['nb_alpha_training_only'] for x in disp if x['validation_year']==y);a=g[g.team_side.eq('away')];h=g[g.team_side.eq('home')];z=a.merge(h,on=['game_id','date','season'],suffixes=('_away','_home'),validate='one_to_one');z['home_win_probability']=[home_probability(x,yh,alpha) for x,yh in zip(z.predicted_runs_away,z.predicted_runs_home)];z['away_win_probability']=1-z.home_win_probability;z['actual_home_win']=z.actual_team_runs_home.gt(z.actual_team_runs_away).astype(int);z['selected_side']=np.where(z.home_win_probability.ge(.5),'home','away');z['favorite_probability']=np.maximum(z.home_win_probability,z.away_win_probability);z['correct']=np.where(z.selected_side.eq('home'),z.actual_home_win.eq(1),z.actual_home_win.eq(0));z['projected_run_difference']=z.predicted_runs_home-z.predicted_runs_away;z['absolute_run_difference']=abs(z.projected_run_difference);z['nb_alpha']=alpha;games.append(z)
 games=pd.concat(games,ignore_index=True).sort_values(['date','game_id']);games['fixed_probability_bucket']=pd.cut(games.favorite_probability,PROB_EDGES,labels=PROB_LABELS,right=False,include_lowest=True);games['run_difference_bucket']=pd.cut(games.absolute_run_difference,DIFF_EDGES,labels=DIFF_LABELS,right=False,include_lowest=True);games['confidence_quantile']=pd.qcut(games.favorite_probability,10,labels=[f'Q{i}' for i in range(1,11)],duplicates='drop');games['run_difference_quantile']=pd.qcut(games.absolute_run_difference,10,labels=[f'Q{i}' for i in range(1,11)],duplicates='drop')
 rng=np.random.default_rng(SEED);rows=[]
 schemes=[('fixed_probability','fixed_probability_bucket',PROB_LABELS),('probability_decile','confidence_quantile',[f'Q{i}' for i in range(1,11)]),('fixed_absolute_run_difference','run_difference_bucket',DIFF_LABELS),('run_difference_decile','run_difference_quantile',[f'Q{i}' for i in range(1,11)])]
 for scheme,col,labels in schemes:
  for scope,frame in [('combined',games)]+[(str(y),games[games.season.eq(y)]) for y in range(2022,2026)]:
   for label in labels:rows.append(summary(frame[frame[col].eq(label)],scheme,label,scope,rng))
 buckets=pd.DataFrame(rows)
 # Predefined cumulative confidence levels, never optimized from results.
 levels=[]
 for measure,col,thresholds in [('favorite_probability','favorite_probability',[.55,.60,.65]),('absolute_run_difference','absolute_run_difference',[.5,1.0,1.5,2.0])]:
  for t in thresholds:
   for scope,frame in [('combined',games)]+[(str(y),games[games.season.eq(y)]) for y in range(2022,2026)]:
    g=frame[frame[col].ge(t)];r=summary(g,'cumulative_'+measure,f'>={t}',scope,rng);r['threshold']=t;r['measure']=measure;levels.append(r)
 levels=pd.DataFrame(levels)
 # Overall and ranking diagnostics.
 overall=[]
 for scope,g in [('combined',games)]+[(str(y),games[games.season.eq(y)]) for y in range(2022,2026)]:
  ph=g.home_win_probability;overall.append({'scope':scope,'games':len(g),'accuracy':g.correct.mean(),'mean_favorite_probability':g.favorite_probability.mean(),'favorite_calibration_error':g.correct.mean()-g.favorite_probability.mean(),'home_outcome_auc':roc_auc_score(g.actual_home_win,ph),'favorite_confidence_correctness_auc':roc_auc_score(g.correct,g.favorite_probability),'run_difference_correctness_auc':roc_auc_score(g.correct,g.absolute_run_difference),'favorite_confidence_spearman':spearmanr(g.favorite_probability,g.correct.astype(float)).statistic,'run_difference_spearman':spearmanr(g.absolute_run_difference,g.correct.astype(float)).statistic})
 overall=pd.DataFrame(overall)
 # Monotonicity on combined bucket accuracies.
 mono=[]
 for scheme in ('fixed_probability','probability_decile','fixed_absolute_run_difference','run_difference_decile'):
  q=buckets[(buckets.scheme.eq(scheme))&buckets.scope.eq('combined')&buckets.games.gt(0)];mono.append({'scheme':scheme,'buckets':len(q),'accuracy_monotonic_non_decreasing':bool(q.accuracy.is_monotonic_increasing),'bucket_order_accuracy_spearman':spearmanr(np.arange(len(q)),q.accuracy).statistic,'lowest_bucket_accuracy':q.accuracy.iloc[0],'highest_bucket_accuracy':q.accuracy.iloc[-1]})
 mono=pd.DataFrame(mono)
 games.to_csv(R/'v11_2_confidence_oos_game_predictions_2022_2025.csv',index=False);pd.DataFrame(disp).to_csv(R/'v11_2_confidence_training_dispersion.csv',index=False);buckets.to_csv(R/'v11_2_confidence_accuracy_buckets.csv',index=False);levels.to_csv(R/'v11_2_confidence_cumulative_levels.csv',index=False);overall.to_csv(R/'v11_2_confidence_overall_diagnostics.csv',index=False);mono.to_csv(R/'v11_2_confidence_monotonicity.csv',index=False)
 c=overall[overall.scope.eq('combined')].iloc[0];fixed=buckets[(buckets.scheme.eq('fixed_probability'))&buckets.scope.eq('combined')];high=levels[(levels.measure.eq('favorite_probability'))&levels.scope.eq('combined')];report=f"""# V11.2 model confidence versus winner accuracy\n\nThis diagnostic uses {len(games):,} chronological OOS games from 2022-2025, frozen V11.2 expected runs, and training-only NB2 dispersion. No sportsbook or betting data were loaded.\n\nOverall winner accuracy: {c.accuracy:.3%}; mean favorite probability: {c.mean_favorite_probability:.3%}; home-outcome AUC: {c.home_outcome_auc:.4f}. Raw favorite-confidence/correctness Spearman: {c.favorite_confidence_spearman:.4f}; absolute-run-difference/correctness Spearman: {c.run_difference_spearman:.4f}.\n\nFixed probability bucket accuracy monotonic: {mono.loc[mono.scheme.eq('fixed_probability'),'accuracy_monotonic_non_decreasing'].iloc[0]}. Probability-decile accuracy monotonic: {mono.loc[mono.scheme.eq('probability_decile'),'accuracy_monotonic_non_decreasing'].iloc[0]}. Absolute-difference decile accuracy monotonic: {mono.loc[mono.scheme.eq('run_difference_decile'),'accuracy_monotonic_non_decreasing'].iloc[0]}.\n\nPredefined cumulative probability levels:\n{high[['bucket','games','accuracy','expected_accuracy','calibration_error','accuracy_ci_low','accuracy_ci_high']].to_string(index=False)}\n""";(R/'v11_2_confidence_vs_winner_accuracy_report.md').write_text(report,encoding='utf-8');print(report);print('\nOVERALL\n',overall.to_string(index=False));print('\nMONOTONICITY\n',mono.to_string(index=False))
if __name__=='__main__':main()
