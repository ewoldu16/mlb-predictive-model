"""V11.2 chronological feature-efficiency and ablation study. No odds are loaded."""
from pathlib import Path
import importlib.util,json,time,warnings,tracemalloc
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom,spearmanr,pearsonr
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error,mean_poisson_deviance,mean_squared_error,roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;R=ROOT/'results';F=R/'figures';SEED=112
FOLDS=[([2021],2022),([2021,2022],2023),([2021,2022,2023],2024)];COUNTS=[20,30,40,50,60,80,100,120,150,173]
AUDIT=[777007,777874,777382,777481,777249,777782,778175,777067,776866,777980,778279]

def v11mod():
 s=importlib.util.spec_from_file_location('v11',ROOT/'evaluate-v11-unified-team-runs.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def family(x):
 z=x.removeprefix('opp_')
 if z.startswith('sp_matchup') or z.startswith('recent100') or z.startswith('arsenal'):return 'starter_matchup'
 if z.startswith('sp_official_l30') or z.startswith('sp_official_recent3') or z.startswith('sp_l30') or z.startswith('sp_rich_l30'):return 'starter_recent'
 if z.startswith('sp_days') or 'pitch_count' in z:return 'starter_workload'
 if z.startswith('sp_'):return 'starter_quality'
 if z.startswith('bp_avail'):return 'bullpen_availability'
 if z.startswith('bp_official_l30') or z.startswith('bp_l30') or z.startswith('bp_l7'):return 'bullpen_recent'
 if z.startswith('bp_'):return 'bullpen_quality'
 if z.startswith('lineup_'):return 'lineup_quality'
 if 'risp' in z:return 'RISP'
 if 'platoon' in z or z.startswith('ctx_hand') :return 'platoon_handedness'
 if z.startswith('ctx_venue') or z.startswith('ctx_combined'):return 'venue_context'
 if z.startswith('oq_'):return 'opponent_quality_offense'
 if z.startswith('sit_run_diff') or z.startswith('sit_pythagorean') or z.startswith('sit_actual_minus'):return 'run_differential_pythagorean'
 if z.startswith('sit_'):return 'team_record_form'
 if z.startswith('off_l7') or z.startswith('off_l15') or z.startswith('off_l30') or z.startswith('l30_'):return 'recent_offense'
 if z in ('home_indicator',):return 'park_home_context'
 return 'offensive_baseline'
def source(x):
 z=x.removeprefix('opp_')
 if z.startswith('sp_matchup'):return 'games_YEAR_starter_lineup_matchup_features.csv'
 if z.startswith('arsenal'):return 'features_arsenal_lineup_matchup_YEAR.csv'
 if z.startswith('recent100'):return 'features_statsimpl_starter_recent100_YEAR.csv'
 if z.startswith('sp_official'):return 'features_official_starter_pitching_YEAR.csv'
 if z.startswith('bp_official'):return 'features_official_bullpen_pitching_YEAR.csv'
 if z.startswith('bp_avail'):return 'features_v6_bullpen_availability_YEAR.csv'
 if z.startswith('sp_rich'):return 'features_richer_starter_YEAR.csv'
 if z.startswith('ctx_'):return 'features_v8_contextual_offense_YEAR.csv'
 if z.startswith('oq_'):return 'features_opponent_quality_offense_YEAR.csv'
 if z.startswith('sit_'):return 'features_situational_YEAR.csv'
 if 'risp' in z or z in ('season_avg','season_obp','season_slg','season_ops','l30_avg','l30_obp','l30_slg','l30_ops'):return 'features_statsimpl_offense_risp_YEAR.csv'
 if z.startswith('off_'):return 'features_v7_offensive_form_YEAR.csv'
 return 'games_YEAR_starter_lineup_matchup_features.csv'
def window(x):
 for w in ('recent100','recent3','l7','l10','l15','l30','season'):
  if w in x.lower():return w
 return 'current_pregame_context'
def why(x):
 return {'starter_matchup':'association between the opposing starter-lineup matchup and scoring','run_differential_pythagorean':'association between underlying team strength and scoring','bullpen_quality':'association between opposing bullpen quality and scoring','venue_context':'association between venue-specific offensive context and scoring','opponent_quality_offense':'association between quality-adjusted offense and scoring','lineup_quality':'association between actual lineup quality and scoring','bullpen_recent':'association between recent opposing bullpen form and scoring','offensive_baseline':'association between baseline offensive quality and scoring','starter_quality':'association between opposing starter quality and scoring','starter_workload':'association between expected starter workload and scoring','platoon_handedness':'association between handedness context and scoring','park_home_context':'association between home context and scoring','recent_offense':'association between recent offensive form and scoring','RISP':'association between prior RISP performance and scoring','starter_recent':'association between recent opposing starter form and scoring','team_record_form':'association between prior team results and scoring','bullpen_availability':'association between bullpen availability and scoring'}.get(x,'predictive association with team scoring')
def metrics(y,p):
 p=np.clip(p,.05,None);return {'rmse':mean_squared_error(y,p)**.5,'mae':mean_absolute_error(y,p),'poisson_deviance':mean_poisson_deviance(y,p),'spearman':spearmanr(y,p).statistic,'pearson':pearsonr(y,p).statistic,'mean_prediction':p.mean(),'actual_mean':np.mean(y)}
def prep(tr,va,features):
 imp=SimpleImputer(strategy='median');sc=StandardScaler();xt=sc.fit_transform(imp.fit_transform(tr[features]));xv=sc.transform(imp.transform(va[features]));return xt,xv,imp,sc
def fitpred(xt,y,xv,cols):
 t=time.perf_counter();m=PoissonRegressor(alpha=10,max_iter=3000).fit(xt[:,cols],y);train=time.perf_counter()-t;t=time.perf_counter();p=np.clip(m.predict(xv[:,cols]),.05,None);return m,p,train,time.perf_counter()-t
def game(df):
 a=df[df.team_side.eq('away')];h=df[df.team_side.eq('home')];g=a.merge(h,on=['game_id','date','season'],suffixes=('_away','_home'));g['pt']=g.predicted_runs_away+g.predicted_runs_home;g['atot']=g.actual_team_runs_away+g.actual_team_runs_home;g['pdiff']=g.predicted_runs_home-g.predicted_runs_away;g['adiff']=g.actual_team_runs_home-g.actual_team_runs_away;g['hw']=g.adiff.gt(0).astype(int);return g
def nbdiag(y,mu,a,model):
 size=1/a;p=size/(size+mu);rows=[{'model':model,'diagnostic':'nll','actual':np.nan,'predicted':float(-np.mean(nbinom.logpmf(y,size,p))),'alpha':a},{'model':model,'diagnostic':'variance','actual':np.var(y,ddof=1),'predicted':np.mean(mu+a*mu**2),'alpha':a}]
 for k in (0,1):rows.append({'model':model,'diagnostic':f'{k}_runs','actual':np.mean(y==k),'predicted':np.mean(nbinom.pmf(k,size,p)),'alpha':a})
 for k in (4,5,6,7):rows.append({'model':model,'diagnostic':f'{k}_plus','actual':np.mean(y>=k),'predicted':np.mean(1-nbinom.cdf(k-1,size,p)),'alpha':a})
 return rows
def main():
 warnings.filterwarnings('ignore');R.mkdir(exist_ok=True);F.mkdir(exist_ok=True);v=v11mod();spec=json.loads((R/'v11_frozen_specification.json').read_text());features=spec['features'];idx={f:i for i,f in enumerate(features)}
 # Development universe only; 2025 is not constructed until after compact freeze.
 devdata=v.add_scoring_baselines(pd.concat([v.long_year(y) for y in range(2021,2025)],ignore_index=True));saved=pd.read_csv(R/'v11_model_selection_folds.csv');saved=saved[(saved.feature_family.eq('F_plus_team_strength'))&saved.model_config.eq('poisson_a10')].set_index('validation_year')
 cache={};control=[];coef=[];perm=[]
 for yrs,vy in FOLDS:
  tr=devdata[devdata.season.isin(yrs)];va=devdata[devdata.season.eq(vy)];xt,xv,imp,sc=prep(tr,va,features);m,p,ft,pt=fitpred(xt,tr.actual_team_runs.to_numpy(),xv,np.arange(len(features)));z={'validation_year':vy,**metrics(va.actual_team_runs,p),'training_seconds':ft,'prediction_seconds':pt};control.append(z)
  for k in ('rmse','mae','poisson_deviance'):
   if abs(z[k]-float(saved.loc[vy,k]))>1e-8:raise RuntimeError(f'control mismatch {vy} {k}')
  pi=permutation_importance(m,xv,va.actual_team_runs,n_repeats=2,random_state=SEED,scoring='neg_root_mean_squared_error')
  for j,f in enumerate(features):coef.append({'validation_year':vy,'feature':f,'coefficient':m.coef_[j]});perm.append({'validation_year':vy,'feature':f,'permutation_rmse_importance':pi.importances_mean[j]})
  cache[vy]=(tr,va,xt,xv,imp,sc)
 control=pd.DataFrame(control);base_rmse=control.rmse.mean();base_mae=control.mae.mean()
 # Single-feature ablation, chronologically refitted.
 ab=[]
 for j,f in enumerate(features):
  for vy,(tr,va,xt,xv,_,_) in cache.items():
   cols=np.delete(np.arange(len(features)),j);_,p,_,_=fitpred(xt,tr.actual_team_runs.to_numpy(),xv,cols);q=metrics(va.actual_team_runs,p);ab.append({'feature':f,'validation_year':vy,**q})
 ab=pd.DataFrame(ab);co=pd.DataFrame(coef);pe=pd.DataFrame(perm);importance=ab.groupby('feature',as_index=False).agg(ablation_rmse=('rmse','mean'),ablation_mae=('mae','mean'));importance['ablation_rmse_delta']=importance.ablation_rmse-base_rmse;importance['ablation_mae_delta']=importance.ablation_mae-base_mae
 cs=co.groupby('feature').coefficient.agg(['mean',lambda x:(np.sign(x)==np.sign(x.iloc[0])).mean(),'std']).reset_index();cs.columns=['feature','mean_coefficient','coefficient_sign_consistency','coefficient_sd'];ps=pe.groupby('feature',as_index=False).permutation_rmse_importance.mean();importance=importance.merge(cs,on='feature').merge(ps,on='feature');importance['family']=importance.feature.map(family)
 # Mandatory family ablation.
 fam=[]
 for fa,fs in pd.Series(features).groupby(pd.Series(features).map(family)):
  keep=[idx[x] for x in features if x not in set(fs)]
  for vy,(tr,va,xt,xv,_,_) in cache.items():
   _,p,_,_=fitpred(xt,tr.actual_team_runs.to_numpy(),xv,keep);fam.append({'family':fa,'features_removed':len(fs),'validation_year':vy,**metrics(va.actual_team_runs,p)})
 fam=pd.DataFrame(fam);fams=fam.groupby(['family','features_removed'],as_index=False).agg(rmse=('rmse','mean'),mae=('mae','mean'),spearman=('spearman','mean'),pearson=('pearson','mean'),poisson_deviance=('poisson_deviance','mean'));fams['delta_rmse']=fams.rmse-base_rmse;fams['delta_mae']=fams.mae-base_mae;fams=fams.sort_values('delta_rmse',ascending=False)
 # Correlations use latest-fold training only (2021-2023), never 2024 validation.
 corrdata=devdata[devdata.season.le(2023)][features];ci=SimpleImputer(strategy='median').fit_transform(corrdata);cr=pd.DataFrame(ci,columns=features).corr();pairs=[]
 for i in range(len(features)):
  for j in range(i+1,len(features)):
   r=cr.iat[i,j]
   if abs(r)>=.8:pairs.append({'feature_a':features[i],'feature_b':features[j],'correlation':r,'abs_correlation':abs(r),'threshold_80':True,'threshold_90':abs(r)>=.9,'threshold_95':abs(r)>=.95,'threshold_98':abs(r)>=.98,'same_family':family(features[i])==family(features[j])})
 pairs=pd.DataFrame(pairs).sort_values('abs_correlation',ascending=False)
 # Development-safe ranking combines ablation, permutation, stable magnitude; retain domain diversity naturally through score.
 rank=importance.copy();rank['score']=rank.ablation_rmse_delta.rank(pct=True)+rank.permutation_rmse_importance.rank(pct=True)+rank.mean_coefficient.abs().rank(pct=True)*rank.coefficient_sign_consistency;rank=rank.sort_values(['score','ablation_rmse_delta'],ascending=False);ordered=rank.feature.tolist()
 cand=[];candpred={};eff=[]
 for n in COUNTS:
  fs=features if n==173 else ordered[:n];rows=[];pp=[];tt=ptt=0;tracemalloc.start()
  for vy,(tr,va,xt,xv,_,_) in cache.items():
   cols=[idx[x] for x in fs];_,p,t1,t2=fitpred(xt,tr.actual_team_runs.to_numpy(),xv,cols);tt+=t1;ptt+=t2;q=metrics(va.actual_team_runs,p);rows.append({'feature_count':n,'validation_year':vy,**q});z=va[['game_id','date','season','team','team_side','actual_team_runs']].copy();z['predicted_runs']=p;pp.append(z)
  _,peak=tracemalloc.get_traced_memory();tracemalloc.stop();rr=pd.DataFrame(rows);cand.extend(rows);candpred[n]=pd.concat(pp);eff.append({'feature_count':n,'training_seconds':tt,'prediction_seconds':ptt,'peak_memory_mb':peak/1048576})
 cand=pd.DataFrame(cand);summ=cand.groupby('feature_count',as_index=False).agg(mean_rmse=('rmse','mean'),sd_rmse=('rmse','std'),mean_mae=('mae','mean'),mean_spearman=('spearman','mean'),mean_pearson=('pearson','mean'),mean_poisson_deviance=('poisson_deviance','mean'),worst_rmse=('rmse','max')).sort_values('feature_count');full=summ[summ.feature_count.eq(173)].iloc[0]
 eligible=summ[summ.feature_count.between(50,80)&summ.mean_rmse.le(full.mean_rmse+.01)&summ.mean_mae.le(full.mean_mae+.01)&summ.mean_spearman.ge(full.mean_spearman-.005)&summ.worst_rmse.le(full.worst_rmse+.03)];chosen=int(eligible.feature_count.min()) if len(eligible) else int(summ[summ.mean_rmse.le(full.mean_rmse+.01)].feature_count.min());chosen_features=features if chosen==173 else ordered[:chosen]
 # Pareto: no smaller/equal model with lower/equal RMSE and at least one strict advantage.
 summ['pareto_efficient']=[not any((summ.feature_count<=r.feature_count)&(summ.mean_rmse<=r.mean_rmse)&((summ.feature_count<r.feature_count)|(summ.mean_rmse<r.mean_rmse))) for r in summ.itertuples()] ;se=summ.mean_rmse.std(ddof=1)/np.sqrt(len(summ));one_se=int(summ[summ.mean_rmse.le(summ.mean_rmse.min()+se)].feature_count.min())
 frozen={'experiment':'V11.2 FEATURE EFFICIENCY','reference_features':173,'feature_count':chosen,'features':chosen_features,'poisson_alpha':10,'imputation':'training-fold median','scaling':'training-fold StandardScaler','prediction_floor':.05,'selection_data':'chronological OOS 2022-2024 only','selection_rule':'smallest 50-80 model within .01 RMSE/.01 MAE/.005 Spearman/.03 worst-fold RMSE of V11; fallback smallest within .01 RMSE','one_standard_error_diagnostic_count':one_se,'random_seed':SEED,'family_mapping':{x:family(x) for x in chosen_features},'sportsbook_data_used':False};(R/'v11_2_compact_frozen_specification.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
 # Only now construct/load 2025 and fit final full/compact models.
 hold=v.add_scoring_baselines(v.long_year(2025));train=devdata[devdata.season.le(2024)];xt,xh,imp,sc=prep(train,hold,features);outs=[];pred25={};finalmods={}
 for label,fs in [('V11_full',features),('V11_2_compact',chosen_features)]:
  cols=[idx[x] for x in fs];m,p,t1,t2=fitpred(xt,train.actual_team_runs.to_numpy(),xh,cols);pred25[label]=hold[['game_id','date','season','team','team_side','actual_team_runs']].assign(predicted_runs=p);finalmods[label]=(m,cols);outs.append({'model':label,'feature_count':len(fs),**metrics(hold.actual_team_runs,p),'training_seconds':t1,'prediction_seconds':t2})
 compare=pd.DataFrame(outs);games=[]
 for label,d in pred25.items():
  g=game(d);games.append({'model':label,'game_total_rmse':mean_squared_error(g.atot,g.pt)**.5,'game_total_mae':mean_absolute_error(g.atot,g.pt),'game_total_spearman':spearmanr(g.atot,g.pt).statistic,'winner_auc':roc_auc_score(g.hw,g.pdiff),'winner_accuracy':((g.pdiff>0)==g.hw).mean()})
 compare=compare.merge(pd.DataFrame(games),on='model')
 # Error environments use 2021-2024 thresholds only.
 compact=pred25['V11_2_compact'].merge(hold[['game_id','team_side','season_woba','opp_sp_official_season_era','opp_bp_official_season_era']],on=['game_id','team_side']);fullp=pred25['V11_full'][['game_id','team_side','predicted_runs']].rename(columns={'predicted_runs':'v11_pred'});compact=compact.merge(fullp,on=['game_id','team_side']);compact['outcome_bucket']=pd.cut(compact.actual_team_runs,[-1,0,1,3,5,7,9,np.inf],labels=['shutout','1','2-3','4-5','6-7','8-9','10+']);er=[]
 conditions={'outcome_bucket':compact.outcome_bucket}
 for f in ('season_woba','opp_sp_official_season_era','opp_bp_official_season_era'):
  lo,hi=train[f].quantile([.1,.9]);conditions[f'bottom_decile_{f}']=compact[f].le(lo);conditions[f'top_decile_{f}']=compact[f].ge(hi)
 for typ,val in conditions.items():
  if typ=='outcome_bucket': groups=compact.groupby(val,observed=True)
  else:groups=[(typ,compact[val])]
  for name,g in groups:er.append({'diagnostic':typ,'group':str(name),'observations':len(g),'v11_rmse':np.sqrt(np.mean((g.actual_team_runs-g.v11_pred)**2)),'compact_rmse':np.sqrt(np.mean((g.actual_team_runs-g.predicted_runs)**2)),'v11_bias':(g.v11_pred-g.actual_team_runs).mean(),'compact_bias':(g.predicted_runs-g.actual_team_runs).mean()})
 # NB diagnostics use frozen training-only alpha.
 alpha=.25962778329405606;dist=[]
 for label,d in pred25.items():dist+=nbdiag(d.actual_team_runs.to_numpy(),d.predicted_runs.to_numpy(),alpha,label)
 # Same fixed 11 games and largest changed coefficient contributions.
 aud=pred25['V11_full'].merge(pred25['V11_2_compact'],on=['game_id','date','season','team','team_side','actual_team_runs'],suffixes=('_v11','_compact'));aud=aud[aud.game_id.isin(AUDIT)].copy();aud['difference']=aud.predicted_runs_compact-aud.predicted_runs_v11
 # Final importance uses 2021-2024 fit coefficients plus development diagnostics.
 cm,ccols=finalmods['V11_2_compact'];fi=importance[importance.feature.isin(chosen_features)].copy();coefmap=dict(zip(chosen_features,cm.coef_));fi['standardized_coefficient_2021_2024']=fi.feature.map(coefmap);fi['coefficient_sign']=np.where(fi.standardized_coefficient_2021_2024>=0,'positive','negative');fi['why_it_matters']=fi.family.map(why);fi['interpretation_caveat']='predictive association, not causation';miss=train[chosen_features].isna().mean();fi['missing_rate_2021_2024']=fi.feature.map(miss);fi=fi.sort_values('ablation_rmse_delta',ascending=False)
 finalfam=fi.groupby('family',as_index=False).agg(features=('feature','count'),mean_ablation_rmse_delta=('ablation_rmse_delta','mean'),sum_permutation_importance=('permutation_rmse_importance','sum'),mean_abs_coefficient=('standardized_coefficient_2021_2024',lambda x:np.mean(abs(x)))).merge(fams[['family','delta_rmse','delta_mae']],on='family',how='left').sort_values('delta_rmse',ascending=False)
 # Inventory after freeze may report 2025 missingness but cannot influence selection.
 inventory=[]
 for f in features:
  z={'feature':f,'family':family(f),'interpretation':f.replace('_',' '),'source_dataset':source(f),'window':window(f),'orientation':'opponent' if f.startswith('opp_') else ('home indicator' if f=='home_indicator' else 'batting team'),'variance_2021_2024':train[f].var(),'mean_2021_2024':train[f].mean(),'std_2021_2024':train[f].std(),'potentially_redundant':bool((pairs.feature_a.eq(f)|pairs.feature_b.eq(f)).any())}
  for y in range(2021,2026):z[f'missing_pct_{y}']=(devdata if y<2025 else hold).query('season==@y')[f].isna().mean()
  inventory.append(z)
 # Contribution-change diagnostic on audit rows: compact/full standardized terms.
 contrib=[]
 fm,fcols=finalmods['V11_full'];cm,ccols=finalmods['V11_2_compact'];zh=sc.transform(imp.transform(hold[features]));rows=hold.reset_index(drop=True)
 for rr in aud.itertuples():
  ii=rows.index[(rows.game_id.eq(rr.game_id))&(rows.team_side.eq(rr.team_side))][0]
  fullterms={features[j]:fm.coef_[j]*zh[ii,j] for j in range(len(features))};compactterms={features[col]:cm.coef_[j]*zh[ii,col] for j,col in enumerate(ccols)}
  for f in features:contrib.append({'game_id':rr.game_id,'team_side':rr.team_side,'feature':f,'v11_log_contribution':fullterms[f],'compact_log_contribution':compactterms.get(f,0),'contribution_difference':compactterms.get(f,0)-fullterms[f]})
 contributions=pd.DataFrame(contrib);top=contributions.reindex(contributions.contribution_difference.abs().sort_values(ascending=False).index).groupby(['game_id','team_side']).head(5).groupby(['game_id','team_side']).apply(lambda g:' | '.join(f'{r.feature}:{r.contribution_difference:+.4f}' for r in g.itertuples()),include_groups=False).rename('largest_contribution_differences').reset_index();aud=aud.merge(top,on=['game_id','team_side'])
 # Efficiency relative to full candidate.
 eff=pd.DataFrame(eff);baseeff=eff[eff.feature_count.eq(173)].iloc[0];eff['training_time_relative']=eff.training_seconds/baseeff.training_seconds;eff['prediction_time_relative']=eff.prediction_seconds/baseeff.prediction_seconds
 # Plots.
 plt.figure();plt.plot(summ.feature_count,summ.mean_rmse,'o-');plt.axhline(full.mean_rmse,ls='--');plt.xlabel('Features');plt.ylabel('Development RMSE');plt.tight_layout();plt.savefig(F/'v11_2_feature_count_vs_rmse.png',dpi=160);plt.close()
 plt.figure();plt.plot(summ.feature_count,summ.mean_spearman,'o-');plt.xlabel('Features');plt.ylabel('Development Spearman');plt.tight_layout();plt.savefig(F/'v11_2_feature_count_vs_spearman.png',dpi=160);plt.close()
 plt.figure(figsize=(9,6));q=fams.sort_values('delta_rmse');plt.barh(q.family,q.delta_rmse);plt.xlabel('RMSE deterioration when removed');plt.tight_layout();plt.savefig(F/'v11_2_family_ablation.png',dpi=160);plt.close()
 plt.figure(figsize=(9,6));q=fi.nlargest(20,'ablation_rmse_delta').sort_values('ablation_rmse_delta');plt.barh(q.feature,q.ablation_rmse_delta);plt.xlabel('Single-feature ablation RMSE delta');plt.tight_layout();plt.savefig(F/'v11_2_top_feature_importance.png',dpi=160);plt.close()
 cp=pred25['V11_2_compact'].copy();cp['bucket']=pd.qcut(cp.predicted_runs,10,duplicates='drop');ca=cp.groupby('bucket',observed=True).agg(predicted=('predicted_runs','mean'),actual=('actual_team_runs','mean'));plt.figure();plt.plot(ca.predicted,ca.actual,'o-');lo=min(ca.min());hi=max(ca.max());plt.plot([lo,hi],[lo,hi],'--');plt.xlabel('Predicted runs');plt.ylabel('Actual runs');plt.tight_layout();plt.savefig(F/'v11_2_predicted_actual_calibration.png',dpi=160);plt.close()
 plt.figure();q=compare.set_index('model')[['rmse','mae']];q.plot(kind='bar');plt.xticks(rotation=0);plt.tight_layout();plt.savefig(F/'v11_2_model_comparison.png',dpi=160);plt.close()
 plt.figure();plt.hist(cp.actual_team_runs-cp.predicted_runs,bins=30);plt.xlabel('Actual - predicted runs');plt.tight_layout();plt.savefig(F/'v11_2_residual_distribution.png',dpi=160);plt.close()
 # Machine-readable CV metrics.
 official=list((ROOT/'data/raw/official_mlb_pitching').glob('official_pitcher_game_lines_*.csv')) if (ROOT/'data/raw/official_mlb_pitching').exists() else [];official_rows=sum(len(pd.read_csv(x,usecols=[0])) for x in official)
 cv={'historical_games_processed':int(train.game_id.nunique()+hold.game_id.nunique()),'team_game_observations':int(len(train)+len(hold)),'official_pitcher_game_records':int(official_rows),'original_feature_count':173,'compact_feature_count':chosen,'feature_reduction_pct':(173-chosen)/173,'development_rmse':float(summ.loc[summ.feature_count.eq(chosen),'mean_rmse'].iloc[0]),'untouched_2025_rmse':float(compare.loc[compare.model.eq('V11_2_compact'),'rmse'].iloc[0]),'untouched_2025_mae':float(compare.loc[compare.model.eq('V11_2_compact'),'mae'].iloc[0]),'game_total_rmse':float(compare.loc[compare.model.eq('V11_2_compact'),'game_total_rmse'].iloc[0]),'winner_auc':float(compare.loc[compare.model.eq('V11_2_compact'),'winner_auc'].iloc[0]),'chronological_validation_folds':3,'sportsbook_data_used':False}
 (R/'v11_2_cv_metrics.json').write_text(json.dumps(cv,indent=2),encoding='utf-8')
 # Save.
 pd.DataFrame(inventory).to_csv(R/'v11_2_feature_inventory.csv',index=False);pairs.to_csv(R/'v11_2_correlation_clusters.csv',index=False);importance.to_csv(R/'v11_2_feature_importance.csv',index=False);fams.to_csv(R/'v11_2_family_ablation.csv',index=False);cand.to_csv(R/'v11_2_compact_candidate_results.csv',index=False);summ.to_csv(R/'v11_2_pareto_frontier.csv',index=False);eff.to_csv(R/'v11_2_computational_efficiency.csv',index=False);pd.DataFrame(er).to_csv(R/'v11_2_error_analysis.csv',index=False);compare.to_csv(R/'v11_2_untouched_2025_comparison.csv',index=False);pd.DataFrame(dist).to_csv(R/'v11_2_distribution_diagnostics.csv',index=False);aud.to_csv(R/'v11_2_11_game_audit.csv',index=False);contributions.to_csv(R/'v11_2_11_game_contribution_differences.csv',index=False);fi.to_csv(R/'v11_2_final_feature_importance.csv',index=False);finalfam.to_csv(R/'v11_2_final_family_importance.csv',index=False)
 chosenrow=summ[summ.feature_count.eq(chosen)].iloc[0];hfull=compare[compare.model.eq('V11_full')].iloc[0];hcomp=compare[compare.model.eq('V11_2_compact')].iloc[0];report=f"""# V11.2 feature-efficiency report\n\nControl reproduced exactly. Development selected {chosen} features: RMSE {chosenrow.mean_rmse:.4f} versus full {full.mean_rmse:.4f}; MAE {chosenrow.mean_mae:.4f}; Spearman {chosenrow.mean_spearman:.4f}. One-SE diagnostic count: {one_se}.\n\nUntouched 2025 full/compact RMSE {hfull.rmse:.4f}/{hcomp.rmse:.4f}, MAE {hfull.mae:.4f}/{hcomp.mae:.4f}, Spearman {hfull.spearman:.4f}/{hcomp.spearman:.4f}; game-total RMSE {hfull.game_total_rmse:.4f}/{hcomp.game_total_rmse:.4f}; winner AUC {hfull.winner_auc:.4f}/{hcomp.winner_auc:.4f}.\n\nRecommendation is based solely on chronological predictive efficiency. No sportsbook data were loaded.\n""";(R/'v11_2_final_report.md').write_text(report,encoding='utf-8')
 print(summ.to_string(index=False));print('\nCHOSEN',chosen);print(compare.to_string(index=False));print('\nFAMILIES\n',fams.to_string(index=False))
if __name__=='__main__':main()
