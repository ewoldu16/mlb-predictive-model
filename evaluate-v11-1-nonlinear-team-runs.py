"""V11.1 nonlinear team-run challenge. No sportsbook data is loaded."""
from pathlib import Path
import importlib.util, json, warnings
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import nbinom, poisson, spearmanr, pearsonr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT=Path(__file__).resolve().parent; R=ROOT/'results'
FOLDS=[([2021],2022),([2021,2022],2023),([2021,2022,2023],2024)]
BUCKET_EDGES=[-np.inf,3,3.5,4,4.5,5,5.5,6,np.inf]
BUCKETS=['<3','3-3.5','3.5-4','4-4.5','4.5-5','5-5.5','5.5-6','6+']
AUDIT_IDS=[777007,777874,777382,777481,777249,777782,778175,777067,776866,777980,778279]
SEED=111

def load_v11():
 spec=importlib.util.spec_from_file_location('v11',ROOT/'evaluate-v11-unified-team-runs.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

INTERACTIONS=[
 ('offense_x_starter_xwoba','season_woba','opp_sp_season_xwoba_allowed'),
 ('lineup_x_starter_xwoba','lineup_season_woba','opp_sp_season_xwoba_allowed'),
 ('lineup_x_starter_era','lineup_season_woba','opp_sp_official_season_era'),
 ('ops_x_starter_whip','season_ops','opp_sp_official_season_whip'),
 ('recent_offense_x_recent_starter','off_l30_woba','opp_sp_l30_xwoba_allowed'),
 ('offense_k_x_starter_k','off_l30_k_pct','opp_sp_l30_k_pct'),
 ('offense_bb_x_starter_bb','off_l30_bb_pct','opp_sp_l30_bb_pct'),
 ('power_x_starter_hr','off_season_hr_fb','opp_sp_rich_season_hr_per_fb'),
 ('hardhit_x_starter_hardhit','off_season_hardhit_pct','opp_sp_rich_season_hardhit_pct'),
 ('offense_x_bullpen_era','season_woba','opp_bp_official_season_era'),
 ('ops_x_bullpen_whip','season_ops','opp_bp_official_season_whip'),
 ('recent_offense_x_recent_bullpen','off_l30_woba','opp_bp_official_l30_era'),
 ('offense_x_available_bullpen','season_woba','opp_bp_official_available_pool_era'),
 ('offense_x_bullpen_fatigue','season_woba','opp_bp_avail_mean_fatigue'),
 ('hand_offense_x_matchup_xwoba','ctx_hand_season_woba','opp_sp_matchup_season_xwoba_allowed'),
 ('platoon_x_recent100_lhb','season_platoon_woba','opp_recent100_lhb_xwoba_allowed'),
 ('platoon_x_recent100_rhb','season_platoon_woba','opp_recent100_rhb_xwoba_allowed'),
 ('poor_starter_x_poor_bullpen','opp_sp_official_season_era','opp_bp_official_season_era'),
 ('starter_workload_x_bullpen_quality','opp_sp_season_avg_pitch_count','opp_bp_official_available_pool_era'),
 ('offense_x_venue_hand','season_woba','ctx_combined_season_woba')]
SPLINES=['season_woba','season_ops','lineup_season_woba','off_l30_woba','ctx_hand_season_woba','opp_sp_season_xwoba_allowed','opp_sp_official_season_era','opp_sp_official_season_whip','opp_sp_l30_xwoba_allowed','opp_bp_official_season_era','opp_bp_official_available_pool_era','sit_run_diff_per_game']

def add_interactions(x):
 z=x.copy()
 for n,a,b in INTERACTIONS:z[n]=z[a]*z[b]
 return z
def metric(y,p):
 p=np.clip(np.asarray(p,float),.05,None);y=np.asarray(y,float)
 return dict(rmse=mean_squared_error(y,p)**.5,mae=mean_absolute_error(y,p),poisson_deviance=mean_poisson_deviance(y,p),spearman=spearmanr(y,p).statistic,pearson=pearsonr(y,p).statistic,mean_prediction=p.mean(),actual_mean=y.mean())
def make_model(kind,param,features):
 if kind in ('control','interaction'):
  return Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',PoissonRegressor(alpha=param,max_iter=3000))])
 if kind=='spline':
  spl=[x for x in SPLINES if x in features]
  pre=ColumnTransformer([('linear',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),features),('curves',Pipeline([('imp',SimpleImputer(strategy='median')),('spline',SplineTransformer(n_knots=4,degree=2,knots='quantile',include_bias=False)),('scale',StandardScaler())]),spl)])
  return Pipeline([('pre',pre),('model',PoissonRegressor(alpha=param,max_iter=3000))])
 lr,leaves,l2=param
 return Pipeline([('imputer',SimpleImputer(strategy='median')),('model',HistGradientBoostingRegressor(loss='poisson',learning_rate=lr,max_iter=180,max_leaf_nodes=leaves,min_samples_leaf=40,l2_regularization=l2,early_stopping=False,random_state=SEED))])
def prep(x,kind):return add_interactions(x) if kind=='interaction' else x
def calibrate(df,model,scope):
 x=df.copy();x['bucket']=pd.cut(x.predicted_runs,BUCKET_EDGES,labels=BUCKETS,right=False);out=[]
 for b in BUCKETS:
  g=x[x.bucket.eq(b)];r={'model':model,'scope':scope,'bucket':b,'observations':len(g),'mean_prediction':g.predicted_runs.mean(),'actual_mean':g.actual_team_runs.mean(),'calibration_error':g.predicted_runs.mean()-g.actual_team_runs.mean(),'median_actual':g.actual_team_runs.median()}
  for k in range(2,8):r[f'pct_{k}_plus']=g.actual_team_runs.ge(k).mean()
  out.append(r)
 return out
def tail_rows(df,model,scope):
 out=[]
 for label,mask in [('low_lt_3.5',df.predicted_runs.lt(3.5)),('high_ge_5.5',df.predicted_runs.ge(5.5)),('extreme_high_ge_6',df.predicted_runs.ge(6))]:
  g=df[mask];r={'model':model,'scope':scope,'environment':label,'observations':len(g),'predicted_mean':g.predicted_runs.mean(),'actual_mean':g.actual_team_runs.mean(),'bias':g.predicted_runs.mean()-g.actual_team_runs.mean(),'rmse':np.sqrt(np.mean((g.actual_team_runs-g.predicted_runs)**2)),'mae':np.mean(abs(g.actual_team_runs-g.predicted_runs))}
  for k in (4,5,6,7):r[f'actual_{k}_plus']=g.actual_team_runs.ge(k).mean()
  out.append(r)
 return out
def rank_rows(df,model,scope):
 x=df.copy();x['rank_pct']=x.predicted_runs.rank(method='first',pct=True);x['decile']=np.minimum(np.ceil(x.rank_pct*10),10).astype(int);out=[]
 for d,g in x.groupby('decile'):out.append({'model':model,'scope':scope,'group':f'decile_{d}','observations':len(g),'mean_prediction':g.predicted_runs.mean(),'actual_mean':g.actual_team_runs.mean()})
 for name,mask in [('bottom_5pct',x.rank_pct.le(.05)),('top_5pct',x.rank_pct.gt(.95))]:
  g=x[mask];out.append({'model':model,'scope':scope,'group':name,'observations':len(g),'mean_prediction':g.predicted_runs.mean(),'actual_mean':g.actual_team_runs.mean()})
 return out
def game_metrics(df,model,scope):
 a=df[df.team_side.eq('away')];h=df[df.team_side.eq('home')];g=a.merge(h,on=['game_id','date','season'],suffixes=('_away','_home'),validate='one_to_one');g['projected_total']=g.predicted_runs_away+g.predicted_runs_home;g['actual_total']=g.actual_team_runs_away+g.actual_team_runs_home;g['projected_diff']=g.predicted_runs_home-g.predicted_runs_away;g['actual_diff']=g.actual_team_runs_home-g.actual_team_runs_away;g['home_win']=g.actual_diff.gt(0).astype(int)
 return g,{'model':model,'scope':scope,'games':len(g),'total_rmse':mean_squared_error(g.actual_total,g.projected_total)**.5,'total_mae':mean_absolute_error(g.actual_total,g.projected_total),'total_spearman':spearmanr(g.actual_total,g.projected_total).statistic,'winner_auc':roc_auc_score(g.home_win,g.projected_diff),'winner_accuracy':((g.projected_diff.gt(0))==g.home_win).mean()}
def nb_alpha(y,mu):return max(1e-8,float(np.sum((y-mu)**2-y)/np.sum(mu**2)))
def dist_rows(y,mu,a,scope):
 rows=[];size=1/a;prob=size/(size+mu)
 for name in ('poisson','negative_binomial'):
  nll=np.mean(mu-y*np.log(mu)+gammaln(y+1)) if name=='poisson' else -np.mean(nbinom.logpmf(y,size,prob));rows.append({'scope':scope,'distribution':name,'diagnostic':'nll','actual':np.nan,'predicted':nll,'alpha':0 if name=='poisson' else a})
  for k in (0,1,2):rows.append({'scope':scope,'distribution':name,'diagnostic':f'{k}_runs','actual':np.mean(y==k),'predicted':np.mean(poisson.pmf(k,mu) if name=='poisson' else nbinom.pmf(k,size,prob)),'alpha':0 if name=='poisson' else a})
  for k in (4,5,6,7):rows.append({'scope':scope,'distribution':name,'diagnostic':f'{k}_plus','actual':np.mean(y>=k),'predicted':np.mean(1-(poisson.cdf(k-1,mu) if name=='poisson' else nbinom.cdf(k-1,size,prob))),'alpha':0 if name=='poisson' else a})
  rows.append({'scope':scope,'distribution':name,'diagnostic':'variance','actual':np.var(y,ddof=1),'predicted':np.mean(mu if name=='poisson' else mu+a*mu**2),'alpha':0 if name=='poisson' else a})
 return rows

def main():
 warnings.filterwarnings('ignore');R.mkdir(exist_ok=True);v=load_v11();spec=json.loads((R/'v11_frozen_specification.json').read_text());features=spec['features']
 data=v.add_scoring_baselines(pd.concat([v.long_year(y) for y in range(2021,2026)],ignore_index=True));meta=['game_id','date','season','team','opponent','team_side','actual_team_runs','opponent_actual_runs']
 saved=pd.read_csv(R/'v11_model_selection_folds.csv');saved=saved[(saved.feature_family.eq('F_plus_team_strength'))&(saved.model_config.eq('poisson_a10'))].set_index('validation_year')
 configs=[('V11_control','control',10),*[ (f'interactions_a{a}','interaction',a) for a in (1,3,10,30,100)],*[ (f'spline_a{a}','spline',a) for a in (1,3,10,30,100)],('hist_lr03_l7_l2','hist',(.03,7,1)),('hist_lr05_l15_l2','hist',(.05,15,1)),('hist_lr10_l15_l2','hist',(.10,15,1)),('hist_lr05_l31_l2','hist',(.05,31,1)),('hist_lr05_l15_l0','hist',(.05,15,0)),('hist_lr05_l15_l10','hist',(.05,15,10))]
 defs=[]
 for label,kind,param in configs:defs.append({'candidate':label,'kind':kind,'parameters':str(param),'base_features':len(features),'added_features':len(INTERACTIONS) if kind=='interaction' else (len(SPLINES)*3 if kind=='spline' else 0)})
 pd.DataFrame(defs).to_csv(R/'v11_1_candidate_definitions.csv',index=False);pd.DataFrame(INTERACTIONS,columns=['interaction','left_feature','right_feature']).to_csv(R/'v11_1_interaction_definitions.csv',index=False)
 folds=[];preds={}
 for label,kind,param in configs:
  pp=[]
  for yrs,vy in FOLDS:
   tr=data[data.season.isin(yrs)];va=data[data.season.eq(vy)];xf=features+[n for n,_,_ in INTERACTIONS] if kind=='interaction' else features;m=make_model(kind,param,xf);m.fit(prep(tr[features],kind),tr.actual_team_runs);p=np.clip(m.predict(prep(va[features],kind)),.05,None);r={'candidate':label,'kind':kind,'validation_year':vy,'features':len(xf),**metric(va.actual_team_runs,p)};folds.append(r);q=va[meta].copy();q['predicted_runs']=p;pp.append(q)
   if label=='V11_control':
    for k in ('rmse','mae','poisson_deviance'):
     if abs(r[k]-float(saved.loc[vy,k]))>1e-8:raise RuntimeError(f'V11 control mismatch {vy} {k}: {r[k]} vs {saved.loc[vy,k]}')
  preds[label]=pd.concat(pp,ignore_index=True)
 folds=pd.DataFrame(folds);summary=folds.groupby(['candidate','kind','features'],as_index=False).agg(mean_rmse=('rmse','mean'),sd_rmse=('rmse','std'),mean_mae=('mae','mean'),sd_mae=('mae','std'),mean_poisson_deviance=('poisson_deviance','mean'),mean_spearman=('spearman','mean'),mean_pearson=('pearson','mean')).sort_values('mean_rmse');control=summary[summary.candidate.eq('V11_control')].iloc[0];chall=summary[~summary.candidate.eq('V11_control')].iloc[0]
 # Predeclared materiality: challenger must improve RMSE >=.01, MAE non-worse, ranking improve, and RMSE improve in >=2 folds without any fold worsening >.03.
 cf=folds[folds.candidate.eq('V11_control')].set_index('validation_year');bf=folds[folds.candidate.eq(chall.candidate)].set_index('validation_year');years_better=int((bf.rmse<cf.rmse).sum());promote=bool(control.mean_rmse-chall.mean_rmse>=.01 and chall.mean_mae<=control.mean_mae and chall.mean_spearman>control.mean_spearman and years_better>=2 and (bf.rmse-cf.rmse).max()<=.03)
 decision=f"Best challenger: {chall.candidate}. Mean RMSE change {chall.mean_rmse-control.mean_rmse:+.6f}; MAE {chall.mean_mae-control.mean_mae:+.6f}; Spearman {chall.mean_spearman-control.mean_spearman:+.6f}; RMSE improved {years_better}/3 years. Promotion rule passed: {promote}.\n"
 (R/'v11_1_model_selection_decision.md').write_text('# V11.1 development decision\n\n'+decision,encoding='utf-8')
 chosen=str(chall.candidate);crow=next(x for x in configs if x[0]==chosen);_,ckind,cparam=crow;frozen={'experiment':'V11.1 NONLINEAR TEAM-RUN CHALLENGE','control_verified':True,'challenger':chosen,'kind':ckind,'parameters':cparam,'base_features':features,'interactions':INTERACTIONS if ckind=='interaction' else [],'spline_variables':SPLINES if ckind=='spline' else [],'imputation':'training-fold median','scaling':'training-fold StandardScaler where applicable','prediction_floor':0.05,'random_seed':SEED,'promotion_rule_passed':promote,'recommended_model':'V11.1 challenger' if promote else 'retain frozen V11','sportsbook_data_used':False}
 (R/'v11_1_frozen_specification.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
 # Diagnostics on development after freeze.
 cal=[];tails=[];ranks=[]
 for label,d in preds.items():
  for y,g in d.groupby('season'):cal+=calibrate(g,label,str(y));tails+=tail_rows(g,label,str(y));ranks+=rank_rows(g,label,str(y))
  cal+=calibrate(d,label,'combined_development');tails+=tail_rows(d,label,'combined_development');ranks+=rank_rows(d,label,'combined_development')
 # One untouched holdout evaluation of control and frozen challenger.
 tr=data[data.season.le(2024)];ho=data[data.season.eq(2025)];hold={};models={}
 for label,kind,param in [('V11_control','control',10),crow]:
  xf=features+[n for n,_,_ in INTERACTIONS] if kind=='interaction' else features;m=make_model(kind,param,xf);m.fit(prep(tr[features],kind),tr.actual_team_runs);p=np.clip(m.predict(prep(ho[features],kind)),.05,None);q=ho[meta].copy();q['predicted_runs']=p;hold[label]=q;models[label]=m
 hm=pd.DataFrame([{'candidate':k,**metric(g.actual_team_runs,g.predicted_runs)} for k,g in hold.items()]);hcal=[];htail=[];hrank=[]
 for label,g in hold.items():hcal+=calibrate(g,label,'2025');htail+=tail_rows(g,label,'2025');hrank+=rank_rows(g,label,'2025')
 # Game level and projected run-difference calibration.
 grows=[];gameframes={}
 for scope,collection in [('development',preds),('2025',hold)]:
  for label in ('V11_control',chosen):
   g,r=game_metrics(collection[label],label,scope);g['model']=label;g['scope']=scope;gameframes[(scope,label)]=g;grows.append(r)
 # Negative-binomial dispersion for final challenger from training predictions only.
 cm=models[chosen];ptr=np.clip(cm.predict(prep(tr[features],ckind)),.05,None);alpha=nb_alpha(tr.actual_team_runs.to_numpy(),ptr);dist=dist_rows(ho.actual_team_runs.to_numpy(),hold[chosen].predicted_runs.to_numpy(),alpha,'2025_frozen_challenger')
 # Training-only deciles for extreme historical pregame conditions, applied to OOS records.
 ext=[]
 for label,d in list(preds.items())+[(f'{k}_2025',g) for k,g in hold.items()]:
  base=label.replace('_2025','');scope='2025' if label.endswith('_2025') else 'development';source=ho if scope=='2025' else data[data.season.between(2022,2024)];z=d.merge(source[['game_id','team_side','season_woba','opp_sp_official_season_era','opp_bp_official_season_era']],on=['game_id','team_side'],how='left');trainref=tr if scope=='2025' else data[data.season.lt(z.season.min())]
  for f,direction in [('season_woba','top'),('season_woba','bottom'),('opp_sp_official_season_era','top'),('opp_sp_official_season_era','bottom'),('opp_bp_official_season_era','top'),('opp_bp_official_season_era','bottom')]:
   lo,hi=trainref[f].quantile([.1,.9]);mask=z[f].ge(hi) if direction=='top' else z[f].le(lo);g=z[mask];ext.append({'candidate':base,'scope':scope,'condition':direction+'_'+f,'observations':len(g),'predicted_mean':g.predicted_runs.mean(),'actual_mean':g.actual_team_runs.mean(),'bias':g.predicted_runs.mean()-g.actual_team_runs.mean(),'rmse':np.sqrt(np.mean((g.actual_team_runs-g.predicted_runs)**2))})
 # Permutation importance on 2024 development only; honest diagnostic of selected challenger.
 va24=data[data.season.eq(2024)];m24=make_model(ckind,cparam,features);m24.fit(prep(data[data.season.le(2023)][features],ckind),data[data.season.le(2023)].actual_team_runs);pi=permutation_importance(m24,prep(va24[features],ckind),va24.actual_team_runs,n_repeats=3,random_state=SEED,scoring='neg_root_mean_squared_error');names=list(prep(va24[features],ckind).columns);interpret=pd.DataFrame({'diagnostic_type':'permutation_importance','feature':names,'importance_mean_rmse':pi.importances_mean,'importance_sd':pi.importances_std}).sort_values('importance_mean_rmse',ascending=False)
 # Frozen-model response curves and local median-replacement explanations.
 extra=[];reference=tr[features].median(numeric_only=True).reindex(features).to_frame().T
 if ckind=='spline':
  for f in SPLINES:
   for raw in np.unique(tr[f].dropna().quantile(np.linspace(.02,.98,25)).to_numpy()):
    z=reference.copy();z[f]=raw;extra.append({'diagnostic_type':'spline_response_curve','feature':f,'raw_value':raw,'predicted_runs':float(np.clip(models[chosen].predict(z),.05,None)[0]),'reference':'all other inputs at 2021-2024 median'})
 reps=[]
 for label,g in [('low',hold[chosen].nsmallest(1,'predicted_runs')),('average',hold[chosen].iloc[(hold[chosen].predicted_runs-hold[chosen].predicted_runs.mean()).abs().argsort()[:1]]),('high',hold[chosen][hold[chosen].predicted_runs.ge(5.5)].nsmallest(1,'predicted_runs')),('extreme_high',hold[chosen].nlargest(1,'predicted_runs'))]:
  if len(g):reps.append((label,g.iloc[0]))
 audit_team_keys={(gid,s) for gid in AUDIT_IDS for s in ('away','home')}
 for label,row in reps+[(f'audit_{r.game_id}_{r.team_side}',r) for r in hold[chosen].itertuples() if (r.game_id,r.team_side) in audit_team_keys]:
  base=ho[(ho.game_id.eq(row.game_id))&(ho.team_side.eq(row.team_side))][features].copy();bp=float(row.predicted_runs)
  for f in features:
   alt=base.copy();alt[f]=tr[f].median();delta=bp-float(np.clip(models[chosen].predict(prep(alt,ckind)),.05,None)[0]);extra.append({'diagnostic_type':'local_median_replacement','representative':label,'game_id':row.game_id,'team_side':row.team_side,'team':row.team,'feature':f,'raw_value':base[f].iloc[0],'training_median':tr[f].median(),'predicted_runs':bp,'local_delta_runs':delta})
 interpret=pd.concat([interpret,pd.DataFrame(extra)],ignore_index=True,sort=False)
 # Same 11-game audit; old totals/ML/Run Score are predictive comparisons, never inputs.
 gc=gameframes[('2025','V11_control')];gn=gameframes[('2025',chosen)];aud=gc[gc.game_id.isin(AUDIT_IDS)][['game_id','date','team_away','team_home','actual_team_runs_away','actual_team_runs_home','predicted_runs_away','predicted_runs_home','projected_total','projected_diff']].rename(columns={'predicted_runs_away':'v11_away','predicted_runs_home':'v11_home','projected_total':'v11_total','projected_diff':'v11_run_diff'});aud=aud.merge(gn[['game_id','predicted_runs_away','predicted_runs_home','projected_total','projected_diff']].rename(columns={'predicted_runs_away':'v11_1_away','predicted_runs_home':'v11_1_home','projected_total':'v11_1_total','projected_diff':'v11_1_run_diff'}),on='game_id')
 oldt=pd.read_csv(R/'totals_oos_predictions_2022_2025.csv')[['game_id','predicted_total']];oldm=pd.read_csv(R/'v5_team_strength_oos_predictions_2022_2025.csv')[['game_id','predicted_home_probability']];rs=pd.read_csv(R/'team_run_score_complete_team_games_2021_2025.csv')[['game_id','team_side','run_score']].pivot(index='game_id',columns='team_side',values='run_score').add_prefix('run_score_').reset_index();aud=aud.merge(oldt,on='game_id',how='left').merge(oldm,on='game_id',how='left').merge(rs,on='game_id',how='left').sort_values('game_id')
 report=['# V11.1 same 11-game re-audit','Architecture was frozen using 2022-2024 only; these outcomes did not select it.','']
 for x in aud.itertuples():report += [f'## {x.game_id}: {x.team_away} at {x.team_home}',f'Pregame V11 {x.v11_away:.3f}/{x.v11_home:.3f}; V11.1 {x.v11_1_away:.3f}/{x.v11_1_home:.3f}; changes {x.v11_1_away-x.v11_away:+.3f}/{x.v11_1_home-x.v11_home:+.3f}. Totals {x.v11_total:.3f} -> {x.v11_1_total:.3f}; old totals {x.predicted_total:.3f}; old ML home P {x.predicted_home_probability:.3f}; Run Scores {x.run_score_away:.2f}/{x.run_score_home:.2f}.',f'Outcome shown after predictions: {x.actual_team_runs_away:.0f}-{x.actual_team_runs_home:.0f}.','']
 # Save all new artifacts.
 folds.to_csv(R/'v11_1_development_folds.csv',index=False);summary.to_csv(R/'v11_1_development_summary.csv',index=False);pd.DataFrame(cal).to_csv(R/'v11_1_development_calibration.csv',index=False);pd.DataFrame(tails+ext).to_csv(R/'v11_1_tail_diagnostics.csv',index=False);pd.DataFrame(ranks).to_csv(R/'v11_1_prediction_deciles.csv',index=False);hm.to_csv(R/'v11_1_untouched_2025_metrics.csv',index=False);pd.DataFrame(hcal).to_csv(R/'v11_1_untouched_2025_calibration.csv',index=False);pd.DataFrame(htail).to_csv(R/'v11_1_untouched_2025_tail_diagnostics.csv',index=False);pd.DataFrame(hrank).to_csv(R/'v11_1_untouched_2025_prediction_deciles.csv',index=False);pd.DataFrame(grows).to_csv(R/'v11_1_game_level_comparison.csv',index=False);pd.DataFrame(dist).to_csv(R/'v11_1_distribution_diagnostics.csv',index=False);interpret.to_csv(R/'v11_1_feature_interpretability.csv',index=False);aud.to_csv(R/'v11_1_11_game_reaudit.csv',index=False);(R/'v11_1_11_game_reaudit_report.md').write_text('\n\n'.join(report),encoding='utf-8')
 h0=hm[hm.candidate.eq('V11_control')].iloc[0];h1=hm[hm.candidate.eq(chosen)].iloc[0];final=f"""# V11.1 final report\n\n{decision}\nControl reproduction passed exactly.\n\nUntouched 2025: V11 RMSE {h0.rmse:.4f}, MAE {h0.mae:.4f}, Spearman {h0.spearman:.4f}; {chosen} RMSE {h1.rmse:.4f}, MAE {h1.mae:.4f}, Spearman {h1.spearman:.4f}.\n\nRecommendation: {'replace V11 with frozen challenger' if promote else 'keep frozen V11; nonlinear evidence was not materially reliable'}. No odds or betting data were loaded.\n""";(R/'v11_1_final_report.md').write_text(final,encoding='utf-8')
 print(decision);print(hm.to_string(index=False));print(pd.DataFrame(grows).to_string(index=False))
if __name__=='__main__':main()
