"""Build and dry-run the outcome-blind V13 AI validator infrastructure."""
from pathlib import Path
import importlib.util,json,hashlib,os
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from ai_validator.context_builder import audit_packet,clean
from ai_validator.prompts import PROMPT_HASH,PROMPT_VERSION,SYSTEM_PROMPT
from ai_validator.providers import MockValidatorProvider,ExternalLLMValidatorProvider,external_configured
from ai_validator.validator import CachedValidator

ROOT=Path(__file__).resolve().parent;R=ROOT/'results';P=ROOT/'data/processed';CACHE=ROOT/'data/cache/v13_ai_validator';SEED=1313;SAMPLE_TEAM_GAMES=400
ALPHA={2022:.25399455027714585,2023:.26562511422760027,2024:.2629530600894081}

def load_v11():
 s=importlib.util.spec_from_file_location('v11',ROOT/'evaluate-v11-unified-team-runs.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def qtile(mu,a,q):
 size=1/a;p=size/(size+mu);return int(nbinom.ppf(q,size,p))
def family(feature):
 z=feature.removeprefix('opp_')
 if z.startswith(('sp_matchup','recent100','arsenal')):return 'matchups'
 if z.startswith('sp_'):return 'opposing_starter'
 if z.startswith('bp_'):return 'opposing_bullpen'
 if z.startswith('lineup'):return 'lineup'
 if z.startswith(('ctx_hand','ctx_combined')) or 'platoon' in z:return 'handedness_context'
 if z.startswith('ctx_venue'):return 'venue_context'
 if z.startswith('sit_'):return 'team_strength'
 if 'risp' in z:return 'RISP'
 if z.startswith(('off_l','l30_')):return 'recent_offense'
 return 'offense'
def warnings_for(row,features,zvals):
 out=[];missing=row[features].isna();rate=float(missing.mean())
 if rate>.35:out.append({'reason_code':'MISSING_CRITICAL_DATA','message':f'{rate:.1%} of compact features missing'})
 if pd.isna(row.get('lineup_season_woba')):out.append({'reason_code':'PARTIAL_LINEUP','message':'actual-lineup season wOBA unavailable'})
 if pd.isna(row.get('opp_sp_official_season_era')):out.append({'reason_code':'SPARSE_STARTER_HISTORY','message':'opposing starter official season ERA unavailable'})
 match=[x for x in features if family(x)=='matchups']
 if match and row[match].isna().mean()>.5:out.append({'reason_code':'SPARSE_MATCHUP_HISTORY','message':'more than half of compact matchup fields unavailable'})
 if np.nanmax(abs(zvals))>6:out.append({'reason_code':'EXTREME_INPUT','message':f'compact feature absolute z-score reaches {np.nanmax(abs(zvals)):.2f}'})
 if 'opp_bp_avail_pool_size' in row and pd.notna(row['opp_bp_avail_pool_size']) and row['opp_bp_avail_pool_size']<3:out.append({'reason_code':'BULLPEN_AVAILABILITY_CONFLICT','message':'pregame available bullpen pool smaller than three'})
 return out
def team_context(row,features,all_features,z,coef):
 full=[]
 for j,f in enumerate(features):full.append({'feature':f,'family':family(f),'raw_value':row[f],'standardized_value':z[j],'coefficient':coef[j],'log_mean_contribution':z[j]*coef[j],'interpretation':f.replace('_',' ')})
 pos=sorted(full,key=lambda x:x['log_mean_contribution'],reverse=True)[:6];neg=sorted(full,key=lambda x:x['log_mean_contribution'])[:6]
 sections={}
 for item in full:sections.setdefault(item['family'],{})[item['feature']]={'raw_value':item['raw_value'],'standardized_value':item['standardized_value'],'contribution':item['log_mean_contribution']}
 diag={}
 for f in all_features:
  if f not in features:diag.setdefault(family(f),{})[f]=row[f]
 return sections,diag,pos,neg,full
def sample_games(d):
 rng=np.random.default_rng(SEED);g=d.groupby('game_id').agg(season=('season','first'),date=('date','first'),max_pred=('predicted_runs','max'),min_pred=('predicted_runs','min'),mean_pred=('predicted_runs','mean'),max_offense=('season_woba','max'),max_starter_era=('opp_sp_official_season_era','max'),min_starter_era=('opp_sp_official_season_era','min'),max_bp_era=('opp_bp_official_season_era','max'),min_bp_era=('opp_bp_official_season_era','min'),max_variance=('predictive_variance','max')).reset_index();chosen={};n_each=22
 tests={'low_prediction':g.min_pred.le(g.min_pred.quantile(.15)),'average_prediction':g.mean_pred.between(g.mean_pred.quantile(.425),g.mean_pred.quantile(.575)),'high_prediction':g.max_pred.ge(g.max_pred.quantile(.85)),'extreme_prediction':g.max_pred.ge(g.max_pred.quantile(.975)),'elite_offense':g.max_offense.ge(g.max_offense.quantile(.9)),'weak_offense':g.max_offense.le(g.max_offense.quantile(.1)),'weak_starter':g.max_starter_era.ge(g.max_starter_era.quantile(.9)),'strong_starter':g.min_starter_era.le(g.min_starter_era.quantile(.1)),'weak_bullpen':g.max_bp_era.ge(g.max_bp_era.quantile(.9)),'high_uncertainty':g.max_variance.ge(g.max_variance.quantile(.9))}
 for label,mask in tests.items():
  ids=g.loc[mask,'game_id'].to_numpy();take=rng.choice(ids,min(n_each,len(ids)),replace=False)
  for x in take:chosen.setdefault(int(x),[]).append(label)
 remaining=g[~g.game_id.isin(chosen)].game_id.to_numpy();need=SAMPLE_TEAM_GAMES//2-len(chosen)
 for x in rng.choice(remaining,max(0,need),replace=False):chosen[int(x)]=['random_ordinary']
 return chosen
def main():
 R.mkdir(exist_ok=True);P.mkdir(exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);v=load_v11();frozen=json.loads((R/'v11_2_compact_frozen_specification.json').read_text());features=frozen['features'];all_features=json.loads((R/'v11_frozen_specification.json').read_text())['features']
 # Reproduce the frozen pipeline in development folds; no selection or alteration occurs.
 data=v.add_scoring_baselines(pd.concat([v.long_year(y) for y in range(2021,2025)],ignore_index=True));parts=[];models={}
 for yrs,vy in [([2021],2022),([2021,2022],2023),([2021,2022,2023],2024)]:
  tr=data[data.season.isin(yrs)];va=data[data.season.eq(vy)].copy();imp=SimpleImputer(strategy='median');sc=StandardScaler();xt=sc.fit_transform(imp.fit_transform(tr[features]));xv=sc.transform(imp.transform(va[features]));model=PoissonRegressor(alpha=10,max_iter=3000).fit(xt,tr.actual_team_runs);va['predicted_runs']=np.clip(model.predict(xv),.05,None);va['predictive_variance']=va.predicted_runs+ALPHA[vy]*va.predicted_runs**2;va['prediction_interval_low']=[qtile(mu,ALPHA[vy],.05) for mu in va.predicted_runs];va['prediction_interval_high']=[qtile(mu,ALPHA[vy],.95) for mu in va.predicted_runs];va['_z']=[x for x in xv];parts.append(va);models[vy]=(imp,sc,model)
 oos=pd.concat(parts,ignore_index=True);selected=sample_games(oos);sample=oos[oos.game_id.isin(selected)].copy();sample_rows=[];packets=[];audits=[]
 for gid,g in sample.groupby('game_id'):
  away=g[g.team_side.eq('away')].iloc[0];home=g[g.team_side.eq('home')].iloc[0];vy=int(away.season);_,_,model=models[vy];contexts={};diagnostics={};explanations={};warnings=[]
  for side,row in [('away',away),('home',home)]:
   sec,diag,pos,neg,full=team_context(row,features,all_features,np.asarray(row['_z']),model.coef_);contexts[side]=sec;diagnostics[side]=diag;explanations[f'{side}_top_positive']=pos;explanations[f'{side}_top_negative']=neg;warnings += [{**w,'team_side':side} for w in warnings_for(row,features,np.asarray(row['_z']))]
  packet={'schema_version':'v13-context-1.0.0','game':{'game_id':int(gid),'date':str(away.date.date()),'away_team':away.team,'home_team':home.team},'prediction':{'away_expected_runs':away.predicted_runs,'home_expected_runs':home.predicted_runs,'projected_total':away.predicted_runs+home.predicted_runs,'projected_run_difference':home.predicted_runs-away.predicted_runs,'negative_binomial':{'parameterization':'variance=mu+alpha*mu^2','alpha_training_only':ALPHA[vy],'away_variance':away.predictive_variance,'home_variance':home.predictive_variance,'away_90pct_interval':[away.prediction_interval_low,away.prediction_interval_high],'home_90pct_interval':[home.prediction_interval_low,home.prediction_interval_high]}},'away_context':contexts['away'],'home_context':contexts['home'],'diagnostic_context':{'label':'diagnostic_only_not_v11_2_features','away':diagnostics['away'],'home':diagnostics['home']},'data_quality':{'warnings':warnings,'warning_count':len(warnings),'compact_missing_rate_away':away[features].isna().mean(),'compact_missing_rate_home':home[features].isna().mean()},'model_explanation':explanations}
  audit=audit_packet(clean(packet));audits.append({'game_id':gid,'clean':audit['clean'],'forbidden_paths':' | '.join(audit['forbidden_paths']),'context_hash':audit['context_hash'],'packet_keys':' | '.join(packet.keys())});packets.append(clean(packet));sample_rows.append({'game_id':gid,'date':packet['game']['date'],'season':vy,'away_team':away.team,'home_team':home.team,'away_expected_runs':away.predicted_runs,'home_expected_runs':home.predicted_runs,'projected_total':packet['prediction']['projected_total'],'strata':' | '.join(selected[int(gid)]),'team_game_predictions':2,'context_hash':audit['context_hash']})
 auditdf=pd.DataFrame(audits)
 if not auditdf.clean.all():raise RuntimeError('OUTCOME LEAKAGE DETECTED; stopping before provider execution')
 # Save packets before any outcome join. No outcome join occurs in unconfigured mode.
 packet_path=P/'v13_ai_context_packets_development.jsonl';packet_path.write_text('\n'.join(json.dumps(x,separators=(',',':')) for x in packets),encoding='utf-8');pd.DataFrame(sample_rows).to_csv(R/'v13_ai_validator_sample.csv',index=False);auditdf.to_csv(R/'v13_ai_context_packet_audit.csv',index=False);auditdf[['game_id','clean','forbidden_paths','context_hash']].to_csv(R/'v13_ai_outcome_leakage_audit.csv',index=False)
 prompt_record={'prompt_version':PROMPT_VERSION,'prompt_hash':PROMPT_HASH,'system_prompt':SYSTEM_PROMPT,'schema_version':'v13-validation-result-1.0.0','reason_codes_frozen':sorted(__import__('ai_validator.schemas',fromlist=['REASON_CODES']).REASON_CODES),'frozen_before_outcome_evaluation':True};(R/'v13_ai_frozen_prompt_and_schema.json').write_text(json.dumps(prompt_record,indent=2),encoding='utf-8')
 configured=external_configured();provider=ExternalLLMValidatorProvider() if configured else MockValidatorProvider();validator=CachedValidator(provider,CACHE/'responses');results=[]
 # External mode would run the fixed development packets. Mock mode verifies every packet/schema/cache path only.
 for packet in packets:
  row=validator.validate(packet);resp=row.pop('response');results.append({'game_id':packet['game']['game_id'],'run_type':'REAL_EXTERNAL_LLM' if configured else 'MOCK_INFRASTRUCTURE_ONLY_NOT_AI_EVALUATION',**row,**resp})
 res=pd.DataFrame(results);res.to_csv(R/'v13_ai_validation_results.csv',index=False)
 # Re-read through cache to prove deterministic cache keys work.
 cache_check=[validator.validate(x)['cache_hit'] for x in packets]
 if not all(cache_check):raise RuntimeError('cache validation failed')
 # Outcome analyses intentionally remain not-run when no external provider exists.
 columns={'v13_ai_status_error_analysis.csv':['run_status','status','observations','mae','median_ae','rmse','mean_signed_error','p90_ae','p95_ae'],'v13_ai_confidence_analysis.csv':['run_status','confidence_bucket','observations','mae','rmse','extreme_error_rate','spearman_confidence_negative_error'],'v13_ai_reason_code_analysis.csv':['run_status','reason_code','count','mean_absolute_error','baseline_absolute_error','error_difference','rmse','extreme_error_rate'],'v13_ai_baseline_comparison.csv':['run_status','validator','coverage','flagged_error_rate','auc_extreme_error'],'v13_ai_false_flag_analysis.csv':['run_status','status_group','pct_error_le_1','pct_error_le_2','pct_error_ge_5'],'v13_ai_2025_untouched_results.csv':['run_status','note']}
 stop='NOT_RUN_NO_EXTERNAL_LLM' if not configured else 'PENDING_EXTERNAL_DECISIONS_FROZEN_BEFORE_OUTCOME_JOIN'
 for name,cols in columns.items():pd.DataFrame([{c:(stop if c=='run_status' else ('2025 remains untouched' if c=='note' else None)) for c in cols}]).to_csv(R/name,index=False)
 cost=pd.DataFrame([{'run_type':'external' if configured else 'mock_infrastructure_only','provider':provider.name,'model':provider.model,'requests':len(res),'input_tokens':res.input_tokens.sum(min_count=1),'output_tokens':res.output_tokens.sum(min_count=1),'estimated_cost':None,'cost_note':'No pricing configured; cost not invented. Mock requests incur no API cost.' if not configured else 'Configure pricing separately if authoritative.','average_latency_seconds':res.latency_seconds.mean(),'estimated_normal_15_game_day_requests':15,'token_estimate_15_game_day':None}]);cost.to_csv(R/'v13_ai_cost_analysis.csv',index=False)
 first=packets[0];mock=res.iloc[0];human=f"""{first['game']['away_team']} @ {first['game']['home_team']}\n\nV11.2:\n{first['game']['away_team']} expected runs: {first['prediction']['away_expected_runs']:.2f}\n{first['game']['home_team']} expected runs: {first['prediction']['home_expected_runs']:.2f}\nProjected total: {first['prediction']['projected_total']:.2f}\n\nValidator dry run:\n{mock.status}\nConfidence: {mock.confidence_score}/100\nReason codes: {mock.reason_codes}\n\n{mock.explanation}\n""";(R/'v13_ai_human_readable_example.txt').write_text(human,encoding='utf-8')
 if not configured:
  report=f"""# V13 AI contextual validator infrastructure\n\n**Real LLM validation has not yet been evaluated.**\n\nNo external provider variables were configured. The deterministic mock provider reviewed {len(packets)} game packets / {2*len(packets)} team-game predictions solely to verify construction, strict schema parsing, caching, reason-code enforcement, prompt hashing, and outcome-blind safeguards. It must not be interpreted as AI performance.\n\nLeakage audit: {auditdf.clean.sum()}/{len(auditdf)} packets clean; zero forbidden outcome fields. 2025 was not loaded or evaluated. No outcome/error tables were calculated.\n\nFrozen prompt version `{PROMPT_VERSION}`, SHA-256 `{PROMPT_HASH}`.\n\nTo run a real development evaluation, set `V13_VALIDATOR_PROVIDER`, `V13_VALIDATOR_MODEL`, `V13_VALIDATOR_ENDPOINT`, and `V13_VALIDATOR_API_KEY` in the process environment, then run `python evaluate-v13-ai-validator.py`. The endpoint must accept an OpenAI-compatible chat-completions JSON request and return strict JSON content. Do not place credentials in source or output files. After external decisions are cached and frozen, a separate outcome-join evaluation may be run; only after development methodology remains frozen should a fixed 2025 sample be evaluated.\n\nThe validator should not be added to production until it beats or complements deterministic baselines on development and untouched 2025.\n"""
 else:report='# V13 external development inference completed\n\nExternal decisions are frozen. Outcome evaluation must be executed as a separate, explicit phase before any 2025 inference.\n'
 (R/'v13_ai_final_report.md').write_text(report,encoding='utf-8');print(report);print('Packets',len(packets),'team predictions',2*len(packets),'cache verified',all(cache_check))
if __name__=='__main__':main()
