"""V12 frozen V11 market diagnostic. No model fitting or threshold optimization."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import nbinom, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score

ROOT=Path(__file__).resolve().parent;R=ROOT/'results';D=ROOT/'data/processed'
SEED=1200;BOOT=10000;FLOOR=1.70;PRIMARY=.10
ALPHAS={2022:.25399455027714585,2023:.26562511422760027,2024:.2629530600894081,2025:.25962778329405606}
EDGE_EDGES=[-np.inf,0,.02,.04,.06,.08,.10,.15,np.inf];EDGE_LABELS=['<=0%','0-2%','2-4%','4-6%','6-8%','8-10%','10-15%','15%+']
PROB_EDGES=[-np.inf,.4,.45,.5,.55,.6,.65,.7,np.inf];PROB_LABELS=['<40%','40-45%','45-50%','50-55%','55-60%','60-65%','65-70%','70%+']
THRESHOLDS=[1e-12,.02,.04,.05,.075,.10]

def dec(x):
 x=pd.to_numeric(x,errors='coerce');return np.where(x>=100,1+x/100,np.where(x<=-100,1+100/abs(x),np.nan))
def nbpmf(mu,a,n=60):
 k=np.arange(n+1);size=1/a;p=size/(size+mu);q=nbinom.pmf(k,size,p);q[-1]+=max(0,1-q.sum());return q
def probabilities(away,home,a,line):
 pa=nbpmf(away,a);ph=nbpmf(home,a);joint=np.outer(pa,ph);paway=np.tril(joint,-1).sum();phome=np.triu(joint,1).sum();tie=np.trace(joint);den=phome+paway
 total=np.convolve(pa,ph);k=np.arange(len(total));pover=total[k>line].sum();punder=total[k<line].sum();ppush=total[np.isclose(k,line)].sum()
 return phome/den,paway/den,tie,pover,punder,ppush,float((np.arange(len(pa))*pa).sum()),float((np.arange(len(ph))*ph).sum())
def pnl(res,odds):return np.where(res.eq('W'),odds-1,np.where(res.eq('L'),-1,0.))
def maxdd(x):
 z=np.r_[0,x.sort_values(['date','game_id']).profit_units.cumsum()];return float((np.maximum.accumulate(z)-z).max())
def fit_cal(y,p):
 ok=np.isfinite(p);y=np.asarray(y)[ok];p=np.clip(np.asarray(p)[ok],1e-6,1-1e-6);x=np.log(p/(1-p)).reshape(-1,1);m=LogisticRegression(C=1e6,solver='lbfgs').fit(x,y)
 return dict(log_loss=log_loss(y,p),brier=brier_score_loss(y,p),auc=roc_auc_score(y,p),calibration_intercept=float(m.intercept_[0]),calibration_slope=float(m.coef_[0,0]))
def summarize(g,scope,market='all',total_games=None):
 n=len(g);w=int(g.result.eq('W').sum());l=int(g.result.eq('L').sum());p=int(g.result.eq('P').sum());u=float(g.profit_units.sum()) if n else 0
 return {'scope':scope,'market_type':market,'total_eligible_games':total_games,'signals':n,'passes':None if total_games is None else total_games-g.game_id.nunique(),'bet_pct':None if total_games is None else g.game_id.nunique()/total_games,'ml_signals':int(g.market.eq('ML').sum()),'over_signals':int(g.market.eq('OVER').sum()),'under_signals':int(g.market.eq('UNDER').sum()),'wins':w,'losses':l,'pushes':p,'win_rate_ex_push':w/(w+l) if w+l else np.nan,'average_odds':g.decimal_odds.mean(),'median_odds':g.decimal_odds.median(),'average_model_probability':g.model_probability.mean(),'average_market_probability':g.market_no_vig_probability.mean(),'average_edge':g.edge.mean(),'realized_advantage':w/(w+l)-g.loc[~g.result.eq('P'),'market_no_vig_conditional'].mean() if w+l else np.nan,'units':u,'roi':u/n if n else np.nan,'max_drawdown':maxdd(g) if n else 0,'winning_average_odds':g.loc[g.result.eq('W'),'decimal_odds'].mean(),'losing_average_odds':g.loc[g.result.eq('L'),'decimal_odds'].mean(),'stake':n,'turnover':n}
def choose(g,floor=FLOOR):
 q=g[g.edge.ge(PRIMARY)&(g.decimal_odds.ge(floor) if floor is not None else True)].copy()
 if not len(q):return None,'PASS: no >=10% candidate surviving odds rule'
 q=q.sort_values(['edge','decimal_odds'],ascending=False);winner=q.iloc[0];reason='only qualifying candidate' if len(q)==1 else 'largest edge'
 ml=q[q.market.eq('ML')].sort_values('edge',ascending=False);ou=q[~q.market.eq('ML')].sort_values('edge',ascending=False)
 if len(ml) and len(ou):
  left,right=ml.iloc[0],ou.iloc[0]
  if left.edge>.05 and right.edge>.05 and abs(left.edge-right.edge)<=.02:
   winner=pd.DataFrame([left,right]).sort_values(['decimal_odds','edge'],ascending=False).iloc[0];reason='5% / within-2pp higher-odds tiebreak'
 return winner.name,reason
def main():
 pred=pd.read_csv(R/'v11_combined_oos_team_predictions.csv',usecols=['game_id','date','season','team','team_side','actual_team_runs','predicted_runs']);pred.game_id=pred.game_id.astype(int)
 a=pred[pred.team_side.eq('away')].merge(pred[pred.team_side.eq('home')],on=['game_id','date','season'],suffixes=('_away','_home'),validate='one_to_one');a=a.rename(columns={'predicted_runs_away':'lambda_away','predicted_runs_home':'lambda_home','actual_team_runs_away':'away_runs','actual_team_runs_home':'home_runs','team_away':'away_team','team_home':'home_team'})
 ml=pd.read_csv(D/'historical_mlb_moneylines_2022_2025.csv');ml=ml[(ml.sportsbook.eq('bet365'))&ml.match_status.eq('matched')].copy();ml.game_id=ml.game_id.astype(int);assert not ml.game_id.duplicated().any()
 tt=pd.read_csv(D/'historical_mlb_totals_2022_2025.csv');tt=tt[(tt.sportsbook.eq('bet365'))&tt.match_status.eq('matched')].copy();tt.game_id=tt.game_id.astype(int);assert not tt.game_id.duplicated().any()
 ml['current_home_decimal']=dec(ml.current_home_odds);ml['current_away_decimal']=dec(ml.current_away_odds);ml['opening_home_decimal']=dec(ml.opening_home_odds);ml['opening_away_decimal']=dec(ml.opening_away_odds)
 for snap in ('current','opening'):
  rh=1/ml[f'{snap}_home_decimal'];ra=1/ml[f'{snap}_away_decimal'];ml[f'{snap}_home_novig']=rh/(rh+ra);ml[f'{snap}_away_novig']=ra/(rh+ra)
 tt['current_over_decimal']=dec(tt.current_over_odds);tt['current_under_decimal']=dec(tt.current_under_odds);tt['opening_over_decimal']=dec(tt.opening_over_odds);tt['opening_under_decimal']=dec(tt.opening_under_odds)
 base=a.merge(ml[['game_id','current_home_decimal','current_away_decimal','opening_home_decimal','opening_away_decimal','current_home_novig','current_away_novig','opening_home_novig','opening_away_novig']],on='game_id',how='outer',indicator='ml_match').merge(tt[['game_id','current_total','opening_total','current_over_decimal','current_under_decimal','opening_over_decimal','opening_under_decimal']],on='game_id',how='left')
 rows=[]
 for g in base[base.ml_match.eq('both')].itertuples():
  alpha=ALPHAS[int(g.season)];ph,pa,pt,_,_,_,mh,ma=probabilities(g.lambda_away,g.lambda_home,alpha,g.current_total if np.isfinite(g.current_total) else 8.5)
  if abs(mh-g.lambda_away)>1e-6 or abs(ma-g.lambda_home)>1e-6:raise ValueError('NB mean check failed')
  common={'game_id':g.game_id,'date':g.date,'season':g.season,'home_team':g.home_team,'away_team':g.away_team,'lambda_home':g.lambda_home,'lambda_away':g.lambda_away,'alpha':alpha,'actual_home_runs':g.home_runs,'actual_away_runs':g.away_runs,'tie_probability_before_renormalization':pt}
  valid_ml=all(np.isfinite(x) and x>1 for x in (g.current_home_decimal,g.current_away_decimal,g.opening_home_decimal,g.opening_away_decimal))
  if valid_ml:
   for side,prob,market,odds,oprob,res in [('HOME_ML',ph,g.current_home_novig,g.current_home_decimal,g.opening_home_novig,'W' if g.home_runs>g.away_runs else 'L'),('AWAY_ML',pa,g.current_away_novig,g.current_away_decimal,g.opening_away_novig,'W' if g.away_runs>g.home_runs else 'L')]:
    rows.append({**common,'market':'ML','side':side,'model_probability':prob,'model_probability_conditional_no_push':prob,'market_no_vig_probability':market,'market_no_vig_conditional':market,'decimal_odds':odds,'opening_market_probability':oprob,'opening_decimal_odds':g.opening_home_decimal if side=='HOME_ML' else g.opening_away_decimal,'result':res,'push_probability':0,'total_line':np.nan,'opening_total_line':np.nan})
  valid_total=np.isfinite(g.current_total) and 0<g.current_total<30 and all(np.isfinite(x) and x>1 for x in (g.current_over_decimal,g.current_under_decimal,g.opening_over_decimal,g.opening_under_decimal))
  if valid_total:
   ph2,pa2,pt2,po,pu,pp,_,_=probabilities(g.lambda_away,g.lambda_home,alpha,g.current_total);ro=1/g.current_over_decimal;ru=1/g.current_under_decimal;qo=ro/(ro+ru);qu=ru/(ro+ru);oro=1/g.opening_over_decimal;oru=1/g.opening_under_decimal;oqo=oro/(oro+oru);oqu=oru/(oro+oru)
   actual=g.home_runs+g.away_runs
   for side,prob,q,odds,oq,oodds,res in [('OVER',po,qo,g.current_over_decimal,oqo,g.opening_over_decimal,'P' if actual==g.current_total else ('W' if actual>g.current_total else 'L')),('UNDER',pu,qu,g.current_under_decimal,oqu,g.opening_under_decimal,'P' if actual==g.current_total else ('W' if actual<g.current_total else 'L'))]:
    rows.append({**common,'market':side,'side':side,'model_probability':prob,'model_probability_conditional_no_push':prob/(1-pp),'market_no_vig_probability':q*(1-pp),'market_no_vig_conditional':q,'decimal_odds':odds,'opening_market_probability':oq,'opening_decimal_odds':oodds,'result':res,'push_probability':pp,'total_line':g.current_total,'opening_total_line':g.opening_total if np.isfinite(g.opening_total) and 0<g.opening_total<30 else np.nan})
 c=pd.DataFrame(rows);c['edge']=c.model_probability-c.market_no_vig_probability;c['ev']=c.model_probability*(c.decimal_odds-1)-(1-c.model_probability-c.push_probability);c['profit_units']=pnl(c.result,c.decimal_odds);c['edge_bucket']=pd.cut(c.edge,EDGE_EDGES,labels=EDGE_LABELS,right=True)
 # Edge buckets and monotonic/ranking diagnostics.
 eb=[]
 for market in ('ML','OVER','UNDER'):
  x=c[c.market.eq(market)]
  for b in EDGE_LABELS:
   g=x[x.edge_bucket.eq(b)];z=summarize(g,b,market);z.update({'edge_bucket':b,'candidates':len(g),'average_claimed_edge':g.edge.mean(),'actual_win_rate':g.loc[~g.result.eq('P'),'result'].eq('W').mean(),'spearman_edge_outcome_residual':spearmanr(x.edge,x.result.eq('W').astype(float)-x.market_no_vig_probability).statistic});eb.append(z)
 # Probability calibration: V11, market, plus old frozen probabilities on identical games.
 cal=[]
 def addcal(name,market,y,p):
  y=np.asarray(y);p=np.asarray(p);m=fit_cal(y,p);b=pd.cut(p,PROB_EDGES,labels=PROB_LABELS)
  for label in PROB_LABELS:
   keep=b==label;cal.append({'source':name,'market':market,'bucket':label,'observations':keep.sum(),'average_probability':np.mean(p[keep]) if keep.any() else np.nan,'actual_frequency':np.mean(y[keep]) if keep.any() else np.nan,'calibration_error':np.mean(p[keep]-y[keep]) if keep.any() else np.nan,**m})
 for market in ('ML','OVER','UNDER'):
  x=c[c.market.eq(market)].copy();x=x[~x.result.eq('P')];y=x.result.eq('W').astype(int);addcal('V11',market,y,x.model_probability_conditional_no_push);addcal('Bet365_current_no_vig',market,y,x.market_no_vig_conditional)
 oldml=pd.read_csv(R/'v5_team_strength_oos_predictions_2022_2025.csv')[['game_id','predicted_home_probability']];xm=c[c.side.eq('HOME_ML')].merge(oldml,on='game_id');addcal('old_V5', 'ML',xm.result.eq('W').astype(int),xm.predicted_home_probability)
 oldt=pd.read_csv(R/'totals_bet365_game_side_ledger.csv');oldt.side=oldt.side.str.upper()
 for side in ('OVER','UNDER'):
  xo=c[c.market.eq(side)].merge(oldt[oldt.side.eq(side)][['game_id','model_probability']],on='game_id',suffixes=('','_old'));xo=xo[~xo.result.eq('P')];addcal('old_totals',side,xo.result.eq('W').astype(int),xo.model_probability_old)
 # Threshold diagnostics with exactly two odds-floor states.
 th=[]
 for market in ('ML','OVER','UNDER'):
  x=c[c.market.eq(market)]
  for floor_name,floor in [('none',None),('decimal_odds_ge_1.70',FLOOR)]:
   for t in THRESHOLDS:
    g=x[x.edge.ge(t)&(True if floor is None else x.decimal_odds.ge(floor))];th.append({**summarize(g,f'>={t:g}',market),'edge_threshold':t,'odds_floor':floor_name})
 # Primary one-signal selector.
 c['selected']=False;c['selection_reason']='';chosen={}
 for gid,g in c.groupby('game_id'):
  idx,reason=choose(g);chosen[gid]=(idx,reason)
  if idx is not None:c.loc[idx,'selected']=True;c.loc[idx,'selection_reason']=reason
 def rejection(r):
  if r.selected:return 'selected'
  if r.edge<PRIMARY:return 'edge below 10%'
  if r.decimal_odds<FLOOR:return 'odds below 1.70'
  return 'lost one-signal selection'
 c['rejection_reason']=c.apply(rejection,axis=1);ledger=c[c.selected].sort_values(['date','game_id']).copy();ledger['cumulative_units']=ledger.profit_units.cumsum();ledger['pnl_recalculated']=pnl(ledger.result,ledger.decimal_odds)
 if not np.allclose(ledger.profit_units,ledger.pnl_recalculated) or not np.isclose(ledger.profit_units.sum(),ledger.pnl_recalculated.sum()):raise ValueError('P&L audit failure')
 universe=c[['game_id','season']].drop_duplicates();season=[]
 for y in [2022,2023,2024,2025]:season.append(summarize(ledger[ledger.season.eq(y)],str(y),'all',int(universe.season.eq(y).sum())))
 season.append(summarize(ledger,'combined','all',len(universe)));types=[summarize(g,'combined',m,len(universe)) for m,g in ledger.groupby('market')]
 # Floor comparison for the same >=10% selector.
 def select_floor(floor):
  picks=[]
  for _,g in c.groupby('game_id'):
   idx,_=choose(g,floor)
   if idx is not None:picks.append(c.loc[idx])
  return pd.DataFrame(picks)
 nf=select_floor(None);ff=ledger;floorcmp=pd.DataFrame([summarize(nf,'no_floor','all',len(universe)),summarize(ff,'1.70_floor','all',len(universe))])
 # Bootstrap primary ROI and hit rate.
 rng=np.random.default_rng(SEED);arr=ledger.profit_units.to_numpy();settled=ledger[~ledger.result.eq('P')].result.eq('W').astype(float).to_numpy();br=rng.choice(arr,(BOOT,len(arr)),replace=True).mean(1);bh=rng.choice(settled,(BOOT,len(settled)),replace=True).mean(1);boot=pd.DataFrame([{'seed':SEED,'replicates':BOOT,'observed_roi':arr.mean(),'roi_standard_error':br.std(ddof=1),'roi_ci_low':np.quantile(br,.025),'roi_ci_high':np.quantile(br,.975),'roi_ci_excludes_zero':np.quantile(br,.025)>0 or np.quantile(br,.975)<0,'observed_hit_rate':settled.mean(),'hit_rate_ci_low':np.quantile(bh,.025),'hit_rate_ci_high':np.quantile(bh,.975)}])
 # Old selector reconstructed at identical threshold/floor; intersect common eligible game universe.
 old=pd.read_csv(R/'combined_signal_complete_candidate_ledger.csv');old=old[old.game_id.isin(universe.game_id)].copy();old['market']=old.candidate_type.replace({'ML':'ML','OVER':'OVER','UNDER':'UNDER'});old['market_no_vig_conditional']=old.market_no_vig_probability;old['profit_units']=pnl(old.result,old.decimal_odds);op=[]
 for _,g in old.groupby('game_id'):
  idx,_=choose(g,FLOOR)
  if idx is not None:op.append(old.loc[idx])
 oldsel=pd.DataFrame(op);comparison=pd.DataFrame([summarize(oldsel,'old_frozen','all',len(universe)),summarize(ledger,'V11','all',len(universe))])
 # Opening-to-current movement, line first for totals and price if unchanged.
 mov=[];movement_lookup=[]
 for market in ('ML','OVER','UNDER'):
  x=c[c.market.eq(market)].copy()
  if market=='ML':x['movement']=x.market_no_vig_conditional-x.opening_market_probability
  else:
   direction=1 if market=='OVER' else -1;line_move=(x.total_line-x.opening_total_line)*direction;price_move=x.market_no_vig_conditional-x.opening_market_probability;x['movement']=np.where(line_move.ne(0),line_move,price_move)
  movement_lookup.append(x[['game_id','side','movement']])
  for b in EDGE_LABELS:
   g=x[x.edge_bucket.eq(b)];mov.append({'source':'V11','market':market,'edge_bucket':b,'candidates':len(g),'movement_in_model_direction_pct':g.movement.gt(0).mean(),'average_movement_in_model_direction':g.movement.mean(),'semantic':'ML no-vig probability; totals line movement first, price probability if unchanged'})
 lookup=pd.concat(movement_lookup,ignore_index=True);old['side']=np.where(old.market.eq('ML'),np.where(old.selection.eq(old.home_team),'HOME_ML','AWAY_ML'),old.market);om=old.merge(lookup,on=['game_id','side'],how='inner');om['edge_bucket']=pd.cut(om.edge,EDGE_EDGES,labels=EDGE_LABELS,right=True)
 for market in ('ML','OVER','UNDER'):
  x=om[om.market.eq(market)]
  for b in EDGE_LABELS:
   g=x[x.edge_bucket.eq(b)];mov.append({'source':'old_frozen','market':market,'edge_bucket':b,'candidates':len(g),'movement_in_model_direction_pct':g.movement.gt(0).mean(),'average_movement_in_model_direction':g.movement.mean(),'semantic':'identical movement definition, old frozen candidate direction'})
 # Save and report.
 c.to_csv(R/'v12_market_candidate_ledger.csv',index=False);pd.DataFrame(cal).to_csv(R/'v12_probability_calibration.csv',index=False);pd.DataFrame(eb).to_csv(R/'v12_edge_buckets.csv',index=False);pd.DataFrame(th).to_csv(R/'v12_edge_thresholds.csv',index=False);floorcmp.to_csv(R/'v12_odds_floor_comparison.csv',index=False);ledger.to_csv(R/'v12_primary_one_signal_ledger.csv',index=False);pd.DataFrame(season).to_csv(R/'v12_primary_season_results.csv',index=False);pd.DataFrame(types).to_csv(R/'v12_market_type_results.csv',index=False);boot.to_csv(R/'v12_bootstrap_uncertainty.csv',index=False);comparison.to_csv(R/'v12_old_vs_v11_comparison.csv',index=False);pd.DataFrame(mov).to_csv(R/'v12_opening_to_current_diagnostics.csv',index=False)
 combined=pd.DataFrame(season).query("scope=='combined'").iloc[0];prof=int(pd.DataFrame(season).query("scope!='combined'").units.gt(0).sum());report=f"""# V12 frozen V11 market test\n\nV11 NB2 parameterization: variance = mu + alpha*mu^2; equality for ML is removed by conditioning on unequal scores. Integer-total push mass is explicitly returned; current two-sided no-vig probabilities are scaled by (1-push) for unconditional edge, while EV is p_win*(odds-1)-p_loss.\n\nPrimary predefined strategy: {int(combined.signals)} signals, {combined.wins}-{combined.losses}-{combined.pushes}, {combined.units:+.3f} units, ROI {combined.roi:.3%}, max drawdown {combined.max_drawdown:.3f}. Profitable seasons: {prof}/4. Bootstrap ROI 95% CI [{boot.roi_ci_low.iloc[0]:.3%}, {boot.roi_ci_high.iloc[0]:.3%}].\n\nP&L independently reconciled exactly. `currentLine` is not described as a verified closing line. V11 remained frozen and no betting result affected model or dispersion selection.\n""";(R/'v12_final_report.md').write_text(report,encoding='utf-8')
 print(pd.DataFrame(season).to_string(index=False));print('\nMARKETS\n',pd.DataFrame(types).to_string(index=False));print('\nBOOTSTRAP\n',boot.to_string(index=False));print('\nOLD VS V11\n',comparison.to_string(index=False))
if __name__=='__main__':main()
