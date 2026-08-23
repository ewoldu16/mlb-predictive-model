from pathlib import Path
from flask import Flask,render_template,jsonify,request,abort,send_from_directory,redirect,url_for,session,flash
from functools import wraps
from datetime import timedelta
from werkzeug.security import check_password_hash
import hmac,json,os,pandas as pd
from mlb_app.model_service import V112ModelService
from mlb_app.performance import load_performance
from mlb_app.live_pipeline import load_today
from mlb_app.transparency import build_transparency
from mlb_app.storage import database_url,health_state,load_rebuild_request,queue_rebuild_request
from mlb_app.live_tracking import load_live_tracking
from mlb_app.owner_controls import AVAILABILITY_STATUSES,audit_history,bootstrap_team,csrf_token,load_team_state,locked,offense_confidence,record_rebuild,replacement_suggestions,save_lineup,save_team_state,set_availability
from mlb_app.refresh_service import refresh_cycle

ROOT=Path(__file__).resolve().parent
def create_app(test_config=None):
 app=Flask(__name__,template_folder='website/templates',static_folder='website/static');app.config.update(JSON_SORT_KEYS=False,SECRET_KEY=os.getenv('SECRET_KEY') or os.urandom(32),OWNER_USERNAME=os.getenv('OWNER_USERNAME'),OWNER_PASSWORD_HASH=os.getenv('OWNER_PASSWORD_HASH'),PERMANENT_SESSION_LIFETIME=timedelta(minutes=int(os.getenv('OWNER_SESSION_MINUTES','30'))),SESSION_COOKIE_HTTPONLY=True,SESSION_COOKIE_SAMESITE='Strict',SESSION_COOKIE_SECURE=os.getenv('FLASK_ENV')=='production' or bool(os.getenv('RENDER')))
 if test_config:app.config.update(test_config)
 if not app.config.get('TESTING') and (app.config.get('OWNER_USERNAME') or app.config.get('OWNER_PASSWORD_HASH')) and not os.getenv('SECRET_KEY'):raise RuntimeError('SECRET_KEY is required when owner authentication is configured')
 service=V112ModelService(ROOT);performance=load_performance(ROOT);history_path=ROOT/'results/v11_2_confidence_oos_game_predictions_2022_2025.csv'
 def history():return pd.read_csv(history_path)
 def owner_required(fn):
  @wraps(fn)
  def wrapped(*args,**kwargs):
   if not session.get('owner_authenticated'):return redirect(url_for('owner_login',next=request.path))
   return fn(*args,**kwargs)
  return wrapped
 def require_csrf():
  token=request.form.get('_csrf_token') or request.headers.get('X-CSRF-Token');expected=session.get('_csrf_token')
  if not token or not expected or not hmac.compare_digest(token,expected):abort(400,'invalid CSRF token')
 def game_by_id(game_id):return next((x for x in load_today(ROOT).get('games',[]) if int(x['game_id'])==int(game_id)),None)
 def rebuild_if_ready(game):
  states={side:load_team_state(ROOT,game['game_id'],side) for side in ('away','home')}
  if database_url():
   valid=bool(all(states.values()) and all(s.get('valid') for s in states.values()));queue_rebuild_request(game['game_id'],{'game_id':int(game['game_id']),'date':game['date'],'requested_at':pd.Timestamp.utcnow().isoformat(),'valid':valid,'lineup_versions':{side:states[side]['version'] for side in states if states[side]}},'owner_lineup_changed' if valid else 'PROVISIONAL_LINEUP_INCOMPLETE');return None
  if not all(states.values()) or not all(s.get('valid') for s in states.values()):return None
  before=game_by_id(game['game_id']);payload=refresh_cycle(ROOT,service,game['date']);after=next((x for x in payload.get('games',[]) if int(x['game_id'])==int(game['game_id'])),None)
  if before and after and before.get('prediction') and after.get('prediction'):
   def raw(snapshot,side,feature):return next((x.get('raw_value') for x in snapshot.get('feature_vectors',{}).get(side,[]) if x.get('feature')==feature),None)
   impact={'before_timestamp':before.get('snapshot',{}).get('generated_at'),'after_timestamp':after.get('snapshot',{}).get('generated_at'),'away_expected_runs_change':after['prediction']['away']['expected_runs']-before['prediction']['away']['expected_runs'],'home_expected_runs_change':after['prediction']['home']['expected_runs']-before['prediction']['home']['expected_runs'],'home_win_probability_change':after['prediction']['home_win_probability']-before['prediction']['home_win_probability'],'lineup_woba':{side:{'before':raw(before,side,'lineup_season_woba'),'after':raw(after,side,'lineup_season_woba')} for side in ('away','home')}}
   for state in states.values():state['last_impact']=impact;save_team_state(ROOT,state)
  if after:
   for state in states.values():record_rebuild(ROOT,state,session.get('owner_id','owner'),after)
  return after
 @app.context_processor
 def owner_context():return {'csrf_token':lambda:csrf_token(session)}
 @app.template_filter('pct')
 def pct(x,d=1):
  try:value=float(x)
  except (TypeError,ValueError):return 'N/A'
  return f'{value*100:.{d}f}%' if pd.notna(value) and value not in (float('inf'),float('-inf')) else 'N/A'
 @app.template_filter('num')
 def num(x,d=2):
  try:value=float(x)
  except (TypeError,ValueError):return 'N/A'
  return f'{value:.{d}f}' if pd.notna(value) and value not in (float('inf'),float('-inf')) else 'N/A'
 @app.route('/')
 def home():return render_template('home_current.html',payload=load_today(ROOT),performance=performance,refresh=health_state())
 @app.route('/game/<int:game_id>')
 def game_detail(game_id):
  live=next((x for x in load_today(ROOT).get('games',[]) if x['game_id']==game_id),None)
  if live:
   html=render_template('game_live.html',game=live,t=build_transparency(ROOT,live,service,load_today(ROOT)))
   if live.get('lineup_status')=='owner_managed':html=html.replace('Using confirmed lineup','Using owner-managed provisional lineup').replace('Lineup uncertainty: provisional.','Lineup uncertainty: owner-managed provisional lineup.')
   return html
  h=history();row=h[h.game_id.eq(game_id)]
  if row.empty:abort(404)
  r=row.iloc[0].to_dict();return render_template('game.html',game=r,historical=True)
 @app.route('/methodology')
 def methodology():return render_template('methodology.html',metadata=service.meta)
 @app.route('/performance')
 def performance_page():return render_template('performance.html',p=performance)
 @app.route('/live-tracking')
 def live_tracking_page():return render_template('live_tracking.html',tracking=load_live_tracking(ROOT))
 @app.route('/about')
 def about():return render_template('about.html',p=performance)
 @app.route('/owner/login',methods=['GET','POST'])
 def owner_login():
  if not app.config.get('OWNER_USERNAME') or not app.config.get('OWNER_PASSWORD_HASH'):abort(503,'owner authentication is not configured')
  if request.method=='POST':
   require_csrf();username=app.config.get('OWNER_USERNAME') or '';password_hash=app.config.get('OWNER_PASSWORD_HASH') or ''
   if username and password_hash and hmac.compare_digest(request.form.get('username',''),username) and check_password_hash(password_hash,request.form.get('password','')):
    session.clear();session['owner_authenticated']=True;session['owner_id']=username;session.permanent=True;csrf_token(session);return redirect(url_for('owner_dashboard'))
   flash('Invalid owner credentials.');return render_template('owner_login.html'),401
  return render_template('owner_login.html')
 @app.post('/owner/logout')
 @owner_required
 def owner_logout():require_csrf();session.clear();return redirect(url_for('home'))
 @app.route('/owner')
 @owner_required
 def owner_dashboard():
  payload=load_today(ROOT);games=[]
  for game in payload.get('games',[]):
   states={side:load_team_state(ROOT,game['game_id'],side) for side in ('away','home')};games.append({'game':game,'states':states,'locked':locked(game),'rebuild':load_rebuild_request(game['game_id'])})
  return render_template('owner_dashboard.html',date=payload.get('date'),games=games)
 @app.route('/owner/game/<int:game_id>/<side>')
 @owner_required
 def owner_team(game_id,side):
  if side not in {'away','home'}:abort(404)
  game=game_by_id(game_id)
  if not game:abort(404)
  state=bootstrap_team(ROOT,game,side);suggestions={order:replacement_suggestions(state,order) for order in state.get('empty_positions',[])};vector=(game.get('feature_vectors') or {}).get(side)
  return render_template('owner_team.html',game=game,state=state,statuses=AVAILABILITY_STATUSES,suggestions=suggestions,locked=locked(game),audit=audit_history(ROOT,game_id)[-20:],offense_confidence=offense_confidence(state,vector))
 @app.post('/owner/game/<int:game_id>/<side>/availability')
 @owner_required
 def owner_availability(game_id,side):
  require_csrf();game=game_by_id(game_id)
  if not game or side not in {'away','home'}:abort(404)
  try:set_availability(ROOT,game,side,int(request.form['player_id']),request.form['status'],session['owner_id']);rebuild_if_ready(game)
  except PermissionError as exc:abort(409,str(exc))
  except ValueError as exc:abort(400,str(exc))
  return redirect(url_for('owner_team',game_id=game_id,side=side))
 @app.post('/owner/game/<int:game_id>/<side>/lineup')
 @owner_required
 def owner_save_lineup(game_id,side):
  require_csrf();game=game_by_id(game_id)
  if not game or side not in {'away','home'}:abort(404)
  lineup=[]
  for order in range(1,10):
   value=request.form.get(f'player_{order}','').strip()
   if value:lineup.append({'order':order,'player_id':int(value),'position':request.form.get(f'position_{order}')})
  try:save_lineup(ROOT,game,side,lineup,session['owner_id']);rebuild_if_ready(game)
  except PermissionError as exc:abort(409,str(exc))
  except ValueError as exc:flash(str(exc));return redirect(url_for('owner_team',game_id=game_id,side=side))
  return redirect(url_for('owner_team',game_id=game_id,side=side))
 @app.post('/owner/game/<int:game_id>/rebuild')
 @owner_required
 def owner_rebuild(game_id):
  require_csrf();game=game_by_id(game_id)
  if not game:abort(404)
  if locked(game):abort(409,'first-pitch lock is active')
  states=[load_team_state(ROOT,game_id,side) for side in ('away','home')]
  if not all(states) or not all(s.get('valid') for s in states):abort(409,'both teams require exactly nine valid AVAILABLE hitters')
  result=rebuild_if_ready(game);flash('Provisional V11.2 forecast rebuilt.' if result else 'Valid owner lineup saved; the dedicated refresh worker has been asked to rebuild V11.2.');return redirect(url_for('owner_dashboard'))
 @app.route('/history')
 def history_page():
  h=history();team=request.args.get('team','').strip();season=request.args.get('season',type=int);confidence=request.args.get('confidence','').strip().upper();date=request.args.get('date','').strip()
  if team:h=h[h.team_away.str.contains(team,case=False,na=False)|h.team_home.str.contains(team,case=False,na=False)]
  if season:h=h[h.season.eq(season)]
  if date:h=h[h.date.astype(str).str.startswith(date)]
  if confidence:
   labels=pd.cut(h.favorite_probability,[.5,.55,.60,1.01],labels=['LOW','MODERATE','HIGH'],right=False);h=h[labels.eq(confidence)]
  return render_template('history.html',rows=h.sort_values('date',ascending=False).head(250).to_dict('records'),filters={'team':team,'season':season,'confidence':confidence,'date':date})
 @app.route('/api/games/today')
 def api_today():return jsonify(load_today(ROOT))
 @app.route('/api/game/<int:game_id>')
 def api_game(game_id):
  live=next((x for x in load_today(ROOT).get('games',[]) if x['game_id']==game_id),None)
  if live:return jsonify(live)
  h=history();row=h[h.game_id.eq(game_id)]
  if row.empty:return jsonify({'error':'game not found'}),404
  return jsonify(row.iloc[0].where(pd.notna(row.iloc[0]),None).to_dict())
 @app.route('/api/model/performance')
 def api_performance():return jsonify(performance)
 @app.route('/api/model/metadata')
 def api_metadata():return jsonify(service.meta)
 @app.route('/health')
 def health():
  refresh=health_state()
  return jsonify({'healthy':True,'model_loaded':True,'model_version':service.meta['model_version'],'artifact_verified':True,'persistence':'supabase_postgres' if database_url() else 'local_filesystem','refresh_executor':'github_actions' if database_url() else 'local_manual','last_successful_live_refresh':refresh.get('last_successful_refresh'),'refresh_status':refresh.get('status','unknown'),'current_data_date':refresh.get('current_data_date')})
 @app.route('/api/predictions/history')
 def api_history():
  h=history();season=request.args.get('season',type=int);team=request.args.get('team','')
  if season:h=h[h.season.eq(season)]
  if team:h=h[h.team_away.str.contains(team,case=False,na=False)|h.team_home.str.contains(team,case=False,na=False)]
  limit=min(request.args.get('limit',100,type=int),500);frame=h.sort_values('date',ascending=False).head(limit).copy();return jsonify({'count':len(frame),'predictions':frame.where(pd.notna(frame),None).to_dict('records')})
 @app.route('/figures/<path:name>')
 def figures(name):return send_from_directory(ROOT/'results/figures',name)
 @app.errorhandler(404)
 def missing(_):return render_template('404.html'),404
 return app
app=create_app()
if __name__=='__main__':app.run(debug=False,host='127.0.0.1',port=5000)
