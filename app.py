from pathlib import Path
from flask import Flask,render_template,jsonify,request,abort,send_from_directory
import json,pandas as pd
from mlb_app.model_service import V112ModelService
from mlb_app.performance import load_performance
from mlb_app.live_pipeline import load_today
from mlb_app.transparency import build_transparency

ROOT=Path(__file__).resolve().parent
def create_app(test_config=None):
 app=Flask(__name__,template_folder='website/templates',static_folder='website/static');app.config.update(JSON_SORT_KEYS=False)
 if test_config:app.config.update(test_config)
 service=V112ModelService(ROOT);performance=load_performance(ROOT);history_path=ROOT/'results/v11_2_confidence_oos_game_predictions_2022_2025.csv'
 def history():return pd.read_csv(history_path)
 @app.template_filter('pct')
 def pct(x,d=1):return f'{float(x)*100:.{d}f}%'
 @app.template_filter('num')
 def num(x,d=2):return f'{float(x):.{d}f}'
 @app.route('/')
 def home():return render_template('home.html',payload=load_today(ROOT),performance=performance)
 @app.route('/game/<int:game_id>')
 def game_detail(game_id):
  live=next((x for x in load_today(ROOT).get('games',[]) if x['game_id']==game_id),None)
  if live:return render_template('game_live.html',game=live,t=build_transparency(ROOT,live,service,load_today(ROOT)))
  h=history();row=h[h.game_id.eq(game_id)]
  if row.empty:abort(404)
  r=row.iloc[0].to_dict();return render_template('game.html',game=r,historical=True)
 @app.route('/methodology')
 def methodology():return render_template('methodology.html',metadata=service.meta)
 @app.route('/performance')
 def performance_page():return render_template('performance.html',p=performance)
 @app.route('/about')
 def about():return render_template('about.html',p=performance)
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
