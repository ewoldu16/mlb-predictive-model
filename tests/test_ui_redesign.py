import re


def test_editorial_assets_and_mime_types():
    from app import create_app
    client=create_app({'TESTING':True}).test_client()
    css=client.get('/static/css/product.css');js=client.get('/static/js/site.js')
    assert css.status_code==200 and css.content_type.startswith('text/css')
    assert js.status_code==200 and 'javascript' in js.content_type
    text=css.get_data(as_text=True)
    for token in ('--white:#fff','--black:#121212','--cyan:#00a7c4','--pink:#e60067','--green:#12843b','--red:#c62828'):
        assert token in text
    assert 'prefers-reduced-motion' in text and '@media(max-width:430px)' in text


def test_active_home_route_uses_editorial_template(monkeypatch):
    import app as site
    game={'game_id':1,'date':'2099-01-01','start_time':'20:00','away_team':'Away','home_team':'Home','away_starter':'A','home_starter':'H','venue':'Park','forecast_type':'FINAL_PREGAME_PREDICTION','lineup_status':'confirmed','prediction':{'away':{'expected_runs':4.0},'home':{'expected_runs':4.5},'projected_total':8.5,'home_win_probability':.56,'predicted_winner':'Home','winner_probability':.56,'confidence':'MODERATE'}}
    monkeypatch.setattr(site,'load_today',lambda root:{'date':'2099-01-01','games':[game]})
    html=site.create_app({'TESTING':True}).test_client().get('/').get_data(as_text=True)
    assert '/static/css/product.css' in html
    assert 'data-ui="editorial-v11-2"' in html
    assert 'prediction-card' in html and 'probability-track' in html and 'wordmark-mark' in html


def test_central_metric_formatter_is_semantic_and_finite_safe():
    from app import create_app
    app=create_app({'TESTING':True})
    metric=app.jinja_env.filters['metric']
    assert metric(.32718492,'season_woba')=='.327'
    assert metric(.23718492,'season_k_pct')=='23.7%'
    assert metric(4.638192746,'expected_runs')=='4.64'
    assert metric(float('nan'),'season_woba')=='N/A'


def test_home_prediction_has_readable_precision(monkeypatch):
    import app as site
    game={'game_id':1,'date':'2099-01-01','start_time':'20:00','away_team':'Away','home_team':'Home','away_starter':'A','home_starter':'H','venue':'Park','forecast_type':'FINAL_PREGAME_PREDICTION','lineup_status':'confirmed','prediction':{'away':{'expected_runs':4.638192746},'home':{'expected_runs':4.119283746},'projected_total':8.757476492,'home_win_probability':.529183746,'predicted_winner':'Home','winner_probability':.529183746,'confidence':'MODERATE'}}
    monkeypatch.setattr(site,'load_today',lambda root:{'date':'2099-01-01','games':[game]})
    html=site.create_app({'TESTING':True}).test_client().get('/').get_data(as_text=True)
    assert '52.9%' in html and '4.64' in html and '8.76' in html
    assert not re.search(r'4\.638192|0\.529183|8\.757476',html)
