"""Exact bridge from historical V11 feature builders to daily inference rows."""
from pathlib import Path
import importlib.util
import pandas as pd

def _load_v11(root):
    root=Path(root);spec=importlib.util.spec_from_file_location('v11_live_exact',root/'evaluate-v11-unified-team-runs.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def build_daily_feature_rows(root,day,game_ids,features):
    root=Path(root);year=pd.Timestamp(day).year
    required=[root/'data/processed'/f'games_{year}_starter_lineup_matchup_features.csv']
    stems=['features_v7_offensive_form','features_v8_contextual_offense','features_statsimpl_offense_risp','features_richer_starter','features_official_starter_pitching','features_official_bullpen_pitching','features_v6_bullpen_availability','features_statsimpl_starter_recent100','features_arsenal_lineup_matchup','features_opponent_quality_offense','features_situational']
    required += [root/'data/processed'/f'{stem}_{year}.csv' for stem in stems]
    missing=[str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:return None,{'status':'source_data_incomplete','missing_files':missing}
    v11=_load_v11(root);long=v11.long_year(year);target=long[long.game_id.isin([int(x) for x in game_ids])].copy()
    if target.empty:return None,{'status':'target_games_missing_from_feature_universe'}
    complete=[]
    for gid,group in target.groupby('game_id'):
        if set(group.team_side)=={'away','home'} and len(group)==2:complete.append(int(gid))
    target=target[target.game_id.isin(complete)]
    if target.empty:return None,{'status':'target_games_missing_from_feature_universe','missing_game_ids':[int(x) for x in game_ids]}
    out=target[['game_id','team_side','team','opponent']+features].copy();out['feature_cutoff']=str(day);return out,{'status':'ok','source':'exact_historical_v11_builders','available_game_ids':complete,'missing_game_ids':sorted(set(map(int,game_ids))-set(complete))}
