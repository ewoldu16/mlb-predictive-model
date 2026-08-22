from dataclasses import dataclass,asdict

STATUSES={'ACCEPT','LOW_CONFIDENCE','DATA_ISSUE','CONTEXT_CONFLICT'}
ACTIONS={'USE','USE_WITH_CAUTION','RECOMPUTE','MANUAL_REVIEW'}
REASON_CODES={
'CONSISTENT_WITH_INPUTS','MODEL_COMPONENTS_AGREE','MODEL_COMPONENTS_CONFLICT','EXTREME_PREDICTION','EXTREME_INPUT',
'WEAK_OFFENSE_CONFLICT','ELITE_OFFENSE_CONFLICT','STARTER_QUALITY_CONFLICT','STARTER_RECENT_FORM_CONFLICT',
'STARTER_MATCHUP_CONFLICT','BULLPEN_QUALITY_CONFLICT','BULLPEN_AVAILABILITY_CONFLICT','LINEUP_QUALITY_CONFLICT',
'PLATOON_CONFLICT','VENUE_CONTEXT_CONFLICT','TEAM_STRENGTH_CONFLICT','SPARSE_STARTER_HISTORY',
'SPARSE_MATCHUP_HISTORY','PARTIAL_LINEUP','MISSING_CRITICAL_DATA','POSSIBLE_STALE_DATA','POSSIBLE_ID_MISMATCH','OTHER_DATA_QUALITY'}

@dataclass
class ValidationResult:
 status:str;confidence_score:int;prediction_consistency:int;data_quality_score:int;reason_codes:list
 team_flags:dict;primary_concern:str;explanation:str;recommended_action:str
 def to_dict(self):return asdict(self)

def validate_result(x):
 if not isinstance(x,dict):raise ValueError('validator response must be an object')
 need={'status','confidence_score','prediction_consistency','data_quality_score','reason_codes','team_flags','primary_concern','explanation','recommended_action'}
 if set(x)!=need:raise ValueError(f'exact schema required; missing={need-set(x)}, extra={set(x)-need}')
 if x['status'] not in STATUSES or x['recommended_action'] not in ACTIONS:raise ValueError('invalid status/action')
 for k in ('confidence_score','prediction_consistency','data_quality_score'):
  if not isinstance(x[k],int) or not 0<=x[k]<=100:raise ValueError(f'{k} must be integer 0-100')
 if not isinstance(x['reason_codes'],list) or not set(x['reason_codes'])<=REASON_CODES:raise ValueError('invalid reason code')
 if set(x['team_flags'])!={'away','home'} or not all(isinstance(v,list) for v in x['team_flags'].values()):raise ValueError('team_flags requires away/home lists')
 for k in ('primary_concern','explanation'):
  if not isinstance(x[k],str):raise ValueError(f'{k} must be string')
 return ValidationResult(**x)

