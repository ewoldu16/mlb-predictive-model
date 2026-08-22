import hashlib,json,math

FORBIDDEN_KEYS={'actual_team_runs','opponent_actual_runs','actual_runs','away_runs','home_runs','away_score','home_score','final_score','winner','home_win','outcome','result','profit','pnl','roi','units'}

def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 if isinstance(v,float) and (math.isnan(v) or math.isinf(v)):return None
 if hasattr(v,'item'):return clean(v.item())
 return v
def leakage_paths(obj,path='$'):
 bad=[]
 if isinstance(obj,dict):
  for k,v in obj.items():
   if k.lower() in FORBIDDEN_KEYS:bad.append(f'{path}.{k}')
   bad.extend(leakage_paths(v,f'{path}.{k}'))
 elif isinstance(obj,list):
  for i,v in enumerate(obj):bad.extend(leakage_paths(v,f'{path}[{i}]'))
 return bad
def packet_hash(packet):return hashlib.sha256(json.dumps(clean(packet),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def audit_packet(packet):
 bad=leakage_paths(packet)
 return {'clean':not bad,'forbidden_paths':bad,'context_hash':packet_hash(packet)}

