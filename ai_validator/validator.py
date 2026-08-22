from pathlib import Path
import hashlib,json,time
from .context_builder import clean,packet_hash,audit_packet
from .prompts import PROMPT_HASH,SYSTEM_PROMPT
from .schemas import validate_result

class CachedValidator:
 def __init__(self,provider,cache_dir,retries=2):self.provider=provider;self.cache=Path(cache_dir);self.cache.mkdir(parents=True,exist_ok=True);self.retries=retries
 def validate(self,packet):
  audit=audit_packet(packet)
  if not audit['clean']:raise RuntimeError('OUTCOME LEAKAGE: '+str(audit['forbidden_paths']))
  key=hashlib.sha256(f'{self.provider.name}|{self.provider.model}|{PROMPT_HASH}|{packet_hash(packet)}'.encode()).hexdigest();path=self.cache/f'{key}.json'
  if path.exists():
   row=json.loads(path.read_text());validate_result(row['response']);row['cache_hit']=True;return row
  error=None
  for retry in range(self.retries+1):
   try:
    response,meta=self.provider.validate_game(clean(packet),SYSTEM_PROMPT);validate_result(response);row={'provider':self.provider.name,'model':self.provider.model,'prompt_hash':PROMPT_HASH,'context_hash':audit['context_hash'],'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'response':response,'retry_count':retry,'cache_hit':False,**meta};path.write_text(json.dumps(row,indent=2));return row
   except Exception as exc:error=exc
  raise RuntimeError(f'validator failed after retries: {error}')

