from abc import ABC,abstractmethod
import json,os,time,urllib.request
from .schemas import validate_result

class BaseValidatorProvider(ABC):
 name='base';model='unset'
 @abstractmethod
 def validate_game(self,packet,system_prompt):...

class MockValidatorProvider(BaseValidatorProvider):
 """Deterministic plumbing test only; never represents real AI evidence."""
 name='mock';model='deterministic-schema-test-v1'
 def validate_game(self,packet,system_prompt):
  warnings=packet['data_quality']['warnings'];conf=max(5,90-10*len(warnings));status='DATA_ISSUE' if warnings else 'ACCEPT';codes=[w['reason_code'] for w in warnings] or ['CONSISTENT_WITH_INPUTS','MODEL_COMPONENTS_AGREE']
  return validate_result({'status':status,'confidence_score':conf,'prediction_consistency':75,'data_quality_score':max(0,100-15*len(warnings)),'reason_codes':codes,'team_flags':{'away':[],'home':[]},'primary_concern':warnings[0]['message'] if warnings else 'No deterministic packet warning.','explanation':'MOCK SCHEMA/FLOW TEST ONLY. This is not an LLM evaluation.','recommended_action':'MANUAL_REVIEW' if warnings else 'USE'}).to_dict(),{'input_tokens':None,'output_tokens':None,'latency_seconds':0.0}

class ExternalLLMValidatorProvider(BaseValidatorProvider):
 """OpenAI-compatible JSON endpoint configured entirely through environment variables."""
 def __init__(self):
  self.name=os.environ['V13_VALIDATOR_PROVIDER'];self.model=os.environ['V13_VALIDATOR_MODEL'];self.endpoint=os.environ['V13_VALIDATOR_ENDPOINT'];self._key=os.environ['V13_VALIDATOR_API_KEY']
 def validate_game(self,packet,system_prompt):
  body={'model':self.model,'temperature':0,'response_format':{'type':'json_object'},'messages':[{'role':'system','content':system_prompt},{'role':'user','content':json.dumps(packet,separators=(',',':'))}]};req=urllib.request.Request(self.endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+self._key});t=time.perf_counter()
  with urllib.request.urlopen(req,timeout=90) as response:data=json.loads(response.read())
  content=data['choices'][0]['message']['content'];result=validate_result(json.loads(content)).to_dict();usage=data.get('usage',{})
  return result,{'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),'latency_seconds':time.perf_counter()-t}

def external_configured():
 return all(os.environ.get(k) for k in ('V13_VALIDATOR_PROVIDER','V13_VALIDATOR_MODEL','V13_VALIDATOR_ENDPOINT','V13_VALIDATOR_API_KEY'))

