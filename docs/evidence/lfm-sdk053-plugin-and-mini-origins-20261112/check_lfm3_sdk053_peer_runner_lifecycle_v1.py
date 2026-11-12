import copy,contextlib,hashlib,importlib,importlib.abc,importlib.util,json,os,resource,sys,types
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20));resource.setrlimit(resource.RLIMIT_CPU,(60,60));os.sched_setaffinity(0,{7});os.nice(19)
class NoNative(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_neuronx','torch_xla','numpy','ml_dtypes','nki','neuronxcc','neuronx_distributed','neuronx_distributed_inference','vllm','ctypes'):raise AssertionError('native import: '+fullname)
sys.meta_path.insert(0,NoNative())
r=Path('/home/ubuntu/src/vllm-neuron-ultra-sdk053-lfm3-20261112')
p=r/'docs/lfm3-sdk053/source-manifest.json';raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()=='d71d49ad0d4023a3aeb89d14fda3f0ce0040a3b709ee280aa695c0731ef3c0b9'
manifest=json.loads(raw)
for item in manifest['files']:
 b=(r/item['path']).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256'],item['path']
spec=importlib.util.spec_from_file_location('lfm_peer_stubs',r/'tests/source/test_lfm3_sdk053.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
class Batch:
 def __init__(self):self.req_id_to_index={};self.num_computed_tokens_cpu={};self.block_table=types.SimpleNamespace(append_row=lambda *a:None)
 def remove_request(self,req):self.req_id_to_index.pop(req,None)
 def add_request(self,state):self.req_id_to_index[state.req_id]=len(self.req_id_to_index)
 def condense(self):self.req_id_to_index={req:i for i,req in enumerate(self.req_id_to_index)}
 def refresh_metadata(self):pass
with t.patch.dict(os.environ,t.ENV,clear=True),t.external_stubs():
 mod=importlib.import_module('vllm_neuron.worker.neuronx_distributed_model_runner')
 obj=mod.NeuronxDistributedModelRunner(t.vconfig(),'cpu');obj.max_prompt_length=2048;obj.input_batch=Batch();obj.model=types.SimpleNamespace(architecture='Lfm3ForCausalLM',neuron_config=types.SimpleNamespace(on_device_sampling_config=None))
 obj.maybe_get_kv_connector_output=lambda out:contextlib.nullcontext(None)
 observations=[]
 def prepare(out):
  data=mod.IntermediateInputData()
  for request in out.scheduled_new_reqs:obj._process_new_request_for_continuous_batching(request,data)
  for index in range(len(out.scheduled_cached_reqs.req_ids)):obj._process_cached_request_for_continuous_batching(out.scheduled_cached_reqs,index,data)
  observations.append(copy.deepcopy(vars(data)));return data
 obj._prepare_model_input=prepare;obj._execute_model_for_text=lambda *a:['mock-logits']
 def step(news=(),cached=(),finished=()):
  out=t.output(news,cached,finished)
  out.scheduled_cached_reqs.num_computed_tokens=[len(obj.requests[x].prompt_token_ids)+len(obj.requests[x].output_token_ids)-1 for x in cached]
  obj.execute_model(out)
  for request_id in out.num_scheduled_tokens:obj.requests[request_id].output_token_ids.append(9)
  obj._cached_logits=None;obj._cached_model_input=None
 step([t.new('a',1)]);a_slot=obj.vllm_req_to_neuron_seq_id_mapping['a']
 step(cached=['a']);assert observations[-1]['position_ids']==[[2]]
 step([t.new('b',2,tokens=[3,4,5])]);assert obj.vllm_req_to_neuron_seq_id_mapping['a']==a_slot and set(obj.requests)=={'a','b'}
 step(cached=['b','a']);assert observations[-1]['position_ids']==[[3],[3]] and observations[-1]['request_ids']==['b','a']
 step([t.new('a',1,tokens=[8])],finished=['a']);assert obj.requests['a'].prompt_token_ids==[8] and len(obj.requests['a'].output_token_ids)==1
 step(cached=['a','b']);assert observations[-1]['position_ids']==[[1],[4]]
 step(finished=['a','b']);assert obj.free_seq_ids==set(range(4)) and not obj.requests and not obj.vllm_req_to_neuron_seq_id_mapping
 print(json.dumps({'manifest_files_verified':len(manifest['files']),'full_runner_execute_steps':7,'actual_runner_update_and_new_cached_slot_methods_used':True,'covered':['prefill','decode','held-back live request','reordered cached IDs','finish/new same ID','mixed prompt lengths','final complete release'],'tensor_model_and_sampling_payloads_mocked':True,'native_imports_blocked':True,'checkout_modified':False},indent=2))
