"""Source/metadata/mock hooks only: never importsTorch or runs model arithmetic."""
import ast,copy,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import minicpm5_frame1_upstream_worker_v1 as w
import minicpm5_frame1_mlp_worker_v2 as previous
import minicpm5_frame1_upstream_replay_v1 as replay

def read(ref):
 raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256'];return json.loads(raw)

class Source(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.plan=json.loads((R/'research/minicpm5-layer0-phase-cpu-input-plan-20261107-v1.json').read_text());cls.plan['frame1_upstream']=w.science()
  for ref in (*w.CPU_ANCHORS,*w.FULL_CPU_ANCHORS,*w.R3_CAPTURES):read(ref);cls.plan['inputs'][ref['path']]=dict(sha256=ref['sha256'])
 def test_no_native_import(self):self.assertFalse(set(sys.modules)&{'torch','numpy','transformers','torch_neuronx'})
 def test_original_runtime_functions_preserved(self):
  def bodies(path):return {n.name:ast.dump(n,include_attributes=False)for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef)}
  original,new=bodies(previous.__file__),bodies(w.__file__)
  for name in ('execute','weights','activation_source','activation_dispatch','anchor','original_dispatch','check_dispatch','validate_baseline','cpu_route','checked_layout','capture_value','metadata','raw'):
   with self.subTest(name=name):self.assertEqual(original[name],new[name])
 def test_real_science(self):self.assertEqual(w.validate_science(self.plan),w.science())
 def test_science_changes_rejected(self):
  for key,value in [('native_length',1230),('cpu_bucket',128),('precision_interventions',['oproj']),('capture_fields',list(w.FIELDS)[:-1]),('norm_eps',1e-5),('actual_capture_bytes',4096),('norm_weight_bytes',0)]:
   with self.subTest(key=key):
    plan=copy.deepcopy(self.plan);plan['frame1_upstream'][key]=value
    with self.assertRaises(RuntimeError):w.validate_science(plan)
 def test_prompt_and_anchor_rejected(self):
  for kind in ('prompt','r3_anchor','position'):
   with self.subTest(kind=kind):
    plan=copy.deepcopy(self.plan)
    if kind=='prompt':plan['fixtures'][7]['input_ids'].pop()
    elif kind=='r3_anchor':plan['inputs'][w.R3_CAPTURES[0]['path']]['sha256']='0'*64
    else:plan['graph_calls'][1]['call']['logical_positions']=[1231]
    with self.assertRaises(RuntimeError):w.validate_science(plan)
 def test_full_original_shapes_and_budget(self):
  self.assertEqual(w.capture_bytes(True),35295232);self.assertEqual(w.capture_bytes(False),58720256)
  self.assertEqual(w.science()['actual_capture_bytes'],152735744);self.assertLess(152735744+24576,w.OUTPUT_CAP)
  for native in (True,False):
   self.assertEqual(len(w.expected_shapes(native)),7)
   for shape in w.expected_shapes(native).values():self.assertEqual(shape,[1,1231 if native else 2048,2048])
 def test_same_original_norm_bodies(self):
  proofs=[w.norm_source(path.read_text())for path in w.NORM_SOURCES.values()]
  self.assertEqual(proofs[0]['arithmetic_ast_sha256'],proofs[1]['arithmetic_ast_sha256'])
 def test_changed_norm_order_rejected(self):
  text=w.NORM_SOURCES['native'].read_text();changed=text.replace('return self.weight * hidden_states.to(input_dtype)','return (self.weight * hidden_states).to(input_dtype)')
  with self.assertRaises(RuntimeError):w.norm_source(changed)
 def test_changed_norm_epsilon_rejected(self):
  text=w.NORM_SOURCES['native'].read_text().replace('variance + self.variance_epsilon','variance + 1e-5')
  with self.assertRaises(RuntimeError):w.norm_source(text)
 def test_changed_norm_dispatch_decorator_rejected(self):
  text=w.NORM_SOURCES['native'].read_text().replace('use_kernel_forward_from_hub("RMSNorm")','use_kernel_forward_from_hub("Other")')
  with self.assertRaises(RuntimeError):w.norm_source(text)
 def test_old_source_untouched(self):
  self.assertEqual(hashlib.sha256(Path(previous.__file__).read_bytes()).hexdigest(),'8fdd1a477e8e5159a77f3a4c176eea2120f4c0f970d9deef57e0a50a269ceb85')

class FullAnchors(unittest.TestCase):
 def fixture(self,native,rank):
  rec=read(w.R3_CAPTURES[0 if native else rank+1]);original=next(x for x in rec['captures']if x['name']=='input')['metadata']
  values={key:dict(metadata=copy.deepcopy(original))for key in ('mlp_input','postnorm_output','attention_input','inputnorm_output')}
  context=types.SimpleNamespace(read_json=lambda path,wanted:read(dict(path=path,sha256=wanted)))
  old=read(w.CPU_ANCHORS[rank])if not native else None
  return context,values,old
 def test_real_r3_native_and_both_ranks(self):
  for native,rank in ((True,-1),(False,0),(False,1)):
   context,values,old=self.fixture(native,rank);self.assertTrue(w.check_full_anchor(context,values,old,rank,native)['full_mlp_input_exact'])
 def test_changed_original_mlp_input_rejected(self):
  context,values,old=self.fixture(True,-1);values['mlp_input']['metadata']['sha256']='f'*64
  with self.assertRaises(RuntimeError):w.check_full_anchor(context,values,old,-1,True)
 def test_adjacent_input_mismatch_rejected(self):
  for field in ('postnorm_output','attention_input'):
   context,values,old=self.fixture(False,0);values[field]['metadata']['sha256']='f'*64
   with self.assertRaises(RuntimeError):w.check_full_anchor(context,values,old,0,False)
 def test_native_source_layout_rejected(self):
  context,values,old=self.fixture(True,-1);values['mlp_input']['metadata']['stride']=[1,1,1231]
  with self.assertRaises(RuntimeError):w.check_full_anchor(context,values,old,-1,True)
 def test_wrong_rank_baseline_rejected(self):
  context,values,old=self.fixture(False,0);bad=read(w.R3_CAPTURES[2]);context.read_json=lambda *args:bad
  with self.assertRaises(RuntimeError):w.check_full_anchor(context,values,old,0,False)

class Handle:
 def __init__(self,items,row):self.items,self.row=items,row
 def remove(self):self.items.remove(self.row)
class Module:
 def __init__(self):self.pre=[];self.post=[]
 def register_forward_pre_hook(self,fn,with_kwargs=False):
  item=(fn,with_kwargs);self.pre.append(item);return Handle(self.pre,item)
 def register_forward_hook(self,fn):self.post.append(fn);return Handle(self.post,fn)
 def enter(self,args=(),kwargs=None):
  for fn,kw in list(self.pre):fn(self,args,kwargs or {})if kw else fn(self,args)
 def leave(self,value):
  for fn in list(self.post):fn(self,(),value)
class KV:
 def get_cache(self):pass
 def update_cache(self):pass
class Prediction:
 def __init__(self,*args):self.closed=False
 def close(self):self.closed=True

def core():
 layer=Module()
 for name in ('input_layernorm','self_attn','post_attention_layernorm','mlp'):setattr(layer,name,Module())
 config=types.SimpleNamespace(is_prefix_caching=False,windowed_context_encoding_size=None,qkv_kernel_enabled=False,mlp_kernel_enabled=False,qkv_kernel_fuse_residual_add=False,mlp_kernel_fuse_residual_add=False,sequence_parallel_enabled=False,is_eagle_draft=False)
 layer.neuron_config=config
 return types.SimpleNamespace(layers=[layer,Module()],norm=Module(),kv_mgr=KV(),neuron_config=config)

class Hooks(unittest.TestCase):
 def setUp(self):
  self.c=core();self.patches=[patch.object(w,'PredictionHooks',Prediction),patch.object(w,'capture_value',side_effect=lambda torch,directory,name,value,native:dict(name=name,value=value))]
  for item in self.patches:item.start();self.addCleanup(item.stop)
 def observer(self,native):
  obs=w.Observers(None,self.c,native,Path('/unused'),lambda:None);self.addCleanup(obs.close);return obs
 def enter(self,native):
  kwargs=dict(past_key_values=types.SimpleNamespace(get_seq_length=lambda index:0),position_ids=[0])if native else dict(past_key_value=None,get_kv_per_layer=False,update_kv_per_layer=False,is_for_context_encoding=True,idx=0)
  self.c.layers[0].enter(('residual-input',),kwargs)
 def drive(self,native,omit=None,wrong_attn=False):
  self.enter(native);layer=self.c.layers[0]
  if omit!='inputnorm_output':layer.input_layernorm.leave('normalized')
  layer.self_attn.enter(kwargs=dict(hidden_states='normalized'))
  if native:attn=('attention',None)
  else:
   cls=type('NeuronAttentionBaseOutput',(),{'__module__':'neuronx_distributed_inference.modules.attention.attention_base'})
   attn=cls();attn.hidden_states='attention';attn.residual='unexpected'if wrong_attn else None
  layer.self_attn.leave(attn)
  layer.post_attention_layernorm.enter(('residual-sum',));layer.post_attention_layernorm.leave('mlp-input');layer.mlp.enter(('mlp-input',))
 def test_native_all_seven_ordered(self):
  obs=self.observer(True);self.drive(True)
  with self.assertRaises(w.PrefixDone)as raised:self.c.layers[0].leave('block-output')
  self.assertIs(raised.exception,obs.done);self.assertEqual(list(obs.values),list(w.FIELDS));self.assertEqual(obs.completed,1)
 def test_cpu_all_seven_and_cleanup(self):
  original=(self.c.kv_mgr.get_cache,self.c.kv_mgr.update_cache);obs=self.observer(False);self.drive(False)
  with self.assertRaises(w.PrefixDone):self.c.layers[0].leave(('block-output',))
  obs.close();self.assertEqual((self.c.kv_mgr.get_cache,self.c.kv_mgr.update_cache),original)
  self.assertEqual(self.c.layers[0].pre,[]);self.assertEqual(self.c.layers[0].self_attn.post,[])
 def test_skipped_boundary_rejected(self):
  obs=self.observer(True);self.drive(True,omit='inputnorm_output')
  with self.assertRaises(RuntimeError):self.c.layers[0].leave('output')
  self.assertEqual(obs.completed,0)
 def test_duplicate_entry_rejected(self):
  self.observer(True);self.enter(True)
  with self.assertRaises(RuntimeError):self.enter(True)
 def test_downstream_and_history_rejected(self):
  self.observer(False)
  for target in [lambda:self.c.layers[1].enter(),lambda:self.c.norm.enter(),self.c.kv_mgr.get_cache,self.c.kv_mgr.update_cache]:
   with self.assertRaises(RuntimeError):target()
 def test_fused_residual_result_rejected(self):
  self.observer(False)
  with self.assertRaises(RuntimeError):self.drive(False,wrong_attn=True)
 def test_fused_config_rejected(self):
  self.observer(False);self.c.neuron_config.mlp_kernel_enabled=True
  with self.assertRaises(RuntimeError):self.enter(False)
 def test_unexpected_attention_output_rejected(self):
  self.observer(False);self.enter(False);self.c.layers[0].input_layernorm.leave('normalized');self.c.layers[0].self_attn.enter(kwargs=dict(hidden_states='normalized'))
  with self.assertRaises(RuntimeError):self.c.layers[0].self_attn.leave(('wrong-sdk-output',))
 def test_ambiguous_hidden_input_rejected(self):
  self.observer(True)
  with self.assertRaises(RuntimeError):self.c.layers[0].self_attn.enter(('x',),dict(hidden_states='y'))
 def test_install_failure_removes_prior_hooks(self):
  self.c.layers[0].self_attn.register_forward_hook=lambda *args:(_ for _ in ()).throw(RuntimeError('install failure'))
  with self.assertRaises(RuntimeError):self.observer(True)
  self.assertEqual(self.c.layers[0].pre,[]);self.assertEqual(self.c.layers[0].input_layernorm.post,[]);self.assertEqual(self.c.layers[0].self_attn.pre,[])

class Replay(unittest.TestCase):
 def controls(self):
  return {role:{name:dict(raw_exact=True,layout_exact=True)for name in ('inputnorm_output','attention_input','postnorm_input','postnorm_output','mlp_input')}for role in ('native','cpu-rank0','cpu-rank1')}
 def test_all_original_controls_required(self):self.assertTrue(replay.control_passed(self.controls()))
 def test_false_control_blocks_counterfactuals(self):
  controls=self.controls();controls['native']['postnorm_input']['raw_exact']=False;self.assertFalse(replay.control_passed(controls))
 def test_layout_only_change_blocks_counterfactuals(self):
  controls=self.controls();controls['cpu-rank0']['postnorm_output']['layout_exact']=False;self.assertFalse(replay.control_passed(controls))
 def test_missing_role_rejected(self):
  controls=self.controls();controls.pop('cpu-rank1')
  with self.assertRaises(RuntimeError):replay.control_passed(controls)
 def test_missing_control_rejected(self):
  controls=self.controls();controls['native'].pop('inputnorm_output')
  with self.assertRaises(RuntimeError):replay.control_passed(controls)
 def test_non_boolean_control_rejected(self):
  controls=self.controls();controls['native']['postnorm_output']['raw_exact']='true'
  with self.assertRaises(RuntimeError):replay.control_passed(controls)
 def test_unchanged_full_layout_restore(self):
  import minicpm5_frame1_mlp_replay_v2 as old
  def function(path,name):return ast.dump(next(node for node in ast.parse(Path(path).read_text()).body if isinstance(node,ast.FunctionDef)and node.name==name),include_attributes=False)
  self.assertEqual(function(replay.__file__,'restore'),function(old.__file__,'restore'))
 def test_bind_original_functions_without_native_execution(self):
  for path in w.NORM_SOURCES.values():
   sentinel=object();function=replay.bind_norm(sentinel,path,hashlib.sha256(path.read_bytes()).hexdigest())
   self.assertEqual(function.__name__,'forward');self.assertIs(function.__globals__['torch'],sentinel)
  self.assertFalse(set(sys.modules)&{'torch','numpy','transformers','torch_neuronx'})
 def test_bind_stale_source_rejected(self):
  with self.assertRaises(RuntimeError):replay.bind_norm(object(),w.NORM_SOURCES['native'],'0'*64)

if __name__=='__main__':unittest.main(verbosity=2)
