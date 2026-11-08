"""Run inherited observer checks and actual-control wiring without native imports."""
import copy,hashlib,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import check_minicpm5_frame1_attention_source_v1 as inherited
import minicpm5_frame1_attention_observer_v2 as a
import minicpm5_frame1_attention_worker_v2 as w
import minicpm5_attention_k_consumer_layout_v1 as k
import minicpm5_attention_failed_r1_history_v1 as history
import review_minicpm5_frame1_attention_complete_v2 as complete
inherited.a=a;inherited.w=w

class Wiring(unittest.TestCase):
 def run_case(self,native):
  values={}
  for name,shape in a.geometry(native).items():
   dims=shape['shape'];stride=[];current=1
   for n in reversed(dims):stride.insert(0,current);current*=n
   values[name]=types.SimpleNamespace(meta=dict(shape=dims,stride=stride,offset=0,dtype=shape['dtype'],sha256=hashlib.sha256(name.encode()).hexdigest()))
  producer=values['k_post_rope'].meta
  if native:producer['stride']=[315136,128,256,1]
  consumer=copy.deepcopy(producer)
  if native:consumer['stride']=[315136,157568,128,1]
  boundary=k.compare(producer,consumer,native)
  rows=[dict(name=name,metadata=copy.deepcopy(x.meta),sha256=x.meta['sha256'])for name,x in values.items()]
  report=dict(complete=True,native=native,rank=-1 if native else 0,captures=rows,k_consumer_boundary=dict(record=boundary))
  def produce(name):return types.SimpleNamespace(meta=copy.deepcopy(values[name].meta))
  def projection(name):return lambda x:produce(name+'_projection')
  def rope(*args):return (produce('q_post_rope'),produce('k_post_rope'))+(()if native else(values['rope_cos'],values['rope_sin']))
  called=[]
  def context(*args,**kwargs):
   args=args[1:]if native else args
   self.assertEqual(args[1].meta,consumer);self.assertEqual(values['k_post_rope'].meta,producer);called.append(True)
   return produce('context_heads'),types.SimpleNamespace(name='NONE')
  projections=types.SimpleNamespace(**{n+'_proj':projection(n)for n in ('q','k','v')})
  if native:attention=projections;attention.o_proj=lambda x:produce('projection_output');module=types.SimpleNamespace(apply_rotary_pos_emb=rope,eager_attention_forward=context)
  else:attention=types.SimpleNamespace(qkv_proj=projections,o_proj=types.SimpleNamespace(o_proj=lambda x:produce('projection_output')),apply_rotary_embedding=rope,perform_prefill=context);module=None
  torch=types.SimpleNamespace(is_inference_mode_enabled=lambda:True,get_num_threads=lambda:2 if native else 1)
  def restore(t,row,d,n,g):return types.SimpleNamespace(meta=copy.deepcopy(row['metadata']))
  with tempfile.TemporaryDirectory()as tmp,patch.object(a,'restore',side_effect=restore),patch.object(a,'save',side_effect=lambda t,d,name,v,n,g:dict(name=name,metadata=copy.deepcopy(v.meta))),patch.object(a.old,'metadata',side_effect=lambda t,x:x.meta):
   result=a.own_input_controls(torch,attention,module,native,report['rank'],report,Path(tmp),lambda:True)
  self.assertEqual(called,[True]);self.assertTrue(result['passed']);self.assertEqual(result['k_consumer_boundary'],boundary)
 def test_native_replay_uses_actual_consumer_layout(self):self.run_case(True)
 def test_cpu_exact_replay_preserved(self):self.run_case(False)
 def test_actual_failure_predecessor_validates(self):
  t=history.validate(dict(prior_attention_failure=dict(terminal=history.TERMINAL,verifier_exit=history.EXIT)));self.assertFalse(t['complete_capture'])
 def test_new_complete_review_sources_authenticate(self):complete.authentic_sources()
 def test_bounded_science_unchanged(self):
  import minicpm5_frame1_attention_observer_v1 as original
  self.assertEqual(a.science(),original.science())

if __name__=='__main__':
 suite=unittest.TestSuite()
 for name in ['Profile','Source','Geometry','Upstream','Hooks','Controls']:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(inherited,name)))
 suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Wiring))
 result=unittest.TextTestRunner(verbosity=1).run(suite);raise SystemExit(not result.wasSuccessful())
