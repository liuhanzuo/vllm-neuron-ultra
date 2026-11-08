"""Source/metadata and lifecycle checks only; no native model or service starts."""
import ast,copy,hashlib,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import minicpm5_frame1_upstream_runner_v1 as r
import check_minicpm5_frame1_startup_v5 as legacy_tests
import minicpm5_post_dory_history_v1 as history
import prepare_minicpm5_frame1_upstream_runtime_v1 as builder

class Startup(legacy_tests.Checks):
 def setUp(self):
  switch=patch.object(legacy_tests,'r',r);switch.start();self.addCleanup(switch.stop)
  super().setUp()
 def test_science_and_original_capture_code_unchanged(self):
  def nodes(name):return {x.name:ast.dump(x,include_attributes=False)for x in ast.parse((R/name).read_text()).body if isinstance(x,ast.FunctionDef)}
  before=nodes('minicpm5_frame1_mlp_runner_v5.py');after=nodes('minicpm5_frame1_upstream_runner_v1.py')
  changed={'load_history_helpers','environment','current_history','load_spec','cpu_coordinator','phase_main','outer'}
  self.assertEqual({k:v for k,v in before.items()if k not in changed},{k:after[k]for k in before if k not in changed})
 def test_fresh_original_prerequisites_then_actual_current_catalog(self):
  # Actual read-only prerequisite check must precede enlarged history imports.
  code="import sys;sys.path[:0]=["+repr(str(R))+","+repr(str(Q))+"];import minicpm5_frame1_upstream_runner_v1 as r;r.old.check_prerequisites(r.p.read_ref(r.BASE_SPEC));assert 'torch'not in sys.modules"
  result=subprocess.run([sys.executable,'-B','-c',code],cwd=R,capture_output=True,text=True,timeout=120)
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)

class HistoryBinding(unittest.TestCase):
 def setUp(self):
  self.receipt=json.loads(Path(history.EXIT['path']).read_text())
  self.catalog=dict(rows=[None]*978,processes=[None]*190,dory_rotary_append=dict(finalizer_owner=copy.deepcopy(self.receipt['owner'])))
 def check(self):return history.check_exit(self.catalog,self.receipt)
 def test_actual_receipt(self):self.assertTrue(self.check())
 def test_missing_external_exit(self):
  self.receipt['actual_tool_exit_observed']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_nonzero_finalizer_exit(self):
  self.receipt['actual_exit']['exit_code']=1
  with self.assertRaises(RuntimeError):self.check()
 def test_wrong_owner(self):
  self.receipt['owner']['start_ticks']='1'
  with self.assertRaises(RuntimeError):self.check()
 def test_omitted_history(self):
  self.catalog['rows'].pop()
  with self.assertRaises(RuntimeError):self.check()
 def test_model_promotion_rejected(self):
  self.receipt['full_model_qualified']=True
  with self.assertRaises(RuntimeError):self.check()
 def test_actual_large_v14_spec_uses_original_reader(self):
  reference=history.dory.prior.SPEC
  raw=history.text(reference['path'],reference['sha256'])
  self.assertGreater(len(raw),8*2**20)
  self.assertEqual(hashlib.sha256(raw).hexdigest(),reference['sha256'])
  self.assertEqual(json.loads(raw)['schema'],'nemotron9b_package_runtime_v14')
 def test_large_v14_spec_wrong_hash_rejected(self):
  with self.assertRaises(RuntimeError):history.text(history.dory.prior.SPEC['path'],'a'*64)
 def test_old_catalog_rejected(self):
  with self.assertRaises(RuntimeError):history.validate_binding(dict(current_catalog=dict(path='/old',sha256='a'*64)),self.catalog)

class Scope(unittest.TestCase):
 def test_no_native_import(self):self.assertFalse({'torch','torch_neuronx','numpy','transformers'}&set(sys.modules))
 def test_original_budgets(self):
  self.assertEqual(r.CAP,192*2**20);self.assertEqual(r.worker.CAPTURE_FILE_CAP,16*2**20)
  self.assertEqual(r.props('native')['MemoryMax'],str(16*2**30));self.assertEqual(r.props('replay')['MemoryMax'],str(4*2**30))
  self.assertEqual(r.props('replay')['RuntimeMaxSec'],'600')
 def test_complete_saved_geometry_and_remaining_space(self):
  import minicpm5_frame1_upstream_replay_v2 as replay
  self.assertEqual(replay.capture_plan()['all_raw_bytes'],177926144)
  self.assertGreater(r.CAP-replay.capture_plan()['all_raw_bytes'],20*2**20)
  self.assertEqual(r.worker.expected_shapes(True)['attention_output'],[1,1231,2048])
  self.assertEqual(r.worker.expected_shapes(False)['attention_output'],[1,2048,2048])
 def test_all_seven_original_anchors(self):
  anchors=builder.actual_anchors()
  self.assertEqual(set(anchors),{x['path']for x in (*r.worker.CPU_ANCHORS,*r.worker.FULL_CPU_ANCHORS,*r.worker.R3_CAPTURES)})
 def test_new_executable_and_preserved_original_environments(self):
  import minicpm5_frame1_mlp_runner_v5 as old
  for phase in r.PHASES:
   self.assertEqual(r.environment(phase,'a'*32,29571),old.environment(phase,'a'*32,29571))
   self.assertIn('upstream_runner_v1.py',' '.join(r.arguments('phase',dict(path='/fixture',sha256='a'*64),phase)))
 def test_plan_preserves_all_other_science(self):
  base=r.p.read_ref(r.BASE_PLAN)
  spec=dict(output_root=str(R/'minicpm5-frame1-upstream-fixture'),new_sources={},added_json=builder.actual_anchors(),science=r.worker.science())
  plan=builder.expanded_plan(spec,base)
  self.assertEqual({k:v for k,v in base.items()if k not in ('output_root','source_sha256','inputs')},{k:plan[k]for k in base if k not in ('output_root','source_sha256','inputs')})
  self.assertEqual(plan['frame1_upstream'],r.worker.science())
 def test_reduced_scientific_shape_rejected(self):
  base=r.p.read_ref(r.BASE_PLAN)
  spec=dict(output_root=str(R/'minicpm5-frame1-upstream-fixture'),new_sources={},added_json=builder.actual_anchors(),science=r.worker.science())
  spec['science']['native_length']=1
  with self.assertRaises(RuntimeError):builder.expanded_plan(spec,base)
 def test_output_symlink_rejected(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);(root/'a').symlink_to('/dev/null')
   with self.assertRaises(RuntimeError):r.usage(root,0)
 def test_whole_output_cap_enforced(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);p=root/'actual';p.write_bytes(b'x')
   with patch.object(r,'CAP',0):
    with self.assertRaises(RuntimeError):r.usage(root,0)

if __name__=='__main__':unittest.main(verbosity=2)
