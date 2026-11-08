import copy,hashlib,sys,unittest
from pathlib import Path
from unittest.mock import patch
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q.parent),str(Q)]
import run_minicpm5_upstream_review_v1 as r
class Tests(unittest.TestCase):
 def setUp(self):
  self.t=dict(schema='minicpm5_frame1_upstream_root_terminal_v1',reviewer='/root',actual_tool_exit_observed=True,exact_outer_owner_gone=True,all_three_groups_physically_drained=True,actual_exit=dict(session_id=17,exit_code=0),all_three_phases_succeeded=True,phase_owners={x:{}for x in r.m.PHASES},rank_owners=[{},{}],full_model_qualified=False,new_speedup_claim=False)
 def check(self):return r.validate_terminal(self.t)
 def test_actual_shape(self):self.assertTrue(self.check())
 def test_missing_actual_exit(self):
  self.t['actual_tool_exit_observed']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_failed_execution(self):
  self.t['actual_exit']['exit_code']=1
  with self.assertRaises(RuntimeError):self.check()
 def test_boolean_exit_not_zero(self):
  self.t['actual_exit']['exit_code']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_live_outer(self):
  self.t['exact_outer_owner_gone']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_undrained_group(self):
  self.t['all_three_groups_physically_drained']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_omitted_phase(self):
  del self.t['phase_owners']['cpu']
  with self.assertRaises(RuntimeError):self.check()
 def test_missing_rank(self):
  self.t['rank_owners'].pop()
  with self.assertRaises(RuntimeError):self.check()
 def test_no_speedup_promotion(self):
  self.t['new_speedup_claim']=True
  with self.assertRaises(RuntimeError):self.check()
 def test_original_callback_pin(self):self.assertEqual(hashlib.sha256(r.CALLBACK.read_bytes()).hexdigest(),r.CALLBACK_SHA)
 def test_no_model_import(self):self.assertFalse({'torch','numpy','torch_neuronx','transformers'}&set(sys.modules))
 def test_prelaunch_refuses_failed_terminal(self):
  self.t['actual_exit']['exit_code']=1
  with patch.object(r.m.p,'read_ref',return_value=self.t),patch.object(r.m.old.post.external,'acquire_lock')as lock:
   with self.assertRaises(RuntimeError):r.review_run({}, {})
   lock.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
