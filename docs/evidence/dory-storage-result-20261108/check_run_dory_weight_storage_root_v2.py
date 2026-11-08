"""Actual completed Mini receipts and inherited inert request regressions."""
import copy,json,sys,unittest
from pathlib import Path
Q=Path(__file__).resolve().parent;sys.path[:0]=[str(Q),str(Q.parent)]
import run_dory_weight_storage_root_v2 as s
import check_run_dory_weight_storage_root_v1 as inherited
inherited.s=s
h=s.predecessor_review
class ActualBinding(unittest.TestCase):
 def setUp(self):
  self.t=s.load(h.recovery.TERMINAL);self.run=s.load(h.RUN);self.ended=s.load(h.EXIT);self.report=s.load(self.run['result'])
 def check(self):return h.bind(self.run,self.ended,self.report,self.t)
 def test_real_capture_recovery_receipts(self):self.assertEqual(self.check(),self.run['owner'])
 def test_original_failed_exit_not_promoted(self):
  self.run['original_runner_actual_exit']['exit_code']=0
  with self.assertRaises(RuntimeError):self.check()
 def test_original_full_payload_count_retained(self):
  self.report['files']=95
  with self.assertRaises(RuntimeError):self.check()
 def test_original_full_payload_bytes_retained(self):
  self.report['raw_bytes']-=1
  with self.assertRaises(RuntimeError):self.check()
 def test_no_unobserved_external_exit(self):
  self.ended['actual_tool_exit_observed']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_boolean_tool_exit_rejected(self):
  self.ended['actual_exit']['exit_code']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_changed_external_owner_rejected(self):
  self.ended['owner']['pid']+=1
  with self.assertRaises(RuntimeError):self.check()
 def test_rerun_claim_rejected(self):
  self.run['model_rerun']=True
  with self.assertRaises(RuntimeError):self.check()
 def test_consistent_false_scientific_verdict_allowed(self):
  self.run['all21_own_input_controls_exact']=self.report['passed']=self.report['all21_own_input_controls_exact']=False;self.check()
 def test_false_verdict_promotion_rejected(self):
  self.run['all21_own_input_controls_exact']=False
  with self.assertRaises(RuntimeError):self.check()
 def test_full_model_promotion_rejected(self):
  self.report['full_model_qualified']=True
  with self.assertRaises(RuntimeError):self.check()
 def test_current_source_closure_retains_failed_histories(self):
  pins=s.sources()
  for n in ['minicpm5_attention_failed_r1_history_v1.py','minicpm5_attention_failed_r2_history_v1.py']:self.assertIn(str(s.R/n),pins)
  for n in ['dory_after_mini_attention_r3_v1.py','review_minicpm5_frame1_attention_metadata_v3.py','recover_minicpm5_attention_r3_saved_review_v1.py']:self.assertIn(str(Q/n),pins)
if __name__=='__main__':
 suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(inherited.Request));suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(ActualBinding));result=unittest.TextTestRunner().run(suite);raise SystemExit(not result.wasSuccessful())
