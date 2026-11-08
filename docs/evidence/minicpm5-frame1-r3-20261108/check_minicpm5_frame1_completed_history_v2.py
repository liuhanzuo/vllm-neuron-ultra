"""Counterexamples for the one documented started-before-ACK failure."""
import copy
import json
from pathlib import Path
import sys
import unittest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import minicpm5_frame1_completed_history_v2 as h
import finalize_minicpm5_frame1_r2_history_v1 as f
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.prior=h.previous.load(h.PRIOR);cls.terminal,_,cls.spec=h.evidence()
  # Synthetic finalizer identity used ONLY for an in-memory validation fixture.
  cls.owner=dict(pid=2147483000,start_ticks='999999999',ppid=1,argv=['python3','-B',str(h.FINALIZER)],cgroup='0::/fixture')
  cls.value=f.compose(cls.prior,cls.terminal,cls.spec,cls.owner)
 def check(self,v):return h.validate_catalog(v,self.prior,self.terminal,self.spec)
 def test_actual_failed_evidence_and_complete_extension(self):
  self.assertTrue(self.check(self.value));self.assertEqual(len(self.value['rows']),922)
 def test_never_launched_lie_rejected(self):
  v=copy.deepcopy(self.value);v['rows'][919]['never_launched']=True
  with self.assertRaises(RuntimeError):self.check(v)
 def test_manager_pid_cannot_become_owner(self):
  v=copy.deepcopy(self.value);v['rows'][919]['owner']={'pid':1583,'start_ticks':'1'}
  with self.assertRaises(RuntimeError):self.check(v)
 def test_invocation_replacement_rejected(self):
  v=copy.deepcopy(self.value);v['rows'][919]['invocation_id']='0'*32
  with self.assertRaises(RuntimeError):self.check(v)
 def test_second_missing_ack_exception_rejected(self):
  v=copy.deepcopy(self.value);v['rows'][920]['never_launched']=False
  with self.assertRaises(RuntimeError):self.check(v)
 def test_prior_row_omission_rejected(self):
  v=copy.deepcopy(self.value);v['rows'].pop(0)
  with self.assertRaises(RuntimeError):self.check(v)
 def test_prior_process_rewrite_rejected(self):
  v=copy.deepcopy(self.value);v['processes'][0]['pid']+=1
  with self.assertRaises(RuntimeError):self.check(v)
 def test_unknown_actual_owner_not_filled(self):
  v=self.value['rows'][919];self.assertIsNone(v['owner']);self.assertIsNone(v['captured_process_owner']);self.assertIsNone(v['installed_argv'])
 def test_success_promotion_rejected(self):
  v=copy.deepcopy(self.value);v['latest_producer_qualified']=True
  with self.assertRaises(RuntimeError):self.check(v)
 def test_current_source_pin_omission_rejected(self):
  v=copy.deepcopy(self.value);v['source_sha256'].pop(str(h.FINALIZER))
  with self.assertRaises(RuntimeError):self.check(v)
 def test_wrong_finalizer_rejected(self):
  v=copy.deepcopy(self.value);v['processes'][-1]['argv']=['python3','-B','-']
  with self.assertRaises(RuntimeError):self.check(v)
 def test_unpinned_catalog_read_rejected(self):
  with self.assertRaises(RuntimeError):h.text(h.PATH)
 def test_json_roundtrip(self):
  self.assertTrue(self.check(json.loads(json.dumps(self.value,allow_nan=False))))
if __name__=='__main__':unittest.main(verbosity=2)
