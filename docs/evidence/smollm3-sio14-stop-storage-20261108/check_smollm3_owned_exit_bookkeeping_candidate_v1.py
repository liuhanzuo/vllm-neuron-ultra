"""Synthetic transition controls; no services, native imports or model work."""
import copy,datetime,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import smollm3_owned_exit_bookkeeping_candidate_v1 as candidate
import smollm3_observer_invariance_lifecycle_v24 as original
from smollm3_authorized_stop_transition_v1 import AuthorizedStopAbsent
OWNER=dict(pid=101,start_ticks='202',ppid=11,argv=['python3','worker.py'],cgroup='0::/owned',invocation_id='a'*32)
TERMINAL=dict(InvocationID='a'*32,MainPID='0',ControlPID='0',ExecMainPID='101',Job='',ActiveState='inactive',SubState='dead',ExecMainStatus='0')
STOP=dict(TERMINAL,ActiveState='deactivating',SubState='stop-sigterm')

class Controls(unittest.TestCase):
 def invoke(self,samples,process=None,gone=None,refresh_error=None,use_original=False):
  clock=[0.0];count=[0];samples=copy.deepcopy(samples)
  def refresh():
   count[0]+=1
   if refresh_error is not None:raise refresh_error
   return samples[min(count[0],len(samples)-1)]
  def absent(pid):raise FileNotFoundError(pid)
  module=original if use_original else candidate
  fake_time=types.SimpleNamespace(monotonic=lambda:clock[0],sleep=lambda delay:clock.__setitem__(0,clock[0]+delay))
  with patch.object(module,'time',fake_time),patch.object(module,'process',process or absent),patch.object(module,'gone',gone or (lambda owner:True)):
   value=module.owned_live_or_retained_exit(OWNER,samples[0],refresh,timeout=.12)
  return value,count[0]
 def test_original_reproduces_mainpid_zero_stop_failure(self):
  with self.assertRaisesRegex(RuntimeError,'Exact retained exit PID'):self.invoke([STOP,TERMINAL,TERMINAL],use_original=True)
 def test_deactivating_zero_main_pid_waits_for_two_terminal_samples(self):
  value,count=self.invoke([STOP,TERMINAL,TERMINAL]);self.assertEqual(value,TERMINAL);self.assertEqual(count,2)
 def test_pending_stop_job_waits(self):
  value,count=self.invoke([dict(TERMINAL,Job='98'),TERMINAL,TERMINAL]);self.assertEqual(value,TERMINAL);self.assertEqual(count,2)
 def test_pending_control_process_waits(self):
  value,count=self.invoke([dict(STOP,ControlPID='102'),TERMINAL,TERMINAL]);self.assertEqual(value,TERMINAL);self.assertEqual(count,2)
 def test_live_exact_owner_still_returns_immediately(self):
  value,count=self.invoke([dict(STOP,MainPID='101')],process=lambda pid:dict(OWNER));self.assertEqual(value['MainPID'],'101');self.assertEqual(count,0)
 def test_missing_owner_with_nonzero_mainpid_waits(self):
  value,count=self.invoke([dict(STOP,MainPID='101'),TERMINAL,TERMINAL]);self.assertEqual(value,TERMINAL);self.assertEqual(count,2)
 def test_empty_argv_original_identity_waits(self):
  value,count=self.invoke([dict(STOP,MainPID='101'),TERMINAL,TERMINAL],process=lambda pid:dict(OWNER,argv=[]));self.assertEqual(value,TERMINAL);self.assertEqual(count,2)
 def test_zero_main_pid_never_terminal_times_out(self):
  with self.assertRaisesRegex(RuntimeError,'Bounded exiting process transition expired'):self.invoke([STOP])
 def test_foreign_exec_pid_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Exact retained exit PID'):self.invoke([dict(STOP,ExecMainPID='999')])
 def test_zero_exec_pid_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Exact retained exit PID'):self.invoke([dict(STOP,ExecMainPID='0')])
 def test_foreign_main_pid_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Exit transition PID changed'):self.invoke([dict(STOP,MainPID='999')])
 def test_new_invocation_rejected_on_refresh(self):
  with self.assertRaisesRegex(RuntimeError,'Exit transition invocation changed'):self.invoke([STOP,dict(TERMINAL,InvocationID='b'*32)])
 def test_foreign_nonempty_argv_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Foreign nonempty argv'):self.invoke([dict(STOP,MainPID='101')],process=lambda pid:dict(OWNER,argv=['foreign']))
 def test_reused_start_rejected_when_live(self):
  with self.assertRaisesRegex(RuntimeError,'Exit transition identity changed'):self.invoke([dict(STOP,MainPID='101')],process=lambda pid:dict(OWNER,start_ticks='999'))
 def test_refresh_authentication_failure_propagates(self):
  error=RuntimeError('Resource binding changed')
  with self.assertRaises(RuntimeError)as caught:self.invoke([STOP],refresh_error=error)
  self.assertIs(caught.exception,error)
 def test_authorized_absence_remains_typed(self):
  error=AuthorizedStopAbsent({'synthetic':True})
  with self.assertRaises(AuthorizedStopAbsent)as caught:self.invoke([STOP],refresh_error=error)
  self.assertIs(caught.exception,error)
 def test_changed_terminal_status_requires_two_identical_samples(self):
  last=dict(TERMINAL,ExecMainStatus='1');value,count=self.invoke([TERMINAL,last,last]);self.assertEqual(value,last);self.assertEqual(count,2)
 def test_retained_owner_not_gone_never_qualifies(self):
  def live(owner):raise RuntimeError('Prior process remains')
  with self.assertRaisesRegex(RuntimeError,'Bounded exiting process transition expired'):self.invoke([TERMINAL],gone=live)

def run():
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Controls);result=unittest.TextTestRunner(verbosity=2).run(suite)
 if not result.wasSuccessful():raise SystemExit(1)
 forbidden=('torch','numpy','transformers','vllm','neuronxcc');assert not any(n.split('.')[0]in forbidden for n in sys.modules)
 ref=lambda path:dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 report=dict(schema='smollm3_owned_exit_bookkeeping_source_controls_v1',reviewer='/root',observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(Path(candidate.__file__)),checker=ref(Path(__file__)),frozen_original=ref(Path(original.__file__)),tests_passed=result.testsRun,failures=0,errors=0,synthetic_samples_only=True,original_bad_edge_reproduced=True,actual_failed_property_snapshot_available=False,original_sio14_cause_proven=False,source_candidate_integrated=False,native_or_service_execution=False,original_two_second_default_retained=True,new_speedup_claim=False)
 out=Q/'smollm3-owned-exit-bookkeeping-source-controls-20261108-v1.json'
 with out.open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(dict(report=ref(out),passed=result.testsRun)))
if __name__=='__main__':run()
