import copy,json,sys,unittest
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');sys.path.insert(0,str(R))
from nemotron9b_exec_transition_v1 import await_exec
CHILD=dict(pid=101,start_ticks='200',ppid=100,cgroup='0::/owned',argv=['python','worker'])
TARGET=['python','-B','-m','exact.module']
class TestTransition(unittest.TestCase):
 def run_case(self,snapshots,*,exit_code=None,intent_error=False,observe_error=False,deadline=1.0,step=.005):
  rows=iter(snapshots);last=[snapshots[-1]];now=[0.0];saved=[];intent=[];observed=[]
  def read():
   last[0]=next(rows,last[0]);return copy.deepcopy(last[0])
  def sleep(seconds):now[0]+=step
  def validate():
   intent.append(True)
   if intent_error:raise RuntimeError('Bad intent')
  def observe():
   observed.append(True)
   if observe_error:raise RuntimeError('Unstable double observation')
   return dict(exact=True)
  self.saved=saved;self.intent=intent;self.observed=observed
  return await_exec(child_owner=CHILD,target_argv=TARGET,read_process=read,poll=lambda:exit_code,guard=lambda:None,validate_intent=validate,observe_target=observe,save_diagnostic=saved.append,deadline=deadline,clock=lambda:now[0],sleep=sleep)
 def test_direct_exact_target(self):
  self.assertEqual(self.run_case([dict(CHILD,argv=TARGET)]),dict(exact=True));self.assertEqual(self.saved,[]);self.assertEqual(len(self.observed),1)
 def test_original_then_target(self):self.assertEqual(self.run_case([CHILD,dict(CHILD,argv=TARGET)]),dict(exact=True))
 def test_authenticated_empty_resolves(self):
  self.run_case([CHILD,dict(CHILD,argv=[]),dict(CHILD,argv=TARGET)])
  self.assertEqual(self.saved[0]['status'],'transient_empty_argv_resolved');self.assertFalse(self.saved[0]['qualified']);self.assertEqual(len(self.intent),2)
 def test_pid_reuse_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'kernel identity'):self.run_case([dict(CHILD,start_ticks='201',argv=[])])
  self.assertEqual(self.saved[0]['last']['start_ticks'],'201')
 def test_parent_group_pid_changes_rejected(self):
  for key,value in [('pid',102),('ppid',99),('cgroup','0::/unowned')]:
   with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'kernel identity'):self.run_case([dict(CHILD,**{key:value},argv=TARGET)])
 def test_empty_without_bound_intent_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Bad intent'):self.run_case([dict(CHILD,argv=[])],intent_error=True)
 def test_persistent_empty_timeout(self):
  with self.assertRaisesRegex(RuntimeError,'exec grace'):self.run_case([dict(CHILD,argv=[])],step=.1)
  self.assertFalse(self.observed);self.assertEqual(self.saved[0]['status'],'failed')
 def test_late_target_does_not_bypass_grace(self):
  with self.assertRaisesRegex(RuntimeError,'exec grace'):self.run_case([dict(CHILD,argv=[]),dict(CHILD,argv=TARGET)],step=.3)
 def test_unexpected_nonempty_argv_saved_and_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Unexpected server argv'):self.run_case([dict(CHILD,argv=['vLLM::Other'])])
  self.assertEqual(self.saved[0]['last']['argv'],['vLLM::Other'])
 def test_empty_reverts_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'reverted'):self.run_case([dict(CHILD,argv=[]),CHILD])
 def test_child_exit_not_accepted(self):
  with self.assertRaisesRegex(RuntimeError,'exited'):self.run_case([dict(CHILD,argv=TARGET)],exit_code=0)
 def test_missing_child_not_accepted(self):
  with self.assertRaisesRegex(RuntimeError,'disappeared'):self.run_case([None])
 def test_original_never_execs(self):
  with self.assertRaisesRegex(RuntimeError,'deadline'):self.run_case([CHILD],deadline=.02)
 def test_double_observation_still_required(self):
  with self.assertRaisesRegex(RuntimeError,'Unstable'):self.run_case([dict(CHILD,argv=TARGET)],observe_error=True)
 def test_target_intent_still_required(self):
  with self.assertRaisesRegex(RuntimeError,'Bad intent'):self.run_case([dict(CHILD,argv=TARGET)],intent_error=True)
if __name__=='__main__':unittest.main()
