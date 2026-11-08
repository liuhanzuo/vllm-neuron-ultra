"""Real retained r7 failure plus adversarial metadata checks; no process work."""
import copy,json,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'research')]
import finalize_nemotron9b_package_r7_root_v2 as f

class RetainedFailed(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.spec=f.m.load(f.SPEC);cls.before=f.m.load(f.m.ref(f.C/'external-pre-stop.json'))
 def run_case(self,change=None):
  saved=copy.deepcopy(self.before['rows']['compile']);rows={}
  for name in ['compile-owner.json','compile-cleanup-before-stop.json','compile-launch.json','failed-terminal.json']:rows[name]=f.m.load(f.m.ref(f.C/name))
  if change:change(saved,rows)
  return f.m.retained_stage(self.spec,'compile',f.SPEC,f.REVIEW,f.p,f.C,saved,read=lambda path:rows[Path(path).name])
 def test_actual_failed_stage(self):
  row,out=self.run_case();self.assertFalse(row['never_launched']);self.assertFalse(out['original_exit_success']);self.assertTrue(out['retained_failed_service']);self.assertEqual(out['original_exit_properties']['Result'],'timeout')
 def test_all53_actual_stages(self):
  values=[f.m.retained_stage(self.spec,stage,f.SPEC,f.REVIEW,f.p,f.C,self.before['rows'][stage])for stage in ['controller',*self.spec['stages']]]
  self.assertEqual(len(values),53);self.assertEqual(sum(bool(b.get('worker_reported_passed'))for a,b in values),42)
 def test_reject_success_promotion(self):
  with self.assertRaises(RuntimeError):self.run_case(lambda s,r:s.update(successful=True))
 def test_reject_live_pid(self):
  with self.assertRaises(RuntimeError):self.run_case(lambda s,r:s['properties'].update(MainPID='3209603'))
 def test_reject_foreign_invocation(self):
  with self.assertRaises(RuntimeError):self.run_case(lambda s,r:s['properties'].update(InvocationID='f'*32))
 def test_reject_zero_exit(self):
  with self.assertRaises(RuntimeError):self.run_case(lambda s,r:s['properties'].update(Result='success',ExecMainCode='1',ExecMainStatus='0'))
 def test_reject_unverified_owner(self):
  with self.assertRaises(RuntimeError):self.run_case(lambda s,r:s.update(identity_verified=False))
 def test_reject_missing_second_drain(self):
  def change(s,r):next(x['drain']for x in r['failed-terminal.json']['cleanup']if x['phase']=='compile')['samples'].pop()
  with self.assertRaises(RuntimeError):self.run_case(change)
 def test_reject_live_group(self):
  def change(s,r):next(x['drain']for x in r['failed-terminal.json']['cleanup']if x['phase']=='compile')['samples'][0]['group']['absent']=False
  with self.assertRaises(RuntimeError):self.run_case(change)
 def test_reject_changed_terminal_sample(self):
  def change(s,r):next(x['drain']for x in r['failed-terminal.json']['cleanup']if x['phase']=='compile')['samples'][0]['properties']['ExecMainStatus']='9'
  with self.assertRaises(RuntimeError):self.run_case(change)
 def test_reject_changed_launch(self):
  with self.assertRaises(RuntimeError):self.run_case(lambda s,r:r['compile-launch.json']['command'].append('--foreign'))
 def test_reject_controller_success(self):
  with self.assertRaises(RuntimeError):self.run_case(lambda s,r:r['failed-terminal.json'].update(passed=True))

if __name__=='__main__':unittest.main()
