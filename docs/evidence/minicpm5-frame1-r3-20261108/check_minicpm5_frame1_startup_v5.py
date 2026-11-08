import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import minicpm5_frame1_mlp_runner_v5 as r
class Checks(unittest.TestCase):
 def setUp(self):
  self.spec=json.loads((R/'research/minicpm5-frame1-mlp-r2-20261108-v1-spec.json').read_text())
  self.a=dict(path='/tmp/fixture-admission.json',sha256='a'*64);self.attempt='test-only';self.phase='native'
  self.argv=r.arguments('phase',self.a,self.phase);self.group=r.p.GROUP_PREFIX+self.spec['units'][self.phase]+'.service'
  cmd=r.service_command(self.spec,self.a,self.attempt,self.phase)
  self.env=cmd[cmd.index('/usr/bin/env'):]
  self.v=dict(r.props(self.phase),Id=self.spec['units'][self.phase]+'.service',LoadState='loaded',InvocationID='a'*32,
    ExecStart='{ argv[]='+ ' '.join(self.env)+' ; }',ExecStopPost='',RuntimeMaxUSec='900s',TimeoutStopUSec='15s',
    ControlGroup=self.group,MainPID='4242',ControlPID='0',ExecMainPID='4242',Job='',ActiveState='active',SubState='running')
  self.proc=dict(pid=4242,start_ticks='123456',ppid=1583,argv=self.argv,cgroup='0::'+self.group)
  self.state={}
 def ready(self,proc=None,v=None,elapsed=0):
  return r.phase_exec_ready(self.proc if proc is None else proc,self.spec,self.a,self.attempt,self.phase,
                           self.v if v is None else v,self.state,elapsed)
 def test_final_argv_ack(self):self.assertTrue(self.ready())
 def test_empty_then_env_then_final_same_identity(self):
  self.assertFalse(self.ready(dict(self.proc,argv=[])))
  self.assertFalse(self.ready(dict(self.proc,argv=self.env),elapsed=1))
  self.assertTrue(self.ready(elapsed=2))
 def test_missing_proc_is_bounded(self):
  self.assertFalse(r.phase_exec_ready(None,self.spec,self.a,self.attempt,self.phase,self.v,self.state,1))
  with self.assertRaises(RuntimeError):self.ready(elapsed=30)
 def test_replaced_start_ticks_rejected(self):
  self.ready(dict(self.proc,argv=[]))
  with self.assertRaises(RuntimeError):self.ready(dict(self.proc,start_ticks='123457'))
 def test_replaced_pid_rejected(self):
  self.ready(dict(self.proc,argv=[]))
  with self.assertRaises(RuntimeError):self.ready(dict(self.proc,pid=4243),dict(self.v,MainPID='4243',ExecMainPID='4243'))
 def test_replaced_invocation_rejected(self):
  self.ready(dict(self.proc,argv=[]))
  with self.assertRaises(RuntimeError):self.ready(v=dict(self.v,InvocationID='b'*32))
 def test_wrong_cgroup_rejected(self):
  with self.assertRaises(RuntimeError):self.ready(dict(self.proc,cgroup='0::/other'))
 def test_wrong_installed_argv_rejected(self):
  with self.assertRaises(RuntimeError):self.ready(v=dict(self.v,ExecStart='{ argv[]=/bin/false ; }'))
 def test_unexpected_argv_rejected(self):
  with self.assertRaises(RuntimeError):self.ready(dict(self.proc,argv=['python3','unrelated.py']))
 def test_terminal_before_ack_rejected(self):
  with self.assertRaises(RuntimeError):self.ready(v=dict(self.v,MainPID='0',ActiveState='inactive',SubState='dead'))
 def test_rejected_snapshot_preserved(self):
  with tempfile.TemporaryDirectory()as tmp:
   actual=dict(self.proc,argv=['unexpected']);root=Path(tmp)
   r.startup_record(root,self.phase,0,self.v,actual,0)
   with self.assertRaises(RuntimeError):self.ready(actual)
   row=json.loads((root/'native-startup-observations.jsonl').read_text())
   self.assertEqual(row['process'],actual);self.assertEqual(row['properties'],self.v)
 def test_journal_finite_and_exclusive(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);r.startup_record(root,self.phase,0,self.v,self.proc,0)
   with self.assertRaises(FileExistsError):r.startup_record(root,self.phase,0,self.v,self.proc,0)
   with self.assertRaises(RuntimeError):r.startup_record(root,self.phase,601,self.v,self.proc,29)
 def run_cleanup(self,after):
  self.ready(dict(self.proc,argv=[]))
  with tempfile.TemporaryDirectory()as tmp,patch.object(r.p,'properties',side_effect=[self.v,after]),patch.object(r.p,'command')as stop,patch.object(r.p,'gone')as gone,patch.object(r.p,'empty_group',return_value={'absent':True})as group:
   result=r.cleanup_phase(self.spec,self.a,self.attempt,self.phase,None,self.state,Path(tmp))
   stop.assert_called_once_with(['systemctl','--user','stop',self.spec['units'][self.phase]],timeout=30)
   gone.assert_called_once_with(self.state['process']);group.assert_called_once_with(self.group)
   return result
 def test_stop_to_collected_service(self):
  after=dict(self.v,LoadState='not-found',InvocationID='',MainPID='0',ControlPID='0',ExecMainPID='0',ActiveState='inactive',SubState='dead')
  self.assertTrue(self.run_cleanup(after))
 def test_stop_to_retained_same_service(self):
  self.assertTrue(self.run_cleanup(dict(self.v,MainPID='0',ActiveState='inactive',SubState='dead')))
 def test_cleanup_replacement_rejected(self):
  with self.assertRaises(RuntimeError):self.run_cleanup(dict(self.v,InvocationID='b'*32,MainPID='0',ActiveState='inactive',SubState='dead'))
 def test_primary_error_survives_cleanup_failure(self):
  primary=ValueError('primary')
  def fail():raise RuntimeError('cleanup')
  try:
   try:raise primary
   finally:r.cleanup_preserving_error(primary,fail)
  except ValueError as caught:self.assertIs(caught,primary)
 def test_cleanup_error_without_primary_propagates(self):
  with self.assertRaises(RuntimeError):r.cleanup_preserving_error(None,lambda:(_ for _ in ()).throw(RuntimeError('cleanup')))
 def test_science_and_original_capture_code_unchanged(self):
  old={x.name:ast.dump(x,include_attributes=False)for x in ast.parse((R/'minicpm5_frame1_mlp_runner_v4.py').read_text()).body if isinstance(x,ast.FunctionDef)}
  new={x.name:ast.dump(x,include_attributes=False)for x in ast.parse((R/'minicpm5_frame1_mlp_runner_v5.py').read_text()).body if isinstance(x,ast.FunctionDef)}
  changed={'load_history_helpers','service_identity','phase_exec_ready','outer'}
  self.assertEqual({k:v for k,v in old.items()if k not in changed},{k:new[k]for k in old if k not in changed})
  pins={'minicpm5_frame1_mlp_worker_v2.py':'8fdd1a477e8e5159a77f3a4c176eea2120f4c0f970d9deef57e0a50a269ceb85','minicpm5_frame1_mlp_replay_v2.py':'8c1a96c664eee0775416d2ce9a6b1cbdf9d44ed04b42d074df66a855198b109e'}
  for name,sha in pins.items():self.assertEqual(hashlib.sha256((R/name).read_bytes()).hexdigest(),sha)
 def test_fresh_original_prerequisites_then_actual_current_catalog(self):
  code="""import sys,json
sys.path.insert(0,'/home/ubuntu/experiments/trn1-latest-models-20261031')
import minicpm5_frame1_mlp_runner_v5 as r
b,o,p=r.old.check_prerequisites(r.p.read_ref(r.BASE_SPEC))
r.load_history_helpers()
h=r.history_metadata
c=h.load(dict(path=str(h.PATH),sha256='d3941454ea1c95f4b79d96887ec9203b4a8261f0f14503d5a3e5a6f1e3842993'))
assert len(c['rows'])==922 and len(c['processes'])==173 and 'torch'not in sys.modules
print(json.dumps(dict(prerequisites_passed=True,rows=922,processes=173,native_imported=False)))
"""
  result=subprocess.run([sys.executable,'-B','-c',code],cwd=R,capture_output=True,text=True,timeout=120)
  self.assertEqual(result.returncode,0,result.stdout+result.stderr);print(result.stdout,flush=True)
if __name__=='__main__':unittest.main(verbosity=2)
