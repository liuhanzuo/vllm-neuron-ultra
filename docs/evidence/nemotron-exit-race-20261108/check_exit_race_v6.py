"""Bounded source tests for exit TOCTOU; no framework, device or model load."""
import ast, copy, hashlib, io, json, sys, unittest
from pathlib import Path
from unittest.mock import patch
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v5 as old
import nemotron9b_package_protocol_v6 as p
C=R/'nemotron9b-package-r1-20261108-control'
spec=json.loads((R/'research/nemotron9b-package-runtime-v5-r1-20261108-spec.json').read_text())
live=json.loads((R/'research/nemotron9b-package-runtime-v5-r1-20261108-root-live.json').read_text())
stage='alias-control-persistent-finite_nonzero'
saved=json.loads((C/(stage+'-cleanup-before-stop.json')).read_text());owner=saved['owner'];terminal=saved['properties']
spec_ref=live['execution_spec'];review_ref=live['source_review']
running=dict(terminal,MainPID=str(owner['pid']),ControlGroup=p.group(spec,stage),ActiveState='active',SubState='running',ExecMainCode='0')
actual=p.proc_owner(owner)

def call(module=p,owner_arg=owner,first=running,following=None,process=None):
 with patch.object(p.low,'properties',return_value=terminal if following is None else following),patch.object(p.low,'proc',return_value=process),patch.object(p.time,'sleep'),patch.object(p.time,'monotonic',return_value=0):
  return module.service_identity(spec,stage,spec_ref,review_ref,owner_arg,observation=first)

def stat_bytes(state='Z',start=None):
 fields=['0']*22;fields[0]=state;fields[19]=owner['start_ticks'] if start is None else start
 return (str(owner['pid'])+' (python) '+' '.join(fields)).encode()

class ExitRace(unittest.TestCase):
 def test_old_reproduces_absent_process_race(self):
  with self.assertRaisesRegex(RuntimeError,'Actual process'):call(module=old)
 def test_same_invocation_terminal_recovery(self):
  seen,value=call();self.assertEqual(seen,owner);self.assertEqual(value,terminal)
 def test_normal_live_observation(self):self.assertEqual(call(process=actual),(owner,running))
 def test_normal_retained_terminal(self):self.assertEqual(call(first=terminal),(owner,terminal))
 def test_no_owner_cannot_recover(self):
  with self.assertRaises(RuntimeError):call(owner_arg=None)
 def test_failed_exit_is_returned_without_success(self):
  failed=dict(terminal,Result='exit-code',ExecMainStatus='7',ActiveState='failed',SubState='failed')
  self.assertEqual(call(following=failed)[1],failed)
  with patch.object(p.low,'properties',return_value=failed):self.assertFalse(p.snapshot(spec,stage,spec_ref,review_ref,owner)['successful'])
 def test_replaced_invocation_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Original invocation'):call(following=dict(terminal,InvocationID='f'*32))
 def test_replaced_pid_rejected(self):
  with self.assertRaises(RuntimeError):call(following=dict(terminal,ExecMainPID='17'))
 def test_changed_installed_command_rejected(self):
  with self.assertRaises(RuntimeError):call(following=dict(terminal,ExecStart='bad'))
 def test_changed_resource_rejected(self):
  with self.assertRaises(RuntimeError):call(following=dict(terminal,MemoryMax='1'))
 def test_gc_cannot_create_success(self):
  with self.assertRaises(RuntimeError):call(following=dict(terminal,LoadState='not-found',InvocationID=''))
 def test_live_wrong_argv_rejected(self):
  with self.assertRaises(RuntimeError):call(process=dict(actual,argv=['unexpected']))
 def test_live_wrong_cgroup_rejected(self):
  with self.assertRaises(RuntimeError):call(process=dict(actual,cgroup='0::/'))
 def test_pid_reuse_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Owner replaced'):call(process=dict(actual,start_ticks='1'))
 def test_zombie_requires_exact_original_identity(self):
  with patch.object(Path,'open',return_value=io.BytesIO(stat_bytes())):self.assertTrue(p._original_exit_sample(dict(actual,argv=[]),owner))
  self.assertFalse(p._original_exit_sample(dict(actual,argv=[],start_ticks='1'),owner))
 def test_running_empty_argv_rejected(self):
  with patch.object(Path,'open',return_value=io.BytesIO(stat_bytes('R'))):self.assertFalse(p._original_exit_sample(dict(actual,argv=[]),owner))
 def test_zombie_pid_reuse_rejected(self):
  with patch.object(Path,'open',return_value=io.BytesIO(stat_bytes(start='1'))):self.assertFalse(p._original_exit_sample(dict(actual,argv=[]),owner))
 def test_transition_wait_is_bounded(self):
  with self.assertRaisesRegex(RuntimeError,'did not reach'):call(following=running)
 def test_original_science_resources_and_storage(self):
  self.assertEqual(p.STAGES,old.STAGES);self.assertEqual(len(p.STAGES),52)
  self.assertEqual([p.resources(x) for x in (*p.STAGES,'controller')],[old.resources(x) for x in (*old.STAGES,'controller')])
  self.assertEqual((p.CAPS,p.ROOT_CAP,p.NVME_CAP,p.ROOT_FLOOR,p.NVME_FLOOR),(old.CAPS,old.ROOT_CAP,old.NVME_CAP,old.ROOT_FLOOR,old.NVME_FLOOR))
 def test_all_original_source_pins_unchanged(self):
  for f in R.glob('nemotron9b_package_*v5.py'):self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),spec['source_sha256'][str(f)])
 def test_other_fourteen_modules_only_rename_imports(self):
  for f in R.glob('nemotron9b_package_*v5.py'):
   if f.name in ('nemotron9b_package_protocol_v5.py','nemotron9b_package_metadata_v5.py'):continue
   self.assertEqual(ast.dump(ast.parse(f.read_text().replace('_v5','_v6'))),ast.dump(ast.parse(f.with_name(f.name.replace('_v5','_v6')).read_text())))
 def test_all_other_protocol_functions_unchanged(self):
  def functions(path):return {n.name:ast.dump(n) for n in ast.parse(path.read_text().replace('_v5','_v6')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
  before=functions(R/'nemotron9b_package_protocol_v5.py');after=functions(R/'nemotron9b_package_protocol_v6.py')
  for name,tree in before.items():
   if name not in ('service_identity','required_sources'):self.assertEqual(tree,after[name],name)
 def test_named_metadata_retains_old_envelopes(self):
  a=old.metadata.ENVELOPE_SCHEMAS;b=p.metadata.ENVELOPE_SCHEMAS
  for name,shape in a.items():self.assertEqual(shape,b[name])
  self.assertEqual(old.metadata.LIMITS,p.metadata.LIMITS);self.assertEqual(old.metadata.NON_UTF8,p.metadata.NON_UTF8)

if __name__=='__main__':unittest.main(verbosity=2)
