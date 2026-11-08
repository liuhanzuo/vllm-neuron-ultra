"""Run actual v8 HTTP lifecycle functions against kernel/HTTP failure fixtures."""
import ast,hashlib,json,sys,types,unittest
from pathlib import Path
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v7 as prior
import nemotron9b_package_protocol_v8 as p
class Closure(unittest.TestCase):
 def test_all_v7_source_bytes_still_original(self):
  s=json.loads((R/'research/nemotron9b-package-runtime-v7-r3-20261108-spec.json').read_text())
  for f in R.glob('nemotron9b_package_*v7.py'):self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),s['source_sha256'][str(f)])
 def test_all_science_and_other_modules_only_version_rename(self):
  for f in R.glob('nemotron9b_package_*v7.py'):
   if f.name in ('nemotron9b_package_protocol_v7.py','nemotron9b_package_metadata_v7.py','nemotron9b_package_worker_v7.py'):continue
   self.assertEqual(ast.dump(ast.parse(f.read_text().replace('_v7','_v8'))),ast.dump(ast.parse(f.with_name(f.name.replace('_v7','_v8')).read_text())))
 def test_other_protocol_and_worker_functions_unchanged(self):
  for prefix,allowed in [('protocol',{'required_sources'}),('worker',{'http_stage'})]:
   def funcs(version):return {n.name:ast.dump(n)for n in ast.parse((R/('nemotron9b_package_'+prefix+'_'+version+'.py')).read_text().replace('_v7','_v8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
   a,b=funcs('v7'),funcs('v8');self.assertEqual(set(a),set(b))
   for name in a:
    if name not in allowed:self.assertEqual(a[name],b[name],name)
 def test_historical_envelope_support_preserved(self):
  for name,shape in prior.metadata.ENVELOPE_SCHEMAS.items():self.assertEqual(shape,p.metadata.ENVELOPE_SCHEMAS[name])
 def test_source_closure_pins_transition_helper(self):
  required=set(p.required_sources())
  self.assertIn(str(R/'nemotron9b_exec_transition_v1.py'),required)
  self.assertIn(str(R/'research/check_nemotron9b_package_exec_v8.py'),required)
  self.assertTrue(all(str(f)in required for f in R.glob('nemotron9b_package_*v7.py')))
 def test_science_and_caps_unchanged(self):
  for name in ['STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP','CAPS','BUCKETS','SPLIT_FLAGS','PACKAGE','COMMIT']:self.assertEqual(getattr(prior,name),getattr(p,name),name)
if __name__=='__main__':
 module=types.ModuleType('http_v8_replay');module.__file__=str(R/'check_nemotron9b_package_server_lifecycle_v4.py')
 s=Path(module.__file__).read_text().replace('_v4','_v8');exec(compile(s,module.__file__,'exec'),module.__dict__)
 inventory=types.ModuleType('inventory_v8_replay');inventory.__file__=str(R/'research/check_nemotron9b_package_inventory_v7.py')
 s=Path(inventory.__file__).read_text().replace('import nemotron9b_package_protocol_v7 as p','import nemotron9b_package_protocol_v8 as p');exec(compile(s,inventory.__file__,'exec'),inventory.__dict__)
 class TransientHttp(unittest.TestCase):
  def test_real_http_stage_accepts_only_resolved_authenticated_empty_argv(self):
   import tempfile
   with tempfile.TemporaryDirectory()as tmp:
    h=module.Harness(Path(tmp));original_proc=h.proc;empty=[False]
    def proc(pid):
     if h.executed and not empty[0]:empty[0]=True;return dict(h.child_owner,argv=[])
     return original_proc(pid)
    h.proc=proc;result=h.run();h.check(result)
    diagnostic=p.read(Path(tmp)/'quality-server-exec-diagnostic.json')
    self.assertTrue(empty[0]);self.assertEqual(diagnostic['status'],'transient_empty_argv_resolved')
    self.assertFalse(diagnostic['qualified']);self.assertTrue(result['passed'])
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(TransientHttp),unittest.defaultTestLoader.loadTestsFromTestCase(Closure),unittest.defaultTestLoader.loadTestsFromTestCase(module.ActualHttpRegression)])
 for name in unittest.defaultTestLoader.getTestCaseNames(inventory.Inventory):
  if name not in ('test_other_fourteen_modules_only_import_rename','test_all_other_protocol_functions_unchanged'):suite.addTest(inventory.Inventory(name))
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful() else 1)
