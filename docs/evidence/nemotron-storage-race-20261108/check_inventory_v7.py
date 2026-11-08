"""Actual filesystem rename race and retained lifecycle regression checks."""
import ast,hashlib,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path('/home/ubuntu/experiments/trn1-latest-models-20261031');sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v6 as old
import nemotron9b_package_protocol_v7 as p
original_walk=os.walk

def charged(path):
 s=path.lstat();return max(s.st_size,s.st_blocks*512)

class Inventory(unittest.TestCase):
 def setUp(self):self.directory=tempfile.TemporaryDirectory();self.root=Path(self.directory.name)
 def tearDown(self):self.directory.cleanup()
 def test_original_disappearing_tmp_failure_reproduced(self):
  tmp=self.root/'highwater.json.tmp';tmp.write_bytes(b'pending')
  def walking(*args,**kwargs):
   for item in original_walk(*args,**kwargs):tmp.unlink();yield item
  with patch.object(old.os,'walk',side_effect=walking):
   with self.assertRaises(FileNotFoundError):old.usage(self.root)
 def test_atomic_rename_triggers_full_rescan_and_includes_new_file(self):
  tmp=self.root/'highwater.json.tmp';tmp.write_bytes(b'pending');dest=self.root/'highwater.json';calls=[]
  def walking(*args,**kwargs):
   calls.append(1)
   for item in original_walk(*args,**kwargs):
    if len(calls)==1:tmp.replace(dest)
    yield item
  with patch.object(p.os,'walk',side_effect=walking):self.assertEqual(p.usage(self.root),charged(dest))
  self.assertEqual(len(calls),2)
 def test_all_live_temporary_files_are_charged(self):
  files=[self.root/'done.json',self.root/'next.json.tmp',self.root/'.pending-file']
  for x in files:x.write_bytes(b'x'*6000)
  self.assertEqual(p.usage(self.root),sum(charged(x)for x in files))
 def test_rescan_discards_partial_sum(self):
  a=self.root/'a';a.write_bytes(b'a');tmp=self.root/'z.tmp';tmp.write_bytes(b'z');calls=[]
  def walking(*args,**kwargs):
   calls.append(1)
   if len(calls)==1:
    tmp.unlink();yield(str(self.root),[],['a','z.tmp'])
   else:yield from original_walk(*args,**kwargs)
  with patch.object(p.os,'walk',side_effect=walking):self.assertEqual(p.usage(self.root),charged(a))
 def test_persistent_churn_fails_boundedly(self):
  with patch.object(p.os,'walk',side_effect=FileNotFoundError('rename'))as walk,patch.object(p.time,'sleep'):
   with self.assertRaisesRegex(RuntimeError,'did not stabilize'):p.usage(self.root)
   self.assertEqual(walk.call_count,8)
 def test_permission_error_is_not_skipped(self):
  def walking(*args,**kwargs):kwargs['onerror'](PermissionError('denied'));return iter(())
  with patch.object(p.os,'walk',side_effect=walking):
   with self.assertRaises(PermissionError):p.usage(self.root)
 def test_disappearing_output_root_is_not_zero_usage(self):
  def walking(*args,**kwargs):self.root.rmdir();raise FileNotFoundError('removed root')
  with patch.object(p.os,'walk',side_effect=walking):
   with self.assertRaisesRegex(RuntimeError,'root disappeared'):p.usage(self.root)
 def test_absent_uncreated_root_is_zero(self):self.assertEqual(p.usage(self.root/'uncreated'),0)
 def test_directory_symlink_is_rejected(self):
  (self.root/'alias').symlink_to(self.root,target_is_directory=True)
  with self.assertRaisesRegex(RuntimeError,'directory symlinks'):p.usage(self.root)
 def test_file_symlink_outside_prepared_view_rejected(self):
  (self.root/'file').symlink_to('/dev/null')
  with self.assertRaises(RuntimeError):p.usage(self.root)
 def test_existing_model_view_symlink_rule_preserved(self):
  (self.root/'view').mkdir();(self.root/'view'/'weight').symlink_to('/dev/null')
  self.assertEqual(p.usage(self.root,True),0)
 def test_hardlinked_evidence_rejected(self):
  a=self.root/'a';a.write_bytes(b'x');os.link(a,self.root/'b')
  with self.assertRaises(RuntimeError):p.usage(self.root)
 def test_all_original_v6_files_unchanged(self):
  spec=json.loads((R/'research/nemotron9b-package-runtime-v6-r2-20261108-spec.json').read_text())
  for f in R.glob('nemotron9b_package_*v6.py'):self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),spec['source_sha256'][str(f)])
 def test_other_fourteen_modules_only_import_rename(self):
  for f in R.glob('nemotron9b_package_*v6.py'):
   if f.name in ('nemotron9b_package_protocol_v6.py','nemotron9b_package_metadata_v6.py'):continue
   self.assertEqual(ast.dump(ast.parse(f.read_text().replace('_v6','_v7'))),ast.dump(ast.parse(f.with_name(f.name.replace('_v6','_v7')).read_text())))
 def test_all_other_protocol_functions_unchanged(self):
  def functions(path):return {x.name:ast.dump(x)for x in ast.parse(path.read_text().replace('_v6','_v7')).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
  before=functions(R/'nemotron9b_package_protocol_v6.py');after=functions(R/'nemotron9b_package_protocol_v7.py')
  for name,tree in before.items():
   if name not in ('usage','required_sources'):self.assertEqual(tree,after[name],name)
 def test_all_v6_envelopes_preserved(self):
  for name,shape in old.metadata.ENVELOPE_SCHEMAS.items():self.assertEqual(shape,p.metadata.ENVELOPE_SCHEMAS[name])

if __name__=='__main__':
 # Re-run all23 exit-race/source tests with v7 as the module under test.
 previous=types.ModuleType('exit_race_v7_replay');previous.__file__=str(R/'research/check_nemotron9b_package_exit_race_v6.py')
 source=Path(previous.__file__).read_text().replace('import nemotron9b_package_protocol_v6 as p','import nemotron9b_package_protocol_v7 as p')
 exec(compile(source,previous.__file__,'exec'),previous.__dict__)
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Inventory),unittest.defaultTestLoader.loadTestsFromTestCase(previous.ExitRace)])
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful() else 1)
