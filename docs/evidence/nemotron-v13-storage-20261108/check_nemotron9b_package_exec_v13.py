"""v13 preserves every scientific stage; only maintenance history dispatch changes."""
import ast,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v12 as prior
import nemotron9b_package_protocol_v13 as p
import nemotron9b_post_smol_maintenance_v2 as maintenance
from research.check_nemotron9b_package_exec_v12 import replay

def funcs(path):
 text=path.read_text().replace('_v12','_v13').replace('nemotron9b_post_smol_maintenance_v1','nemotron9b_post_smol_maintenance_v2')
 return {n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
class Closure(unittest.TestCase):
 def test_frozen_v12_bytes_preserved(self):
  prep=json.loads((R/'research/nemotron9b-v12-maintenance-runtime-source-preparation-20261108-v1.json').read_text())
  for name,digest in prep['source_sha256'].items():self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(),digest,name)
 def test_fourteen_runtime_modules_only_renamed(self):
  files=list(R.glob('nemotron9b_package_*v12.py'));self.assertEqual(len(files),16)
  for old in files:
   if old.name in {'nemotron9b_package_protocol_v12.py','nemotron9b_package_metadata_v12.py'}:continue
   self.assertEqual(old.read_text().replace('_v12','_v13'),old.with_name(old.name.replace('_v12','_v13')).read_text(),old.name)
 def test_protocol_all_functions_preserved_except_additive_sources(self):
  old,new=funcs(Path(prior.__file__)),funcs(Path(p.__file__));self.assertEqual(set(old),set(new))
  for name in old:
   if name!='required_sources':self.assertEqual(old[name],new[name],name)
 def test_metadata_functions_and_all_prior_schemas_preserved(self):
  self.assertEqual(funcs(Path(prior.metadata.__file__)),funcs(Path(p.metadata.__file__)))
  for schema,value in prior.metadata.ENVELOPE_SCHEMAS.items():self.assertEqual(p.metadata.ENVELOPE_SCHEMAS[schema],value)
  self.assertEqual(set(p.metadata.ENVELOPE_SCHEMAS)-set(prior.metadata.ENVELOPE_SCHEMAS),{'nemotron9b_package_runtime_v13','nemotron9b_package_runtime_source_review_v13'})
  self.assertEqual(p.metadata.LIMITS,prior.metadata.LIMITS)
 def test_required_source_superset(self):
  required=set(p.required_sources());self.assertLess(set(prior.required_sources()),required)
  self.assertTrue(all(Path(name).is_file() for name in required))
  self.assertTrue(all(str(path) in required for path in R.glob('nemotron9b_package_*v13.py')))
  for name in ['nemotron9b_post_smol_maintenance_v2.py','research/check_nemotron9b_post_smol_maintenance_v2.py','research/check_nemotron9b_package_exec_v13.py']:self.assertIn(str(R/name),required)
 def test_all_science_and_resources_unchanged(self):
  for name in ['STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP','CAPS','BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT']:self.assertEqual(getattr(prior,name),getattr(p,name),name)
  self.assertEqual(len(p.STAGES),52);self.assertEqual(prior.alias_jobs(),p.alias_jobs());self.assertEqual(prior.split_jobs(),p.split_jobs())
  for stage in (*p.STAGES,'controller'):self.assertEqual(prior.resources(stage),p.resources(stage))
 def test_new_catalog_reader_requires_hash(self):
  from research.check_nemotron9b_post_smol_maintenance_v2 import fixture
  spec,catalog,_=fixture();data=json.dumps(catalog).encode();sha=hashlib.sha256(data).hexdigest()
  with patch.object(p.metadata.low,'clear_window'),patch.object(p.metadata,'stable_bytes',return_value=data) as read:
   self.assertEqual(p.metadata.read(maintenance.CURRENT,sha),catalog);read.assert_called_once_with(maintenance.CURRENT,16*2**20)
   with self.assertRaisesRegex(RuntimeError,'fingerprint'):p.metadata.read(maintenance.CURRENT,'0'*64)
 def test_generic_raw_path_still_rejected(self):
  with patch.object(p.metadata.low,'clear_window'),patch.object(p.metadata,'stable_bytes') as read:
   with self.assertRaisesRegex(RuntimeError,'Reviewed UTF8'):p.metadata.text(R/'unrelated.raw','a'*64)
   read.assert_not_called()
if __name__=='__main__':
 inherited=replay('inherited_v13',R/'research/check_nemotron9b_package_exec_v11.py',[
  ('_v11','_v13'),("if __name__=='__main__':",'if True:'),('[Closure,IntegratedStorage,','[IntegratedStorage,'),
  ('result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)','pass')])
 oldbridge=replay('old_bridge_v13',R/'research/check_nemotron9b_post_smol_maintenance_v1.py')
 chain=replay('chain_v13',R/'research/check_nemotron9b_smol_terminal_chain_v1.py',[('metadata_v9','metadata_v13')])
 from research.check_nemotron9b_post_smol_maintenance_v2 import Checks as NewBridge
 from research.check_finalize_completed_cache_xz_maintenance_20261108_v1 import Checks as Finalizer
 suite=unittest.TestSuite([inherited.suite,*[unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in [Closure,oldbridge.Checks,chain.Checks,NewBridge,Finalizer]]])
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful() else 1)
