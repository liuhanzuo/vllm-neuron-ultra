"""v12 source integration and inherited regression checks; no native admission."""
import ast,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v11 as prior
import nemotron9b_package_protocol_v12 as p
import nemotron9b_post_smol_maintenance_v1 as maintenance

def replay(name,path,replacements=()):
 module=types.ModuleType(name);module.__file__=str(path)
 source=path.read_text()
 for before,after in replacements:source=source.replace(before,after)
 exec(compile(source,str(path),'exec'),module.__dict__)
 return module

def funcs(path):
 return {node.name:ast.dump(node)for node in ast.parse(path.read_text().replace('_v11','_v12')).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}

class Closure(unittest.TestCase):
 def test_published_v11_source_hashes_unchanged(self):
  prep=json.loads((R/'research/nemotron9b-v11-socket-runtime-source-preparation-20261108-v2.json').read_text())
  for path,digest in prep['source_sha256'].items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest,path)
 def test_fourteen_runtime_modules_only_version_rename(self):
  files=list(R.glob('nemotron9b_package_*v11.py'));self.assertEqual(len(files),16)
  for old in files:
   if old.name in {'nemotron9b_package_protocol_v11.py','nemotron9b_package_metadata_v11.py'}:continue
   self.assertEqual(old.read_text().replace('_v11','_v12'),old.with_name(old.name.replace('_v11','_v12')).read_text(),old.name)
 def test_protocol_only_source_closure_and_anchor_selection_changed(self):
  before,after=funcs(Path(prior.__file__)),funcs(Path(p.__file__));self.assertEqual(set(before),set(after))
  for name in before:
   if name not in {'required_sources','bundle'}:self.assertEqual(before[name],after[name],name)
  source=Path(prior.__file__).read_text().replace('_v11','_v12')
  original="    validate_smol_terminal_chain(s,catalog,metadata.load)"
  replacement="    from nemotron9b_post_smol_maintenance_v1 import validate as validate_maintenance\n    anchor=validate_maintenance(s,catalog,metadata.load)\n    validate_smol_terminal_chain(dict(s,final_predecessor_catalog=anchor['catalog_reference']),anchor['catalog'],metadata.load)"
  expected=next(node for node in ast.parse(source.replace(original,replacement)).body if isinstance(node,ast.FunctionDef)and node.name=='bundle')
  self.assertEqual(ast.dump(expected),after['bundle'])
 def test_all_historical_metadata_schemas_and_limits_retained(self):
  for schema,shape in prior.metadata.ENVELOPE_SCHEMAS.items():self.assertEqual(p.metadata.ENVELOPE_SCHEMAS[schema],shape)
  self.assertEqual(set(p.metadata.ENVELOPE_SCHEMAS)-set(prior.metadata.ENVELOPE_SCHEMAS),{'nemotron9b_package_runtime_v12','nemotron9b_package_runtime_source_review_v12'})
  self.assertEqual(p.metadata.LIMITS,prior.metadata.LIMITS)
  old,new=funcs(Path(prior.metadata.__file__)),funcs(Path(p.metadata.__file__));self.assertEqual(set(old),set(new))
  for name in old:
   if name!='text':self.assertEqual(old[name],new[name],name)
 def test_required_sources_strict_superset_and_exist(self):
  required=set(p.required_sources());self.assertLess(set(prior.required_sources()),required)
  for name in ['nemotron9b_post_smol_maintenance_v1.py','research/check_nemotron9b_post_smol_maintenance_v1.py','research/check_nemotron9b_package_exec_v12.py']:self.assertIn(str(R/name),required)
  self.assertTrue(all(str(f)in required for f in R.glob('nemotron9b_package_*v12.py')))
  self.assertTrue(all(Path(f).is_file()for f in required))
 def test_full_science_and_resources_retained(self):
  for name in ['STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP','CAPS','BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT']:self.assertEqual(getattr(prior,name),getattr(p,name),name)
  self.assertEqual(len(p.STAGES),52);self.assertEqual(prior.alias_jobs(),p.alias_jobs());self.assertEqual(prior.split_jobs(),p.split_jobs())
  for stage in (*p.STAGES,'controller'):self.assertEqual(prior.resources(stage),p.resources(stage))
 def test_named_maintenance_reader_is_hash_bound(self):
  fixture=replay('bridge_reader_fixture',R/'research/check_nemotron9b_post_smol_maintenance_v1.py')
  spec,catalog,_=fixture.fixture();data=json.dumps(catalog).encode();digest=hashlib.sha256(data).hexdigest();path=Path(spec['final_predecessor_catalog']['path'])
  with patch.object(p.metadata.low,'clear_window'),patch.object(p.metadata,'stable_bytes',return_value=data)as read:
   self.assertEqual(p.metadata.read(path,digest),catalog);read.assert_called_once_with(path,16*2**20)
   with self.assertRaisesRegex(RuntimeError,'fingerprint'):p.metadata.read(path,'0'*64)
 def test_no_generic_raw_payload_reader_exception(self):
  with patch.object(p.metadata.low,'clear_window'),patch.object(p.metadata,'stable_bytes')as read:
   with self.assertRaisesRegex(RuntimeError,'Reviewed UTF8'):p.metadata.text(R/'unrelated.raw','a'*64)
   read.assert_not_called()
 def test_direct_historical_anchor_retains_failed_outcomes(self):
  original=replay('chain_direct_v12',R/'research/check_nemotron9b_smol_terminal_chain_v1.py',[('metadata_v9','metadata_v12')])
  anchor=maintenance.validate(original.SPEC,original.CURRENT,original.loaded)
  self.assertEqual(anchor['catalog_reference'],original.CATALOG);self.assertFalse(anchor['fresh_physical_drain_proven'])
  result=original.chain.validate(dict(original.SPEC,final_predecessor_catalog=anchor['catalog_reference']),anchor['catalog'],original.loaded)
  self.assertEqual([row['original_exit_code']for row in result['producers']],[1,1]);self.assertFalse(result['scientific_success_inferred'])

if __name__=='__main__':
 # Reuse the original meaningful socket/storage/HTTP checks, with v12 imports.
 # The old Closure whitelist describes v10->v11 and is replaced above.
 inherited=replay('inherited_v12',R/'research/check_nemotron9b_package_exec_v11.py',[
  ('_v11','_v12'),("if __name__=='__main__':",'if True:'),
  ('[Closure,IntegratedStorage,','[IntegratedStorage,'),
  ('result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)','pass')])
 bridge=replay('bridge_v12',R/'research/check_nemotron9b_post_smol_maintenance_v1.py')
 chain=replay('chain_v12',R/'research/check_nemotron9b_smol_terminal_chain_v1.py',[('metadata_v9','metadata_v12')])
 suite=unittest.TestSuite([inherited.suite,*[unittest.defaultTestLoader.loadTestsFromTestCase(cls)for cls in [Closure,bridge.Checks,chain.Checks]]])
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)
