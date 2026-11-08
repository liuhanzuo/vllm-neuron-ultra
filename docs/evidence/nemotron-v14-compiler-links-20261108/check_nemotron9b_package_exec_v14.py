"""v14 actual storage/evidence integration, exact history and unchanged science."""
import ast,copy,hashlib,json,os,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import nemotron9b_package_protocol_v13 as prior
import nemotron9b_package_protocol_v14 as p
import nemotron9b_package_evidence_v14 as evidence
import nemotron9b_completed_history_drain_v1 as drain
from research.check_nemotron9b_package_exec_v12 import replay
from research.check_nemotron9b_compiler_asset_links_v1 import Checks as AssetChecks
CATALOG=dict(path=str(p.completed_history.CURRENT),sha256='f58472178de26c52ecbf5cfdefdde05a866d981772e9d0dc109a9767007aeec4')

def funcs(path):
 return {n.name:ast.dump(n) for n in ast.parse(path.read_text().replace('_v13','_v14')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
class Closure(unittest.TestCase):
 def test_frozen_v13_bytes_preserved(self):
  old=json.loads((R/'research/nemotron9b-package-runtime-v13-r5-20261108-spec.json').read_text())
  for path in R.glob('nemotron9b_package_*v13.py'):self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),old['source_sha256'][str(path)])
 def test_ten_science_runtime_modules_only_renamed(self):
  files=list(R.glob('nemotron9b_package_*v13.py'));self.assertEqual(len(files),16)
  changed={'protocol','metadata','evidence','controller','external','window'}
  checked=[]
  for old in files:
   if old.name.removeprefix('nemotron9b_package_').removesuffix('_v13.py')in changed:continue
   self.assertEqual(old.read_text().replace('_v13','_v14'),old.with_name(old.name.replace('_v13','_v14')).read_text(),old.name);checked.append(old)
  self.assertEqual(len(checked),10)
 def test_protocol_unchanged_outside_explicit_integration(self):
  old,new=funcs(Path(prior.__file__)),funcs(Path(p.__file__));self.assertEqual(set(old),set(new))
  for name in old:
   if name not in {'required_sources','bundle','_usage_once','usage','storage'}:self.assertEqual(old[name],new[name],name)
 def test_metadata_unchanged_except_named_history_dispatch(self):
  old,new=funcs(Path(prior.metadata.__file__)),funcs(Path(p.metadata.__file__));self.assertEqual(set(old),set(new))
  for name in old:
   if name!='text':self.assertEqual(old[name],new[name],name)
  for schema,value in prior.metadata.ENVELOPE_SCHEMAS.items():self.assertEqual(p.metadata.ENVELOPE_SCHEMAS[schema],value)
  self.assertEqual(set(p.metadata.ENVELOPE_SCHEMAS)-set(prior.metadata.ENVELOPE_SCHEMAS),{'nemotron9b_package_runtime_v14','nemotron9b_package_runtime_source_review_v14'})
  self.assertEqual(p.metadata.LIMITS,prior.metadata.LIMITS)
 def test_evidence_unchanged_except_asset_tree_and_row_verification(self):
  old,new=funcs(R/'nemotron9b_package_evidence_v13.py'),funcs(Path(evidence.__file__));self.assertEqual(set(old),set(new))
  for name in old:
   if name not in {'tree','verify_rows'}:self.assertEqual(old[name],new[name],name)
 def test_controller_external_window_change_only_drain_routing(self):
  for name in ['controller','external','window']:
   old=(R/('nemotron9b_package_'+name+'_v13.py')).read_text().replace('_v13','_v14')
   new=(R/('nemotron9b_package_'+name+'_v14.py')).read_text()
   old=old.replace('p.low.predecessor_drain(catalog)',"p.predecessor_drain(catalog,allowed_units=[spec['units']['controller']])"if name=='controller'else'p.predecessor_drain(catalog)')
   self.assertEqual(old,new,name)
 def test_required_source_superset(self):
  required=set(p.required_sources());self.assertLess(set(prior.required_sources()),required);self.assertTrue(all(Path(name).is_file()for name in required))
  self.assertTrue(all(str(path)in required for path in R.glob('nemotron9b_package_*v14.py')))
 def test_all_science_and_resources_unchanged(self):
  for name in ['STAGES','SCIENTIFIC_STAGES','ROOT_FLOOR','NVME_FLOOR','ROOT_CAP','NVME_CAP','CAPS','BUCKETS','SPLIT_FLAGS','ALIAS_FLAGS','PACKAGE','COMMIT']:self.assertEqual(getattr(prior,name),getattr(p,name),name)
  self.assertEqual(len(p.STAGES),52);self.assertEqual(prior.alias_jobs(),p.alias_jobs());self.assertEqual(prior.split_jobs(),p.split_jobs())
  for stage in (*p.STAGES,'controller'):self.assertEqual(prior.resources(stage),p.resources(stage))
 def test_actual_large_history_complete_and_hash_bound(self):
  value=p.metadata.load(CATALOG);self.assertEqual(len(value['rows']),925);self.assertEqual(len(value['processes']),182)
  row=next(row for row in value['rows']if row['unit']==drain.history.prior.previous.UNIT)
  self.assertIsNone(row['owner']);self.assertIs(row['never_launched'],False)
  with self.assertRaisesRegex(RuntimeError,'history bytes'):p.metadata.load(dict(CATALOG,sha256='0'*64))
 def test_generic_raw_path_still_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Reviewed UTF8'):p.metadata.text(R/'unrelated.raw','a'*64)

class IntegratedAssets(AssetChecks):
 def setUp(self):
  super().setUp();self.spec=dict(work_root=str(self.work),control_root=str(self.work.parent/'control'),evidence_root=str(self.work.parent/'evidence'),compiler_asset_policy=dict(p.compiler_links.ASSET_REF))
  Path(self.spec['control_root']).mkdir();Path(self.spec['evidence_root']).mkdir()
 def test_real_storage_and_final_evidence_accept_all_ten(self):
  actual=p.storage(self.spec,reserve=True);self.assertEqual(actual['nvme_used'],self.inventory())
  rows=evidence.tree(self.work,lambda:None,spec=self.spec);self.assertEqual(len(rows),10);evidence.verify_rows(rows,lambda:None,self.spec)
  self.assertTrue(all(row['compiler_asset']for row in rows.values()))
  category=evidence.tree(self.compiler,lambda:None,spec=self.spec);self.assertEqual(category,rows)
 def test_storage_scope_and_counted_category_cap(self):
  with patch.object(p,'usage',wraps=p.usage)as calls:p.storage(self.spec)
  for call in calls.call_args_list:
   enabled=call.args[0]in (self.work,self.compiler)
   self.assertEqual(call.kwargs.get('compiler_root'),self.compiler if enabled else None)
   self.assertEqual(call.kwargs.get('compiler_assets'),self.policy if enabled else None)
  with patch.dict(p.CAPS,{'compile-work':self.inventory()-1}):
   with self.assertRaisesRegex(RuntimeError,'category cap compile-work'):p.storage(self.spec)
 def test_missing_policy_and_unrelated_context_rejected(self):
  spec=dict(self.spec);del spec['compiler_asset_policy']
  with self.assertRaises(KeyError):p.storage(spec)
  with self.assertRaises(KeyError):evidence.tree(self.work,lambda:None,spec=spec)
  with self.assertRaises(RuntimeError):evidence.tree(self.work,lambda:None)
 def test_final_evidence_rejects_link_replacement(self):
  rows=evidence.tree(self.work,lambda:None,spec=self.spec);path=self.group/'act_info.json';path.unlink();path.symlink_to(self.policy['assets']['silu_and_others.json']['path'])
  with self.assertRaises(RuntimeError):evidence.verify_rows(rows,lambda:None,self.spec)
 def test_final_evidence_rejects_modified_identity_or_target_bytes(self):
  rows=evidence.tree(self.work,lambda:None,spec=self.spec);changed=copy.deepcopy(rows);changed[str(self.group/'act_info.json')]['stat']['st_ino']+=1
  with self.assertRaisesRegex(RuntimeError,'identity changed'):evidence.verify_rows(changed,lambda:None,self.spec)
  with patch.object(p.compiler_links.os,'read',side_effect=[b'x',b'']):
   with self.assertRaises(RuntimeError):evidence.verify_rows(rows,lambda:None,self.spec)
 def test_final_evidence_rejects_wrong_directory_and_name(self):
  path=self.work/'act_info.json';path.symlink_to(self.policy['assets']['act_info.json']['path'])
  with self.assertRaises(RuntimeError):evidence.tree(self.work,lambda:None,spec=self.spec)

class DrainChecks(unittest.TestCase):
 def setUp(self):
  from research.finalize_smollm3_sio14_root_20261108_v1 import life
  self.life=life;self.allowed='trn1-latest-verify-nemotron9b-package-r6-fixture-controller'
  self.catalog=dict(rows=[],processes=[])
  for obj,name,value in [(drain.history,'validate',lambda x:x),(life,'command',lambda args:self.allowed+'.service loaded active running'),(life,'gone',lambda owner:None)]:
   context=patch.object(obj,name,side_effect=value);context.start();self.addCleanup(context.stop)
 def test_only_explicit_current_controller_may_be_live(self):
  self.assertEqual(drain.physical_drain(self.catalog,[self.allowed]),[])
  with self.assertRaises(RuntimeError):drain.physical_drain(self.catalog)
 def test_other_live_unit_is_not_exempted(self):
  with patch.object(self.life,'command',return_value=self.allowed+'.service running\ntrn1-other.service running'):
   with self.assertRaises(RuntimeError):drain.physical_drain(self.catalog,[self.allowed])
 def test_historical_controller_cannot_be_exempted(self):
  self.catalog['rows']=[dict(unit=self.allowed)]
  with self.assertRaisesRegex(RuntimeError,'historical'):drain.physical_drain(self.catalog,[self.allowed])
 def test_noncontroller_or_multiple_exemptions_rejected(self):
  for allowed in [['trn1-other'],[self.allowed,self.allowed],['trn1-latest-verify-nemotron9b-package-r6-compile']]:
   with self.assertRaises(RuntimeError):drain.physical_drain(self.catalog,allowed)
 def test_missing_ack_row_requires_actual_absence(self):
  name=drain.history.prior.previous.UNIT;self.catalog['rows']=[dict(unit=name,owner=None,never_launched=False,recorded_cgroups=[])]
  value=dict(Id=name+'.service',LoadState='not-found',InvocationID='',MainPID='0',ControlPID='0',ExecMainPID='0')
  with patch.object(self.life,'props',return_value=value),patch.object(self.life,'terminal',return_value=True),patch.object(self.life,'cgroup_inventory',return_value=[]),patch.object(drain.time,'sleep'):
   self.assertEqual(len(drain.physical_drain(self.catalog,[self.allowed])),1)
   value['InvocationID']='unexpected'
   with self.assertRaisesRegex(RuntimeError,'missing-ACK'):drain.physical_drain(self.catalog,[self.allowed])

if __name__=='__main__':
 inherited=replay('inherited_v14',R/'research/check_nemotron9b_package_exec_v11.py',[
  ('_v11','_v14'),("if __name__=='__main__':",'if True:'),('[Closure,IntegratedStorage,','[IntegratedStorage,'),
  ('for root in self.spec.values():Path(root).mkdir()',"for root in self.spec.values():Path(root).mkdir()\n  self.spec['compiler_asset_policy']=dict(p.compiler_links.ASSET_REF)"),
  ('result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)','pass')])
 suite=unittest.TestSuite([inherited.suite,*[unittest.defaultTestLoader.loadTestsFromTestCase(cls)for cls in [Closure,IntegratedAssets,DrainChecks]]])
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)
