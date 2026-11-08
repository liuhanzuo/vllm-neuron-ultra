"""Real completed history and rejection checks for v15 preparation."""
import copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research import prepare_nemotron9b_package_runtime_v15 as b
class Checks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.parent=b.p.metadata.load(b.PARENT);cls.catalog=b.p.metadata.load(b.CATALOG)
  cls.pins={name:b.p.metadata.sha(name)for name in b.p.required_sources()}
  cls.pins[str(b.TESTS_PATH)]='a'*64
  cls.observer=dict(path=str(b.Q/'nemotron9b-v14-fixture-observer.json'),sha256='a'*64)
 def compose(self,**changes):
  args=dict(parent=self.parent,catalog=self.catalog,run_id='r7-builder-fixture',observer_ref=self.observer,pins=self.pins);args.update(changes);return b.compose(**args)
 def test_actual_parent_all_scientific_and_provenance_fields_preserved(self):
  result=self.compose()
  for key,value in self.parent.items():
   if key not in b.CHANGED:self.assertEqual(result[key],value,key)
  self.assertEqual(set(result)-set(self.parent),b.ADDED);self.assertEqual(len(result['stages']),52)
  self.assertFalse(result['runtime_readiness']['ready']);self.assertIsNone(result['speedup_denominator'])
 def test_actual_completed_history_and_earlier_failures_verified(self):
  proof=b.validate_history(self.compose(),self.catalog)
  self.assertEqual((proof['complete_services'],proof['complete_processes']),(981,196));self.assertFalse(proof['fresh_physical_drain_proven'])
 def test_original_pin_cannot_be_rebound(self):
  pins=dict(self.pins);pins[next(iter(self.parent['source_sha256']))]='0'*64
  with self.assertRaisesRegex(RuntimeError,'Never rebind'):self.compose(pins=pins)
 def test_new_worker_pin_cannot_be_omitted(self):
  pins=dict(self.pins);del pins[str(R/'nemotron9b_package_worker_v15.py')]
  with self.assertRaisesRegex(RuntimeError,'Complete new source'):self.compose(pins=pins)
 def test_model_commit_and_science_and_budgets_cannot_change(self):
  for field in ['package_commit','stages','resources','disk_budget','automatic_successor_launch','speedup_denominator']:
   value=copy.deepcopy(self.parent)
   if field=='package_commit':value[field]='0'*40
   elif field=='stages':value[field].remove('long')
   elif field=='resources':value[field]['compile']['MemoryMax']='1'
   elif field=='disk_budget':value[field]['root_floor']-=1
   elif field=='automatic_successor_launch':value[field]=True
   else:value[field]='failed-r5'
   with self.subTest(field=field),self.assertRaises(RuntimeError):self.compose(parent=value)
 def test_run_identifier_reuse_and_escape_rejected(self):
  for run_id in [self.parent['run_id'],'../escape','bad_name','x'*41]:
   with self.assertRaises(RuntimeError):self.compose(run_id=run_id)
 def test_historical_services_and_processes_cannot_be_dropped(self):
  for field in ['rows','processes']:
   catalog=dict(self.catalog);catalog[field]=catalog[field][:-1]
   with self.assertRaises(RuntimeError):self.compose(catalog=catalog)
 def test_existing_service_name_cannot_be_reused(self):
  catalog=dict(self.catalog);catalog['rows']=list(catalog['rows']);catalog['rows'][0]=dict(unit='trn1-latest-verify-nemotron9b-package-r7-builder-fixture-long')
  with self.assertRaisesRegex(RuntimeError,'never-reused'):self.compose(catalog=catalog)
 def test_unfinalized_history_rejected(self):
  with self.assertRaises(RuntimeError):self.compose(catalog=dict(self.catalog,root_finalized=False))
 def test_missing_catalog_does_not_write_any_files(self):
  original=b.p.metadata.load
  def load(ref):
   if ref==b.CATALOG:raise FileNotFoundError('catalog absent')
   return original(ref)
  with patch.object(b.p.metadata,'load',side_effect=load),patch.object(b.p,'once')as publish:
   with self.assertRaises(FileNotFoundError):b.build('r7-builder-fixture')
   publish.assert_not_called()
 def test_original_root_only_observer_preserved(self):
  old=b.p.metadata.load(self.parent['external_observer_contract']);new=b.observer_policy(self.parent,old,'f'*64)
  self.assertEqual(new['participants'],['/root']);self.assertEqual({k:v for k,v in old.items()if k!='source_sha256'},{k:v for k,v in new.items()if k!='source_sha256'})
 def test_new_observer_cannot_relax_isolation(self):
  old=b.p.metadata.load(self.parent['external_observer_contract']);old['participants'].append('/other')
  with self.assertRaises(RuntimeError):b.observer_policy(self.parent,old,'f'*64)
 def test_no_native_imports(self):
  self.assertFalse(any(name.split('.')[0]in ['torch','numpy','vllm','transformers','neuronxcc','torch_neuronx']for name in sys.modules))

 def test_maintenance_receipt_requires_actual_successful_exit(self):
  spec=self.compose();original=b.p.metadata.load
  def load(ref):
   value=original(ref)
   if ref==b.history.MAINTENANCE:
    value=copy.deepcopy(value);value['actual_exit']['exit_code']=1
   return value
  with self.assertRaisesRegex(RuntimeError,'maintenance external exit'):
   b.history.validate_maintenance(spec,load)
 def test_maintenance_evidence_cannot_be_omitted(self):
  spec=self.compose();del spec['source_sha256'][b.history.MAINTENANCE['path']]
  with self.assertRaisesRegex(RuntimeError,'maintenance evidence closure'):
   b.history.validate_maintenance(spec,b.p.metadata.load)
 def test_live_maintenance_owner_is_rejected(self):
  with patch.object(b.history.h.m.p,'gone',side_effect=RuntimeError('still live')):
   with self.assertRaisesRegex(RuntimeError,'still live'):
    b.history.validate_maintenance(self.compose(),b.p.metadata.load)
 def test_old_finalizer_exit_cannot_replace_latest(self):
  value=b.p.metadata.load(b.FINALIZER);value['actual_exit']=dict(session_id=84203,exit_code=0,tool_chunk='029ea3')
  with self.assertRaisesRegex(RuntimeError,'observed finalizer exit'):
   b.history.finalizer(self.catalog,value)

if __name__=='__main__':unittest.main(verbosity=2)
