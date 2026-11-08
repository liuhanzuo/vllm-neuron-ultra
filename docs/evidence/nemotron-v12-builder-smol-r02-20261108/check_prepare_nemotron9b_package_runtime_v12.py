"""Real parent preservation plus synthetic future transformation rejection checks.

No future completion records are published and no native/model work is run.
"""
import copy,hashlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from research import prepare_nemotron9b_package_runtime_v12 as b
from research import check_nemotron9b_post_smol_maintenance_v1 as fixture
PARENT=b.p.metadata.load(b.PARENT);PREP=b.p.metadata.load(b.PREPARATION)
PINS=b.merge(PREP['source_sha256'],{name:b.p.metadata.sha(name)for name in b.p.required_sources()})
ORIGINAL_OBSERVER=b.p.metadata.load(PARENT['external_observer_contract'])

class BuilderChecks(unittest.TestCase):
 def setUp(self):
  self.parent=copy.deepcopy(PARENT);self.fixture_spec,self.catalog,self.docs=fixture.fixture();self.catalog_ref=dict(path=str(b.CURRENT),sha256='a'*64)
  self.chain=[dict(catalog=self.catalog['maintenance_extension']['prior_catalog'])]
  self.observer_ref=dict(path=str(b.Q/'nemotron9b-package-fixture-observer.json'),sha256='b'*64);self.pins=dict(PINS)
 def compose(self,**change):
  args=dict(parent=self.parent,catalog=self.catalog,catalog_ref=self.catalog_ref,run_id='r5-source-fixture',observer_ref=self.observer_ref,chain=self.chain,pins=self.pins);args.update(change);return b.compose(**args)
 def test_real_parent_inputs_science_and_failure_history_preserved(self):
  result=self.compose()
  for key in ['payloads','runtime_software_identity','runtime_python_source_sha256','runtime_python_non_utf8_sha256','runtime_binary_stat','runtime_binary_sha256','references','reference_metadata','quality_suite','historical_quality','stages','resources','disk_budget','metadata_aliases','unlaunched_spec_preparation_failure','exit_race_successor','inventory_race_successor','smol_terminal_chain_successor']:
   self.assertEqual(result[key],PARENT[key],key)
  self.assertEqual(result['stages'],list(b.p.STAGES));self.assertFalse(result['runtime_readiness']['ready']);self.assertFalse(result['automatic_successor_launch']);self.assertIsNone(result['speedup_denominator']);self.assertEqual(self.parent,PARENT)
 def test_composed_fixture_passes_independent_maintenance_contract(self):
  result=self.compose();proof=b.maintenance.validate(result,self.catalog,lambda ref:self.docs[ref['path']]);self.assertFalse(proof['fresh_physical_drain_proven']);self.assertFalse(proof['new_speedup_claim'])
 def test_original_source_pin_rebind_rejected(self):
  name=next(iter(self.parent['source_sha256']));self.pins[name]='0'*64
  with self.assertRaisesRegex(RuntimeError,'Never rebind'):self.compose()
 def test_omitted_required_source_rejected(self):
  self.pins.pop(str(R/'nemotron9b_package_worker_v12.py'))
  with self.assertRaisesRegex(RuntimeError,'Complete new source'):self.compose()
 def test_original_model_commit_change_rejected(self):
  self.parent['package_commit']='0'*40
  with self.assertRaisesRegex(RuntimeError,'Original model'):self.compose()
 def test_stage_omission_rejected(self):
  self.parent['stages'].remove('long')
  with self.assertRaisesRegex(RuntimeError,'Every original'):self.compose()
 def test_resource_relaxation_rejected(self):
  self.parent['resources']['compile']['MemoryMax']='1'
  with self.assertRaisesRegex(RuntimeError,'resource bounds'):self.compose()
 def test_storage_floor_reduction_rejected(self):
  self.parent['disk_budget']['root_floor']-=1
  with self.assertRaisesRegex(RuntimeError,'storage policy'):self.compose()
 def test_failed_reference_cannot_become_speedup_denominator(self):
  self.parent['speedup_denominator']='failed-r4'
  with self.assertRaisesRegex(RuntimeError,'denominator'):self.compose()
 def test_automatic_successor_launch_rejected(self):
  self.parent['automatic_successor_launch']=True
  with self.assertRaisesRegex(RuntimeError,'automatic'):self.compose()
 def test_reused_run_identifier_and_unsafe_path_rejected(self):
  for run in [self.parent['run_id'],'../escape','bad_name','x'*41]:
   with self.assertRaisesRegex(RuntimeError,'run identifier'):self.compose(run_id=run)
 def test_existing_service_collision_rejected(self):
  self.catalog['rows'].append(dict(unit='trn1-latest-verify-nemotron9b-package-r5-source-fixture-long'))
  with self.assertRaisesRegex(RuntimeError,'never-reused'):self.compose()
 def test_control_names_normalized_without_changing_stage_names(self):
  result=self.compose();self.assertIn('alias-control-functional-finite_nonzero',result['stages']);self.assertTrue(result['units']['alias-control-functional-finite_nonzero'].endswith('finite-nonzero'));self.assertEqual(len(set(result['units'].values())),53)
 def test_unfinalized_catalog_rejected(self):
  self.catalog['root_finalized']=False
  with self.assertRaisesRegex(RuntimeError,'completed predecessor'):self.compose()
 def test_wrong_current_catalog_namespace_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'maintenance namespace'):self.compose(catalog_ref=dict(path='/tmp/other.json',sha256='a'*64))
 def test_omitted_or_wrong_smol_anchor_rejected(self):
  for chain in [[],[dict(catalog=dict(path='/tmp/old.json',sha256='0'*64))]]:
   with self.assertRaisesRegex(RuntimeError,'Smol anchor'):self.compose(chain=chain)
 def test_published_v12_and_v11_bytes_unchanged(self):
  old=b.p.metadata.load(PREP['parent_preparation'])
  for prep in [PREP,old]:
   for path,digest in prep['source_sha256'].items():self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(),digest,path)
 def test_missing_actual_history_cannot_emit_any_specification(self):
  original=b.p.metadata.load
  def load(ref):
   if ref['path']==str(b.CURRENT):raise FileNotFoundError('actual completed maintenance catalog absent')
   return original(ref)
  with patch.object(b.p.metadata,'load',side_effect=load),patch.object(b.p,'once')as publish:
   with self.assertRaises(FileNotFoundError):b.build(self.catalog_ref,'r5-source-fixture')
   publish.assert_not_called()
 def test_no_native_imports(self):
  self.assertFalse(any(name.split('.')[0]in ['torch','numpy','vllm','transformers','neuronxcc','torch_neuronx']for name in sys.modules))

class ChainAndObserver(unittest.TestCase):
 def test_real_prior_chain_selected_and_original_failures_preserved(self):
  ref=PARENT['final_predecessor_catalog'];catalog=b.p.metadata.load(ref);chain=b.selected_chain(ref,catalog,b.p.metadata.load);self.assertEqual(chain,PARENT['required_smol_predecessor_chain'])
  result=b.smol.validate(PARENT,catalog,b.p.metadata.load);self.assertEqual([x['original_exit_code']for x in result['producers']],[1,1]);self.assertFalse(result['scientific_success_inferred'])
 def test_cycle_rejected(self):
  ref=PARENT['final_predecessor_catalog'];catalog=b.p.metadata.load(ref);catalog=copy.deepcopy(catalog);catalog['appended_from']=ref;original=b.p.metadata.load
  def load(reference):return catalog if reference==ref else original(reference)
  with self.assertRaisesRegex(RuntimeError,'acyclic'):b.selected_chain(ref,catalog,load)
 def test_non_smol_anchor_cannot_hide_missing_smol_chain(self):
  with self.assertRaisesRegex(RuntimeError,'completed sio14'):b.selected_chain(dict(path='/tmp/non-smol.json',sha256='a'*64),{},lambda ref:{})
 def test_observer_policy_changes_only_current_runtime_source_hash(self):
  result=b.observer_policy(PARENT,ORIGINAL_OBSERVER,'f'*64);self.assertEqual({k:v for k,v in result.items()if k!='source_sha256'},{k:v for k,v in ORIGINAL_OBSERVER.items()if k!='source_sha256'})
 def test_observer_policy_cannot_relax_exclusive_window(self):
  row=copy.deepcopy(ORIGINAL_OBSERVER);row['shared_gate_covers_entire_operation']=False
  with self.assertRaisesRegex(RuntimeError,'isolation'):b.observer_policy(PARENT,row,'f'*64)
 def test_no_unregistered_or_concurrent_observer_added(self):
  row=copy.deepcopy(ORIGINAL_OBSERVER);row['participants'].append('/other')
  with self.assertRaisesRegex(RuntimeError,'Root-only'):b.observer_policy(PARENT,row,'f'*64)
if __name__=='__main__':unittest.main(verbosity=2)
