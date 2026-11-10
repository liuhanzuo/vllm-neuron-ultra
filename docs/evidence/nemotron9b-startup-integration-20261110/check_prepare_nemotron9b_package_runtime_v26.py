"""v25 builder and history rejection tests over actual authenticated metadata."""
import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path[:0]=[str(R),str(Q)]
import prepare_nemotron9b_package_runtime_v26 as b
h,p=b.history,b.p

class Builder(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.records={}
  def read(ref):
   value=p.metadata.load(ref);cls.records[ref['path']]=value;return value
  cls.done=h.completed(read)
  cls.pins={name:p.metadata.sha(name)for name in p.required_sources()}
  cls.oref=dict(path=str(Q/'nemotron9b-package-runtime-v25-fixture-observer.json'),sha256='c'*64)
 def compose(self,**kw):
  args=dict(done=self.done,run_id='r11-v25-fixture',pins=self.pins,observer_ref=self.oref);args.update(kw)
  return b.compose(**args)
 def test_all_original_model_inputs_science_and_quotas_retained(self):
  s=self.compose();parent=self.done['parent']
  self.assertEqual(set(s),set(parent)|h.ADDED)
  for key,value in parent.items():
   if key not in h.CHANGED:self.assertEqual(s[key],value,key)
  self.assertEqual(s['disk_budget'],parent['disk_budget'])
  self.assertEqual(len(s['stages']),52);self.assertIsNone(s['speedup_denominator'])
  self.assertEqual(s['retirement_policy'],'none')
 def test_unique_nvme_namespaces_and_no_automatic_execution(self):
  s=self.compose()
  self.assertEqual(len(set(s[k]for k in ('control_root','evidence_root','work_root'))),3)
  for k in ('control_root','evidence_root','work_root'):self.assertEqual(Path(s[k]).parent,Path('/mnt/nvme'))
  self.assertFalse(s['automatic_successor_launch']);self.assertFalse(s['runtime_readiness']['ready'])
 def test_reused_run_or_historical_unit_rejected(self):
  with self.assertRaisesRegex(RuntimeError,'Fresh'):self.compose(run_id=self.done['parent']['run_id'])
  altered=dict(self.done,history=dict(self.done['history'],rows=[dict(unit='trn1-latest-verify-nemotron9b-package-r11-v25-fixture-controller')]))
  with self.assertRaisesRegex(RuntimeError,'reuse'):self.compose(done=altered)
 def test_missing_helper_or_rebound_source_rejected(self):
  pins=dict(self.pins);pins.pop(str(R/'nemotron9b_package_startup_v26.py'))
  with self.assertRaisesRegex(RuntimeError,'closure'):self.compose(pins=pins)
  pins=dict(self.pins);pins[next(iter(self.done['source_sha256']))]='0'*64
  with self.assertRaisesRegex(RuntimeError,'rebound'):self.compose(pins=pins)
 def test_history_preserves_all1202_services_and_appends_exact_two_owners(self):
  before=self.records[h.SNAPSHOT['path']];after=self.done['history']
  self.assertEqual(after['rows'],before['rows']);self.assertEqual(after['processes'][:-2],before['processes'])
  self.assertEqual(len(after['processes']),399)
  self.assertEqual([x['pid']for x in after['processes'][-2:]],[254877,358511])
  self.assertFalse(after['nemotron_outcome']['producer_qualified']);self.assertTrue(after['nemotron_outcome']['failed_release'])
 def test_actual_exit_or_owner_mismatch_rejected(self):
  for pin,key,value in [(h.MINI_EXIT,'exit_code',1),(h.TIMING_EXIT,'exit_code',1),(h.MINI_EXIT,'exact_owner_gone',False),(h.TIMING_EXIT,'session_id',1)]:
   bad=dict(self.records[pin['path']]);bad[key]=value
   def read(ref):return bad if ref['path']==pin['path'] else self.records[ref['path']]
   with self.subTest(pin=pin,key=key),self.assertRaises(RuntimeError):h.completed(read)
 def test_incomplete_or_reclassified_history_rejected(self):
  original=self.records[h.SNAPSHOT['path']]
  changes=[dict(rows=original['rows'][:-1]),dict(processes=original['processes'][:-1]),
   dict(nemotron_outcome=dict(original['nemotron_outcome'],producer_qualified=True)),
   dict(previous_mini_failure_promoted=True)]
  for changed in changes:
   bad=dict(original,**changed)
   def read(ref):return bad if ref['path']==h.SNAPSHOT['path']else self.records[ref['path']]
   with self.subTest(change=list(changed)),self.assertRaises(RuntimeError):h.completed(read)
 def test_spec_cannot_remove_prior_science_or_add_unknown_relaxation(self):
  s=self.compose();history=self.done['history']
  base=dict(rows=history['rows'][:1149],processes=history['processes'][:350],canonical_sha256=history['canonical_sha256'])
  for key,value in [('references',{}),('payloads',{}),('resources',{}),('disk_budget',{}),('new_permission',True)]:
   bad=dict(s);bad[key]=value
   with self.subTest(key=key),patch.object(h,'completed',return_value=self.done),self.assertRaises(RuntimeError):
    h.validate_spec(bad,base,p.metadata.load)
 def test_lost_original_owner_source_or_reused_historical_unit_rejected(self):
  s=self.compose();history=self.done['history']
  base=dict(rows=history['rows'][:1149],processes=history['processes'][:350],canonical_sha256=history['canonical_sha256'])
  with patch.object(h,'completed',return_value=self.done):
   self.assertEqual(h.validate_spec(s,base,p.metadata.load)['rows'],history['rows'])
   bad=dict(base,processes=base['processes'][:-1])
   with self.assertRaises(RuntimeError):h.validate_spec(s,bad,p.metadata.load)
   bad=dict(s,source_sha256={})
   with self.assertRaises(RuntimeError):h.validate_spec(bad,base,p.metadata.load)
   bad=dict(s,units=dict(s['units'],controller=history['rows'][0]['unit']))
   with self.assertRaises(RuntimeError):h.validate_spec(bad,base,p.metadata.load)
 def test_spec_and_review_complete_maps_fit_unchanged_envelope(self):
  s=self.compose()
  raw=p.metadata.encode(Q/'nemotron9b-package-runtime-v25-fixture-spec.json',s)
  self.assertLessEqual(len(raw),32*2**20)
  self.assertGreater(len(raw),8*2**20)
 def test_unknown_large_metadata_has_no_blanket_exemption(self):
  with self.assertRaises(RuntimeError):p.metadata.envelope_shape('/mnt/nvme/unrelated/big.json',{'source_sha256':{}})
 def test_native_library_partition_and_all20_payload_pins_retained(self):
  s=self.compose();catalog=p.metadata.load(s['final_predecessor_catalog'])
  parts=p.metadata.classify_sources(s,catalog)
  self.assertEqual(len(parts['historical_payload_sha256']),20)
  self.assertEqual(len(parts['query_native_binary_sha256']),9)
  self.assertEqual(len(parts['python_bytes_sha256']),3)
 def test_import_parser_receives_only_local_python_but_every_external_pin_is_retained(self):
  seen=[]
  def graph(*,entries):
   seen.extend(entries)
   self.assertTrue(entries)
   self.assertTrue(all(x.is_relative_to(R) and x.suffix=='.py'for x in entries))
   return dict(source_sha256={})
  with patch.object(b,'local_imports',side_effect=graph):s,_,_=b.assemble('r11-v25-r2-test-only')
  for path in p.required_sources():self.assertIn(path,s['source_sha256'])
  external=[x for x in p.required_sources()if not Path(x).is_relative_to(R)]
  self.assertEqual(len(external),28)
  self.assertTrue(all(Path(x)not in seen for x in external))
 def test_successor_assembly_does_not_write_or_launch(self):
  with patch.object(b,'local_imports',return_value=dict(source_sha256=self.pins)),patch.object(p,'once')as write,patch.object(p,'command')as launch:
   s,observer,graph=b.assemble('r11-v25-test-only')
  write.assert_not_called();launch.assert_not_called()
  self.assertEqual(observer['participants'],['/root']);self.assertFalse(s['runtime_readiness']['ready'])
if __name__=='__main__':unittest.main(verbosity=2)
