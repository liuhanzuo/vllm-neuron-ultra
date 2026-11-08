"""Real retained prepare/r4-failure evidence and explicit synthetic append controls.
No actual future terminal records, stop operations or native execution.
"""
import copy,hashlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
R=Path(__file__).resolve().parents[1];Q=R/'research';sys.path.insert(0,str(R));sys.path.insert(0,str(Q))
import finalize_nemotron9b_package_r5_root_20261108_v1 as f
import nemotron9b_r4_terminal_metadata_v2 as m
SPEC,REVIEW,LIVE=f.setup();PRIOR=f.p.metadata.load(f.PRIOR)

def fixture():
 result=copy.deepcopy(PRIOR);rows=[dict(unit=SPEC['units'][s],synthetic_fixture_only=True)for s in ['controller',*SPEC['stages']]]
 process=dict(LIVE['owner'],role='synthetic_future_exit_owner')
 release=dict(path='/explicit-source-fixture/release.json',sha256='a'*64);audit=dict(path='/explicit-source-fixture/audit.json',sha256='b'*64);exited=dict(path='/explicit-source-fixture/exit.json',sha256='c'*64);outcome=dict(producer_qualified=False)
 result['rows']+=rows;result['processes'].append(process);result['root_release_refs'].append(release)
 result.update(schema='root_post_nemotron_package_complete_predecessors_v1',complete_count=916,appended_from=f.PRIOR,appended_terminal_audit=audit,latest_producer='nemotron9b_package_r5',latest_producer_qualified=False,latest_producer_exit=exited,latest_producer_terminal_audit=audit,root_finalized=True,all_planned_launches_accounted_for=True,nemotron9b_package_r5_append=dict(services=copy.deepcopy(rows),processes=[copy.deepcopy(process)],actual_exit=exited,terminal_review=audit,all_planned_services=53))
 return result,release,audit,exited,outcome

class Append(unittest.TestCase):
 def setUp(self):self.result,self.release,self.audit,self.exited,self.outcome=fixture()
 def validate(self):return f.validate_catalog_extension(PRIOR,self.result,SPEC,self.release,self.audit,self.exited,self.outcome)
 def reject(self):
  with self.assertRaises(RuntimeError):self.validate()
 def test_exact_synthetic_append_only(self):self.assertTrue(self.validate());self.assertEqual(self.result['maintenance_extension'],PRIOR['maintenance_extension'])
 def test_historical_row_omission(self):self.result['rows'].pop(0);self.reject()
 def test_planned_stage_omission(self):self.result['rows'].pop();self.reject()
 def test_planned_stage_reordering(self):self.result['rows'][-2:]=reversed(self.result['rows'][-2:]);self.reject()
 def test_prior_maintenance_rewrite(self):self.result['maintenance_extension']['failed_exit']['sha256']='0'*64;self.reject()
 def test_prior_process_omission(self):self.result['processes'].pop(0);self.reject()
 def test_new_owner_omission(self):self.result['processes'].pop();self.reject()
 def test_reused_owner_identity(self):self.result['processes'][-1]=copy.deepcopy(self.result['processes'][0]);self.reject()
 def test_old_release_drop(self):self.result['root_release_refs'].pop(0);self.reject()
 def test_source_rebinding(self):name=next(iter(PRIOR['source_sha256']));self.result['source_sha256'][name]='0'*64;self.reject()
 def test_missing_source_pin(self):name=next(iter(PRIOR['source_sha256']));self.result['source_sha256'].pop(name);self.reject()
 def test_false_qualification(self):self.result['latest_producer_qualified']=True;self.reject()
 def test_wrong_current_reference(self):self.result['appended_from']=self.audit;self.reject()
 def test_new_unreviewed_field(self):self.result['invented']=True;self.reject()
 def test_relabel_original_failure_field(self):self.result['failed_experiment_retained']=not self.result['failed_experiment_retained'];self.reject()
 def test_wrong_new_slice(self):self.result['nemotron9b_package_r5_append']['services'].pop();self.reject()

class ActualRetained(unittest.TestCase):
 def test_actual_current_prepare_exit_metadata(self):
  saved=m.load(m.ref(f.C/'prepare-retained-exit.json'));row,summary=m.retained_stage(SPEC,'prepare',f.SPEC,f.REVIEW,f.p,f.C,saved)
  self.assertFalse(row['never_launched']);self.assertTrue(summary['normal_retained_exit_success']);self.assertTrue(summary['worker_reported_complete']);self.assertTrue(summary['worker_reported_passed'])
 def test_actual_prepare_wrong_owner_rejected(self):
  saved=m.load(m.ref(f.C/'prepare-retained-exit.json'));saved['owner']['start_ticks']='1'
  with self.assertRaises(RuntimeError):m.retained_stage(SPEC,'prepare',f.SPEC,f.REVIEW,f.p,f.C,saved)
 def test_absence_cannot_replace_actual_captured_owner(self):
  saved=m.load(m.ref(f.C/'prepare-retained-exit.json'))
  with self.assertRaises(RuntimeError):m.retained_stage(SPEC,'prepare',f.SPEC,f.REVIEW,f.p,f.C,saved,exists=lambda p:False if p.name=='prepare-owner.json'else p.exists())
 def test_no_default_future_exit_and_no_outputs_created(self):
  with self.assertRaisesRegex(RuntimeError,'Real external exit'):f.main(types.SimpleNamespace(actual_exit=None,actual_exit_sha256=None))
  self.assertFalse(f.AUDIT.exists());self.assertFalse(f.CATALOG.exists())
 def test_live_current_owner_stops_finalizer_before_drain_or_catalog(self):
  receipt=dict(schema='nemotron9b_package_r5_actual_external_exit_v1',reviewer='/root',actual_tool_exit_observed=True,execution_spec=f.SPEC,source_review=f.REVIEW,owner=copy.deepcopy(LIVE['owner']),exact_owner_gone=True,actual_exit=dict(session_id=92674,exit_code=0,tool_chunk='abcdef'))
  original=m.load;reference=dict(path='/synthetic-future/exit.json',sha256='d'*64)
  def load(ref):return receipt if ref==reference else original(ref)
  with patch.object(m,'load',side_effect=load),patch.object(f,'physical_drain')as drain,patch.object(f,'record')as write:
   with self.assertRaises(RuntimeError):f.main(types.SimpleNamespace(actual_exit=reference['path'],actual_exit_sha256=reference['sha256']))
   drain.assert_not_called();write.assert_not_called()
 def test_frozen_original_helper_and_finalizer_unchanged(self):
  self.assertEqual(hashlib.sha256((Q/'finalize_nemotron9b_package_r4_root_20261108_v4.py').read_bytes()).hexdigest(),'32137fa6aa7c76cef8d7c4021245783e478b338b415c8010086eab41498f0899')
  self.assertEqual(f.DIAGNOSTICS,[]);self.assertEqual(SPEC['stages'],list(f.p.STAGES));self.assertEqual(len(SPEC['stages']),52)
if __name__=='__main__':
 # Reuse independent original outcome controls and actual interrupted-r4 proof.
 import check_nemotron9b_r4_terminal_metadata_v2 as old
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(cls)for cls in [Append,ActualRetained,old.Outcomes,old.StoppedLive,old.Processes]])
 result=unittest.TextTestRunner(verbosity=2).run(suite);raise SystemExit(0 if result.wasSuccessful()else 1)
