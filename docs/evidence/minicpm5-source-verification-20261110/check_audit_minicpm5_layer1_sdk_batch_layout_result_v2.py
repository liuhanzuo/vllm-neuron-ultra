"""Synthetic complete auditor orchestration and ordering; no model/raw IO."""
import base64
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import audit_minicpm5_layer1_sdk_batch_layout_result_v2 as a
from research.check_minicpm5_layer1_sdk_batch_layout_post_exit_core_v1 import fixture,ref


class Audit(unittest.TestCase):
    def setUp(self):
        self.f=fixture();ended,live,spec,plan,review,records=self.f
        ended['live']=ref('live');self.exit_ref=ref('model-exit');spec['canonical_sha256']={'canonical.json':'c'*64};spec['root_floor_bytes']=28*2**30
        records['admission']['attempt_id']='d'*32;self.owner={'pid':50,'start_ticks':'150','argv':['synthetic-auditor']};self.lock=ended['lock_identity']
        units=[dict(unit='trn1-synthetic-%d'%i,samples=[dict(properties=dict(Id='trn1-synthetic-%d.service'%i,
            MainPID='0',ControlPID='0',ActiveState='inactive',SubState='dead',Job=''),
            group={'group':a.physical_evidence.GROUP_PREFIX+'trn1-synthetic-%d.service'%i,'absent':True})]*2)for i in range(1308)]
        self.done=dict(binding=plan['latest_completion'],lock_identity=self.lock,owners=[{'pid':90,'start_ticks':'190'}],
            collector={},native={},current_history={},nemotron=dict(history=dict(rows=[{'unit':r['unit']}for r in units])))
        self.actual=dict(complete=True,physical_checks_performed=True,all_current_and_prior_services_preserved=True,
            lock_identity=self.lock,all_additional_owners_checked=self.done['owners'],
            nemotron=dict(complete=True,fresh_physical_drain_verified=True,services=1308,observations=units))
        records['drains']=dict(complete=True,latest_completion=plan['latest_completion'],outer_owner=ended['owner'],
            lock_identity=self.lock,complete_observation_sha256='a'*64,latest_physical=copy.deepcopy(self.actual))
        records['coordinator']['validated']['descriptive']={'synthetic':True}
        for rank in (0,1):
            records['rank%d_result'%rank]['guard_checks']=dict(schema='minicpm5_layer1_sdk_batch_layout_guard_checks_v2',
                fast_checks=100,full_checks=8,period_seconds=1.0,every_source_fully_authenticated=True,
                full_source_checks_not_sampled=True,every_call_keeps_original_serial_lock_process_and_performance_checks=True,
                events=[dict(boundary=boundary,monotonic=float(i),owners=1,metadata=12 if i>=2 else 0)
                    for i,boundary in enumerate(a.GUARD_BOUNDARIES)])
        self.events=[]
        self.p=SimpleNamespace(lock_proof=Mock(side_effect=lambda *x:self.events.append('lock')),clear_window=Mock(),
            gone=Mock(side_effect=lambda *x:self.events.append('gone')),active_heavy=Mock(),empty_group=Mock(),properties=Mock(),sha=Mock(return_value='c'*64))
        self.runner=SimpleNamespace(p=self.p,read_bundle=Mock(return_value=(spec,plan)),
            verify_owned_drain=Mock(side_effect=lambda *x:self.events.append('model-drain')or {'physically_drained':True}),
            admitted_bundles=Mock(return_value=(self.done,lambda *x:None)),latest=SimpleNamespace(physical=Mock(
                side_effect=lambda *x:self.events.append('predecessor-drain')or copy.deepcopy(self.actual))),
            layer1_results=SimpleNamespace(validate_rank_pair=Mock(side_effect=lambda *x:self.events.append('raw')or copy.deepcopy(records['coordinator']['validated']))),
            verify_loaded_sources=Mock(return_value={'complete':True}),usage=Mock(),reference=lambda x:ref('source'))
        self.data={self.exit_ref['path']:ended,ended['live']['path']:live,ended['review']['path']:review}
        self.data.update({ended['outputs'][k]['path']:v for k,v in records.items()})
        self.patches=[patch.object(a.sources,'load',side_effect=lambda r:copy.deepcopy(self.data[r['path']])),
            patch.object(a,'identity',return_value={'identity':'stable'}),patch.object(a.sources,'classify',return_value={}),
            patch.object(a.sources,'validate_native_receipt',side_effect=lambda *x:self.events.append('native-receipt')or {'valid':True}),
            patch.object(a.sources,'check_native_identities',side_effect=lambda *x:self.events.append('native-identities')),
            patch.object(a.sources,'verify_text_sources',side_effect=lambda *x:self.events.append('text'))]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop()for p in reversed(self.patches)])
    def run_audit(self):return a.audit(self.runner,self.exit_ref,self.owner,self.lock)
    def test_complete_order_and_lossless_physical(self):
        r=self.run_audit();self.assertTrue(r['complete']);self.assertFalse(r['self_exit_claimed'])
        self.assertLess(self.events.index('model-drain'),self.events.index('raw'))
        self.assertLess(self.events.index('predecessor-drain'),self.events.index('raw'))
        self.assertLess(self.events.index('native-receipt'),self.events.index('raw'))
        packed=r['actual_predecessor_drain'];raw=zlib.decompress(base64.b64decode(packed['data']));self.assertEqual(json.loads(raw),dict(schema=a.physical_evidence.PAIR_SCHEMA,
            first=dict(tree=self.actual,supplemental_observations=[]),second=dict(tree=self.actual,supplemental_observations=[])))
        self.assertEqual(len(a.drain_units(json.loads(raw))),1308)
        self.assertEqual(self.runner.latest.physical.call_count,2)
        self.assertEqual(r['schema'],'minicpm5_layer1_sdk_batch_layout_post_exit_audit_v2')
        self.assertTrue(r['legacy_saved_physical_semantics_not_requalified'])
        self.runner.layer1_results.validate_rank_pair.assert_called_once();self.runner.read_bundle.assert_called_once()
    def test_nonzero_model_tool_exit_before_sources_or_raw(self):
        self.f[0]['actual_exit_code']=1
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.read_bundle.assert_not_called();self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_lock_mismatch_before_read_bundle(self):
        self.f[0]['lock_identity']={'inode':999}
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.read_bundle.assert_not_called()
    def test_live_model_owner_prevents_raw(self):
        self.p.gone.side_effect=RuntimeError('live')
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_unreleased_model_service_prevents_raw(self):
        self.runner.verify_owned_drain.side_effect=RuntimeError('live service')
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_false_saved_physical_prevents_raw(self):
        self.f[5]['drains']['latest_physical']['physical_checks_performed']=False
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_replaced_saved_owner_prevents_raw(self):
        self.f[5]['drains']['outer_owner']={}
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_bad_native_receipt_prevents_raw(self):
        with patch.object(a.sources,'validate_native_receipt',side_effect=RuntimeError('wrong receipt')):
            with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_changed_native_identity_prevents_raw(self):
        with patch.object(a.sources,'check_native_identities',side_effect=RuntimeError('changed identity')):
            with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_failed_predecessor_physical_prevents_raw(self):
        self.runner.latest.physical.side_effect=RuntimeError('not drained')
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_missing_historical_unit_prevents_raw(self):
        self.f[5]['drains']['latest_physical']['nemotron']['observations'].pop()
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_saved_live_unit_prevents_raw(self):
        self.f[5]['drains']['latest_physical']['nemotron']['observations'][0]['samples'][0]['properties']['MainPID']='99'
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_raw_mismatch_no_completed_result(self):
        self.runner.layer1_results.validate_rank_pair.side_effect=lambda *x:{'changed':True}
        with self.assertRaises(RuntimeError):self.run_audit()
    def test_source_change_after_raw_rejects(self):
        with patch.object(a.sources,'verify_text_sources',side_effect=RuntimeError('changed source')):
            with self.assertRaises(RuntimeError):self.run_audit()
    def test_loaded_helper_closure_rechecked(self):
        self.run_audit();self.runner.verify_loaded_sources.assert_called_once()
    def test_output_restriction_before_any_production_import(self):
        with self.assertRaises(RuntimeError):a.output_path('/tmp/anything.json')
    def test_original_limits_retained(self):
        self.assertEqual(a.LIMITS,dict(address_space_bytes=2*2**30,cpu_seconds=600,hard_cpu_seconds=605,
            output_file_bytes=2*2**20,wall_seconds=900,cpu=6,nice=19))
    def test_guard_boundary_omission_prevents_raw(self):
        g=self.f[5]['rank0_result']['guard_checks'];g['events'].pop(3);g['full_checks']-=1
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_saved_single_sample_omission_prevents_raw(self):
        self.f[5]['drains']['latest_physical']['nemotron']['observations'][0]['samples'].pop()
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_saved_sample_group_omission_prevents_raw(self):
        for sample in self.f[5]['drains']['latest_physical']['nemotron']['observations'][0]['samples']:sample.pop('group',None)
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_guard_owner_count_matches_actual_history(self):
        for e in self.f[5]['rank0_result']['guard_checks']['events']:e['owners']=2
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_unsourced_rank_cannot_claim_complete(self):
        self.f[5]['rank0_result']['guard_checks']['every_source_fully_authenticated']=False
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_rank_metadata_watch_cannot_shrink(self):
        self.f[5]['rank1_result']['guard_checks']['events'][-1]['metadata']=0
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_periodic_full_events_may_interleave(self):
        g=self.f[5]['rank1_result']['guard_checks'];g['events'].insert(4,dict(boundary='periodic',monotonic=3.5,owners=1,metadata=12));g['full_checks']+=1
        self.assertTrue(self.run_audit()['complete'])
    def test_first_missing_all_samples_prevents_raw(self):
        self.actual['nemotron']['observations']=[{'unit':r['unit']}for r in self.actual['nemotron']['observations']]
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_first_nonempty_group_prevents_raw(self):
        for sample in self.actual['nemotron']['observations'][0]['samples']:
            sample['group'].update(absent=False,pids=[777],directories=[])
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_first_mismatched_unit_Id_prevents_raw(self):
        for sample in self.actual['nemotron']['observations'][0]['samples']:
            sample['properties']['Id']='trn1-synthetic-1.service'
        with self.assertRaises(RuntimeError):self.run_audit()
    def test_first_unit_samples_deleted_prevents_raw(self):
        self.actual['nemotron']['observations'][0].pop('samples')
        with self.assertRaises(RuntimeError):self.run_audit()
    def test_first_pending_Job_prevents_raw(self):
        for sample in self.actual['nemotron']['observations'][0]['samples']:sample['properties']['Job']='1234'
        with self.assertRaises(RuntimeError):self.run_audit()
    def test_second_complete_tree_difference_prevents_raw(self):
        second=copy.deepcopy(self.actual);second['unprojected_marker']='changed'
        self.runner.latest.physical.side_effect=[copy.deepcopy(self.actual),second]
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_second_full_physical_exception_prevents_raw(self):
        self.runner.latest.physical.side_effect=[copy.deepcopy(self.actual),RuntimeError('second physical failed')]
        with self.assertRaises(RuntimeError):self.run_audit()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_saved_group_only_extra_retained_and_supplemented(self):
        name='trn1-extra-old-mini';path=a.physical_evidence.GROUP_PREFIX+name+'.service'
        self.actual['model_group']={'group':path,'absent':True}
        self.f[5]['drains']['latest_physical']['model_group']={'group':path,'absent':True}
        self.p.properties.return_value=dict(Id=name+'.service',MainPID='0',ControlPID='0',
            ActiveState='inactive',SubState='dead',Job='')
        self.p.empty_group.return_value={'group':path,'absent':True}
        result=self.run_audit();decoded=json.loads(zlib.decompress(base64.b64decode(result['actual_predecessor_drain']['data'])))
        self.assertEqual(len(a.drain_units(decoded)),1309)
        self.assertEqual([c.args[0]for c in self.p.properties.call_args_list],[name,name])
    def test_new_unknown_group_rejected_before_supplement_access(self):
        self.actual['model_group']={'group':a.physical_evidence.GROUP_PREFIX+'trn1-unbound.service','absent':True}
        with self.assertRaises(RuntimeError):self.run_audit()
        self.p.properties.assert_not_called()
        self.runner.layer1_results.validate_rank_pair.assert_not_called()
    def test_no_native_import(self):
        self.assertFalse({'torch','numpy','neuronxcc','torch_neuronx'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
