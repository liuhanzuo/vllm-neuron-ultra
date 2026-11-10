"""Observed-exit recorder tests using synthetic completed model metadata only."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import record_minicpm5_layer1_sdk_batch_layout_exit_v1 as r
from research.check_minicpm5_layer1_sdk_batch_layout_post_exit_core_v1 import fixture,ref


class Recorder(unittest.TestCase):
    def setUp(self):
        self.ended,self.live,self.spec,self.plan,self.review,self.records=fixture()
        self.records['admission']['attempt_id']='d'*32;self.lock=self.ended['lock_identity'];self.recorder={'pid':80,'start_ticks':'180'}
        self.live_ref=ref('live');self.done=dict(binding=self.plan['latest_completion'],lock_identity=self.lock,owners=[])
        physical=dict(complete=True,physical_checks_performed=True,all_current_and_prior_services_preserved=True,
            lock_identity=self.lock,all_additional_owners_checked=[],nemotron=dict(complete=True,fresh_physical_drain_verified=True,services=1308))
        self.records['drains']=dict(complete=True,latest_completion=self.plan['latest_completion'],outer_owner=self.live['owner'],
            lock_identity=self.lock,complete_observation_sha256='a'*64,latest_physical=physical)
        self.data={self.live_ref['path']:self.live,self.ended['review']['path']:self.review}
        self.data.update({self.ended['outputs'][k]['path']:v for k,v in self.records.items()})
        self.p=SimpleNamespace(lock_proof=Mock(),clear_window=Mock(),gone=Mock(),active_heavy=Mock())
        self.runner=SimpleNamespace(p=self.p,read_bundle=Mock(return_value=(self.spec,self.plan)),
            reference=lambda p:dict(path=str(p),sha256='a'*64),verify_owned_drain=Mock(return_value={'physically_drained':True}),
            admitted_bundles=Mock(return_value=(self.done,None)),verify_loaded_sources=Mock())
        self.patches=[patch.object(r.sources,'load',side_effect=lambda x:copy.deepcopy(self.data[x['path']])),
            patch.object(r.sources,'classify',return_value={}),patch.object(r.sources,'validate_native_receipt'),
            patch.object(r.sources,'check_native_identities')]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop()for p in reversed(self.patches)])
    def record(self,session=99999,chunk='abcdef',code=0):
        return r.record(self.runner,self.live_ref,session,chunk,code,self.recorder,self.lock)
    def test_actual_success_records_four_gone_and_pending_independent_audit(self):
        out=self.record();self.assertTrue(out['independent_post_exit_raw_reread_pending'])
        self.assertEqual(out['actual_execution_handle'],self.live['actual_execution_handle']);self.assertEqual(out['tool_chunk'],'abcdef')
        self.assertEqual([x.args[0]for x in self.p.gone.call_args_list],[self.live['owner'],self.ended['coordinator'],*self.ended['ranks']])
        self.assertFalse(out['full_model_qualified']);self.assertFalse(out['new_speedup_claim'])
    def test_wrong_session_rejects_before_sources(self):
        with self.assertRaises(RuntimeError):self.record(session=99998)
        self.runner.read_bundle.assert_not_called()
    def test_nonterminal_nonzero_bool_or_bad_chunk_rejected_before_live_read(self):
        for kwargs in ({'code':1},{'code':False},{'session':False},{'chunk':''},{'chunk':'not-a-tool-chunk'}):
            with self.subTest(kwargs=kwargs),self.assertRaises(RuntimeError):self.record(**kwargs)
        self.runner.read_bundle.assert_not_called()
    def test_live_external_owner_rejects(self):
        self.p.gone.side_effect=RuntimeError('live')
        with self.assertRaises(RuntimeError):self.record()
        self.runner.read_bundle.assert_not_called()
    def test_service_not_drained_rejects(self):
        self.runner.verify_owned_drain.return_value={'physically_drained':False}
        with self.assertRaises(RuntimeError):self.record()
    def test_child_failure_rejects(self):
        self.records['children']['exit_codes']=[0,1]
        with self.assertRaises(RuntimeError):self.record()
    def test_native_authentication_must_be_checked(self):
        with patch.object(r.sources,'validate_native_receipt',side_effect=RuntimeError('wrong source auth')):
            with self.assertRaises(RuntimeError):self.record()
    def test_changed_native_identity_rejects(self):
        with patch.object(r.sources,'check_native_identities',side_effect=RuntimeError('native changed')):
            with self.assertRaises(RuntimeError):self.record()
    def test_full_loaded_source_closure_checked(self):
        self.record();self.runner.verify_loaded_sources.assert_called_once()
    def test_no_native_import(self):self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
