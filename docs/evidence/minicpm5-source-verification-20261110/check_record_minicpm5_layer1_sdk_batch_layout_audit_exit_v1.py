"""Synthetic audit terminal verification; no actual model or terminal binding."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import record_minicpm5_layer1_sdk_batch_layout_audit_exit_v1 as r
from research.check_minicpm5_layer1_sdk_batch_layout_post_exit_core_v1 import fixture,ref,owner


class AuditExit(unittest.TestCase):
    def setUp(self):
        self.model,self.model_live,self.spec,self.plan,self.review,self.records=fixture()
        self.model['live']=ref('model-live');self.lock=self.model['lock_identity']
        self.model_ref=ref('model-exit');self.live_ref=ref('audit-live');self.result_ref=ref('audit-result')
        self.session=88888;self.auditor=owner(50,['python3','-B',str(r.AUDITOR),'--model-exit',self.model_ref['path'],
            '--model-exit-sha256',self.model_ref['sha256'],'--output',self.result_ref['path']])
        self.live=dict(schema='minicpm5_sdk_batch_layout_audit_live_v1',reviewer='/root',actual_identity_verified=True,
            actual_execution_handle=dict(kind='tool_session',session_id=self.session),owner=self.auditor,model_actual_exit=self.model_ref)
        self.units=[dict(unit='trn1-synthetic-%d'%i,samples=[dict(properties=dict(Id='trn1-synthetic-%d.service'%i,
            MainPID='0',ControlPID='0',ActiveState='inactive',SubState='dead'),group={'absent':True})]*2)for i in range(1308)]
        self.done=dict(binding=self.plan['latest_completion'],lock_identity=self.lock,owners=[owner(80,['old'])],
            nemotron=dict(history=dict(rows=[{'unit':u['unit']}for u in self.units])))
        self.physical=dict(complete=True,physical_checks_performed=True,all_current_and_prior_services_preserved=True,
            lock_identity=self.lock,all_additional_owners_checked=self.done['owners'],
            nemotron=dict(complete=True,fresh_physical_drain_verified=True,services=1308,observations=self.units))
        self.records['drains']=dict(complete=True,latest_completion=self.plan['latest_completion'],outer_owner=self.model['owner'],
            lock_identity=self.lock,complete_observation_sha256='a'*64,latest_physical=copy.deepcopy(self.physical))
        self.records['admission']['attempt_id']='d'*32
        proof=copy.deepcopy(self.records['coordinator']['validated']);proof['descriptive']={'synthetic':True}
        self.result=dict(schema='minicpm5_layer1_sdk_batch_layout_post_exit_audit_v1',reviewer='/root',complete=True,
            self_exit_claimed=False,owner=self.auditor,model_exit=self.model_ref,lock_identity=self.lock,
            source=dict(path=str(r.AUDITOR),sha256=hashlib.sha256(b'synthetic-source').hexdigest()),
            every_raw_byte_reread_after_all_model_processes_exited=True,all_previous_controls_full_bytes_exact=True,
            every_view_formula_reconstructed=True,rope_payload_independently_reread=False,other_arms_reexecuted=False,
            attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False,result=proof,
            descriptive=proof['descriptive'],actual_predecessor_drain=r.auditor.packed_physical(self.physical),
            source_authentication=self.model['outputs']['source_authentication'],inherited_native_authentication=dict(
                native_hash_authentication_inherited=True,native_bytes_read_here=0,historical_payload_bytes_read_here=0))
        self.data={self.live_ref['path']:self.live,self.result_ref['path']:self.result,self.model_ref['path']:self.model,
            self.model['live']['path']:self.model_live,self.model['review']['path']:self.review}
        self.data.update({self.model['outputs'][k]['path']:v for k,v in self.records.items()})
        self.p=SimpleNamespace(lock_proof=Mock(),clear_window=Mock(),gone=Mock(),active_heavy=Mock())
        self.runner=SimpleNamespace(p=self.p,read_bundle=Mock(return_value=(self.spec,self.plan)),
            verify_owned_drain=Mock(return_value={'physically_drained':True}),admitted_bundles=Mock(return_value=(self.done,None)),
            verify_loaded_sources=Mock())
        self.patches=[patch.object(r.sources,'load',side_effect=lambda ref:copy.deepcopy(self.data[ref['path']])),
            patch.object(r,'identity',return_value={'stable':True}),patch.object(r.sources,'text',return_value=b'synthetic-source'),
            patch.object(r.sources,'classify',return_value={}),patch.object(r.sources,'validate_native_receipt'),
            patch.object(r.sources,'check_native_identities')]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop()for p in reversed(self.patches)])
    def record(self,session=88888,chunk='abcdef',code=0):
        return r.record(self.runner,self.live_ref,self.result_ref,session,chunk,code,owner(60,['recorder']),self.lock)
    def repack(self):self.result['actual_predecessor_drain']=r.auditor.packed_physical(self.physical)
    def test_complete_separate_actual_exit_metadata_only(self):
        out=self.record();self.assertTrue(out['actual_tool_exit_observed']);self.assertEqual(out['decoded_historical_unit_count'],1308)
        self.assertEqual(out['tool_chunk'],'abcdef');self.assertFalse(out['raw_tensor_bytes_reread_here']);self.assertFalse(out['new_speedup_claim'])
        self.assertEqual([c.args[0]for c in self.p.gone.call_args_list],[self.auditor,self.model['owner'],self.model['coordinator'],*self.model['ranks']])
        self.runner.verify_loaded_sources.assert_called_once()
    def test_wrong_audit_session_rejects(self):
        with self.assertRaises(RuntimeError):self.record(session=88887)
    def test_bad_actual_exit_rejects_before_read(self):
        for kw in ({'code':1},{'code':False},{'session':False},{'chunk':''},{'chunk':'invented'}):
            with self.subTest(kw=kw),self.assertRaises(RuntimeError):self.record(**kw)
        r.sources.load.assert_not_called()
    def test_same_model_audit_session_rejected(self):
        self.model['actual_execution_handle']['session_id']=self.session
        with self.assertRaises(RuntimeError):self.record()
    def test_nonterminal_model_rejected(self):
        self.model['actual_exit_observed']=False
        with self.assertRaises(RuntimeError):self.record()
    def test_wrong_command_rejected(self):
        self.auditor['argv'][-1]='/different.json'
        with self.assertRaises(RuntimeError):self.record()
    def test_wrong_model_ref_rejected(self):
        self.result['model_exit']=ref('different')
        with self.assertRaises(RuntimeError):self.record()
    def test_self_claimed_exit_rejected(self):
        self.result['self_exit_claimed']=True
        with self.assertRaises(RuntimeError):self.record()
    def test_live_auditor_rejected_before_source_read(self):
        self.p.gone.side_effect=RuntimeError('live')
        with self.assertRaises(RuntimeError):self.record()
        r.sources.text.assert_not_called()
    def test_model_service_not_drained_rejected(self):
        self.runner.verify_owned_drain.return_value={'physically_drained':False}
        with self.assertRaises(RuntimeError):self.record()
    def test_incomplete_output_counts_rejected(self):
        self.result['result']['raw_output_bytes']-=1
        with self.assertRaises(RuntimeError):self.record()
    def test_incomplete_byte_guarantee_rejected(self):
        self.result['result']['every_output_independently_reread']=False
        with self.assertRaises(RuntimeError):self.record()
    def test_native_receipt_substitution_rejected(self):
        self.result['source_authentication']=ref('different')
        with self.assertRaises(RuntimeError):self.record()
    def test_native_identity_change_rejected(self):
        r.sources.check_native_identities.side_effect=RuntimeError('changed')
        with self.assertRaises(RuntimeError):self.record()
    def test_source_hash_change_rejected(self):
        r.sources.text.return_value=b'changed'
        with self.assertRaises(RuntimeError):self.record()
    def test_metadata_changed_rejected(self):
        r.identity.side_effect=[{'inode':1},{'inode':2}]
        with self.assertRaises(RuntimeError):self.record()
    def test_unit_substitution_same_count_rejected(self):
        self.physical['nemotron']['observations'][0]=copy.deepcopy(self.units[-1])
        self.physical['nemotron']['observations'][0]['unit']='trn1-other'
        for s in self.physical['nemotron']['observations'][0]['samples']:s['properties']['Id']='trn1-other.service'
        self.repack()
        with self.assertRaises(RuntimeError):self.record()
    def test_missing_second_sample_rejected(self):
        self.physical['nemotron']['observations'][0]['samples'].pop();self.repack()
        with self.assertRaises(RuntimeError):self.record()
    def test_prior_owner_omission_rejected(self):
        self.physical['all_additional_owners_checked']=[];self.repack()
        with self.assertRaises(RuntimeError):self.record()
    def test_live_historical_unit_rejected(self):
        self.units[0]['samples'][0]['properties']['MainPID']='77';self.repack()
        with self.assertRaises(RuntimeError):self.record()
    def test_full_model_promotion_rejected(self):
        self.result['full_model_qualified']=True
        with self.assertRaises(RuntimeError):self.record()
    def test_false_rope_claim_rejected(self):
        self.result['rope_payload_independently_reread']=True
        with self.assertRaises(RuntimeError):self.record()
    def test_lossless_unpack(self):self.assertEqual(r.unpack(self.result['actual_predecessor_drain']),self.physical)
    def test_bad_encoded_or_truncated_stream_rejected(self):
        for change in ('invalid','truncated','trailing','multistream'):
            p=copy.deepcopy(self.result['actual_predecessor_drain']);data=base64.b64decode(p['data'])
            if change=='invalid':p['data']='!'
            else:
                data={'truncated':data[:-1],'trailing':data+b'x','multistream':data+zlib.compress(b'{}')}[change]
                p.update(data=base64.b64encode(data).decode(),compressed_bytes=len(data))
            with self.subTest(change=change),self.assertRaises((RuntimeError,ValueError)):r.unpack(p)
    def test_bounded_decompression_and_digest(self):
        for key,value in [('uncompressed_bytes',1),('uncompressed_bytes',32*2**20+1),('compressed_bytes',2*2**20),('sha256','0'*64)]:
            p=copy.deepcopy(self.result['actual_predecessor_drain']);p[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(RuntimeError):r.unpack(p)
    def test_no_native_import(self):self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
