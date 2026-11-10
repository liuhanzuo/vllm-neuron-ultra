"""Synthetic actual-exit chain tests. No real process or payload access."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_after_saved_audit_v1 as h
from research.check_minicpm5_layer1_saved_audit_runner_v1 import fixture as collector_fixture


def fixture():
    ce,cl,cs,cr,cp,records,crefs=copy.deepcopy(collector_fixture())
    ref=lambda name:dict(path=str(Q/name),sha256='a'*64)
    ce_ref=ref('synthetic-collector-exit.json');cl_ref=ref('synthetic-collector-live.json');ce['live']=cl_ref
    cs['canonical_sha256']={}
    refs=dict(spec=ref('synthetic-audit-spec.json'),review=ref('synthetic-audit-review.json'),
        terminal=ref('minicpm5-layer1-saved-audit-synthetic.json'))
    spec=dict(schema='minicpm5_layer1_saved_audit_execution_spec_v1',runner=h.audit.SELF,
        limits=h.audit.LIMITS,output=refs['terminal']['path'],collector_exit=ce_ref,source_sha256={},canonical_sha256={})
    review=dict(schema='minicpm5_layer1_saved_audit_runtime_review_v1',reviewer='/root',passed=True,
        blockers=[],spec=refs['spec'],source_sha256={})
    argv=[h.AUDIT_PYTHON,'-B',h.audit.SELF]
    for key in ('spec','review'):argv+=['--'+key,refs[key]['path'],'--'+key+'-sha256',refs[key]['sha256']]
    owner=dict(pid=104,start_ticks='1004',ppid=99,argv=argv,cgroup='0::/synthetic')
    numeric=dict(schema='minicpm5_layer1_full_logits_numeric_audit_v1',complete=True,
        logits_payload_count=84,official_reference_frames=32,candidate_metric_frames=96,
        original_selected_tap_bytes=3145728,replicated_ranks_verified_and_counted_once=True,
        metrics_independently_recomputed=True,exact_classification_and_argmax=True)
    rows=[dict(rank=i//42,arm=h.stream.raw.ARMS[(i%42)//14],call=i%14,
        complete_cache_bytes=2*h.stream.raw.CACHE_BYTES)for i in range(84)]
    cache=dict(schema='minicpm5_layer1_independent_byte_verification_v1',complete=True,ranks=2,arms=3,
        calls=84,frames_per_arm=32,full_cache_records=168,complete_cache_bytes=168*h.stream.raw.CACHE_BYTES,
        every_tap_and_full_cache_hash_exact=True,all_own_history_links_exact=True,raw_records=rows)
    result=dict(schema='minicpm5_layer1_complete_saved_data_audit_v1',complete=True,numeric=numeric,
        layer1_cache=cache,attention_causality_proven=False,full_model_qualified=False,new_speedup_claim=False)
    terminal=dict(schema='minicpm5_layer1_saved_audit_terminal_v1',complete=True,spec=refs['spec'],
        review=refs['review'],collector_exit=ce_ref,owner=owner,lock_identity=ce['lock_identity'],
        self_exit_claimed=False,requires_separate_runtime_admission=False,full_model_qualified=False,
        new_speedup_claim=False,result=result)
    live=dict(schema='minicpm5_layer1_saved_audit_actual_live_v1',reviewer='/root',actual_identity_verified=True,
        owner=copy.deepcopy(owner),spec=refs['spec'],review=refs['review'],actual_execution_handle=dict(kind='tool_session',session_id=2))
    live_ref=ref('synthetic-audit-live.json');ended_ref=ref('synthetic-audit-exit.json')
    ended=dict(schema='minicpm5_layer1_saved_audit_actual_exit_v1',reviewer='/root',actual_exit_observed=True,
        actual_exit_code=0,actual_execution_handle=live['actual_execution_handle'],tool_chunk='abcdef',
        owner=copy.deepcopy(owner),live=live_ref,lock_identity=ce['lock_identity'],actual_owner_gone=True,**refs)
    binding=dict(schema='minicpm5_layer1_completed_collection_and_audit_v1',collector_exit=ce_ref,saved_audit_exit=ended_ref)
    documents={ended_ref['path']:ended,live_ref['path']:live,refs['spec']['path']:spec,
        refs['review']['path']:review,refs['terminal']['path']:terminal,ce_ref['path']:ce,cl_ref['path']:cl,
        ce['spec']['path']:cs,ce['review']['path']:cr,cs['plan']['path']:cp}
    for key,value in records.items():documents[crefs[key]['path']]=value
    return binding,documents,(ended,live,spec,review,terminal,refs)


class AuditExitTests(unittest.TestCase):
    def reject(self,mutate):
        binding,docs,values=fixture();mutate(*values)
        with self.assertRaises((ValueError,RuntimeError,AssertionError)):
            h.validate_audit_exit(*values)
    def test_complete_actual_chain_is_metadata_only_and_retains_physical_pending(self):
        binding,docs,_=fixture()
        with patch.object(h.p,'gone')as gone:
            result=h.completed(binding,lambda ref:docs[ref['path']])
        self.assertTrue(result['metadata_complete']);self.assertTrue(result['physical_drain_still_required'])
        self.assertEqual(len(result['collector_chain']['owners']),4);gone.assert_not_called()
    def test_nonzero_missing_and_boolean_exit_rejected(self):
        for key,value in (('actual_exit_code',1),('actual_exit_code',False),('actual_exit_observed',False),('actual_owner_gone',False)):
            with self.subTest(key=key,value=value):self.reject(lambda e,*_:e.update({key:value}))
    def test_live_identity_and_tool_session_required(self):
        self.reject(lambda e,l,*_:l.update(actual_identity_verified=False))
        self.reject(lambda e,l,*_:l.update(actual_execution_handle=dict(kind='tool_session',session_id=99)))
        self.reject(lambda e,*_:e.update(tool_chunk='not-a-tool-chunk'))
    def test_live_and_exit_owner_cannot_be_substituted(self):
        self.reject(lambda e,l,*_:l['owner'].update(start_ticks='9999'))
        self.reject(lambda e,l,s,r,t,*_:t['owner'].update(pid=999))
    def test_exact_original_audit_argv_required(self):
        def change(e,l,s,r,t,refs):
            for value in (e,l,t):value['owner']['argv']=[h.AUDIT_PYTHON,'-B','fake.py']
        self.reject(change)
    def test_complete_actual_source_review_required(self):
        self.reject(lambda e,l,s,r,*_:r.update(passed=False))
        self.reject(lambda e,l,s,r,*_:r.update(blockers=['missing']))
        self.reject(lambda e,l,s,r,*_:r.update(source_sha256={'fake':'a'*64}))
    def test_original_limits_and_terminal_path_required(self):
        self.reject(lambda e,l,s,*_:s.update(limits={}))
        self.reject(lambda e,l,s,*_:s.update(output='/tmp/other.json'))
    def test_self_claimed_exit_or_model_qualification_rejected(self):
        self.reject(lambda e,l,s,r,t,*_:t.update(self_exit_claimed=True))
        self.reject(lambda e,l,s,r,t,*_:t.update(full_model_qualified=True))
    def test_exact_collector_and_original_lock_binding_required(self):
        self.reject(lambda e,l,s,r,t,*_:t.update(collector_exit={'path':'/tmp/fake','sha256':'b'*64}))
        self.reject(lambda e,l,s,r,t,*_:t.update(lock_identity=dict(device=99,inode=99)))
    def test_incomplete_numeric_or_full_cache_audit_rejected(self):
        self.reject(lambda e,l,s,r,t,*_:t['result']['numeric'].update(official_reference_frames=31))
        self.reject(lambda e,l,s,r,t,*_:t['result']['layer1_cache'].update(full_cache_records=167))
    def test_collector_missing_actual_exit_propagates(self):
        binding,docs,_=fixture();docs[binding['collector_exit']['path']]['actual_exit_code']=1
        with self.assertRaises((ValueError,RuntimeError)):
            h.completed(binding,lambda ref:docs[ref['path']])
    def test_different_collector_cannot_supply_audit(self):
        binding,docs,_=fixture();binding['collector_exit']=dict(path='/tmp/other',sha256='a'*64)
        with self.assertRaisesRegex(ValueError,'exact actual collected'):
            h.completed(binding,lambda ref:docs[ref['path']])
    def test_all_collector_sources_retained_by_audit(self):
        binding,docs,_=fixture();ce=docs[binding['collector_exit']['path']]
        docs[ce['spec']['path']]['source_sha256']['fake']='a'*64
        docs[ce['review']['path']]['source_sha256']['fake']='a'*64
        docs[docs[ce['spec']['path']]['plan']['path']]['source_sha256']['fake']='a'*64
        with self.assertRaisesRegex(ValueError,'every collector source'):
            h.completed(binding,lambda ref:docs[ref['path']])
    def test_physical_stage_checks_all_five_owners_and_actual_service(self):
        binding,docs,_=fixture();bundle=h.completed(binding,lambda ref:docs[ref['path']]);calls=[]
        with patch.object(h.p,'gone',side_effect=lambda owner:calls.append(owner)),            patch.object(h.collector,'verify_owned_drain',return_value=dict(physically_drained=True))as drain:
            self.assertTrue(h.physical(bundle,lambda:None)['physically_drained'])
        self.assertEqual(len(calls),5);drain.assert_called_once()
    def test_physical_guard_denial_precedes_process_access(self):
        def deny():raise RuntimeError('busy')
        with patch.object(h.p,'gone')as gone,self.assertRaisesRegex(RuntimeError,'busy'):h.physical({},deny)
        gone.assert_not_called()
    def test_no_native_imports(self):
        self.assertFalse({'torch','numpy','torch_neuronx','transformers','vllm'}&set(sys.modules))


if __name__=='__main__':unittest.main(verbosity=2)
