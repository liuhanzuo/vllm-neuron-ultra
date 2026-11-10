"""Synthetic future exit/audit boundary; no actual r12 enrollment or payload IO."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research import minicpm5_layer1_sdk_batch_layout_post_exit_core_v1 as c


def ref(name):return dict(path='/synthetic/'+name+'.json',sha256='a'*64)
def owner(pid,argv,group='0::/synthetic',**kw):return dict(pid=pid,start_ticks=str(pid+100),ppid=1,argv=argv,cgroup=group,**kw)


def fixture():
    binding=dict(schema='minicpm5_layer1_sdk_batch_layout_latest_completion_v1',status='complete',
        current_history=copy.deepcopy(c.latest.CURRENT),terminal_handoff=ref('terminal'),nemotron_r12=dict(
        schema='minicpm5_after_nemotron_r12_binding_v1',actual_exit=ref('nemotron-exit'),failed_release_exit=None))
    root='/synthetic/result';refs={k:dict(path=root+'/'+v,sha256='a'*64)for k,v in c.NAMES.items()}
    sref,rref,pref=ref('spec'),ref('review'),ref('plan');lock=dict(device=1,inode=2)
    spec=dict(schema=c.SCHEMA+'_cpu_execution_spec_v2',runner=c.RUNNER,worker=c.WORKER,source_sha256={},latest_completion=binding,
        source_contract=ref('contract'),output_root=root,plan=pref,unit='trn1-mini-layer1-sdk-batch-layout-synthetic')
    plan=dict(source_sha256={},latest_completion=binding,source_contract=spec['source_contract'],output_root=root)
    review=dict(schema=c.SCHEMA+'_runtime_source_review_v2',reviewer='/root',passed=True,blockers=[],spec=sref,source_sha256={})
    outer=owner(1,[c.PYTHON,'-B',c.RUNNER,'--mode','outer','--spec',sref['path'],'--spec-sha256',sref['sha256'],
        '--review',rref['path'],'--review-sha256',rref['sha256']])
    common=['--admission',refs['admission']['path'],'--admission-sha256',refs['admission']['sha256']]
    group='0::'+c.GROUP+spec['unit']+'.service'
    co=owner(2,[c.PYTHON,'-B',c.RUNNER,'--mode','coordinator',*common],group,unit=spec['unit'],invocation_id='b'*32)
    ranks=[owner(i,[c.PYTHON,'-B',c.RUNNER,'--mode','rank',*common],group)for i in (3,4)]
    for r in ranks:r['ppid']=2
    rr=[dict(process=r,rank=i,complete=True,admission=refs['admission'],self_exit_claimed=False,latest_completion=binding)for i,r in enumerate(ranks)]
    a=dict(schema=c.SCHEMA+'_cpu_admission_v2',outer_owner=outer,lock_identity=lock,spec=sref,review=rref,plan=pref,
        runtime_admitted=True,latest_completion=binding,source_contract=spec['source_contract'],source_authentication=refs['source_authentication'],observed_drains=refs['drains'],
        predecessors_checked=True,full_model_qualified=False,new_speedup_claim=False)
    command=['systemd-run','/usr/bin/env','-i','SYNTHETIC=1',*co['argv']]
    props=dict(MainPID='0',ControlPID='0',Job='',ActiveState='active',SubState='exited',InvocationID='b'*32,
        ExecMainPID='2',Result='success',ExecMainCode='1',ExecMainStatus='0',ExecStart='{ argv[]='+' '.join(command[1:])+' ; }')
    proof=dict(schema='minicpm5_layer1_sdk_batch_layout_pair_verification_v1',complete=True,ranks=2,anchor_decode_calls=12,
        physical_attention_invocations=144,full_B4_invocations=48,B1_invocations=96,assembled_four_row_outputs=72,
        unique_replicated_four_row_outputs=36,unique_decode_frames=24,raw_output_bytes=1179648,
        every_output_independently_reread=True,all_previous_controls_full_bytes_exact=True,every_view_formula_reconstructed=True,
        all_other_arms_verified_not_reexecuted=True,other_arms_reexecuted=False,rope_payload_independently_reread=False,
        full_model_qualified=False,new_speedup_claim=False)
    records=dict(admission=a,intent=dict(admission=refs['admission'],command=command),drains={},source_authentication={},
        coordinator_owner=co,rank0_owner=ranks[0],rank1_owner=ranks[1],rank0_result=rr[0],rank1_result=rr[1],
        children=dict(complete=True,successful=True,owners=ranks,exit_codes=[0,0]),
        coordinator=dict(complete=True,rank_exits=[0,0],rank_owners=ranks,results=rr,self_exit_claimed=False,validated=proof),
        service_exit=dict(successful=True,owner=co,observed_by_outer=outer,properties=props),
        outer=dict(complete=True,owner=outer,self_exit_claimed=False,service_exit=refs['service_exit'],coordinator_result=refs['coordinator'],full_model_qualified=False,new_speedup_claim=False),
        cleanup=dict(physically_drained=True,original_failure=None,owner=outer,owners_checked=[co,*ranks]))
    ended=dict(schema=c.SCHEMA+'_actual_exit_v1',reviewer='/root',actual_exit_observed=True,all_four_processes_gone=True,cgroup_empty=True,
        actual_execution_handle=dict(kind='tool_session',session_id=99999),tool_chunk='abcdef',actual_exit_code=0,
        owner=outer,coordinator=co,ranks=ranks,spec=sref,review=rref,outputs=refs,lock_identity=lock,full_model_qualified=False,new_speedup_claim=False)
    live=dict(schema=c.SCHEMA+'_actual_live_v1',reviewer='/root',owner=outer,actual_identity_verified=True,
        actual_execution_handle=ended['actual_execution_handle'],spec=sref,review=rref)
    return ended,live,spec,plan,review,records


class PostExit(unittest.TestCase):
    def setUp(self):self.f=fixture()
    def test_full_metadata_does_not_claim_physical(self):
        r=c.validate(*self.f);self.assertEqual(len(r['owners']),4);self.assertFalse(r['physical_checks_performed'])
    def test_no_actual_tool_exit(self):
        self.f[0]['actual_exit_observed']=False
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_bool_exit_code_rejected(self):
        self.f[0]['actual_exit_code']=False
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_wrong_live_session(self):
        self.f[1]['actual_execution_handle']=dict(kind='tool_session',session_id=99998)
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_wrong_review_source(self):
        self.f[4]['source_sha256']={'changed':'b'*64}
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_pending_r12(self):
        self.f[2]['latest_completion']['status']='pending'
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_source_receipt_must_bind(self):
        self.f[5]['admission']['source_authentication']=ref('replacement')
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_source_contract_replacement_rejected(self):
        self.f[5]['admission']['source_contract']=ref('replacement')
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_all_outputs_required(self):
        self.f[0]['outputs'].pop('rank1_result')
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_actual_rank_exit_required(self):
        self.f[5]['children']['exit_codes']=[0,1]
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_model_must_be_drained(self):
        self.f[5]['cleanup']['physically_drained']=False
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_replacement_invocation_rejected(self):
        self.f[5]['service_exit']['properties']['InvocationID']='c'*32
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_replacement_argv_rejected(self):
        self.f[5]['service_exit']['properties']['ExecStart']='{ argv[]=replacement ; }'
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_qualification_promotion_rejected(self):
        self.f[0]['full_model_qualified']=True
        with self.assertRaises(RuntimeError):c.validate(*self.f)
    def test_recompute_all_outputs(self):
        records=self.f[5];proof=records['coordinator']['validated'];verify=Mock(return_value=copy.deepcopy(proof));guard=Mock()
        self.assertEqual(c.recompute(records,self.f[3],verify,guard),proof);self.assertEqual(guard.call_count,2)
        verify.assert_called_once_with([records['rank0_result'],records['rank1_result']],self.f[3],guard)
    def test_recompute_mismatch_rejected(self):
        proof=copy.deepcopy(self.f[5]['coordinator']['validated']);proof['raw_output_bytes']-=16384
        with self.assertRaises(RuntimeError):c.recompute(self.f[5],self.f[3],lambda *a:proof,lambda:None)
    def test_equal_but_incomplete_manifest_rejected(self):
        proof=self.f[5]['coordinator']['validated'];proof['physical_attention_invocations']=143
        with self.assertRaises(RuntimeError):c.recompute(self.f[5],self.f[3],lambda *a:proof,lambda:None)
    def test_unsaved_rope_claim_rejected(self):
        proof=self.f[5]['coordinator']['validated'];proof['rope_payload_independently_reread']=True
        with self.assertRaises(RuntimeError):c.recompute(self.f[5],self.f[3],lambda *a:proof,lambda:None)
    def test_guard_before_raw_verifier(self):
        verify=Mock();guard=Mock(side_effect=RuntimeError('lost lock'))
        with self.assertRaises(RuntimeError):c.recompute(self.f[5],self.f[3],verify,guard)
        verify.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
