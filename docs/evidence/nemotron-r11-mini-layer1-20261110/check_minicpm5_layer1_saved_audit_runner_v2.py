"""Synthetic terminal and denied runtime tests. Never opens a real tensor."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path[:0]=[str(R),str(Q)]
from research import minicpm5_layer1_saved_audit_runner_v2 as audit
c=audit.collector


def fixture():
    root=Path('/mnt/nvme/minicpm5-original32-layer0-full-bf16-cpu-layer1-synthetic')
    ref=lambda path:dict(path=str(path),sha256='a'*64)
    sr,rr,pr=map(ref,(Q/'synthetic-spec.json',Q/'synthetic-review.json',Q/'synthetic-plan.json'))
    refs={key:ref(root/name)for key,name in audit.FILES.items()}
    spec=dict(schema='minicpm5_layer1_observation_cpu_execution_spec_v1',runner=c.SELF,
        worker=str(R/'minicpm5_original32_layer1_observation_cpu_worker_v1.py'),
        output_root=str(root),resources=c.PROPS,output_cap_bytes=c.CAP,root_floor_bytes=28*2**30,
        output_floor_bytes=30*2**30,plan=pr,source_sha256={},model_qualification_claim=False,
        automatic_full_gates_launch=False,unit='trn1-mini-layer1-observation-synthetic',master_port=29617)
    review=dict(schema='minicpm5_layer1_observation_runtime_source_review_v1',reviewer='/root',
        passed=True,blockers=[],spec=sr,source_sha256={})
    plan=dict(source_sha256={},output_root=str(root))
    argv=[c.PYTHON,'-B',c.SELF,'--mode','outer','--spec',sr['path'],'--spec-sha256',sr['sha256'],
        '--review',rr['path'],'--review-sha256',rr['sha256']]
    owner=dict(pid=100,start_ticks='1000',ppid=99,argv=argv,cgroup='0::/synthetic')
    group='0::'+audit.p.GROUP_PREFIX+spec['unit']+'.service'
    co=dict(pid=101,start_ticks='1001',ppid=99,argv=c.argv('coordinator',refs['admission']),cgroup=group,
        unit=spec['unit'],invocation_id='a'*32)
    ranks=[dict(pid=102+i,start_ticks=str(1002+i),ppid=101,argv=c.argv('rank',refs['admission']),cgroup=group)for i in (0,1)]
    lock=dict(device=1,inode=2)
    a=dict(schema='minicpm5_layer1_observation_cpu_admission_v1',spec=sr,review=rr,plan=pr,
        outer_owner=owner,lock_identity=lock,runtime_admitted=True,predecessors_checked=True,
        observed_drains=refs['drains'],complete_history_services=1255,attempt_id='b'*32,
        full_model_qualified=False,new_speedup_claim=False)
    props=dict(c.PROPS,LoadState='loaded',InvocationID='a'*32,ExecStopPost='',ExecMainPID='101',
        Result='success',ExecMainCode='1',ExecMainStatus='0',MainPID='0',ControlPID='0',
        ActiveState='active',SubState='exited',Job='',RuntimeMaxUSec='2h',TimeoutStopUSec='15s')
    command=c.service_command(spec,refs['admission'],a['attempt_id'])
    props['ExecStart']='{ argv[]='+ ' '.join(command[command.index('/usr/bin/env'):])+' ; }'
    results=[dict(process=owner,rank=i,complete=True,admission=refs['admission'],self_exit_claimed=False)for i,owner in enumerate(ranks)]
    records=dict(admission=a,intent=dict(admission=refs['admission'],command=command),
        drains=dict(complete=True,services=1255,observations=[dict(samples=[{},{}])for _ in range(1255)]),
        coordinator_owner=co,rank0_owner=ranks[0],rank1_owner=ranks[1],rank0_result=results[0],rank1_result=results[1],
        children=dict(complete=True,successful=True,exit_codes=[0,0],owners=ranks),
        coordinator=dict(complete=True,rank_exits=[0,0],rank_owners=ranks,results=results,self_exit_claimed=False),
        service_exit=dict(owner=co,observed_by_outer=owner,successful=True,properties=props),
        outer=dict(complete=True,owner=owner,service_exit=refs['service_exit'],coordinator_result=refs['coordinator'],
            self_exit_claimed=False,raw_payload_reconstruction_pending=True,full_model_qualified=False,new_speedup_claim=False),
        cleanup=dict(physically_drained=True,owner=owner,original_failure=None,owners_checked=[co,*ranks]))
    live=dict(schema='minicpm5_layer1_collector_actual_live_v1',reviewer='/root',actual_identity_verified=True,
        owner=owner,spec=sr,review=rr,actual_execution_handle=dict(kind='tool_session',session_id=1))
    ended=dict(schema='minicpm5_layer1_collector_actual_exit_v1',reviewer='/root',actual_exit_observed=True,
        actual_exit_code=0,actual_execution_handle=live['actual_execution_handle'],tool_chunk='abcdef',
        spec=sr,review=rr,owner=owner,lock_identity=lock,outputs=refs,all_four_processes_gone=True,cgroup_empty=True)
    return ended,live,spec,review,plan,records,refs


class TerminalTests(unittest.TestCase):
    def test_actual_complete_record_chain_requires_no_tensor_or_process_io(self):
        with patch.object(audit.numeric,'audit_collection')as data,patch.object(audit.p,'gone')as gone:
            chain=audit.validate_chain(*fixture())
        self.assertEqual(len(chain['owners']),4);self.assertEqual(len(chain['rank_results']),2)
        data.assert_not_called();gone.assert_not_called()
    def reject(self,mutate):
        values=copy.deepcopy(fixture());mutate(*values)
        with self.assertRaises((ValueError,AssertionError,RuntimeError)):
            audit.validate_chain(*values)
    def test_failed_or_missing_actual_tool_exit_is_not_completion(self):
        for key,value in (('actual_exit_code',1),('actual_exit_observed',False),('actual_exit_code',False),('tool_chunk','')):
            with self.subTest(key=key,value=value):self.reject(lambda e,*_:e.update({key:value}))
    def test_replaced_tool_session_live_owner_or_argv_rejected(self):
        self.reject(lambda e,l,*_:l.update(actual_execution_handle=dict(kind='tool_session',session_id=2)))
        self.reject(lambda e,l,*_:l.update(owner=dict(l['owner'],start_ticks='999')))
        self.reject(lambda e,l,*_:e['owner'].update(argv=e['owner']['argv']+['--fake']))
    def test_unreviewed_or_changed_source_plan_rejected(self):
        self.reject(lambda e,l,s,r,*_:r.update(passed=False))
        self.reject(lambda e,l,s,r,p,*_:p.update(source_sha256={'x':'0'*64}))
    def test_original_resources_and_nonqualification_required(self):
        self.reject(lambda e,l,s,*_:s.update(output_cap_bytes=c.CAP*2))
        self.reject(lambda e,l,s,*_:s.update(model_qualification_claim=True))
    def test_every_original_record_and_output_path_required(self):
        self.reject(lambda e,l,s,r,p,records,refs:records.pop('children'))
        self.reject(lambda e,l,s,r,p,records,refs:refs['cleanup'].update(path='/tmp/fake.json'))
    def test_original_history_all1255_dual_samples_required(self):
        self.reject(lambda e,l,s,r,p,d,*_:d['drains']['observations'].pop())
        self.reject(lambda e,l,s,r,p,d,*_:d['drains']['observations'][0]['samples'].pop())
    def test_launch_intent_and_admission_must_match(self):
        self.reject(lambda e,l,s,r,p,d,*_:d['intent'].update(command=[]))
        self.reject(lambda e,l,s,r,p,d,*_:d['admission'].update(runtime_admitted=False))
    def test_service_actual_exit_must_succeed_with_exact_owner(self):
        self.reject(lambda e,l,s,r,p,d,*_:d['service_exit']['properties'].update(ExecMainStatus='1'))
        self.reject(lambda e,l,s,r,p,d,*_:d['service_exit']['properties'].update(InvocationID='b'*32))
    def test_rank_parent_cgroup_and_actual_child_exits_required(self):
        self.reject(lambda e,l,s,r,p,d,*_:d['rank0_owner'].update(ppid=999))
        self.reject(lambda e,l,s,r,p,d,*_:d['rank1_owner'].update(cgroup='0::/elsewhere'))
        self.reject(lambda e,l,s,r,p,d,*_:d['children'].update(exit_codes=[0,1]))
    def test_rank_replicas_are_distinct_actual_processes(self):
        self.reject(lambda e,l,s,r,p,d,*_:d['rank1_owner'].update(pid=d['rank0_owner']['pid'],start_ticks=d['rank0_owner']['start_ticks']))
    def test_self_certified_exit_or_early_raw_completion_rejected(self):
        self.reject(lambda e,l,s,r,p,d,*_:d['outer'].update(self_exit_claimed=True))
        self.reject(lambda e,l,s,r,p,d,*_:d['outer'].update(raw_payload_reconstruction_pending=False))
    def test_cleanup_error_live_owner_or_missing_drain_rejected(self):
        self.reject(lambda e,l,s,r,p,d,*_:d['cleanup'].update(original_failure={'error':'failed'}))
        self.reject(lambda e,l,s,r,p,d,*_:d['cleanup'].update(physically_drained=False))
        self.reject(lambda e,*_:e.update(all_four_processes_gone=False))
    def test_denied_serial_lock_prevents_metadata_or_raw_reads(self):
        with patch.object(audit.p,'user_bus'),patch.object(c.post.external,'acquire_lock',side_effect=RuntimeError('busy')),\
             patch.object(audit.p,'read_ref')as read,patch.object(audit.numeric,'audit_collection')as data:
            with self.assertRaisesRegex(RuntimeError,'busy'):audit.run({}, {})
        read.assert_not_called();data.assert_not_called()
    def test_unreviewed_audit_releases_its_own_lock_without_data_or_completion(self):
        spec=dict(schema='minicpm5_layer1_saved_audit_execution_spec_v1',runner=audit.SELF,limits=audit.LIMITS)
        review=dict(schema='minicpm5_layer1_saved_audit_runtime_review_v1',reviewer='/root',passed=False)
        with patch.object(audit.p,'user_bus'),patch.object(c.post.external,'acquire_lock',return_value=(123,{},{})),\
             patch.object(audit.p,'lock_proof'),patch.object(audit.p,'clear_window'),patch.object(audit.p,'active_heavy'),\
             patch.object(audit.p,'read_ref',side_effect=[spec,review]),patch.object(c.post.external,'release_lock')as release,\
             patch.object(audit.numeric,'audit_collection')as data,patch.object(audit.p,'once')as write:
            with self.assertRaisesRegex(ValueError,'reviewed'):audit.run({}, {})
        release.assert_called_once_with(123);data.assert_not_called();write.assert_not_called()
    def test_no_native_runtime_import_and_original_limits(self):
        self.assertFalse({'torch','numpy','torch_neuronx','neuronxcc','vllm','transformers'}&set(sys.modules))
        self.assertEqual(audit.LIMITS['address_space_bytes'],2*2**30)
        self.assertEqual(audit.LIMITS['file_bytes'],2*2**20)
        self.assertEqual(audit.LIMITS['allowed_cpus'],[6])

if __name__=='__main__':unittest.main(verbosity=2)
