"""Record root-observed actual tool exit after physical service-tree drain."""
import argparse,json,sys
from pathlib import Path
Q=Path(__file__).resolve().parent;R=Q.parent;sys.path.insert(0,str(R))
import minicpm5_original32_layer0_full_bf16_cpu_runner_v1 as t

def run(session,code,chunk):
    assert session==17987 and code==0 and len(chunk)==6 and all(c in '0123456789abcdef' for c in chunk)
    live_ref=t.reference(Q/'minicpm5-layer0-full-bf16-cpu-root-live-r1-20261108-v1.json');live=t.p.read_ref(live_ref)
    assert live['actual_execution_handle']==dict(kind='tool_session',session_id=session)
    spec=t.p.read_ref(live['spec']);root=Path(spec['output_root']);a=t.p.read(root/'admission.json');ar=t.reference(root/'admission.json')
    assert ar==live['admission'];assert a['outer_owner']==live['owner']
    for owner in [live['owner'],live['coordinator'],*live['ranks']]:t.p.gone(owner)
    props=t.validate_service(spec,ar,a['attempt_id'],t.p.properties(spec['unit']),live['coordinator'])
    assert props['Result']=='success' and props['ExecMainCode']=='1' and props['ExecMainStatus']=='0' and props['SubState']=='exited'
    group=t.p.empty_group(t.p.GROUP_PREFIX+spec['unit']+'.service')
    outputs={name:t.reference(root/file) for name,file in dict(outer_result='outer-result.json',coordinator_result='coordinator-result.json',rank0_result='rank0-result.json',rank1_result='rank1-result.json',children_exits='children-exits.json',service_exit='service-exit.json').items()}
    outer=t.p.read_ref(outputs['outer_result']);coordinator=t.p.read_ref(outputs['coordinator_result']);children=t.p.read_ref(outputs['children_exits']);service=t.p.read_ref(outputs['service_exit'])
    assert outer['complete'] and outer['owner']==live['owner'] and outer['coordinator_result']==outputs['coordinator_result'] and outer['service_exit']==outputs['service_exit']
    assert coordinator['complete'] and coordinator['rank_exits']==[0,0] and coordinator['rank_owners']==live['ranks']
    assert children['complete'] and children['successful'] and children['exit_codes']==[0,0] and children['owners']==live['ranks']
    assert service['successful'] and service['owner']==live['coordinator'];assert service['properties']['InvocationID']==props['InvocationID']
    assert {n:t.p.sha(R/n) for n in spec['canonical_sha256']}==spec['canonical_sha256']
    value=dict(schema='minicpm5_full_bf16_trajectory_actual_exit_v1',reviewer='/root',actual_exit_observed=True,actual_exit_code=code,actual_tool_exit=dict(session_id=session,exit_code=code,tool_chunk=chunk),actual_execution_handle=live['actual_execution_handle'],root_live=live_ref,
        owner=live['owner'],coordinator=live['coordinator'],ranks=live['ranks'],spec=live['spec'],review=live['review'],source_sha256=spec['source_sha256'],lock_identity=live['lock_identity'],outputs=outputs,actual_properties=props,actual_group=group,all_four_processes_gone=True,cgroup_empty=True,canonical_unchanged=True,full_model_qualified=False,new_speedup_claim=False)
    path=Q/'minicpm5-layer0-full-bf16-cpu-root-exit-r1-20261108-v1.json';t.p.once(path,value);print(json.dumps(t.reference(path)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--session',type=int,required=True);p.add_argument('--exit-code',type=int,required=True);p.add_argument('--chunk',required=True);a=p.parse_args();run(a.session,a.exit_code,a.chunk)
