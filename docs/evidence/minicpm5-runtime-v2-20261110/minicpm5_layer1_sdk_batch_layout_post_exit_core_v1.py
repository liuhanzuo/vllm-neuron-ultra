"""Independent SDK batch/layout post-exit validation, pure metadata core.

Callers must supply authentic reads, separately observed tool exits and the
original serial guard. This module neither reads payloads nor claims an exit.
The full-output verifier is injected only after runtime physical drainage.
"""
from pathlib import Path
import re
from research import minicpm5_layer1_sdk_batch_layout_completion_v1 as latest
from research.nemotron9b_r12_current_history_completion_v1 import handle,same_owner
require=latest.require
ROOT=latest.ROOT
RUNNER=str(ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_runner_v2.py')
WORKER=str(ROOT/'minicpm5_layer1_sdk_batch_layout_cpu_worker_v2.py')
PYTHON='/home/ubuntu/venvs/vllm_neuron/bin/python'
SCHEMA='minicpm5_layer1_sdk_batch_layout'
GROUP='/user.slice/user-1000.slice/user@1000.service/app.slice/'
NAMES=dict(admission='admission.json',intent='launch-intent.json',drains='predecessor-drain.json',
    coordinator_owner='coordinator-owner.json',rank0_owner='rank0-owner.json',rank1_owner='rank1-owner.json',
    rank0_result='rank0-result.json',rank1_result='rank1-result.json',children='children-exits.json',
    coordinator='coordinator-result.json',service_exit='service-exit.json',outer='outer-result.json',cleanup='cleanup-drained.json',
    source_authentication='source-authentication.json')


def terminal(props):
    return props['MainPID']==props['ControlPID']=='0' and not props['Job'] and (
        props['ActiveState']in ('inactive','failed')or(props['ActiveState'],props['SubState'])==('active','exited'))


def validate(ended,live,spec,plan,review,records):
    """Validate future actual successful exit metadata, never admission alone."""
    require(ended['schema']==SCHEMA+'_actual_exit_v1' and ended['reviewer']=='/root'
        and ended['actual_exit_observed']is ended['all_four_processes_gone']is ended['cgroup_empty']is True,
        'Actual successful external model terminal and recorded drain')
    tool=handle(ended);same_owner(ended['owner'],live['owner'])
    require(live['schema']==SCHEMA+'_actual_live_v1' and live['reviewer']=='/root'
        and live['actual_identity_verified']is True and live['actual_execution_handle']==tool
        and live['spec']==ended['spec'] and live['review']==ended['review'],'Original live-to-actual-exit chain')
    require(spec['schema']==SCHEMA+'_cpu_execution_spec_v2' and spec['runner']==RUNNER and spec['worker']==WORKER,
        'Exact successor runner/worker and specification')
    require(review['schema']==SCHEMA+'_runtime_source_review_v2' and review['reviewer']=='/root'
        and review['passed']is True and review['blockers']==[] and review['spec']==ended['spec']
        and review['source_sha256']==spec['source_sha256']==plan['source_sha256'],
        'Original independently reviewed complete scientific source closure')
    latest.require_complete_binding(spec['latest_completion'])
    require(plan['latest_completion']==spec['latest_completion'] and plan['source_contract']==spec['source_contract']
        and plan['output_root']==spec['output_root'],'Actual source partition, completed predecessors and output tree')
    require(ended['owner']['argv']==[PYTHON,'-B',RUNNER,'--mode','outer','--spec',ended['spec']['path'],
        '--spec-sha256',ended['spec']['sha256'],'--review',ended['review']['path'],'--review-sha256',ended['review']['sha256']],
        'Exact externally observed launcher argv')
    refs=ended['outputs'];root=Path(spec['output_root'])
    require(set(refs)==set(NAMES)==set(records),'Every original output and source-authentication receipt retained')
    for name,file in NAMES.items():
        latest.reference(refs[name]);require(refs[name]['path']==str(root/file),'Exact owned receipt path')
    a=records['admission'];co=records['coordinator_owner'];ranks=[records['rank0_owner'],records['rank1_owner']]
    same_owner(ended['owner'],a['outer_owner']);same_owner(ended['owner'],records['outer']['owner'])
    same_owner(ended['coordinator'],co);require(ended['ranks']==ranks,'Both actual rank process identities')
    require(a['schema']==SCHEMA+'_cpu_admission_v2' and a['spec']==ended['spec'] and a['review']==ended['review']
        and a['plan']==spec['plan'] and a['lock_identity']==ended['lock_identity'] and a['runtime_admitted']is True
        and a['latest_completion']==spec['latest_completion'] and a['source_contract']==spec['source_contract']
        and a['source_authentication']==refs['source_authentication']
        and a['observed_drains']==refs['drains'] and a['predecessors_checked']is True,
        'Actual admission binds exact model, source authentication and original lock')
    require(co['unit']==spec['unit'] and re.fullmatch('[a-f0-9]{32}',co['invocation_id'])
        and co['invocation_id']!='0'*32 and co['cgroup']=='0::'+GROUP+spec['unit']+'.service',
        'Exact original service and invocation')
    common=['--admission',refs['admission']['path'],'--admission-sha256',refs['admission']['sha256']]
    require(co['argv']==[PYTHON,'-B',RUNNER,'--mode','coordinator',*common],'Exact coordinator command')
    for index,rank in enumerate(ranks):
        same_owner(rank,rank)
        require(rank['argv']==[PYTHON,'-B',RUNNER,'--mode','rank',*common]
            and rank['ppid']==co['pid'] and rank['cgroup']==co['cgroup'],'Direct actual TP2 rank in owned service')
        rr=records['rank%d_result'%index]
        require(rr['process']==rank and rr['rank']==index and rr['complete']is True
            and rr['admission']==refs['admission'] and rr['self_exit_claimed']is False
            and rr['latest_completion']==a['latest_completion'],'Every complete rank binds actual owner and admission')
    owners=[ended['owner'],co,*ranks]
    require(len({(o['pid'],o['start_ticks'])for o in owners})==4,'Four distinct actual model processes')
    child=records['children'];coord=records['coordinator'];outer=records['outer'];cleanup=records['cleanup']
    require(child['complete']is child['successful']is True and child['owners']==ranks and child['exit_codes']==[0,0]
        and coord['complete']is True and coord['rank_exits']==[0,0] and coord['rank_owners']==ranks
        and coord['results']==[records['rank0_result'],records['rank1_result']] and coord['self_exit_claimed']is False,
        'Both ranks actually exited successfully and coordinator retained complete outputs')
    require(outer['complete']is True and outer['self_exit_claimed']is False
        and outer['service_exit']==refs['service_exit'] and outer['coordinator_result']==refs['coordinator']
        and cleanup['physically_drained']is True and cleanup['original_failure']is None
        and cleanup['owner']==ended['owner'] and cleanup['owners_checked']==[co,*ranks],
        'Outer completed and retained actual cleanup of all model processes')
    service=records['service_exit'];props=service['properties'];intent=records['intent'];command=intent['command']
    require(service['successful']is True and service['owner']==co and service['observed_by_outer']==ended['owner']
        and props['InvocationID']==co['invocation_id'] and props['ExecMainPID']==str(co['pid'])
        and props['Result']=='success' and props['ExecMainCode']=='1' and props['ExecMainStatus']=='0' and terminal(props),
        'Retained successful original service exit')
    require(intent['admission']==refs['admission'] and '/usr/bin/env'in command and command[-9:]==co['argv']
        and re.findall(r'argv\[\]=(.*?)\s*;',props['ExecStart'])==[' '.join(command[command.index('/usr/bin/env'):])],
        'Retained launched command, not a replacement service')
    for row in (ended,outer,a):
        require(row['full_model_qualified']is row['new_speedup_claim']is False,'Local diagnostic never promotes full model or speedup')
    return dict(owners=owners,lock_identity=ended['lock_identity'],group=GROUP+spec['unit']+'.service',
        metadata_complete=True,physical_checks_performed=False,full_model_qualified=False,new_speedup_claim=False)


def recompute(records,plan,validate_pair,guard):
    """Only called under the real audit lock after all four model owners exit."""
    require(callable(validate_pair)and callable(guard),'Full raw verifier and real physical guard')
    guard();proof=validate_pair([records['rank0_result'],records['rank1_result']],plan,guard);guard()
    require(proof==records['coordinator']['validated'],'Independent full reread reproduces complete saved result')
    require(proof['schema']=='minicpm5_layer1_sdk_batch_layout_pair_verification_v1' and proof['complete']is True
        and [proof[k]for k in ('ranks','anchor_decode_calls','physical_attention_invocations','full_B4_invocations',
            'B1_invocations','assembled_four_row_outputs','unique_replicated_four_row_outputs','unique_decode_frames','raw_output_bytes')]
            ==[2,12,144,48,96,72,36,24,1179648],'Complete planned two-rank144-call/72-output/24-frame diagnostic')
    require(proof['every_output_independently_reread']is proof['all_previous_controls_full_bytes_exact']
        is proof['every_view_formula_reconstructed']is proof['all_other_arms_verified_not_reexecuted']is True,
        'Every full output, original control and view verified')
    require(proof['other_arms_reexecuted']is proof['rope_payload_independently_reread']
        is proof['full_model_qualified']is proof['new_speedup_claim']is False,
        'No invented other-arm execution, saved RoPE bytes or model qualification')
    return proof
