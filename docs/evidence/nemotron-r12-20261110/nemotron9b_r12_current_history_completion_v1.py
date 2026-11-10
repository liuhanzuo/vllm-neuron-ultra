"""Authenticate the completed current/history MiniCPM5 model and auditor.

No I/O on import. The maintenance caller owns the shared serial lock. Read-only
metadata reads and physical process/group checks replace trusting handoff prose.
"""
from pathlib import Path
import re

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
ORIGINAL_HANDOFF=dict(path=str(ROOT/'research/continuation-handoff-20261110-v416-mini-native-cache-complete.json'),
    sha256='5d5ca748f83717c6bf5114c3398da72e2ce714227323b9ae2299ad98342c22fd')


def require(value,message):
    if not value:raise RuntimeError(message)


def handle(receipt):
    value=receipt['actual_execution_handle']
    require(type(value)is dict and set(value)=={'kind','session_id'} and value['kind']=='tool_session'
        and type(value['session_id'])is int and value['session_id']>0,'Actual tool session identity')
    require(type(receipt['tool_chunk'])is str and re.fullmatch('[a-f0-9]{6,64}',receipt['tool_chunk']),
        'Actual terminal tool chunk')
    require(type(receipt['actual_exit_code'])is int and receipt['actual_exit_code']==0,'Actual successful tool exit')
    return value


def same_owner(a,b):
    require(a==b,'Exact authenticated process owner')
    require(type(a['pid'])is int and a['pid']>0 and str(a['start_ticks']).isdigit()
        and int(a['start_ticks'])>0 and a['argv'] and a['cgroup'].startswith('0::/'),
        'Recorded process identity includes start time and group')


def model(binding,kind,lock,load,physical=None):
    require(kind in ('native_cache','official_history','current_history_factorial'),'Known exact model type')
    native=kind=='native_cache'
    prefix_group='/user.slice/user-1000.slice/user@1000.service/app.slice/'
    terminal=lambda x:x['MainPID']==x['ControlPID']=='0' and not x['Job'] and (x['ActiveState']in ('inactive','failed')or(x['ActiveState'],x['SubState'])==('active','exited'))
    prefix='minicpm5_layer1_'+kind
    ended=load(binding['actual_exit'])
    require(ended['schema']==prefix+'_actual_exit_v1' and ended['reviewer']=='/root'
        and ended['actual_exit_observed']is True and ended['cgroup_empty']is True,
        'Actual named model terminal receipt')
    require(ended['all_three_processes_gone' if native else 'all_four_processes_gone']is True
        and ended['lock_identity']==lock,'Same serial lock and recorded model drains')
    session=handle(ended);live=load(ended['live'])
    require(live['schema']==prefix+'_actual_live_v1' and live['actual_identity_verified']is True
        and live['reviewer']=='/root' and live['actual_execution_handle']==session
        and live['spec']==ended['spec'] and live['review']==ended['review'], 'Live-to-terminal model binding')
    same_owner(ended['owner'],live['owner'])
    spec=load(ended['spec']);review=load(ended['review'])
    require(review['passed']is True and review['blockers']==[] and review['spec']==ended['spec']
        and review['source_sha256']==spec['source_sha256'], 'Exact successful source review')
    expected_runner=ROOT/('minicpm5_layer1_native_cache_runner_v1.py' if native
        else 'minicpm5_layer1_'+kind+'_cpu_runner_v1.py')
    require(spec['runner']==str(expected_runner),'Exact original runner, not another experiment')
    expected_python='/home/ubuntu/venvs/model_reference_2026/bin/python' if native else '/home/ubuntu/venvs/vllm_neuron/bin/python'
    argv=ended['owner']['argv']
    require(len(argv)==13 and argv[:2]==[expected_python,'-B']
        and str((ROOT/argv[2]).resolve())==str(expected_runner)
        and argv[3:]==['--mode','outer','--spec',ended['spec']['path'],'--spec-sha256',ended['spec']['sha256'],
            '--review',ended['review']['path'],'--review-sha256',ended['review']['sha256']], 'Exact actual launcher argv')
    root=Path(spec['output_root']);refs=ended['outputs']
    names=dict(admission='admission.json',intent='launch-intent.json',outer='outer-result.json',
        coordinator_owner='coordinator-owner.json',coordinator='coordinator-result.json',children='children-exits.json',
        service_exit='service-exit.json',cleanup='cleanup-drained.json')
    names.update(dict(native_owner='native-owner.json') if native else
        dict(rank0_owner='rank0-owner.json',rank1_owner='rank1-owner.json'))
    records={}
    for key,name in names.items():
        require(refs[key]['path']==str(root/name),'Original owned output receipt path')
        records[key]=load(refs[key])
    admission=records['admission'];co=records['coordinator_owner']
    same_owner(co,ended['coordinator'])
    same_owner(admission['outer_owner'],ended['owner']);same_owner(records['outer']['owner'],ended['owner'])
    require(admission['spec']==ended['spec'] and admission['review']==ended['review']
        and admission['plan']==spec['plan'] and admission['lock_identity']==lock
        and admission['runtime_admitted']is True,'Actual admission same spec and serial lock')
    children=[records['native_owner']] if native else [records['rank0_owner'],records['rank1_owner']]
    wanted=[ended['native']] if native else ended['ranks']
    require(children==wanted and len(children)==(1 if native else 2),'Every actual native/rank owner')
    owners=[ended['owner'],co,*children]
    group=prefix_group+spec['unit']+'.service'
    require(co['unit']==spec['unit'] and re.fullmatch('[a-f0-9]{32}',co['invocation_id']) and co['invocation_id']!='0'*32
        and co['cgroup']=='0::'+group,'Owned service coordinator identity')
    for child in children:
        same_owner(child,child)
        require(child['ppid']==co['pid'] and child['cgroup']==co['cgroup'],'Direct original child in owned service')
    require(len({(o['pid'],o['start_ticks'])for o in owners})==len(owners),'Distinct actual model owners')
    zeros=[0]*len(children);code_key='native_exits' if native else 'rank_exits'
    require(records['children']['complete']is True and records['children']['successful']is True
        and records['children']['owners']==children and records['children']['exit_codes']==zeros
        and records['coordinator']['complete']is True and records['coordinator'][code_key]==zeros,
        'Actual child and coordinator successful exits')
    require(records['outer']['complete']is True and records['outer']['self_exit_claimed']is False
        and records['outer']['service_exit']==refs['service_exit']
        and records['outer']['coordinator_result']==refs['coordinator'],'Original outer result binds child results')
    service=records['service_exit'];cleanup=records['cleanup'];props=service['properties']
    require(service['successful']is True and service['owner']==co and service['observed_by_outer']==ended['owner']
        and props['InvocationID']==co['invocation_id'] and props['ExecMainPID']==str(co['pid'])
        and props['Result']=='success' and props['ExecMainCode']=='1' and props['ExecMainStatus']=='0'
        and terminal(props),'Retained original successful service exit')
    require(cleanup['physically_drained']is True and cleanup['original_failure']is None
        and cleanup['owner']==ended['owner'],'Actual cleanup completed without failure')
    command=records['intent']['command']
    require(records['intent']['admission']==refs['admission'] and '/usr/bin/env'in command,
        'Original service launch intent retained')
    expected_command=' '.join(command[command.index('/usr/bin/env'):])
    require(re.findall(r'argv\[\]=(.*?)\s*;',props['ExecStart'])==[expected_command], 'Retained exact launched service argv')
    group_proof=None
    if physical is not None:
        current=physical.properties(spec['unit'])
        if current['LoadState']=='loaded':
            require(current['InvocationID']==co['invocation_id'] and current['ExecMainPID']==str(co['pid'])
                and re.findall(r'argv\[\]=(.*?)\s*;',current['ExecStart'])==[expected_command]
                and terminal(current),'Same original loaded service must be terminal')
        else:
            require(current['LoadState']=='not-found' and not current['InvocationID']
                and current['MainPID']==current['ControlPID']==current['ExecMainPID']=='0' and terminal(current),
                'Original model service completely collected')
        for owner in owners:physical.gone(owner)
        group_proof=physical.empty_group(group)
    return dict(ended=ended,spec=spec,owners=owners,group=group_proof,session=session,records=records,
                physical_checks_performed=physical is not None)


def audit(receipt_ref,model_ref,lock,kind,load,physical=None):
    ended=load(receipt_ref)
    require(ended['schema']=='minicpm5_layer1_'+kind+'_post_exit_audit_actual_exit_v1'
        and ended['reviewer']=='/root' and ended['actual_tool_exit_observed']is True
        and ended['exact_owner_gone']is True and ended['serial_lock_reacquired']is True
        and ended['lock_identity']==lock,'Actual separate post-exit audit receipt')
    session=handle(ended);result=load(ended['result'])
    require(result['schema']=='minicpm5_layer1_'+kind+'_post_exit_audit_v1'
        and result['complete']is True and result['reviewer']=='/root'
        and result['model_exit']==model_ref and result['lock_identity']==lock
        and result['every_raw_byte_reread_after_all_model_processes_exited']is True,
        'Auditor actually reread this completed model run')
    same_owner(ended['owner'],result['owner'])
    for row in (ended,result):
        require(row['full_model_qualified']is False and row['new_speedup_claim']is False,'No diagnostic promotion')
    if physical is not None:physical.gone(ended['owner'])
    return dict(owner=ended['owner'],session=session,result_ref=ended['result'],ended=ended,result=result)


def current(binding,official,storage,lock,load,physical=None):
    require(all(k in binding for k in ('actual_live','spec','review','actual_exit','audit_live',
        'independent_audit','independent_audit_exit')),'Complete latest current/history binding required')
    m=model(binding,'current_history_factorial',lock,load,physical)
    ended=m['ended'];spec=m['spec'];records=m['records']
    runner=str(ROOT/'minicpm5_layer1_current_history_factorial_cpu_runner_v1.py')
    common=['--admission',ended['outputs']['admission']['path'],'--admission-sha256',ended['outputs']['admission']['sha256']]
    interpreter='/home/ubuntu/venvs/vllm_neuron/bin/python'
    require(ended['coordinator']['argv']==[interpreter,'-B',runner,'--mode','coordinator',*common]
        and all(row['argv']==[interpreter,'-B',runner,'--mode','rank',*common]for row in ended['ranks']),
        'Exact original coordinator and both rank commands')
    require(records['intent']['command'][-9:]==ended['coordinator']['argv'],
        'Service launch command ends with the authenticated coordinator command')
    require(ended['live']==binding['actual_live'] and ended['spec']==binding['spec']
        and ended['review']==binding['review'],'Latest handoff binds the actual live/spec/review chain')
    a=audit(binding['independent_audit_exit'],binding['actual_exit'],lock,'current_history_factorial',load,physical)
    require(a['result_ref']==binding['independent_audit'],'Latest handoff binds the complete independent result')
    live=load(binding['audit_live'])
    require(live['schema']=='minicpm5_current_history_factorial_audit_live_v1' and live['reviewer']=='/root'
        and live['actual_identity_verified']is True and live['owner']==a['owner']
        and live['actual_execution_handle']==a['session'] and live['model_actual_exit']==binding['actual_exit'],
        'Actual independent auditor live-to-exit chain')
    expected_audit=ROOT/'research/audit_minicpm5_layer1_current_history_factorial_result_v1.py'
    argv=a['owner']['argv']
    require(len(argv)==3 and argv[:2]==['python3','-B'] and str((ROOT/argv[2]).resolve())==str(expected_audit),
        'Original actual independent auditor command')
    require(a['session']['session_id']!=m['session']['session_id'],'Separate model and independent auditor sessions')
    require(spec['completed_official_history']==dict(schema='minicpm5_layer1_completed_official_history_v1',
        model_exit=official['actual_exit'],audit_exit=official['independent_audit_exit']),
        'Current/history experiment follows exact official-history model and auditor')
    require(spec['completed_storage']['maintenance_exit']==storage['actual_exit']
        and spec['completed_storage']['audit_exit']==storage['independent_audit_exit'],
        'Current/history experiment follows exact completed storage and auditor')
    plan=load(spec['plan']);review=load(ended['review'])
    require(plan['source_sha256']==spec['source_sha256']==review['source_sha256'],'Same actual plan/spec/review source map')
    rank_results=[]
    for index,owner in enumerate(ended['ranks']):
        ref=ended['outputs']['rank%d_result'%index]
        require(ref['path']==str(Path(spec['output_root'])/('rank%d-result.json'%index)),'Exact original rank output path')
        row=load(ref);rank_results.append(row)
        require(row['complete']is True and row['rank']==index and row['process']==owner
            and row['admission']==ended['outputs']['admission'] and row['self_exit_claimed']is False,
            'Each completed rank result binds its actual process and admission')
    require(records['coordinator']['rank_owners']==ended['ranks'] and records['coordinator']['results']==rank_results
        and records['coordinator']['self_exit_claimed']is False,'Coordinator binds both exact complete rank results')
    require(records['cleanup']['owners_checked']==[ended['coordinator'],*ended['ranks']],
        'Original cleanup includes coordinator and both ranks')
    proof=a['result']['result']
    require(proof==records['coordinator']['validated'] and proof['decode_calls']==180
        and proof['raw_bytes']==2949120 and proof['unique_replicated_output_rows']==360
        and proof['previous_real_controls_full_bytes_exact']is True and proof['previous_control_calls']==108,
        'Auditor recomputed the exact complete current/history output manifest')
    owners=[*m['owners'],a['owner']]
    require(len({(o['pid'],o['start_ticks'])for o in owners})==5,'Five distinct actual model/auditor owners')
    # Retain provenance and counts, not several redundant copies of complete
    # rank/coordinator output manifests in every subsequent controller proof.
    return dict(owners=owners,model_actual_exit=binding['actual_exit'],audit_actual_exit=binding['independent_audit_exit'],
        model_group=m['group'],actual_sessions=[m['session'],a['session']],
        decode_calls=proof['decode_calls'],raw_bytes=proof['raw_bytes'],
        previous_control_calls=proof['previous_control_calls'],physical_checks_performed=physical is not None,
        complete_model_and_independent_auditor_authenticated=True)

