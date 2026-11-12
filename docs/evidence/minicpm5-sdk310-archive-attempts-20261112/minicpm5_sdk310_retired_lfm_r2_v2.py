"""Fixed three-service retirement proof; retained r2 plus two audited Mini runs.

The r1 failed terminal remains exit 1. No generic missing-Invocation exception.
No import-time IO, processes, services, locks, downloads or writes.
"""
import copy
from pathlib import Path
import minicpm5_sdk310_retired_lfm_r2_v1 as base

c=base.c
bind=base.bind
PREFIX='/user.slice/user-1000.slice/user@1000.service/app.slice/'
EXTRA = (
    dict(unit='trn1-mini-layer1-operator-native-r1-20261111',invocation_id='bb6e0be35a364c1da9ae697a5ed70055',
         run='minicpm5-layer1-operator-native-r1-20261111',runner='minicpm5_layer1_operator_native_runner_v7.py',
         actual_exit=dict(path=str(c.Q/'minicpm5-layer1-operator-native-r1-model-actual-exit-20261111-v1.json'),sha256='3599cd98904b570fa9a48188321a766ccb2f01eefbb113497de0eab42354a2e0'),
         recorder_exit=dict(path=str(c.Q/'minicpm5-native-r1-model-exit-recorder-actual-exit-20261111-v1.json'),sha256='facdbe36e8c057bc1edd5d3ad66ce5199d803a64b7f9e0a06da164aa8eaace4d'),recorder_tool=dict(kind='tool_session',session_id=70508,terminal_chunk='6dd086',exit_code=0),tool=dict(kind='tool_session',session_id=88265,terminal_chunk='0f7129',exit_code=1),
         outer=(1471058,'418093092'),coordinator=(1478958,'418123278'),native=(1494935,'418180432')),
    dict(unit='trn1-mini-layer1-operator-native-r3-sourceclosure-20261111',invocation_id='45f668cfb23c4a01ae457eca6c0d31ee',
         run='minicpm5-layer1-operator-native-r3-sourceclosure-20261111',runner='minicpm5_layer1_operator_native_runner_v12.py',
         actual_exit=dict(path=str(c.Q/'minicpm5-layer1-operator-native-r3-sourceclosure-model-actual-exit-20261111-v1.json'),sha256='00abd33e34bcc09e7b64814a5a1853973c71581f2b74c5a8b45a2aff85225cab'),
         recorder_exit=dict(path=str(c.Q/'minicpm5-native-r3-sourceclosure-model-exit-recorder-actual-exit-20261111-v1.json'),sha256='f51c5e63448fddd3acd21bda6c7a575135e14ddca4a7059c90a662ea54780ab9'),recorder_tool=dict(kind='tool_session',session_id=10620,terminal_chunk='63a3f0',exit_code=0),tool=dict(kind='tool_session',session_id=8333,terminal_chunk='08b0e2',exit_code=0),
         outer=(1810304,'420364013'),coordinator=(1818189,'420394577'),native=(1834718,'420453315')),
)


def exact(item):
    c.require(any(item['unit']==row['unit'] and item['invocation_id']==row['invocation_id'] for row in EXTRA),
              'Only the two fixed historical Mini unit/invocation pairs')
    return PREFIX+item['unit']+'.service'


def retired(item,props,group):
    target=exact(item)
    c.require(props['Id']==item['unit']+'.service' and props['LoadState']=='not-found'
              and props['InvocationID']==props['Job']==props['ControlGroup']==''
              and props['MainPID']==props['ControlPID']==props['ExecMainPID']=='0'
              and props['ActiveState']=='inactive' and props['SubState']=='dead'
              and props['ExecMainCode']==props['ExecMainStatus']=='0'
              and props['Result']=='success' and base.terminal(props)
              and group==dict(group=target,absent=True), 'Exact retired Mini service absent, no invocation/job/PIDs/cgroup')


def loaded_success(item,props,group,coordinator):
    target=exact(item)
    c.require(props['Id']==item['unit']+'.service' and props['LoadState']=='loaded'
              and props['InvocationID']==item['invocation_id'] and props['ExecMainPID']==str(coordinator['pid'])
              and props['Result']=='success' and props['ExecMainCode']=='1' and props['ExecMainStatus']=='0'
              and base.terminal(props) and group==dict(group=target,absent=True), 'Historical r3 successful service bound to original invocation')


def original_drain(item,value,coordinator,native,was_retired):
    c.require(value['physically_drained'] is True and value['self_exit_claimed'] is False
              and value['owners_checked']==[coordinator,native], 'Complete historical physical drain and exact two owners')
    if was_retired:retired(item,value['properties'],value['group'])
    else:loaded_success(item,value['properties'],value['group'],coordinator)


def bind_extra(read):
    result=[]
    for item in EXTRA:
        base.ref(item['actual_exit']);base.ref(item['recorder_exit'])
        ended=read(item['actual_exit']);recorded=read(item['recorder_exit']);unit=item['unit'];inv=item['invocation_id']
        c.require(ended['reviewer']=='/root' and ended['actual_tool']==item['tool']
                  and type(ended['actual_exit_code']) is int and ended['actual_exit_code']==item['tool']['exit_code']
                  and ended['actual_exit_observed'] is True and ended['exact_owner_gone'] is True
                  and ended['self_exit_claimed'] is False and ended['unit']==unit
                  and ended['original_sdk_model_exit_code']==1 and ended['actual_recorder_tool_exit_verified'] is False
                  and ended['independent_raw_audit_pending'] is True and ended['full_model_qualified'] is False
                  and ended['new_speedup_claim'] is False, 'Original real terminal/status retained without model promotion')
        outer,coordinator,native,recorder=(ended[k] for k in ('owner','coordinator','native','recorder'))
        for key,owner in (('outer',outer),('coordinator',coordinator),('native',native)):
            c.require((owner['pid'],owner['start_ticks'])==item[key], 'Exact historical PID/start identity')
        c.require(coordinator['unit']==unit and coordinator['invocation_id']==inv
                  and coordinator['cgroup']==native['cgroup']=='0::'+exact(item)
                  and native['ppid']==coordinator['pid'], 'Original Mini invocation, group and child ancestry')
        refs={key:copy.deepcopy(ended['outputs'][key]) for key in ('admission','coordinator_owner','native_owner','cleanup_drained')}
        for key,filename in (('admission','admission.json'),('coordinator_owner','coordinator-owner.json'),('native_owner','native-owner.json'),('cleanup_drained','cleanup-drained.json')):
            base.ref(refs[key],Path('/mnt/nvme')/item['run']/filename)
        for role,owner in (('coordinator',coordinator),('native',native)):
            expected=['/home/ubuntu/venvs/model_reference_2026/bin/python','-B',str(c.ROOT/item['runner']),
                      '--mode',role,'--admission',refs['admission']['path'],'--admission-sha256',refs['admission']['sha256']]
            c.require(owner['argv']==expected and read(refs[role+'_owner'])==owner, 'Full original command and saved owner receipt')
        cleanup=read(refs['cleanup_drained']);original_drain(item,cleanup,coordinator,native,item['tool']['exit_code']==1)
        original_drain(item,ended['model_drain'],coordinator,native,True)
        if item['tool']['exit_code']==1:
            c.require(ended['schema']=='minicpm5_native_model_failed_actual_exit_v1'
                      and ended['original_nonzero_exit_preserved'] is True and ended['successful_model_terminal'] is False
                      and ended['missing_runtime_receipts']==['native-result.json','children-exits.json','coordinator-result.json','service-exit.json','outer-result.json','native-failure.json','cleanup-pending.json']
                      and all(key not in ended['outputs'] for key in ('service_exit','children_exits','native_result')), 
                      'Failed r1 stays failed; no invented service success receipt')
            refs['outer_failure']=copy.deepcopy(ended['outputs']['outer_failure'])
            base.ref(refs['outer_failure'],Path('/mnt/nvme')/item['run']/'outer-failure.json')
            failure=read(refs['outer_failure'])
            c.require(failure['error_type']=='RuntimeError' and failure['error']=='Original root28GiB and output30GiB floors remain live'
                      and failure['outer_owner']==outer and failure['self_exit_claimed'] is False
                      and failure['full_model_qualified'] is False and failure['new_speedup_claim'] is False,
                      'Original storage-floor failure evidence unchanged')
        else:
            c.require(ended['schema']=='minicpm5_native_model_actual_exit_v1', 'Original r3 terminal schema retained')
            refs['service_exit']=copy.deepcopy(ended['outputs']['service_exit'])
            base.ref(refs['service_exit'],Path('/mnt/nvme')/item['run']/'service-exit.json');service=read(refs['service_exit'])
            c.require(service['successful'] is True and service['self_exit_claimed'] is False and service['owner']==coordinator
                      and service['observed_by_outer']==outer and service['properties']==cleanup['properties'], 'Exact r3 saved service exit and cleanup binding')
            loaded_success(item,service['properties'],service['group'],coordinator)
            refs['children_exits']=copy.deepcopy(ended['outputs']['children_exits'])
            base.ref(refs['children_exits'],Path('/mnt/nvme')/item['run']/'children-exits.json');children=read(refs['children_exits'])
            c.require(children['complete'] is True and children['successful'] is True and children['self_exit_claimed'] is False
                      and children['owners']==[native] and children['exit_codes']==[0], 'Original r3 native child actual success retained')
        c.require(recorded['schema']=='minicpm5_native_model_exit_recorder_actual_exit_v1' and recorded['reviewer']=='/root'
                  and recorded['model_exit']==item['actual_exit'] and recorded['owner']==recorder and recorded['owner_gone'] is True
                  and recorded['actual_tool']==item['recorder_tool'] and recorded['actual_tool_exit_observed'] is True
                  and recorded['recorder_spec']==ended['recorder_spec'] and recorded['recorder_review']==ended['recorder_review']
                  and recorded['lock_identity']==ended['lock_identity']==dict(device=66306,inode=8294421)
                  and recorded['self_exit_claimed'] is False and recorded['full_model_qualified'] is False
                  and recorded['new_speedup_claim'] is False, 'Separate real recorder terminal binds the exact original report and owner')
        result.append(dict(schema='minicpm5_sdk310_retired_mini_history_binding_v1',unit=unit,invocation_id=inv,
            actual_exit=copy.deepcopy(item['actual_exit']),actual_tool=copy.deepcopy(item['tool']),
            recorder_exit=copy.deepcopy(item['recorder_exit']),outputs=refs,
            owners=copy.deepcopy([outer,coordinator,native,recorder]),original_sdk_model_exit_code=1,
            historical_terminal_schema=ended['schema'],original_missing_runtime_receipts=copy.deepcopy(ended.get('missing_runtime_receipts'))))
    return result


def validate(plan,review=None,read=c.load):
    base.validate(plan,review,read)
    extras=bind_extra(read);c.require(plan['retired_mini_history']==extras,'Both exact independently authenticated Mini retirements')
    if review is not None:c.require(review['retired_mini_history']==extras, 'Root reviewed both precise Mini retirement bindings')
    catalog=read(plan['complete_current_serial_catalog'])
    by_identity={}
    for owner in plan['predecessor_owners']:by_identity.setdefault((owner['pid'],owner['start_ticks']),[]).append(owner)
    for item in extras:
        for owner in item['owners']:
            c.require(owner in by_identity.get((owner['pid'],owner['start_ticks']),[]), 'Every original Mini outer/coordinator/native/recorder remains in fresh drain scope')
        matching=[s for s in plan['predecessor_services'] if s['unit']==item['unit'] and s.get('invocation_id')]
        c.require(matching==[dict(unit=item['unit'],invocation_id=item['invocation_id'])], 'One exact known Mini invocation retained; no wildcard exception')
        for ref in (item['actual_exit'],item['recorder_exit']):
            c.require(any(row['terminal']==ref for row in catalog['runs']), 'Historical model and recorder actual exits retained in complete catalog')
    return extras


def observe_extra(item,p):
    group_name=exact(item);samples=[]
    for _ in range(2):
        props=p.properties(item['unit']);group=p.empty_group(group_name);retired(item,props,group)
        c.require(p.terminal(props),'Original protocol terminal check');samples.append(dict(properties=props,group=group))
    c.require(samples[0]==samples[1],'Two stable current Mini retirement observations')
    return dict(unit=item['unit'],original_invocation_id=item['invocation_id'],retirement_binding=copy.deepcopy(item),samples=samples)


def preflight_history(plan,p,read=c.load):
    extras=validate(plan,read=read);by_key={(item['unit'],item['invocation_id']):item for item in extras}
    c.check_predecessor(read(plan['predecessor_exit']),plan['predecessor_expected_exit_code'])
    for owner in plan['predecessor_owners']:p.gone(owner)
    rows=[]
    for saved in plan['predecessor_services']:
        if saved['unit']==base.UNIT:
            rows.append(base.observe_retired(plan['retired_lfm_r2'],p));continue
        key=(saved['unit'],saved.get('invocation_id'))
        if key in by_key:
            rows.append(observe_extra(by_key[key],p));continue
        # Every other row, including the original unknown r1 invocation, is unchanged.
        samples=[]
        for _ in range(2):
            value=p.properties(saved['unit']);c.require(p.terminal(value),'Original prior service terminal/no Job')
            if saved.get('invocation_id'):c.require(value['InvocationID']==saved['invocation_id'],'Known original invocation')
            group=p.empty_group(p.GROUP_PREFIX+saved['unit']+'.service');samples.append(dict(properties=value,group=group))
        c.require(samples[0]==samples[1],'Two stable prior terminal observations');rows.append(dict(unit=saved['unit'],samples=samples))
    return rows


def verify_recorded_history(plan,rows,p,read=c.load):
    extras=validate(plan,read=read);by_key={(item['unit'],item['invocation_id']):item for item in extras}
    c.require(len(rows)==len(plan['predecessor_services']),'Complete saved predecessor history')
    for row,want in zip(rows,plan['predecessor_services']):
        c.require(row['unit']==want['unit'] and len(row['samples'])==2 and row['samples'][0]==row['samples'][1]
                  and p.terminal(row['samples'][0]['properties']), 'Stable terminal historical pair')
        if want['unit']==base.UNIT:
            c.require(row['retirement_binding']==plan['retired_lfm_r2'] and row['original_invocation_id']==base.INVOCATION,
                      'Original complete r2 retirement binding retained');base.pair(row['samples'],True)
        elif (want['unit'],want.get('invocation_id')) in by_key:
            item=by_key[(want['unit'],want['invocation_id'])]
            c.require(row['retirement_binding']==item and row['original_invocation_id']==item['invocation_id'], 'Exact original Mini retirement binding retained')
            for sample in row['samples']:retired(item,sample['properties'],sample['group'])
        elif want.get('invocation_id'):
            c.require(row['samples'][0]['properties']['InvocationID']==want['invocation_id'], 'Every other saved known invocation unchanged')
    for owner in read(plan['retired_lfm_r2']['scope'])['predecessor_owners']:p.gone(owner)
    for item in extras:
        for owner in item['owners']:p.gone(owner)
    return dict(lfm_r2=base.observe_retired(plan['retired_lfm_r2'],p),mini_history=[observe_extra(item,p) for item in extras])
