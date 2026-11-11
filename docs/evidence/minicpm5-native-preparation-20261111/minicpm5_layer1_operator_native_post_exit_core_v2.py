"""Pure native external-exit/receipt binding; never creates terminal success."""
import copy
from pathlib import Path
import re
from research import minicpm5_layer1_operator_native_lifecycle_core_v2 as life
from minicpm5_layer0_runtime_auth_v1 import digest
require=life.require
OUTPUTS={'admission':'admission.json','intent':'launch-intent.json','source_authentication':'source-authentication.json',
    'drains':'predecessor-drain.json','coordinator_owner':'coordinator-owner.json','native_owner':'native-owner.json',
    'native_result':'native-result.json','children_exits':'children-exits.json','coordinator_result':'coordinator-result.json',
    'service_exit':'service-exit.json','outer_result':'outer-result.json','cleanup_drained':'cleanup-drained.json'}


def actual_tool(value,*,allow_call=False):
    session=(type(value)is dict and value.get('kind')=='tool_session'
        and set(value)=={'kind','session_id','terminal_chunk','exit_code'}
        and type(value['session_id'])is int and value['session_id']>0)
    call=(allow_call and type(value)is dict and value.get('kind')=='tool_call'
        and set(value)=={'kind','terminal_chunk','exit_code'})
    require((session or call)and type(value['exit_code'])is int and value['exit_code']==0
        and type(value['terminal_chunk'])is str and re.fullmatch('[a-f0-9]{6,64}',value['terminal_chunk']),
        'Exact externally observed successful tool terminal; never invent a session for a tool call')
    return value


def validate(exit_ref,ended,recorder_ref,recorder,live,records,resolved):
    """Small new root terminal schema; producers remain external to auditor."""
    life.ref(exit_ref);life.ref(recorder_ref);tool=actual_tool(ended['actual_tool']);rt=actual_tool(recorder['actual_tool'],allow_call=True)
    require(ended['schema']=='minicpm5_native_model_actual_exit_v1'and ended['reviewer']=='/root'
        and ended['actual_exit_observed']is True and ended['exact_owner_gone']is True
        and ended['self_exit_claimed']is False and type(ended['actual_exit_code'])is int and ended['actual_exit_code']==0,
        'Externally recorded original native model outer exit zero')
    require(recorder['schema']=='minicpm5_native_model_exit_recorder_actual_exit_v1'and recorder['reviewer']=='/root'
        and recorder['model_exit']==exit_ref and recorder['owner']==ended['recorder']and recorder['owner_gone']is True
        and recorder['self_exit_claimed']is False
        and (rt['kind']=='tool_call'or rt['session_id']!=tool['session_id']),
        'Separate actual recorder exit binds exact model terminal and recorder identity')
    require(live['schema']=='minicpm5_native_model_actual_live_v1'and live['reviewer']=='/root'
        and live['actual_identity_verified']is True and live['owner']==ended['owner']
        and live['actual_execution_handle']=={'kind':'tool_session','session_id':tool['session_id']}
        and live['spec']==ended['spec']and live['review']==ended['review']and live['admission']==ended['outputs']['admission'],
        'Independent actual live-to-terminal owner/session/spec/admission chain')
    a=records['admission'];contract=a['contract'];life.binding(contract,resolved)
    require(set(ended['outputs'])==set(OUTPUTS)and set(records)==set(OUTPUTS),'All original finite model receipts, no omitted success evidence')
    for key,name in OUTPUTS.items():life.ref(ended['outputs'][key],contract['output_root']+'/'+name)
    require(a['outer_owner']==ended['owner']and a['spec']==ended['spec']and a['review']==ended['review']
        and a['resolved_plan_sha256']==ended['resolved_plan_sha256']==resolved['plan_sha256']
        and a['lock_identity']==ended['lock_identity']and a['unit']==ended['unit']==contract['unit']
        and a['runtime_admitted']is a['predecessors_checked']is True,'Complete actual admission/outer/plan/lock binding')
    release=a['outer_release']
    require(release['reviewer']=='/root'and release['challenge']['owner']==ended['owner']
        and release['challenge']['spec']==ended['spec']and release['challenge']['review']==ended['review']
        and release['tool']==a['current_execution']['tool']==live['actual_execution_handle'],
        'Root launch release and external actual execution handle agree')
    for key,name in [('source_authentication','source_authentication'),('observed_drains','drains')]:
        require(a[key]==ended['outputs'][name],'Original admission receipt reference '+key)
    require(a['current_execution']['intent']==ended['outputs']['intent'],'Original acyclic launch intent reference')
    co=records['coordinator_owner'];native=records['native_owner']
    require(co==ended['coordinator']and native==ended['native'],'Exact acknowledged model process tree')
    co_process=life.coordinator_process(contract,ended['outputs']['admission'],co)
    life.process(native,life.command(contract,'native',ended['outputs']['admission']),co['pid'],co['cgroup'])
    wrapped=records['native_result'];driver=life.unwrap_native_result(wrapped,contract,native,ended['outputs']['admission'],resolved)
    children=records['children_exits']
    require(children==dict(complete=True,owners=[native],exit_codes=[0],successful=True,self_exit_claimed=False),
        'Actual coordinator observed native child reaped zero')
    cr=records['coordinator_result']
    require(cr['schema']=='minicpm5_native_compact_coordinator_v2'and cr['complete']is True
        and cr['contract_sha256']==digest(contract)and cr['admission']==ended['outputs']['admission']
        and cr['coordinator_owner']==co_process and cr['native_owner']==native
        and type(cr['native_exit'])is int and cr['native_exit']==0 and cr['native_result']==ended['outputs']['native_result']
        and cr['independent_raw_audit_pending']is cr['external_model_exit_pending']is True
        and cr['self_exit_claimed']is cr['full_model_qualified']is cr['new_speedup_claim']is False,
        'Exact compact coordinator bindings, not native report presence alone')
    se=records['service_exit'];props=se['properties'];drain=records['cleanup_drained']
    require(se['owner']==co and se['observed_by_outer']==ended['owner']and se['successful']is True and se['self_exit_claimed']is False
        and props['Result']=='success'and props['ExecMainCode']=='1'and props['ExecMainStatus']=='0'
        and props['ExecMainPID']==str(co['pid'])and props['InvocationID']==co['invocation_id']
        and props['MainPID']==props['ControlPID']=='0'and not props['Job'],
        'Original successful service terminal and exact invocation')
    require(drain['physically_drained']is True and drain['owners_checked']==[co,native]
        and drain['self_exit_claimed']is False and drain['group']==se['group'],'Original saved full owned drain')
    outer=records['outer_result']
    require(outer['schema']=='minicpm5_native_outer_terminal_core_v2'and outer['complete']is True
        and outer['contract_sha256']==digest(contract)and outer['outer_owner']==ended['owner']
        and outer['service_exit']==ended['outputs']['service_exit']and outer['coordinator_result']==ended['outputs']['coordinator_result']
        and outer['native_result']==ended['outputs']['native_result']and outer['owned_drain']==ended['outputs']['cleanup_drained']
        and outer['service_observation']==props and outer['independent_raw_audit_pending']is True
        and outer['self_exit_claimed']is outer['actual_external_tool_exit_verified']is outer['full_model_qualified']is outer['new_speedup_claim']is False,
        'Original outer result retains external-exit and independent-audit pending flags')
    for value in (ended,recorder):
        require(value['full_model_qualified']is value['new_speedup_claim']is False,'No model/performance promotion in terminal records')
    return dict(driver=driver,contract=contract,model_owners=[ended['owner'],co,native],
        recorder_owner=recorder['owner'],actual_model_tool=copy.deepcopy(tool),actual_recorder_tool=copy.deepcopy(rt))
