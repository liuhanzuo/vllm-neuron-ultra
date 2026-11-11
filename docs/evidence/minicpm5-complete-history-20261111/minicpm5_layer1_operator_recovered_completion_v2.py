"""Authenticated completed SDK operator data after an actually failed model.

Pure supplied-reader boundary: no IO, locks, native imports or runtime admission.
It consumes the real root success/recorder schemas; it never rewrites exit1.
Historical physical evidence is decoded as metadata, not a fresh drain claim.
"""
import copy
import hashlib
from pathlib import Path
import re
from research import minicpm5_layer1_operator_sdk_failed_recovery_core_v1 as core
from research import minicpm5_layer1_operator_sdk_failed_recovery_adapter_v2 as bridge
from research import audit_minicpm5_layer1_operator_sdk_result_v3 as audit
from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
from research import minicpm5_layer1_operator_sdk_sources_v1 as sources
from research import minicpm5_sdk_physical_evidence_v3 as physical
require=core.require
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'research'
AUDITOR=str(Q/'audit_minicpm5_layer1_operator_sdk_failed_recovery_v2.py')
BINDING=dict(schema='minicpm5_sdk_operator_recovered_data_binding_v1',
    actual_exit=dict(path=str(Q/'minicpm5-operator-sdk-recovery-r3-actual-exit-20261111-v1.json'),
        sha256='b312e9d2bceb4c2af867446666ce2949cd789c2d7a43e15c8f23b11fa9115325'),
    recorder_terminal=dict(path=str(Q/'minicpm5-operator-sdk-recovery-r3-recorder-actual-exit-20261111-v1.json'),
        sha256='491abf36836b296b1ac6f7dbceaf8b98ae562af4644fd03d99159f84be81f4f0'))


def tool(value):
    require(type(value)is dict and set(value)=={'kind','session_id','terminal_chunk','exit_code'}
        and value['kind']=='tool_session' and type(value['session_id'])is int and value['session_id']>0
        and type(value['exit_code'])is int and value['exit_code']==0
        and type(value['terminal_chunk'])is str and re.fullmatch('[a-f0-9]{6,64}',value['terminal_chunk']),
        'Externally observed real successful tool session/chunk; bool and failure rejected')
    return value


def successful(ended):
    require(ended['schema']=='minicpm5_operator_sdk_recovery_success_actual_exit_v1'
        and ended['reviewer']=='/root' and ended['actual_tool_exit_observed']is True
        and ended['exact_owner_gone']is ended['serial_lock_reacquired']is True,
        'Actual independent successful recovery audit; failed recovery schemas reject')
    return tool(ended['actual_tool'])


def source_union(plan,spec):
    pins=dict(plan['source_sha256'])
    for path,sha in spec['source_extension'].items():
        require(path not in pins,'Original source map never replaced/reclassified by recovery')
        pins[path]=sha
    return pins


def validate_terminal(binding,ended,live,result,spec,review,recorder,plan):
    require(binding==BINDING,'Exact actual completed recovery and independent recorder terminal')
    observed=successful(ended);rt=tool(recorder['actual_tool'])
    require(recorder['schema']=='minicpm5_recovery_post_exit_recorder_terminal_v1'
        and recorder['result']==binding['actual_exit'] and recorder['owner']==ended['recorder']
        and recorder['owner_gone']is True and rt['session_id']!=observed['session_id'],
        'Separate actual recorder exit binds the exact recovery terminal and actual recorder owner')
    core.same_owner(ended['owner'],live['owner']);core.same_owner(ended['recorder'],recorder['owner'])
    require(live['schema']=='minicpm5_operator_sdk_recovery_actual_live_v1' and live['reviewer']=='/root'
        and live['actual_identity_verified']is True
        and live['actual_execution_handle']==dict(kind='tool_session',session_id=observed['session_id'])
        and live['spec']==ended['spec'] and live['review']==ended['review'] and live['original_exit_code']==1,
        'Exact root-observed recovery live-to-terminal chain')
    argv=['python3','-B',AUDITOR,'--spec',ended['spec']['path'],'--spec-sha256',ended['spec']['sha256'],
        '--review',ended['review']['path'],'--review-sha256',ended['review']['sha256']]
    require(ended['owner']['argv']==argv and ended['owner']==result['owner'],'Exact recovery auditor command and result owner')
    bridge.validate_spec(ended['spec'],ended['review'],spec,review)
    require(result['schema']=='minicpm5_operator_sdk_failed_run_recovery_audit_v1' and result['reviewer']=='/root'
        and result['audit_complete']is True and result['self_exit_claimed']is False
        and result['recovery_spec']==ended['spec'] and result['recovery_review']==ended['review']
        and ended['result']['path']==str(Path(spec['output_root'])/'result.json')
        and result['failed_terminal']==ended['failed_terminal']==spec['failed_terminal']
        and result['failed_evidence']==spec['failed_evidence']
        and result['original_scientific_plan']==spec['original_plan']
        and result['original_scientific_spec']==spec['original_spec']
        and result['lock_identity']==ended['lock_identity'],'Exact independent recovery report and immutable original science')
    require(type(ended['original_model_exit_code'])is int and ended['original_model_exit_code']==1
        and type(result['original_exit_code'])is int and result['original_exit_code']==1
        and ended['original_model_runtime_successful']is result['original_runtime_successful']is False
        and ended['original_success_receipts_absent']is result['original_missing_success_receipts_preserved']is True
        and result['original_failure_receipts_unchanged']is result['original_source_map_unchanged']is True,
        'Original model exit1 and missing coordinator/outer success receipts retained')
    for row in (ended,result,spec):
        require(row['full_model_qualified']is row['new_speedup_claim']is row['automatic_successor_release']is False,
            'Recovered data does not itself admit native work or qualify model/performance')
    for key in ('every_raw_byte_reread_after_all_model_processes_exited','all_endpoint_controls_full_bytes_exact',
                'all_capture_journals_verified','sdk_rope_payload_independently_reread',
                'complete_original_owner_scope_preserved','strict_paired_physical_semantics_verified'):
        require(result[key]is True,'Every original complete raw and physical recovery guarantee retained')
    require(result['other_arms_reexecuted']is result['attention_causality_proven']is result['model_reexecuted']is False,
        'Only saved-data audit, no model replay or causal promotion')
    require(ended['raw_proof']==result['result'],'Actual exit retains exact complete raw audit proof');core.validate_proof(result['result'])
    require(ended['complete_recovery_metadata_independently_verified']is ended['full_physical_metadata_decoded']
        is ended['every_loaded_auditor_source_hash_verified']is True
        and ended['raw_tensor_bytes_reread_here']is False,'Independent metadata-only recorder guarantees')
    require(result['actual_current_pair_schema']==physical.PAIR_SCHEMA and result['full_physical_passes']==2
        and ended['physical_metadata_sha256']==result['actual_current_physical_pair']['sha256']
        and ended['current_serial_catalog']==result['current_serial_catalog']==spec['current_serial_catalog']
        and ended['serial_freshness']==result['serial_freshness'], 'Exact separately decoded historical physical pair/catalog')
    pins=source_union(plan,spec);loaded=result['loaded_source_closure'];executed=loaded['source_sha256']
    require(result['separately_reviewed_source_extension']==spec['source_extension']
        and loaded['complete']is loaded['original_scientific_source_map_unchanged']
        is loaded['separately_reviewed_recovery_extension']is True
        and type(loaded['count'])is int and loaded['count']==len(executed)==ended['loaded_source_count']
        and executed and all(pins.get(path)==sha for path,sha in executed.items())
        and result['source']==dict(path=AUDITOR,sha256=pins[AUDITOR]) and executed.get(AUDITOR)==pins[AUDITOR],
        'Complete original-plus-separately-reviewed actually executed source closure')
    return dict(source_sha256=pins,loaded_source_sha256=executed,owner=ended['owner'],recorder=ended['recorder'])


def historical_scope(plan,done,spec,read,guard):
    view=dict(latest_completion=plan['latest_completion'],serial_predecessors=spec['current_serial_catalog'],
        source_sha256=source_union(plan,spec))
    serial=latest.serial_catalog(view,read,guard)
    rows={row['terminal']['path']:row for row in serial['runs']}
    require(all(rows.get(row['terminal']['path'])==row for row in done['later_serial']['runs']),
        'Recovery historical serial catalog preserves every original row')
    owners=copy.deepcopy(done['operator_sdk']['owners'])+copy.deepcopy(serial['owners'])
    require(all(owner in owners for owner in done['owners']),'Every exact original owner remains in historical recovery scope')
    units=set(done['complete_known_units'])|set(serial['units'])
    for owner in owners:
        group=owner.get('cgroup')
        if type(group)is str and group.startswith('0::'+physical.GROUP_PREFIX):units.add(physical.group_unit(group[3:]))
    return dict(done,owners=owners,later_serial=serial,complete_known_units=sorted(units))


def completed(binding,read,guard,read_source):
    """Callers authenticate exact refs/stable identities and separately admit IO.

    read_source reads only supplied source text under its original source policy.
    No native bytes or tensor outputs are read. A new outer must freshly drain
    returned owners plus every later serial operation before starting native work.
    """
    require(binding==BINDING and all(callable(x)for x in (read,guard,read_source)),'Fixed recovery binding and explicit guarded readers')
    seen={}
    def load(ref):
        guard();sources.reference(ref);value=read(ref);seen[ref['path']]=copy.deepcopy(ref);guard();return value
    ended=load(binding['actual_exit']);successful(ended)
    recorder=load(binding['recorder_terminal']);live=load(ended['live']);result=load(ended['result'])
    spec=load(ended['spec']);review=load(ended['review']);original_spec=load(spec['original_spec']);plan=load(spec['original_plan'])
    require(original_spec['plan']==spec['original_plan'] and original_spec['source_sha256']==plan['source_sha256'],
        'Unchanged original failed operator runtime spec and plan')
    checked=validate_terminal(binding,ended,live,result,spec,review,recorder,plan)
    failed_terminal=load(spec['failed_terminal']);failed=load(spec['failed_evidence']);original_live=load(failed['live'])
    require(failed_terminal['failed_evidence']==spec['failed_evidence'] and failed['spec']==spec['original_spec']
        and failed['review']==spec['original_review'] and failed['lock_identity']==ended['lock_identity']
        and result['original_failed_tool']==failed_terminal['actual_tool'],'Exact unchanged original failed model chain')
    records={key:load(ref)for key,ref in core.output_refs(failed).items()};original_review=load(spec['original_review'])
    lineage=core.validate(failed_terminal,failed,original_live,original_spec,plan,original_review,records)
    require(result['original_model_outputs']==core.output_refs(failed),'Every actual original output reference retained')
    done=latest.completed(plan,load,guard)
    require(done['lock_identity']==ended['lock_identity'],'Same original lock throughout completed lineage')
    require(audit.verify_outer_drain(records,done,failed['failed_owner'],ended['lock_identity'])==result['saved_original_outer_drain'],
        'Saved outer pair checked against its original scope')
    require(audit.verify_rank_guards(records)==result['rank_guard_verification']
        and all(e['owners']==len(done['owners'])for rank in (0,1)for e in records['rank%d_result'%rank]['guard_checks']['events']),
        'Every original complete rank guard covered all original owners')
    historical=historical_scope(plan,done,spec,load,guard);pair=audit.unpack(result['actual_current_physical_pair'])
    required={row['unit']for row in done['nemotron']['history']['rows']}
    units=physical.validate_pair(pair,required,historical['owners'],ended['lock_identity'],known_units=historical['complete_known_units'])
    require(units==set(historical['complete_known_units']) and len(units)==result['current_service_count']==ended['decoded_current_service_count']
        and len(historical['owners'])==result['complete_current_owner_count']==ended['decoded_current_owner_count'],
        'Both complete historical physical passes match exact owner/service sets, then counts')
    for side in ('first','second'):
        tree=pair[side]['tree']
        require(tree['complete_sdk_binding']==latest.sdk.BINDING and tree['later_serial_catalog']==spec['current_serial_catalog'],
            'Every decoded historical observation binds the original completed layout and recovery serial catalog')
    partition=sources.classify(plan['source_sha256'],plan['source_contract'],load)
    inherited=sources.validate_native_receipt(records['source_authentication'],partition,failed['failed_owner'],
        ended['lock_identity'],records['admission']['attempt_id'])
    require(result['source_authentication']==core.output_refs(failed)['source_authentication']
        and result['inherited_native_authentication']==inherited,'Original inherited native source authentication')
    for path,sha in checked['loaded_source_sha256'].items():
        guard();require(hashlib.sha256(read_source(path,sha)).hexdigest()==sha,'Every actually executed recovery text helper authenticated')
    drain=ended['original_model_drain']
    require(drain['physically_drained']is True and drain['owners_checked']==lineage['owners'][1:]
        and drain['properties']['Result']=='exit-code' and drain['properties']['ExecMainStatus']=='1'
        and drain['properties']['ExecMainCode']=='1'
        and drain['properties']['ExecMainPID']==str(records['coordinator_owner']['pid'])
        and drain['properties']['InvocationID']==records['coordinator_owner']['invocation_id']
        and core.terminal(drain['properties']),'Original failed service drain retained by independent recorder')
    guard();return dict(binding=copy.deepcopy(binding),base_plan=plan,base_plan_ref=spec['original_plan'],
        base_spec=original_spec,done=done,native=done['native'],historical=historical,
        recovery=result,recovery_spec=spec,actual_exit=ended,recorder_terminal=recorder,
        source_sha256=checked['source_sha256'],authenticated_metadata=seen,
        owners=[*historical['owners'],checked['owner'],checked['recorder']],complete_known_units=sorted(units),
        lock_identity=ended['lock_identity'],metadata_complete=True,physical_checks_performed=False,
        raw_payloads_read=False,runtime_admitted=False,original_model_exit_code=1,
        full_model_qualified=False,new_speedup_claim=False)
