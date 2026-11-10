"""Authenticate successful official-history execution and independent post-exit audit.

Metadata verification does not replace physical drainage under the current lock.
"""
from pathlib import Path
from research.closed_journals_after_official_history_completion_v1 import handle, same_owner
import minicpm5_layer1_official_history_cpu_runner_v1 as original
p=original.p
require=p.require


def completed(binding,read):
    require(set(binding)=={'schema','model_exit','audit_exit'} and
        binding['schema']=='minicpm5_layer1_completed_official_history_v1','Explicit completed official-history binding')
    ended=read(binding['model_exit']);ae=read(binding['audit_exit']);audit=read(ae['result'])
    handle(ended); handle(ae)
    spec=read(ended['spec']);plan=read(spec['plan']);review=read(ended['review']);live=read(ended['live'])
    records={k:read(ref)for k,ref in ended['outputs'].items()}
    require(ended['schema']=='minicpm5_layer1_official_history_actual_exit_v1'
        and ended['actual_exit_observed']is True and type(ended['actual_exit_code'])is int
        and ended['actual_exit_code']==0 and ended['all_four_processes_gone']is ended['cgroup_empty']is True,
        'Actual successful official-history model exit')
    require(ended['owner']==live['owner'] and ended['actual_execution_handle']==live['actual_execution_handle']
        and live['spec']==ended['spec'] and live['review']==ended['review'], 'Actual execution identity')
    require(review['passed']is True and review['blockers']==[] and review['spec']==ended['spec']
        and spec['source_sha256']==plan['source_sha256']==review['source_sha256'], 'Reviewed actual official-history sources')
    require(ae['schema']=='minicpm5_layer1_official_history_post_exit_audit_actual_exit_v1'
        and ae['actual_tool_exit_observed']is True and type(ae['actual_exit_code'])is int and ae['actual_exit_code']==0
        and ae['exact_owner_gone']is ae['serial_lock_reacquired']is True, 'Actual successful independent audit exit')
    require(audit['schema']=='minicpm5_layer1_official_history_post_exit_audit_v1'
        and audit['complete']is audit['every_raw_byte_reread_after_all_model_processes_exited']is True
        and audit['own_history_reproduced']is True and audit['model_exit']==binding['model_exit']
        and ae['owner']==audit['owner'] and ae['lock_identity']==audit['lock_identity']==ended['lock_identity'],
        'Independent post-exit bytes cover this exact model run')
    require(audit['source']['path']==str(Path(__file__).parent/'audit_minicpm5_layer1_official_history_result_v1.py')
        and p.sha(audit['source']['path'])==audit['source']['sha256'],'Pinned actual independent auditor source')
    require(records['outer']['complete']is True and records['outer']['owner']==ended['owner']==records['admission']['outer_owner']
        and records['children']['exit_codes']==records['coordinator']['rank_exits']==[0,0]
        and records['service_exit']['successful']is True and records['cleanup']['physically_drained']is True
        and records['cleanup']['original_failure']is None, 'Complete successful model service and cleanup')
    require(records['admission']['spec']==ended['spec'] and records['admission']['review']==ended['review']
        and records['admission']['lock_identity']==ended['lock_identity'], 'Actual runtime admission')
    for rank in (0,1):
        result=records['rank%d_result'%rank]
        require(result['process']==ended['ranks'][rank] and result['admission']==ended['outputs']['admission']
            and result==records['coordinator']['results'][rank], 'Actual complete original rank binding')
    proof=audit['result']
    require(proof==records['coordinator']['validated'] and proof['complete']is True
        and [proof[k]for k in ('ranks','decode_calls','raw_bytes','unique_replicated_output_rows')]==[2,180,2949120,360]
        and proof['all_original_and_restored_outputs_exact']is proof['every_output_independently_reread']is True,
        'All180 official history outputs independently verified')
    for value in (ended,ae,audit,proof):
        require(value['full_model_qualified']is value['new_speedup_claim']is False,'No qualification or speed claim')
    owners=[ended['owner'],ended['coordinator'],*ended['ranks'],ae['owner']]
    require(len({(x['pid'],x['start_ticks'])for x in owners})==5,'Five distinct actual processes')
    return dict(binding=binding,ended=ended,spec=spec,plan=plan,records=records,audit=audit,audit_exit=ae,
        owners=owners,lock_identity=ended['lock_identity'])


def physical(bundle,guard):
    guard()
    for owner in bundle['owners']:p.gone(owner)
    ended=bundle['ended'];a=bundle['records']['admission']
    proof=original.verify_owned_drain(bundle['spec'],ended['outputs']['admission'],a['attempt_id'],ended['coordinator'],True)
    guard()
    return dict(official_history_service=proof,owners_checked=bundle['owners'],independent_auditor_gone=True)


HANDOFF = dict(path=str(Path(__file__).parent/'continuation-handoff-20261110-v420-published-mini-history-storage-complete.json'),
    sha256='d23f1b0cd009442ca618814d4fdb42cc5157debdd62cd64a728b3f92daffdc19')


def storage_completed(binding, official_bundle, read):
    require(set(binding)=={'schema','handoff','maintenance_exit','audit_exit'} and
        binding['schema']=='minicpm5_completed_storage_maintenance_v1' and binding['handoff']==HANDOFF,
        'Exact terminal storage and v420 binding')
    handoff=read(binding['handoff']); ended=read(binding['maintenance_exit']); ae=read(binding['audit_exit'])
    result=read(ended['result']); audit=read(ae['result']); plan=read(result['plan']); review=read(result['review'])
    require(handoff['schema']=='trn1_continuation_handoff_v420' and handoff['live_tool_sessions']==[]
        and handoff['active_heavy_units']==[],'Exact completed serial handoff')
    h=handoff['official_history']; storage=handoff['storage_maintenance']; lock=official_bundle['lock_identity']
    require(h['actual_exit']==official_bundle['binding']['model_exit']
        and h['independent_audit_exit']==official_bundle['binding']['audit_exit']
        and h['independent_audit']==official_bundle['audit_exit']['result'], 'Exact v420 official history chain')
    require(storage['actual_exit']==binding['maintenance_exit'] and storage['independent_audit_exit']==binding['audit_exit']
        and storage['independent_audit']==ae['result'],'Exact v420 storage chain')
    require(ended['schema']=='closed_journals_after_r11_actual_exit_v3' and ended['actual_exit_observed'] is True
        and ae['schema']=='closed_journals_after_r11_post_exit_audit_actual_exit_v1'
        and ae['actual_tool_exit_observed'] is True,'Both actual storage tool exits required')
    handles=[handle(value) for value in (ended,ae)]
    require(len({v['session_id'] for v in handles})==2,'Distinct storage and auditor sessions')
    for value in (ended,ae):
        require(value['exact_owner_gone'] is value['serial_lock_reacquired'] is True
            and value['lock_identity']==lock,'Original lock and successful recorded storage drainage')
    same_owner(ended['owner'],result['owner']); same_owner(ae['owner'],audit['owner'])
    require(ended['owner']!=ae['owner'],'Independent storage auditor')
    require(result['complete'] is result['every_original_byte_retained'] is result['canonical_unchanged'] is True
        and result['automatic_successor_launch'] is result['actual_self_exit_claimed'] is result['new_speedup_claim'] is False,
        'Completed lossless storage maintenance only')
    require(plan['completion_handoff']==handoff['previous'] and plan['serial_lock_identity']==lock
        and plan['canonical_sha256']==handoff['canonical_sha256']==official_bundle['spec']['canonical_sha256'],
        'Maintenance follows completed official history and original canonical files')
    previous=read(plan['completion_handoff'])
    require(previous['official_history']==h and previous['native_cache']==handoff['native_cache'],
        'Maintenance completed after exact official/native history')
    require(review['passed'] is True and review['blockers']==[] and review['plan']==result['plan'],
        'Actual accepted maintenance review')
    require(audit['schema']=='closed_journals_after_r11_post_exit_audit_v1' and audit['complete'] is True
        and audit['actual_exit']==binding['maintenance_exit'] and audit['lock_identity']==lock
        and audit['all_original_bytes_reread_after_maintenance_exit'] is audit['canonical_unchanged'] is True
        and audit['automatic_successor_launch'] is False,'Independent completed full decompression audit')
    require(len(audit['files'])==len(result['retired'])==6 and
        all(v['original_absent'] is v['full_decompression_verified'] is True for v in audit['files'])
        and [audit[k] for k in ('original_bytes','compressed_bytes')]==[805306368,95403896]
        and all(result[k]==audit[k] for k in ('original_bytes','compressed_bytes')),
        'All six original journals fully retained and independently reread')
    owners=[ended['owner'],ae['owner']]
    require(len({(v['pid'],v['start_ticks']) for v in owners+official_bundle['owners']})==7,
        'Five official history and two storage owners are distinct')
    return dict(binding=binding,ended=ended,result=result,audit=audit,audit_exit=ae,plan=plan,
        owners=owners,lock_identity=lock)


def storage_physical(bundle,guard):
    guard()
    for owner in bundle['owners']: p.gone(owner)
    guard()
    return dict(owners_checked=bundle['owners'],maintenance_and_independent_auditor_gone=True)
