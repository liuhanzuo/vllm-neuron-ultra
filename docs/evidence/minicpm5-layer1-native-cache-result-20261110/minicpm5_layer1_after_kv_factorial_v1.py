"""Authenticate successful own-history execution and independent post-exit audit.

Metadata verification does not replace physical drainage under the current lock.
"""
from pathlib import Path
import minicpm5_layer1_kv_factorial_cpu_runner_v1 as original
p=original.p
require=p.require


def completed(binding,read):
    require(set(binding)=={'schema','model_exit','audit_exit'} and
        binding['schema']=='minicpm5_layer1_completed_kv_factorial_v1','Explicit completed own-history binding')
    ended=read(binding['model_exit']);ae=read(binding['audit_exit']);audit=read(ae['result'])
    spec=read(ended['spec']);plan=read(spec['plan']);review=read(ended['review']);live=read(ended['live'])
    records={k:read(ref)for k,ref in ended['outputs'].items()}
    require(ended['schema']=='minicpm5_layer1_kv_factorial_actual_exit_v1'
        and ended['actual_exit_observed']is True and type(ended['actual_exit_code'])is int
        and ended['actual_exit_code']==0 and ended['all_four_processes_gone']is ended['cgroup_empty']is True,
        'Actual successful own-history model exit')
    require(ended['owner']==live['owner'] and ended['actual_execution_handle']==live['actual_execution_handle']
        and live['spec']==ended['spec'] and live['review']==ended['review'], 'Actual execution identity')
    require(review['passed']is True and review['blockers']==[] and review['spec']==ended['spec']
        and spec['source_sha256']==plan['source_sha256']==review['source_sha256'], 'Reviewed actual own-history sources')
    require(ae['schema']=='minicpm5_layer1_kv_factorial_post_exit_audit_actual_exit_v1'
        and ae['actual_tool_exit_observed']is True and type(ae['actual_exit_code'])is int and ae['actual_exit_code']==0
        and ae['exact_owner_gone']is ae['serial_lock_reacquired']is True, 'Actual successful independent audit exit')
    require(audit['schema']=='minicpm5_layer1_kv_factorial_post_exit_audit_v1'
        and audit['complete']is audit['every_raw_byte_reread_after_all_model_processes_exited']is True
        and audit['own_history_reproduced']is True and audit['model_exit']==binding['model_exit']
        and ae['owner']==audit['owner'] and ae['lock_identity']==audit['lock_identity']==ended['lock_identity'],
        'Independent post-exit bytes cover this exact model run')
    require(audit['source']['path']==str(Path(__file__).parent/'audit_minicpm5_layer1_kv_factorial_result_v1.py')
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
        and [proof[k]for k in ('ranks','decode_calls','raw_bytes','unique_replicated_output_rows')]==[2,360,5898240,720]
        and proof['own_history_attention_reproduced']is proof['every_output_independently_reread']is True,
        'All360 factorial outputs independently verified')
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
    return dict(factorial_service=proof,owners_checked=bundle['owners'],independent_auditor_gone=True)
