"""Authenticate actual collection and separate saved-audit completion metadata.

No payload reads, process launches or admission. Physical checks must run again
under the new caller's serial lock; this metadata proof alone never starts work.
"""
from pathlib import Path
import re
from research import minicpm5_layer1_metric_diagnostic_runner_v1 as audit
from research import minicpm5_layer1_decode_stream_v2 as stream
p=audit.p;collector=audit.collector;require=stream.require
AUDIT_PYTHON='/usr/bin/python3'


def validate_audit_exit(ended,live,spec,review,terminal,refs):
    require(ended['schema']=='minicpm5_layer1_metric_diagnostic_actual_exit_v1'
        and ended['reviewer']=='/root'and ended['actual_exit_observed']is True
        and type(ended['actual_exit_code'])is int and ended['actual_exit_code']==0,
        'Actual external successful saved-audit tool exit')
    require(live['schema']=='minicpm5_layer1_metric_diagnostic_actual_live_v1'
        and live['reviewer']=='/root'and live['actual_identity_verified']is True,
        'Independent actual saved-audit live identity')
    handle=ended['actual_execution_handle']
    require(set(handle)=={'kind','session_id'}and handle['kind']=='tool_session'
        and type(handle['session_id'])is int and handle['session_id']>0
        and handle==live['actual_execution_handle']
        and type(ended['tool_chunk'])is str and re.fullmatch('[a-f0-9]{6,64}',ended['tool_chunk']),
        'Same actual saved-audit tool session and exit chunk')
    require(set(refs)=={'spec','review','terminal'}and all(ended[k]==refs[k]for k in refs)
        and live['spec']==refs['spec']and live['review']==refs['review']
        and ended['owner']==live['owner']==terminal['owner'],'Actual saved-audit reference and owner bindings')
    expected=[AUDIT_PYTHON,'-B',audit.SELF]
    for key in ('spec','review'):expected+=['--'+key,refs[key]['path'],'--'+key+'-sha256',refs[key]['sha256']]
    require(ended['owner']['argv']==expected,'Direct original stdlib audit interpreter and argv')
    require(spec['schema']=='minicpm5_layer1_metric_diagnostic_execution_spec_v1'and spec['runner']==audit.SELF
        and spec['limits']==audit.LIMITS and refs['terminal']['path']==spec['output'],
        'Original full separate saved-audit specification and limits')
    require(review['schema']=='minicpm5_layer1_metric_diagnostic_runtime_review_v1'
        and review['reviewer']=='/root'and review['passed']is True and review['blockers']==[]
        and review['spec']==refs['spec']and review['source_sha256']==spec['source_sha256'],
        'Exact reviewed actual saved-audit source closure')
    require(terminal['schema']=='minicpm5_layer1_metric_diagnostic_terminal_v1'and terminal['complete']is True
        and terminal['spec']==refs['spec']and terminal['review']==refs['review']
        and terminal['collector_exit']==spec['collector_exit']
        and terminal['lock_identity']==ended['lock_identity']
        and terminal['self_exit_claimed']is False and ended['actual_owner_gone']is True,
        'Completed independent audit plus actual observed process exit')
    require(terminal['requires_separate_runtime_admission']is False
        and terminal['full_model_qualified']is terminal['new_speedup_claim']is False,
        'Completed diagnostic audit is not model qualification')
    require(terminal['diagnostic_only']is True and terminal['audit_passed']is False
        and terminal['result']['schema']=='minicpm5_layer1_saved_data_diagnostic_v1'
        and terminal['result']['audit_passed']is False,
        'Diagnostic completion must retain the original scalar tolerance failure')
    return dict(owner=ended['owner'],lock_identity=ended['lock_identity'],result=terminal['result'])


def completed(binding,read_ref):
    require(callable(read_ref),'Authenticated metadata reader')
    require(set(binding)=={'schema','collector_exit','diagnostic_exit','metric_equivalence'}
        and binding['schema']=='minicpm5_layer1_completed_collection_reviewed_metrics_v1',
        'Only actual completed collector and saved-audit exit references')
    ended=read_ref(binding['diagnostic_exit']);live=read_ref(ended['live'])
    refs={key:ended[key]for key in ('spec','review','terminal')}
    spec=read_ref(refs['spec']);review=read_ref(refs['review']);terminal=read_ref(refs['terminal'])
    proof=validate_audit_exit(ended,live,spec,review,terminal,refs)
    require(spec['collector_exit']==binding['collector_exit'],'Audit covers this exact actual collected trajectory')
    ce=read_ref(binding['collector_exit']);cl=read_ref(ce['live'])
    cs=read_ref(ce['spec']);cr=read_ref(ce['review']);cp=read_ref(cs['plan'])
    output_refs=ce['outputs'];records={key:read_ref(value)for key,value in output_refs.items()}
    chain=audit.validate_chain(ce,cl,cs,cr,cp,records,output_refs)
    require(all(spec['source_sha256'].get(path)==sha for path,sha in cs['source_sha256'].items())
        and spec['canonical_sha256']==cs['canonical_sha256'],
        'Independent audit preserved every collector source and canonical state')
    require(proof['lock_identity']==ce['lock_identity'],'Original single serial lock inode throughout')
    require(len({(owner['pid'],owner['start_ticks'])for owner in [*chain['owners'],proof['owner']]})==5,
        'Five distinct observed collector and independent audit processes')
    from research.minicpm5_layer1_reviewed_metric_evidence_v1 import validate_equivalence
    checked,extra_owners=validate_equivalence(binding,terminal,ce,cs,cp,read_ref)
    stream.audit_records(checked)
    return dict(binding=binding,collector_spec=cs,collector_review=cr,collector_plan=cp,
        collector_exit=ce,collector_chain=chain,saved_audit_spec=spec,saved_audit_review=review,
        saved_audit_terminal=terminal,saved_audit_owner=proof['owner'],saved_audit_exit=ended,
        lock_identity=proof['lock_identity'],audit_result=checked,additional_owners=extra_owners,metadata_complete=True,
        physical_drain_still_required=True,full_model_qualified=False,new_speedup_claim=False)


def physical(bundle,guard):
    require(callable(guard),'Actual serial lease guard');guard()
    for owner in [*bundle['collector_chain']['owners'],bundle['saved_audit_owner'],*bundle.get('additional_owners',[])]:p.gone(owner)
    chain=bundle['collector_chain']
    proof=collector.verify_owned_drain(bundle['collector_spec'],chain['admission_ref'],
        chain['admission']['attempt_id'],chain['coordinator'],True)
    guard();return proof
