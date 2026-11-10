"""Metadata completion and separate serial drain for the next SDK diagnostic.

No native import, payload read or launch. Pending r12 receipts never qualify.
Actual observed future terminal/release receipts must be supplied by root after
its live tool exits; no such binding is manufactured by this module.
"""
import copy
from pathlib import Path
import re
from research import nemotron9b_r12_current_history_completion_v1 as current_validator
from research import minicpm5_layer1_after_saved_audit_v3 as collector_completion
from research import minicpm5_layer1_after_native_cache_v1 as native_completion

ROOT=Path('/home/ubuntu/experiments/trn1-latest-models-20261031')
Q=ROOT/'research'
HANDOFF=dict(path=str(Q/'continuation-handoff-20261110-v425-mini-current-history-published.json'),
    sha256='6e3f8c89ff55f18423269b8d46aa15ce48669b9c710277d980feae5f5371b0fd')
CURRENT=dict(schema='minicpm5_layer1_completed_current_history_factorial_v1',
    model_exit=dict(path=str(Q/'minicpm5-layer1-current-history-factorial-r1-actual-exit-20261110-v1.json'),
        sha256='89e1fe4ccf37d4915c3503c88ebc3dffd32e783a2e97507917ae072c984d6b92'),
    audit_exit=dict(path=str(Q/'minicpm5-layer1-current-history-factorial-post-exit-audit-actual-exit-20261110-v1.json'),
        sha256='f662695a26562ba01e667be21cf81d1b86910f13f1ecb3f29395ce8221b3cd81'))


def require(value,message):
    if not value:raise RuntimeError(message)


def reference(value):
    require(type(value)is dict and set(value)=={'path','sha256'}
        and type(value['path'])is str and Path(value['path']).is_absolute()
        and Path(value['path']).suffix=='.json' and type(value['sha256'])is str
        and re.fullmatch('[a-f0-9]{64}',value['sha256']), 'Exact immutable JSON reference')


def require_complete_binding(binding):
    require(type(binding)is dict and set(binding)=={'schema','status','nemotron_r12','current_history','terminal_handoff'}
        and binding['schema']=='minicpm5_layer1_sdk_batch_layout_latest_completion_v1'
        and binding['status']=='complete','Current Nemotron actual terminal required; pending binding rejected')
    require(binding['current_history']==CURRENT,'Exact completed180call Mini model and independent auditor')
    n=binding['nemotron_r12']
    require(type(n)is dict and set(n)=={'schema','actual_exit','failed_release_exit'}
        and n['schema']=='minicpm5_after_nemotron_r12_binding_v1','Explicit actual r12 completion binding')
    reference(n['actual_exit'])
    if n['failed_release_exit']is not None:reference(n['failed_release_exit'])
    reference(binding['terminal_handoff'])
    require(binding['terminal_handoff']!=HANDOFF,'Pre-r12 Mini terminal cannot admit a post-r12 diagnostic')
    return binding


def current_history(binding,read,guard):
    require(binding==CURRENT,'Exact completed current/history model and audit')
    guard();handoff=read(HANDOFF);h=handoff['current_history_factorial']
    require(h['actual_exit']==binding['model_exit'] and h['independent_audit_exit']==binding['audit_exit'],
        'Fixed terminal handoff binds both actual exits')
    ended=read(binding['model_exit']);lock=ended['lock_identity']
    proof=current_validator.current(h,handoff['official_history'],handoff['storage_maintenance'],lock,read)
    spec=read(ended['spec']);plan=read(spec['plan'])
    records={key:read(ref)for key,ref in ended['outputs'].items()}
    audit_exit=read(binding['audit_exit']);audit=read(audit_exit['result'])
    require(proof['complete_model_and_independent_auditor_authenticated']is True
        and proof['physical_checks_performed']is False,'Full actual metadata chain, no fake physical proof')
    guard()
    return dict(binding=copy.deepcopy(binding),ended=ended,spec=spec,plan=plan,records=records,
        audit=audit,audit_exit=audit_exit,owners=proof['owners'],lock_identity=lock,proof=proof)


def merge_sources(target,addition):
    for path,digest in addition.items():
        require(path not in target or target[path]==digest,'Every original source identity retained')
        target[path]=digest


def exact_owners(*groups):
    owners=[]
    for group in groups:
        for owner in group:
            require(type(owner)is dict and type(owner['pid'])is int and owner['pid']>0
                and type(owner['start_ticks'])is str and owner['start_ticks'].isdigit()
                and int(owner['start_ticks'])>0,'Actual recorded process identity')
            require('argv'not in owner or(type(owner['argv'])is list
                and all(type(arg)is str for arg in owner['argv'])),
                'Preserve historical PID/start-only and empty-argv observations')
            # Keep exec/argv variants sharing PID/starttime; physical gone checks
            # may repeat, but no historical owner variant is silently omitted.
            if owner not in owners:owners.append(copy.deepcopy(owner))
    return owners


def completed(binding,read,guard):
    require_complete_binding(binding)
    require(callable(read)and callable(guard),'Authenticated reader and explicit serial/preparation guard')
    guard();pins={}
    def load(ref):
        guard();value=read(ref);merge_sources(pins,{ref['path']:ref['sha256']});return value
    terminal=load(binding['terminal_handoff'])
    require(terminal['live_tool_sessions']==[] and terminal['active_heavy_units']==[],
        'Terminal handoff cannot contain live serial execution')
    n=terminal['nemotron_r12']
    require(n['actual_exit']==binding['nemotron_r12']['actual_exit']
        and n.get('failed_release_exit')==binding['nemotron_r12']['failed_release_exit'],
        'Latest terminal explicitly binds actual r12 outcome/release')
    h=terminal['current_history_factorial']
    require(h['actual_exit']==CURRENT['model_exit'] and h['independent_audit_exit']==CURRENT['audit_exit'],
        'Terminal retains original current/history lineage')
    # Import only after rejecting pending/missing metadata; this helper is still
    # stdlib source and does not call its physical path from completed().
    from research import minicpm5_after_nemotron_r12_v1 as nemotron
    n_done=nemotron.completed(binding['nemotron_r12'],load,guard)
    current=current_history(binding['current_history'],load,guard)
    collector=collector_completion.completed(current['spec']['completed_layer1_data'],load)
    native=native_completion.completed(current['spec']['completed_native_cache'],load)
    lock=n_done['lock_identity']
    require(current['lock_identity']==collector['lock_identity']==native['lock_identity']==lock,
        'One unchanged serial lock for all actual predecessors')
    require(terminal['canonical_sha256']==n_done['history']['canonical_sha256']==collector['collector_spec']['canonical_sha256'],
        'All original scientific canonical state retained')
    require(n_done['spec']['post_r11_completion']['current_history_exit']==CURRENT['model_exit']
        and n_done['spec']['post_r11_completion']['current_history_audit_exit']==CURRENT['audit_exit'],
        'Actual r12 was admitted after these exact model and audit exits')
    for source_map in (n_done['source_sha256'],current['spec']['source_sha256'],collector['saved_audit_spec']['source_sha256'],native['spec']['source_sha256']):
        merge_sources(pins,source_map)
    cs=collector['collector_spec'];final=load(cs['finalization']);final_exit=load(cs['finalizer_exit'])
    bound_spec=load(final['references']['execution_spec']);catalog=load(cs['catalog']);bound_plan=load(bound_spec['plan'])
    require(final['passed']is True and final['all_planned_launches_accounted_for']is True
        and final['all_groups_drained']is True and final['finalizer_owner']==final_exit['owner']
        and final_exit['actual_exit_observed']is True and final_exit['actual_exit_code']==0,
        'Original finalizer metadata remains complete and actually exited')
    bound=dict(spec=bound_spec,catalog=catalog,plan=bound_plan,finalizer_owner=final_exit['owner'])
    co=[*collector['collector_chain']['owners'],collector['saved_audit_owner'],*collector.get('additional_owners',[])]
    owners=exact_owners(n_done['owners'],current['owners'],native['owners'],co,[final_exit['owner']])
    guard()
    return dict(binding=copy.deepcopy(binding),terminal_handoff=terminal,nemotron=n_done,
        collector=collector,native=native,current_history=current,bound=bound,owners=owners,
        lock_identity=lock,source_sha256=pins,metadata_complete=True,physical_checks_performed=False,
        runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)


def require_latest_terminal(binding):
    """Fresh preparation/outer launch only, never per worker or post-exit audit."""
    require_complete_binding(binding)
    latest=max(Q.glob('continuation-handoff-*.json'),key=lambda path:path.stat().st_mtime_ns)
    require(str(latest)==binding['terminal_handoff']['path'],'Rebind latest actually completed serial terminal before fresh launch')


def physical(done,guard):
    require_complete_binding(done['binding']);require(done['metadata_complete']is True,'Completed metadata first')
    from research import minicpm5_after_nemotron_r12_v1 as nemotron
    guard();proof=nemotron.physical(done['nemotron'],guard)
    low=nemotron.p.low
    for owner in done['owners']:guard();low.gone(owner)
    low.active_heavy();guard()
    return dict(complete=True,nemotron=proof,all_additional_owners_checked=done['owners'],
        lock_identity=done['lock_identity'],all_current_and_prior_services_preserved=True,
        physical_checks_performed=True,full_model_qualified=False,new_speedup_claim=False)
