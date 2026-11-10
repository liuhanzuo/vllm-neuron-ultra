"""Authenticated post-r11/Mini/storage completion for the fresh v27 package.

Preparation reads source and JSON metadata only; physical checks are separate
and run with the original serial lock. No archive/model payload is reopened.
"""
from pathlib import Path
import os
import re
from research import nemotron9b_r12_r11_metadata_v1 as r11
from research import closed_journals_after_official_history_completion_v1 as mini
from research import minicpm5_layer1_after_saved_audit_v3 as layer1
from research import minicpm5_layer1_after_own_history_v1 as own
from research import minicpm5_layer1_after_cross_history_v1 as crossed
from research import minicpm5_layer1_after_kv_factorial_v1 as factorial
from research import minicpm5_layer1_after_native_cache_v1 as native
from research import nemotron9b_r12_current_history_completion_v1 as current_history

ROOT=Path(__file__).resolve().parent;Q=ROOT/'research'
require=r11.require
def pin(name,digest):return dict(path=str(Q/name),sha256=digest)
HANDOFF=pin('continuation-handoff-20261110-v425-mini-current-history-published.json',
    '6e3f8c89ff55f18423269b8d46aa15ce48669b9c710277d980feae5f5371b0fd')
CURRENT_EXIT=pin('minicpm5-layer1-current-history-factorial-r1-actual-exit-20261110-v1.json',
    '89e1fe4ccf37d4915c3503c88ebc3dffd32e783a2e97507917ae072c984d6b92')
CURRENT_AUDIT_EXIT=pin('minicpm5-layer1-current-history-factorial-post-exit-audit-actual-exit-20261110-v1.json',
    'f662695a26562ba01e667be21cf81d1b86910f13f1ecb3f29395ce8221b3cd81')
STORAGE_EXIT=pin('closed-journals-after-r11-actual-exit-20261110-v3.json',
    '214140449560cdbc8f879e86308e41f741151997501473aec031ec5c83d11e86')
STORAGE_AUDIT_EXIT=pin('closed-journals-after-r11-post-exit-audit-actual-exit-20261110-v1.json',
    '89df8f9f63cf5c25f870e9361091edbe9e3a1d78554825dc8a54b0d77efe431a')
R11_EXIT=pin('nemotron9b-r11-actual-external-exit-20261110-v1.json',
    '827c24eb1ef05e4ebd6aa6555c57bdc3dfe7fbf998e7e833f4b99eb06ed55c94')
R11_RELEASE=pin('nemotron9b-r11-failed-release-actual-exit-20261110-v1.json',
    '34b93488cc80ba911f95f4f3519b1e60f9fa4e832701e54a91651e00cffa0c8d')
BINDING=dict(terminal_handoff=HANDOFF,previous_spec=r11.SPEC,previous_review=r11.REVIEW,
    previous_failed_exit=R11_EXIT,previous_failed_release=R11_RELEASE,
    storage_actual_exit=STORAGE_EXIT,storage_independent_audit_exit=STORAGE_AUDIT_EXIT,
    current_history_exit=CURRENT_EXIT,current_history_audit_exit=CURRENT_AUDIT_EXIT)


def latest_terminal():
    latest=max(Q.glob('continuation-handoff-*.json'),key=lambda path:path.stat().st_mtime_ns)
    require(str(latest)==HANDOFF['path'],'Rebind newest completed model/auditor chain before fresh r12 preparation or launch')


def metadata(read,guard=lambda:None):
    pins={}
    def load(ref):
        guard();value=read(ref);r11.merge(pins,{ref['path']:ref['sha256']});return value
    handoff=load(HANDOFF)
    require(handoff['live_tool_sessions']==[] and handoff['active_heavy_units']==[],
        'Exact latest terminal handoff')
    storage=handoff['storage_maintenance']
    require(storage['actual_exit']==STORAGE_EXIT and storage['independent_audit_exit']==STORAGE_AUDIT_EXIT,
        'Actual completed storage and its independent reader bound')
    ended=load(STORAGE_EXIT);audit_exit=load(STORAGE_AUDIT_EXIT)
    for row,flag in [(ended,'actual_exit_observed'),(audit_exit,'actual_tool_exit_observed')]:
        require(row[flag]is True and row['exact_owner_gone']is True and row['serial_lock_reacquired']is True
            and row['reviewer']=='/root','Actual separately observed storage exits')
        mini.handle(row)
    audit=load(audit_exit['result']);result=load(ended['result'])
    require(audit['complete']is True and audit['actual_exit']==STORAGE_EXIT
        and audit['owner']==audit_exit['owner'] and audit['lock_identity']==ended['lock_identity']==audit_exit['lock_identity']
        and audit['all_original_bytes_reread_after_maintenance_exit']is True and audit['canonical_unchanged']is True,
        'Independent complete post-exit archive reread')
    require(audit['original_bytes']==805306368 and audit['compressed_bytes']==95403896
        and len(audit['files'])==6 and all(x['full_decompression_verified']is True and x['original_absent']is True for x in audit['files']),
        'All six exact originals retained losslessly')
    plan=load(result['plan']);review=load(result['review'])
    require(result['complete']is True and result['owner']==ended['owner'] and plan['serial_lock_identity']==ended['lock_identity']
        and review['passed']is True and review['blockers']==[] and review['plan']==result['plan']
        and review['source_sha256']==plan['source_sha256'],'Exact successful reviewed storage execution')
    require(plan['canonical_sha256']==handoff['canonical_sha256'],'Unchanged original scientific scope')
    require(handoff['current_history_factorial']['actual_exit']==CURRENT_EXIT
        and handoff['current_history_factorial']['independent_audit_exit']==CURRENT_AUDIT_EXIT,
        'Exact latest current/history model and independent auditor terminals')
    current=current_history.current(handoff['current_history_factorial'],handoff['official_history'],
        storage,ended['lock_identity'],load)
    model=load(handoff['official_history']['actual_exit']);model_spec=load(model['spec'])
    require(model_spec['nemotron_completion']==dict(actual_exit=R11_EXIT,failed_release_exit=R11_RELEASE),
        'Latest model follows this exact failed r11 and release')
    model_admission=load(model['outputs']['admission']);drains=load(model['outputs']['drains'])
    require(model_admission['predecessors_checked']is True and model_admission['complete_history_services']==1255
        and model_admission['complete_history_processes']==806 and drains['complete']is True
        and drains['services']==1255 and drains['processes']==806,'Full completed post-r11 enrollment')
    previous=r11.metadata_completed(R11_EXIT,R11_RELEASE,guard)
    require(len(previous['history']['rows'])==1255 and len(previous['history']['processes'])==806
        and previous['history']['nemotron_r11_outcome']['failed_release']is True
        and previous['history']['nemotron_r11_outcome']['producer_qualified']is False,
        'Original r11 failure and all1255/806 retained')
    r11.merge(pins,previous['source_sha256'])
    require(previous['history']['canonical_sha256']==handoff['canonical_sha256'],'Same canonical history')
    stages={}
    for name,module in [('completed_layer1_data',layer1),('completed_own_history',own),
                        ('completed_cross_history',crossed),('completed_factorial',factorial),('completed_native_cache',native)]:
        stages[name]=module.completed(model_spec[name],load)
        require(stages[name]['lock_identity']==ended['lock_identity'],'Same serial lock across completed models')
    return dict(source_sha256=pins,handoff=handoff,plan=plan,previous=previous,
        official_exit=model,official_spec=model_spec,official_admission=model_admission,drains=drains,
        stages=stages,storage_exit=ended,storage_audit_exit=audit_exit,lock_identity=ended['lock_identity'],
        current_history=current)


def validate_spec(spec,read):
    require(spec['post_r11_completion']==BINDING,'Fixed completed r11/Mini/storage bindings')
    # This check stays bounded: full metadata reconstruction is a separate
    # preparation/runtime enrollment step, never repeated for each worker.
    for ref in BINDING.values():
        require(spec['source_sha256'].get(ref['path'])==ref['sha256'],'Completed terminal reference in source closure')
    latest=read(HANDOFF)
    require(latest['live_tool_sessions']==[] and latest['active_heavy_units']==[],'Terminal completion cannot be a running handoff')


def physical(spec,read,low,allowed_units=()):
    require(len(allowed_units)<=1 and all(x==spec['units']['controller']for x in allowed_units),
        'Only actual current controller can be exempted')
    validate_spec(spec,read)
    current_owner=low.proc(os.getpid());lock=read(STORAGE_EXIT)['lock_identity']
    def guard():low.lock_proof(current_owner,lock)
    guard();done=metadata(read,guard)
    require(not set(spec['units'].values())&{row['unit']for row in done['previous']['history']['rows']},
        'Fresh r12 cannot reuse any of all1255 historical services')
    for owner in done['previous']['history']['processes']:low.gone(owner)
    # The existing v27 predecessor drain covers all pre-r11 units; add the
    # exact r11 retained identities without repeating the old1202 scans.
    oldspec=read(r11.SPEC);pre=read(done['previous']['history']['completed_nemotron_r11']['pre'])
    actual=[]
    for stage in ('controller',*r11.p.STAGES):
        saved=pre['rows'][stage]
        props=r11.p.validate_after(oldspec,stage,r11.SPEC,r11.REVIEW,saved,low.properties(oldspec['units'][stage]))
        require(low.terminal(props) and props['ActiveState']in ('inactive','failed'),'Released original r11 service')
        actual.append(dict(stage=stage,properties=props,group=low.empty_group(r11.p.group(oldspec,stage))))
    for owner in done['official_admission']['later_owners']:low.gone(owner)
    for name,module in [('completed_layer1_data',layer1),('completed_own_history',own),
                        ('completed_cross_history',crossed),('completed_factorial',factorial),('completed_native_cache',native)]:
        module.physical(done['stages'][name],lambda:None)
    # Recheck additional original Mini service groups carried by the latest
    # authenticated predecessor receipt, including its original legacy chain.
    groups=set()
    def visit(value):
        if isinstance(value,dict):
            if isinstance(value.get('group'),str)and value['group'].startswith(low.GROUP_PREFIX+'trn1-'):
                groups.add(value['group'])
            for child in value.values():visit(child)
        elif isinstance(value,list):
            for child in value:visit(child)
    visit(done['drains']['legacy'])
    for group in sorted(groups):low.empty_group(group)
    plan=dict(done['plan'],completion_handoff=HANDOFF)
    mini_proof=mini.verify(plan,done['handoff'],read,low)
    current_proof=current_history.current(done['handoff']['current_history_factorial'],done['handoff']['official_history'],
        done['handoff']['storage_maintenance'],lock,read,low)
    for row in (done['storage_exit'],done['storage_audit_exit']):low.gone(row['owner'])
    low.active_heavy(allowed_units);guard()
    return dict(r11=actual,latest_mini=mini_proof,current_history=current_proof,storage_owners_gone=True,
        original_r11_services=53,original_history_processes=806,canonical_sha256=done['handoff']['canonical_sha256'])
