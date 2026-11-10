"""Bind the actual r11 terminal before any layer1 collection. No IO on import.

Source preparation must not call completed(): it authenticates native dependency
files through the original bundle and requires external serial ownership.
"""
import copy
from pathlib import Path
import sys
Q=Path(__file__).resolve().parent;ROOT=Q.parent
sys.path[:0]=[str(ROOT),str(Q)]
import nemotron9b_package_protocol_v26 as p
import nemotron9b_r4_terminal_metadata_v2 as terminal
from minicpm5_after_nemotron_package_v2 import Reader as OriginalReader,merge
from dory_after_nemotron_r9_v2 import extend_observed
from dory_nemotron_r8_retained_failure_v1 import retained_stage
require=p.require
SPEC=dict(path=str(Q/'nemotron9b-package-runtime-v26-r11-20261110-spec.json'),
 sha256='6dbf28f32d003f01b7d0a93759ef1494a1bca9f78b196dfa2c61ce7cebee7d4a')
REVIEW=dict(path=str(Q/'nemotron9b-package-runtime-v26-r11-20261110-source-review.json'),
 sha256='525932a71cf0c39d8142d549b6a2df2d8fd5b686fe616f0cbc0f838e4ae9a833')
LIVE_OWNER=dict(pid=384527,start_ticks='407012946',ppid=433847,
 argv=[p.PYTHON,'-B',str(ROOT/'nemotron9b_package_controller_v26.py'),'--launch',
       '--spec',SPEC['path'],'--spec-sha256',SPEC['sha256'],
       '--review',REVIEW['path'],'--review-sha256',REVIEW['sha256']],
 cgroup='0::/user.slice/user-1000.slice/session-56.scope')
LIVE_SESSION=74703

class Reader(OriginalReader):
    def __call__(self,ref):
        require(type(ref)is dict and set(ref)in ({'path','sha256'},{'path','sha256','bytes'}),
                'Exact reference, with optional authenticated byte length')
        value=super().__call__({k:ref[k]for k in ('path','sha256')})
        if 'bytes'in ref:
            require(type(ref['bytes'])is int and ref['bytes']==self.sizes[ref['path']],
                    'Original optional byte length retained')
        return value


def scope(spec,review,prior):
    require(spec['schema']=='nemotron9b_package_runtime_v26' and spec['run_id']=='r11-20261110'
        and spec['root_finalized']is True and spec['stages']==list(p.STAGES)
        and len(spec['stages'])==52,'Exact complete current r11 scope')
    require(spec['controller']==str(ROOT/'nemotron9b_package_controller_v26.py')
        and spec['worker']==str(ROOT/'nemotron9b_package_worker_v26.py')
        and spec['control_root']=='/mnt/nvme/nemotron9b-package-r11-20261110-control'
        and spec['evidence_root']=='/mnt/nvme/nemotron9b-package-r11-20261110-evidence'
        and spec['work_root']=='/mnt/nvme/nemotron9b-package-r11-20261110',
        'Exact actual source and NVMe output namespaces')
    require(spec['units']=={s:'trn1-latest-verify-nemotron9b-package-r11-20261110-'+s.replace('_','-')
        for s in (*p.STAGES,'controller')}
        and spec['resources']=={s:p.resources(s)for s in (*p.STAGES,'controller')},
        'All53 original unit and resource contracts')
    require(review['schema']=='nemotron9b_package_runtime_source_review_v26'
        and review['execution_spec_sha256']==SPEC['sha256'] and review['source_sha256']==spec['source_sha256']
        and review['reviewer']=='/root' and review['passed']is review['source_only']is review['independent_review']is True
        and review['blockers']==[],'Actual finalized r11 review')
    require(spec['speedup_denominator']is None and spec['automatic_successor_launch']is False,
            'Completion does not invent a baseline or launch successor')
    require(len(prior['rows'])==1202 and len(prior['processes'])==399
        and prior['completed_mini_saved_tap_r2']==p.startup_history.MINI_EXIT
        and prior['completed_nemotron_preexec_timing']==p.startup_history.TIMING_EXIT
        and prior['nemotron_outcome']['failed_release']is True
        and prior['previous_mini_failure_promoted']is False,'Full actual pre-r11 history including failed attempts')
    require(not set(spec['units'].values())&{x['unit']for x in prior['rows']},'No historical unit reuse')

def terminal_chain(spec,receipt,records,refs,failed_release_receipt=None):
    intent,pre,admission,release,audit=(records[k]for k in ('intent','pre','admission','release','audit'))
    require(intent['external_owner']==LIVE_OWNER and receipt['owner']==LIVE_OWNER,
            'Exactly the launcher actually observed live')
    code=terminal.validate_actual_exit(receipt,dict(owner=LIVE_OWNER),SPEC,REVIEW,LIVE_SESSION)
    require(intent['execution_spec']==admission['execution_spec']==SPEC
        and intent['source_review']==admission['source_review']==REVIEW
        and intent['canonical_sha256']==admission['canonical_sha256']
        and intent['lock_identity']==admission['lock_identity'],'Original runtime admission and serial identity')
    require(pre['launch_intent']==refs['intent'] and set(pre['rows'])==set(spec['units'])
        and pre['rows']['controller']['owner']==admission['controller'],'All53 actual retained stage owners')
    auditors=[LIVE_OWNER]
    if release['successful']:
        require(code==0 and failed_release_receipt is None,'Successful original launcher with no failed-release promotion')
    else:
        require(code!=0 and failed_release_receipt is not None,'Actual failed launch needs actual separate failed release')
        released=failed_release_receipt;owner=released['owner'];tool=released['actual_exit']
        expected=[p.PYTHON,'-B',spec['controller'],'--audit','--release-failed',
          '--spec',SPEC['path'],'--spec-sha256',SPEC['sha256'],
          '--review',REVIEW['path'],'--review-sha256',REVIEW['sha256']]
        require(owner['argv']==expected and type(tool['session_id'])is int and tool['session_id']>0
            and tool['session_id']!=LIVE_SESSION
            and (owner['pid'],owner['start_ticks'])!=(LIVE_OWNER['pid'],LIVE_OWNER['start_ticks']),
            'Distinct directly observed supported v26 failed release')
        require(terminal.validate_actual_exit(released,dict(owner=owner),SPEC,REVIEW,tool['session_id'])==0,
                'Actual successful release tool exit')
        auditors.append(owner)
        if audit.get('recovery_only'):
            original=records['original_audit']
            require(audit['original_audit']==refs['original_audit']
                and original['pre_stop']==refs['pre'] and audit['success_not_promoted']is True,
                'Original audit retained across explicit recovery')
    require(pre['auditor']in auditors,'Actual retained snapshot auditor')
    outcome=terminal.classify(code,release,audit,records.get('terminal'),records.get('failed'),
        refs['pre'],refs['audit'],refs['admission'])
    require(outcome['producer_qualified']is release['successful']
        and outcome['failed_release']is release['failed'],'No failed outcome becomes success')
    return outcome

def extend(prior,spec,rows,summaries,owners,refs,outcome,pins):
    require(len(prior['rows'])==1202 and len(prior['processes'])==399,'Full exact pre-r11 snapshot')
    stages=['controller',*spec['stages']]
    require(len(rows)==len(summaries)==53
        and [x['stage']for x in summaries]==stages
        and [x['unit']for x in rows]==[spec['units'][s]for s in stages],
        'Every53 current stage including unlaunched failures')
    require(len({x['unit']for x in rows})==53
        and not {x['unit']for x in rows}&{x['unit']for x in prior['rows']},'No source history overwritten')
    if outcome['producer_qualified']:
        require(all(not x['never_launched']and x['original_exit_success']is True for x in summaries)
            and all(x['worker_reported_complete']is x['worker_reported_passed']
                    is x['normal_retained_exit_success']is True for x in summaries[1:]),
            'Every successful original phase actually complete')
    require(all(x.get('owner')is not None for x in summaries if not x['never_launched']),
            'Every launched phase has its actual owner')
    processes,variants=extend_observed(prior['processes'],owners)
    for process in processes[len(prior['processes']):]:process['role']='nemotron_r11_recorded_owner'
    observed={(x['pid'],x['start_ticks'])for x in processes}
    require(all((x['owner']['pid'],x['owner']['start_ticks'])in observed
        for x in summaries if x.get('owner')is not None),'Every retained owner included')
    result=copy.deepcopy(prior)
    result.update(schema='minicpm5_after_nemotron_r11_history_v1',
        rows=prior['rows']+copy.deepcopy(rows),processes=processes,complete_count=1255,
        predecessor_source_sha256=copy.deepcopy(pins),completed_nemotron_r11=refs,
        nemotron_r11_outcome=outcome,nemotron_r11_stages=summaries,nemotron_r11_owner_variants=variants,
        nemotron_r11_admitted_directories=[spec[key]for key in ('control_root','evidence_root','work_root')],
        current_physical_release_verified=False,full_model_qualified=False,new_speedup_claim=False)
    return result

def completed(actual_exit_ref,failed_release_exit_ref,guard):
    require(callable(guard),'Explicit external serial admission guard');guard()
    reader=Reader(guard)
    receipt=reader(actual_exit_ref)
    terminal.validate_actual_exit(receipt,dict(owner=LIVE_OWNER),SPEC,REVIEW,LIVE_SESSION)
    p.low.gone(LIVE_OWNER);guard()
    # Original dependency authentication only after actual terminal evidence.
    spec,catalog=p.bundle(SPEC,REVIEW);guard()
    failed_release=reader(failed_release_exit_ref)if failed_release_exit_ref is not None else None
    before=p.startup_history.completed(reader)
    prior=before['history'];scope(spec,reader(REVIEW),prior)
    c=Path(spec['control_root'])
    require(not list(c.glob('failed-release-extension*')),'Custom release needs a reviewed successor')
    release=reader.path(c/'root-release.json')
    require(release['audit']['path']in (str(c/'external-audit.json'),str(c/'external-recovery-audit.json')),
            'Only original or standard recovery audit')
    names=dict(intent='external-launch-intent.json',pre='external-pre-stop.json',admission='admission.json',
               release='root-release.json',audit=Path(release['audit']['path']).name)
    for key,name in (('terminal','terminal.json'),('failed','failed-terminal.json')):
        if (c/name).exists():names[key]=name
    if names['audit']=='external-recovery-audit.json':names['original_audit']='external-audit.json'
    refs={k:terminal.ref(c/name)for k,name in names.items()}
    records={k:reader(ref)for k,ref in refs.items()}
    outcome=terminal_chain(spec,receipt,records,refs,failed_release)
    require(records['admission']['canonical_sha256']==prior['canonical_sha256'],'Original canonical history')
    rows=[];summaries=[]
    for stage in ('controller',*p.STAGES):
        guard()
        row,summary=retained_stage(spec,stage,SPEC,REVIEW,p,c,records['pre']['rows'][stage],read=reader.path)
        rows.append(row);summaries.append(summary)
    for path in sorted(c.glob('*.json')):reader.path(path)
    pins={}
    for additions in (before['source_sha256'],spec['source_sha256'],reader.pins):merge(pins,additions)
    endpoints=dict(actual_exit=actual_exit_ref,failed_release_exit=failed_release_exit_ref,
        execution_spec=SPEC,source_review=REVIEW,control_root=str(c),**refs)
    history=extend(prior,spec,rows,summaries,reader.owners,endpoints,outcome,pins)
    for owner in history['processes']:p.low.gone(owner)
    # Preserve original complete physical drain and independently check r11
    # against its saved identities. A release JSON alone is insufficient.
    inherited=p.predecessor_drain(catalog,spec=spec)
    current=[]
    for stage in ('controller',*p.STAGES):
        guard();saved=records['pre']['rows'][stage]
        value=p.validate_after(spec,stage,SPEC,REVIEW,saved,p.low.properties(spec['units'][stage]))
        require(p.low.terminal(value)and value['ActiveState']in ('inactive','failed'),'Released current unit terminal')
        if saved['owner']is not None:p.low.gone(p.proc_owner(saved['owner']))
        current.append(dict(stage=stage,properties=value,group=p.low.empty_group(p.group(spec,stage))))
    p.low.active_heavy();guard()
    require(p.canonical()==prior['canonical_sha256'],'Canonical unchanged after full drain')
    return dict(history=history,source_sha256=pins,metadata_files=len(reader.sizes),
        metadata_bytes=sum(reader.sizes.values()),physical_drain=dict(inherited=inherited,current=current),
        fresh_physical_drain_verified=True,base_catalog=spec['final_predecessor_catalog'],
        runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)


def recheck(done,guard):
    """Fresh physical evidence immediately before the new job admission."""
    reader=Reader(guard);spec=reader(SPEC);catalog=reader(done['base_catalog'])
    pre=reader(done['history']['completed_nemotron_r11']['pre'])
    require(len(done['history']['rows'])==1255,'Complete current history')
    for owner in done['history']['processes']:guard();p.low.gone(owner)
    inherited=p.predecessor_drain(catalog,spec=spec);current=[]
    for stage in ('controller',*p.STAGES):
        samples=[];saved=pre['rows'][stage]
        for _ in range(2):
            guard()
            value=p.validate_after(spec,stage,SPEC,REVIEW,saved,p.low.properties(spec['units'][stage]))
            require(p.low.terminal(value)and value['ActiveState']in ('inactive','failed'),'Every r11 service released')
            if saved['owner']is not None:p.low.gone(p.proc_owner(saved['owner']))
            samples.append(dict(properties=value,group=p.low.empty_group(p.group(spec,stage))))
        require(samples[0]==samples[1],'Stable repeated physical release')
        current.append(dict(stage=stage,samples=samples))
    p.low.active_heavy();guard()
    return dict(complete=True,services=1255,processes=len(done['history']['processes']),
                inherited=inherited,current=current)
