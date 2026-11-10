"""Metadata-only replay of existing r11 terminal reconstruction.

This successor keeps the original retained_stage/terminal_chain/extend logic.
It removes native source hashing and physical checks from preparation; runtime
must separately drain every returned process and service before admission.
No model/tensor reads or service calls are performed by metadata_completed.
"""
from pathlib import Path
from research.minicpm5_after_nemotron_r11_v1 import (
    p,Reader,terminal,SPEC,REVIEW,LIVE_OWNER,LIVE_SESSION,scope,
    terminal_chain,retained_stage,merge,extend)
require=p.require

def metadata_completed(actual_exit_ref,failed_release_exit_ref,guard):
    require(callable(guard),'Explicit external serial admission guard');guard()
    reader=Reader(guard)
    receipt=reader(actual_exit_ref)
    terminal.validate_actual_exit(receipt,dict(owner=LIVE_OWNER),SPEC,REVIEW,LIVE_SESSION)
    guard() # Process existence is checked by the later serial physical drain.
    # Original dependency authentication only after actual terminal evidence.
    spec=reader(SPEC);catalog=reader(spec['final_predecessor_catalog']);guard()
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
    return dict(history=history,source_sha256=pins,metadata_files=len(reader.sizes),
        metadata_bytes=sum(reader.sizes.values()),base_catalog=spec['final_predecessor_catalog'],
        fresh_physical_drain_verified=False,runtime_admitted=False,full_model_qualified=False,new_speedup_claim=False)
