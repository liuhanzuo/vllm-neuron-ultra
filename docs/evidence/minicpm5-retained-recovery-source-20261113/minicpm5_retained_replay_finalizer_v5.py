"""Explicit recovery of the prepublication-only interrupted Mini finalizer.

Frozen v3, its spec/review, and its entire three-file history directory remain
unchanged. An exact separate root review admits only this source and catalog_v2.
The v2 composer differs from v1 solely in checking prior pins against the full
authenticated map, which includes both recovered and extension sources.

The old spec is authenticated with every frozen caller.refs predicate before
the final loaded-source check is explicitly widened by the two reviewed pins.
History receives an in-memory view differing ONLY in source_sha256; its resolved
historical map must still equal the old spec. The composer receives the original
spec, and a pure second step chains recovery provenance and supplemental pins.
No frozen caller or History function is patched or bypassed.

No model, raw replay, deletion, new speedup claim, or fabricated actual-tool
receipt is authorized. The missing old finalizer terminal remains unknown.
Import and --help do not perform any runtime operation.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time

Q=Path(__file__).resolve().parent
for directory in (Q,Q.parent):
    if str(directory)not in sys.path:sys.path.insert(0,str(directory))
import minicpm5_retained_replay_contract_v2 as c
import minicpm5_retained_replay_caller_v2 as caller
import minicpm5_retained_replay_catalog_v2 as catalog
from minicpm5_retained_replay_finalizer_v3 import metadata_cap
from minicpm5_retained_replay_observer_v2 import tool,reconcile,drain_phases

SELF=str(c.Q/'minicpm5_retained_replay_finalizer_v5.py')
COMPOSER=str(c.Q/'minicpm5_retained_replay_catalog_v2.py')
OLD_SELF=str(c.Q/'minicpm5_retained_replay_finalizer_v3.py')
EXTRA_SOURCES=(SELF,COMPOSER)
PARTIAL_NAMES=('history-manifest.json','history-part-000.json','history-part-001.json')
IDENTITY_KEYS=('device','inode','size','mtime_ns','ctime_ns')


def short_ref(value):
    result={k:value[k]for k in ('path','sha256')};c.reference(result);return result


def verify_supplement(io,supplement):
    c.require(set(supplement)==set(EXTRA_SOURCES),'Exactly two separately reviewed recovery sources')
    c.require(str(Path(__file__).resolve())==SELF,'Exact executing recovery finalizer path')
    for path,sha in supplement.items():
        c.reference(dict(path=path,sha256=sha))
        c.require(hashlib.sha256(io.raw(path,1<<20)).hexdigest()==sha,'Recovery executable source changed')


def reviewed_sources(spec_ref,review_ref,recovery,original_request_ref,interruption_ref,read,io):
    """All old refs() admission predicates, then explicit supplemental closure.

    Old refs() cannot admit a new __main__. Its exact spec/review/source/accepted
    predicates are repeated here, not falsely called with an altered old review.
    Tests compare this old-prefix behavior to the frozen function.
    """
    spec=read(spec_ref,c.LIMITS['spec_bytes']);c.validate_spec(spec)
    review=read(review_ref,c.LIMITS['spec_bytes'])
    c.require(review==dict(schema='minicpm5_retained_replay_caller_source_review_v2',reviewer='/root',passed=True,
        blockers=[],spec=spec_ref,source_sha256=spec['source_sha256'],launch_authorized=True,deletion_authorized=False),
        'Exact separate root caller launch review; v1 source acceptance alone does not admit execution')
    for path in c.FILES:
        c.require(hashlib.sha256(io.raw(path,1<<20)).hexdigest()==spec['source_sha256'][path],'Actual caller source changed')
    accepted=read(c.ACCEPTED)
    c.require(accepted['passed']is True and accepted['source_ready_for_caller_integration']is True
        and accepted['production_deletion_admitted']is accepted['launch_authorized']is False,'Retain precise original source-only acceptance')
    supplement=recovery['source_sha256'];name=recovery['recovery_name'];c.names(name)
    c.require(recovery==dict(schema='minicpm5_retained_replay_recovery_source_review_v1',reviewer='/root',passed=True,
        blockers=[],original_request=original_request_ref,interruption=interruption_ref,original_spec=spec_ref,
        original_review=review_ref,source_sha256=supplement,recovery_name=name,launch_authorized=True,
        deletion_authorized=False,model_or_raw_replay_authorized=False),'Exact separate root recovery review')
    c.require(not set(supplement)&set(spec['source_sha256']),'Supplement only adds new sources; no historical replacement')
    verify_supplement(io,supplement)
    full=dict(spec['source_sha256'],**supplement)
    io.loaded_local_sources(full)
    return spec,supplement,full


def validate_interruption(value,request_ref,request,observation,spec):
    c.require(value['schema']=='minicpm5_retained_finalizer_incomplete_observation_v1'
        and value['reviewer']=='/root'and short_ref(value['request'])==request_ref
        and value['original_lock_identity']==c.LOCK,'Exact interrupted request and original lock')
    c.require(value['actual_exit_not_observed']is value['termination_cause_unknown']is True
        and value['actual_exit_code']is value['actual_terminal_chunk']is None
        and value['final_publication_complete']is value['deletion_performed']is value['new_speedup_claim']is False
        and value['partial_history_preserved']is value['replay_completed_and_independently_equal']is True
        and value['observed_lock_rows']==value['exact_finalizer_argv_matches']==[],
        'Retain only the observed incomplete publication; no invented finalizer terminal')
    c.require(set(value['tool'])=={'kind','session_id'}and value['tool']['kind']=='tool_session'
        and type(value['tool']['session_id'])is int and value['tool']['session_id']>0
        and value['tool']['session_id']not in (request['actual_observer_tool']['session_id'],
            observation['actual_outer_exit']['actual_tool']['session_id']), 'Interrupted publisher is not a replay or observer tool')
    c.require(short_ref(value['running_handoff'])==observation['running']
        and {k:observation['running_identity'][k]for k in IDENTITY_KEYS}==value['running_handoff']['identity'],
        'Original running handoff identity remains retained')
    c.require(value['finalizer_source']['path']==OLD_SELF
        and value['finalizer_source']['sha256']==spec['source_sha256'][OLD_SELF], 'Exact frozen interrupted v3 source')
    outputs=[spec['publication_paths'][key]for key in ('outer_actual_exit','observer_actual_exit','successor_catalog')]+[spec['terminal_handoff_path']]
    c.require(value['all_four_publication_paths_absent']==outputs,'Interrupted publication stopped before all four final outputs')
    rows=value['partial_final_history'];root=Path(spec['output_root'])/'final-publication'
    c.require(type(rows)is list and [r['path']for r in rows]==[str(root/n)for n in PARTIAL_NAMES],
        'Exact entire abandoned three-file history namespace')
    for row in [value['request'],value['finalizer_source'],value['running_handoff'],
                value['successful_replay_actual_tool'],value['successful_observer_actual_tool'],*rows]:
        c.require(set(row)=={'path','sha256','bytes','identity'}and set(row['identity'])==set(IDENTITY_KEYS)
            and row['identity']['size']==row['bytes']and type(row['bytes'])is int and 0<row['bytes']<=8<<20,
            'Complete original singleton byte identity')
        short_ref(row)
    return rows


def check_recorded(io,row):
    ident=io.identity(row['path'])
    c.require({k:ident[k]for k in IDENTITY_KEYS}==row['identity'],'Interrupted evidence identity changed')
    data=io.raw(row['path'],row['bytes'])
    c.require(len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256']
        and io.identity(row['path'])==ident,'Interrupted evidence exact bytes changed')


def partial_directory(io,root,rows,expected=None):
    root=Path(root)
    c.require(root.resolve()==root and not any(p.is_symlink()for p in (root,*root.parents)),
        'Canonical nonsymlink abandoned history directory')
    before=root.lstat();c.require(stat.S_ISDIR(before.st_mode),'Abandoned history directory remains a directory')
    identity={k:getattr(before,k)for k in ('st_dev','st_ino','st_mode','st_mtime_ns','st_ctime_ns')}
    c.require(expected is None or identity==expected,'Abandoned history directory identity changed')
    names=[]
    with os.scandir(root)as entries:
        for entry in entries:
            names.append(entry.name);c.require(len(names)<=3,'No extra abandoned history entry or transient')
    c.require(sorted(names)==list(PARTIAL_NAMES),'Entire abandoned history retains exactly three files')
    for row in rows:check_recorded(io,row)
    after=root.lstat()
    c.require(all(getattr(after,k)==v for k,v in identity.items()),'Abandoned directory changed during validation')
    return identity


def old_publisher_gone(io):
    """Bounded exact-argument inspection under the original retained lock."""
    count=0
    with os.scandir('/proc')as entries:
        for entry in entries:
            count+=1;c.require(count<=65536,'Bounded complete process listing')
            if entry.name.isdigit():
                actual=io.process_api().proc(int(entry.name))
                c.require(actual is None or OLD_SELF not in actual['argv'],'Interrupted v3 publisher must be absent')


def rebind(composed,spec,supplement,provenance,read,guard):
    """Pure recovery chaining after the otherwise unchanged v2 composition."""
    from research import minicpm5_layer1_operator_sdk_runtime_completion_v4 as latest
    c.require(set(supplement)==set(EXTRA_SOURCES)and not set(supplement)&set(spec['source_sha256']),
        'Pure rebinding appends precisely the reviewed recovery source pair')
    for path,sha in supplement.items():c.reference(dict(path=path,sha256=sha))
    result=copy.deepcopy(composed);objects=result['objects']
    outer,independent,new,idle=[value for _,value in objects]
    pins=dict(spec['source_sha256'],**supplement)
    delta=dict(result['source_delta'],**supplement);added=dict(result['new_source_delta'],**supplement)
    for value in (outer,independent):
        value['final_publication']['recovery']=copy.deepcopy(provenance)
        value.update(complete_source_map_sha256=c.sha(c.canonical(pins)),source_delta_sha256=c.sha(c.canonical(delta)),
            source_delta_sources=len(delta),full_sources=len(pins),newly_recorded_sources=len(added))
    paths=spec['publication_paths'];outer_ref=c.publication_ref(paths['outer_actual_exit'],outer)
    independent['actual_outer_exit']=outer_ref;observer_ref=c.publication_ref(paths['observer_actual_exit'],independent)
    completed=idle['minicpm5_retained_replay_completed']
    completed['final_publication']['recovery']=copy.deepcopy(provenance)
    completed.update(actual_outer_terminal=outer_ref,actual_observer_terminal=observer_ref)
    idle_ref=c.publication_ref(spec['terminal_handoff_path'],idle)
    new['current_handoff']=idle_ref
    new['runs'][-2].update(terminal=outer_ref,source_sha256=copy.deepcopy(added))
    new['runs'][-1]['terminal']=observer_ref
    c.require(new['runs'][:-2]==composed['objects'][2][1]['runs'][:-2]
        and new['runs'][-1]['source_sha256']=={},'All prior rows unchanged; observer never duplicates source pins')
    catalog_ref=c.publication_ref(paths['successor_catalog'],new)
    objects=[(outer_ref,outer),(observer_ref,independent),(catalog_ref,new),(idle_ref,idle)]
    pending={ref['path']:(ref,value)for ref,value in objects}
    c.require(len(pending)==4 and not set(pending)&set(pins),'Disjoint complete recovery publication chain')
    def virtual_read(ref):
        if ref['path']not in pending:return read(ref)
        expected,value=pending[ref['path']];c.require(ref==expected,'Exact recovery publication reference');return copy.deepcopy(value)
    checked=latest.serial_catalog(dict(serial_predecessors=catalog_ref,source_sha256=pins),virtual_read,guard)
    c.require(checked['runs']==new['runs']and checked['current_handoff']==idle_ref,
        'Original downstream consumer accepts all recovered and supplemental source pins')
    result.update(objects=objects,outer_actual_exit=outer_ref,observer_actual_exit=observer_ref,catalog=catalog_ref,
        idle_handoff=idle_ref,source_delta=delta,new_source_delta=added,full_sources=len(pins))
    guard();return result


def publish_completed(request_ref):
    io,_=caller.api();observed={};spec=None
    def read_saved(ref,cap=None):
        c.metadata_reference(ref);path=ref['path'];before=io.identity(path)
        if cap is None:cap=metadata_cap(ref,spec)
        c.require(before['size']<=cap,'Bounded finalization metadata; no arbitrary 64 MiB JSON')
        c.require('bytes'not in ref or before['size']==ref['bytes'],'Exact optional metadata reference byte length')
        if spec is not None and path in spec['exact_envelopes']:
            c.require(before['size']==spec['exact_envelopes'][path]['bytes'],'Exact source envelope size')
        c.require(path not in observed or observed[path]==before,'No finalization input identity rebound')
        value=io.read(short_ref(ref),cap)
        c.require(io.identity(path)==before,'Finalization input changed during authenticated read')
        observed[path]=copy.deepcopy(before);return value
    def metadata_unchanged():
        for path,ident in observed.items():
            c.require(io.identity(path)==ident,'Finalization input metadata changed after its first read')
    recovery_request=read_saved(request_ref,c.LIMITS['ordinary_metadata_bytes'])
    c.require(set(recovery_request)=={'original_request','interruption','recovery_review','updated_at'},'Exact root recovery request')
    for name in ('original_request','interruption','recovery_review'):c.reference(recovery_request[name])
    c.require(type(recovery_request['updated_at'])is str and 0<len(recovery_request['updated_at'])<=128,'Finite recovery timestamp')
    request=read_saved(recovery_request['original_request'])
    c.require(set(request)=={'observation','observer_live','actual_observer_tool','updated_at'},'Exact original final root request')
    observation=read_saved(request['observation']);live=read_saved(request['observer_live']);tool(request['actual_observer_tool'])
    interruption=read_saved(recovery_request['interruption']);review=read_saved(recovery_request['recovery_review'])
    spec,supplement,full=reviewed_sources(observation['spec'],observation['review'],review,
        recovery_request['original_request'],recovery_request['interruption'],read_saved,io)
    rows=validate_interruption(interruption,recovery_request['original_request'],request,observation,spec)
    for row in [interruption['request'],interruption['finalizer_source'],interruption['running_handoff'],
                interruption['successful_replay_actual_tool'],interruption['successful_observer_actual_tool']]:
        check_recorded(io,row)
        ident=io.identity(row['path'])
        c.require({k:ident[k]for k in IDENTITY_KEYS}==row['identity'],'Recorded identity remains unchanged after byte check')
        c.require(row['path']not in observed or observed[row['path']]==ident,'Recorded input cannot rebind an earlier read')
        observed[row['path']]=copy.deepcopy(ident)
    outer_receipt=read_saved(short_ref(interruption['successful_replay_actual_tool']))
    observer_receipt=read_saved(short_ref(interruption['successful_observer_actual_tool']))
    c.require(outer_receipt['schema']==observer_receipt['schema']=='root_actual_tool_terminal_v1'
        and outer_receipt['actual_tool']==observation['actual_outer_exit']['actual_tool']
        and outer_receipt['outer_terminal']==observation['outer_terminal']
        and observer_receipt['actual_tool']==request['actual_observer_tool']
        and observer_receipt['observation']==request['observation']
        and observer_receipt['full_replay_equality']==observation['full_replay_equality'],
        'Interrupted receipt retains the real successful outer and independent observer tools')
    admission=read_saved(observation['admission']);terminal=read_saved(observation['outer_terminal']);previous=read_saved(observation['running'])
    c.require(reconcile(terminal,previous,observation['actual_outer_exit']['actual_tool'])==observation['actual_outer_exit'],
        'Original actual outer terminal and every phase exit remain bound')
    c.require(observation['actual_outer_exit']['passed']is True and request['actual_observer_tool']['exit_code']==0,
        'This interrupted prepublication recovery retains the independently successful replay')
    c.require(request['observation']['path']==str(Path(spec['observer_root'])/'observation.json')
        and request['observer_live']==observation['independent_live'],'Exact accepted observer evidence')
    expected=dict(path=observation['running']['path'],identity=observation['running_identity'])
    c.require(observed[expected['path']]==expected['identity'],'Exact independently observed running identity')
    io.process_api().gone(observation['observer_owner'])
    old_root=Path(spec['output_root'])/'final-publication'
    partial_identity=partial_directory(io,old_root,rows)
    root=Path(spec['output_root'])/('final-publication-recovery-'+review['recovery_name'])
    outputs=[spec['publication_paths'][key]for key in ('outer_actual_exit','observer_actual_exit','successor_catalog')]+[spec['terminal_handoff_path']];published=[]
    def outputs_unchanged():
        written={r['path']for r in published}
        c.require(all(not os.path.lexists(p)for p in outputs if p not in written),'All remaining final outputs must stay absent')
        c.require(all(not os.path.lexists(p+'.publishing')for p in outputs),'No abandoned or concurrent output transient')
        for ref in published:read_saved(ref,c.LIMITS['ordinary_metadata_bytes'])
        partial_directory(io,old_root,rows,partial_identity)
    c.require(not os.path.lexists(root),'Fresh distinct recovery history namespace')
    outputs_unchanged()
    from minicpm5_retained_replay_history_v1 import History,Authority
    from qwen35_package_preflight_serial_v2 import save_history
    from research.minicpm5_layer1_drain_hold_v1 import retain_until_drained
    lock=io.RetainedLock();lock.acquire();started=time.monotonic();last=[-1.0]
    def guard(force=False):
        lock.check();c.require(time.monotonic()-started<7200,'Finite final physical publication');metadata_unchanged()
        c.require(io.identity(expected['path'])==expected['identity'],'Same final running metadata identity')
        if force or time.monotonic()-last[0]>=.5:
            c.require(caller.latest_observation()==expected,'Unknown newer handoff before final publication')
            io.process_api().gone(observation['observer_owner']);old_publisher_gone(io)
            caller.source_unchanged(spec);verify_supplement(io,supplement);caller.storage(spec)
            outputs_unchanged();last[0]=time.monotonic()
    def drained():
        lock.check();io.process_api().gone(observation['observer_owner'])
        return drain_phases(spec,observation['admission'],admission,terminal)
    def checked_drain():
        outputs_unchanged();value=drained();outputs_unchanged();return value
    try:
        guard(True);checked_drain()
        live_event=dict(event='mini_retained_recovery_publisher_live',owner=copy.deepcopy(lock.owner),
            request=copy.deepcopy(request_ref),lock_identity=copy.deepcopy(c.LOCK),
            interrupted_finalizer_exit_code=None,new_speedup_claim=False)
        encoded=json.dumps(live_event,sort_keys=True,allow_nan=False)
        c.require(len(encoded.encode())<=65536,'Bounded actual recovery owner event')
        print(encoded,flush=True)
        history_spec=copy.deepcopy(spec);history_spec['source_sha256']=copy.deepcopy(full)
        history=History(history_spec,observation['spec'],observation['review'],spec['units'],guard).resolve()
        c.require(history.resolved['plan']['source_sha256']==spec['source_sha256'],
            'Complete resolved historical plan remains exactly the original reviewed source map')
        guard(True);pair=history.physical_pair(Authority(history,guard,lambda:[]));guard(True);history.unchanged()
        root.mkdir(mode=0o700);history_refs=save_history(root,pair)
        provenance=dict(schema='minicpm5_retained_replay_prepublication_recovery_v1',request=request_ref,
            original_request=recovery_request['original_request'],interruption=recovery_request['interruption'],
            recovery_review=recovery_request['recovery_review'],source_sha256=copy.deepcopy(supplement),
            original_source_map_sha256=c.sha(c.canonical(spec['source_sha256'])),original_source_count=len(spec['source_sha256']),
            recovery_history_root=str(root),preserved_abandoned_history=copy.deepcopy(rows),
            interrupted_finalizer_actual_exit_observed=False,interrupted_finalizer_exit_code=None,
            interrupted_finalizer_terminal_chunk=None,interrupted_finalizer_termination_cause_unknown=True,
            deletion_authorized=False,model_or_raw_replay_authorized=False,new_speedup_claim=False)
        finalization=dict(schema='minicpm5_retained_replay_fresh_final_publication_v1',running=observation['running'],
            running_identity=observation['running_identity'],observer_owner=observation['observer_owner'],observer_owner_gone=True,
            actual_observer_tool=request['actual_observer_tool'],final_owned=checked_drain(),complete_history=history_refs,lock_identity=c.LOCK)
        composed=catalog.compose(spec,history.resolved,previous,request['observation'],observation,request['observer_live'],live,
            request['actual_observer_tool'],recovery_request['updated_at'],finalization,admission,terminal,read_saved,lambda:guard(True))
        composed=rebind(composed,spec,supplement,provenance,read_saved,lambda:guard(True))
        history.unchanged();guard(True);checked_drain();history.unchanged();guard(True)
        io.loaded_local_sources(full);outputs_unchanged()
        for ref,value in composed['objects']:
            history.unchanged();guard(True);outputs_unchanged();history.unchanged();guard(True)
            c.require(not os.path.lexists(ref['path']),'Exclusive append-only recovery publication')
            actual=io.once(Path(ref['path']),value)
            c.require(actual==ref,'Published complete bytes match the precomposed recovery chain')
            published.append(actual);read_saved(actual,c.LIMITS['ordinary_metadata_bytes'])
            # The idle is last and intentionally becomes the new latest handoff;
            # do not compare latest discovery to the old running handoff afterward.
            metadata_unchanged();outputs_unchanged();history.unchanged()
        return {k:composed[k]for k in ('outer_actual_exit','observer_actual_exit','catalog','idle_handoff',
            'catalog_rows','full_sources','retained_replay_passed','deletion_authorized','new_speedup_claim')}
    finally:
        proof=retain_until_drained(lambda:lock.check(),drained,
            lambda value:print(json.dumps(dict(event='mini_retained_recovery_publication_drain_blocked',details=value)),flush=True),
            lambda:time.sleep(1))
        lock.release(proof['physically_drained'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);parser.add_argument('--request-sha256',required=True)
    args=parser.parse_args(argv);c.require(sys.dont_write_bytecode and __debug__,'Explicit -B/assertions')
    os.sched_setaffinity(0,{7});os.setpriority(os.PRIO_PROCESS,0,19)
    print(json.dumps(publish_completed(dict(path=args.request,sha256=args.request_sha256))),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
